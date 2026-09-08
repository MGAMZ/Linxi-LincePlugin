"""跨天漂移指标算子（ANALYZE 阶段）：计算逐 session 漂移指标与聚合结果。

`source` 三态：`context` 读流水线已载入的 `ecephys` 槽，`sessions` 直传会话清单，`sweep` 扫描 `data_root`。context 态：逐通道真值列写入 `ecephys[k].channel_summary`，会话级目标指标写入 `context.metrics` 的 `drift_targets` 键（由 `ExportEvalMetrics` 统一写出 eval JSON 侧车）；批量态结果留存于算子实例与日志。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import xarray as xr
from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor
from pynwb import NWBHDF5IO

from ._drift_core import (
    DriftAnalysisResult,
    DriftSessionInput,
    centroid_from_frs,
    compute_drift_metrics,
    cosine,
    parse_session_date,
    session_fr,
)
from .load_lince_session import session_counts_matrix

TARGETS_KEY = "drift_targets"
TASKS = ("MA_CO", "MA_RT")
SUMMARY_FIELDS = (
    "cos_raw_mean_hard", "one_minus_cos_raw_mean_hard", "cos_centered_mean_hard",
    "norm_ratio_mean_hard", "norm_ratio_dev_hard", "pearson_gap_cos",
)


def read_binned_spikes(nwb_path: str | Path) -> np.ndarray:
    """pynwb 只读物化单 session 的 (T, 512) 发放矩阵，行列与 timestamps 不一致时转置纠正。"""
    path = Path(nwb_path)
    with NWBHDF5IO(str(path), "r") as io:
        series = io.read().acquisition["binned_spikes"]
        n_rows = len(np.asarray(series.timestamps[:], dtype=np.float64))
        data = np.asarray(series.data[:])
    if data.ndim == 2 and data.shape[0] != n_rows and data.shape[1] == n_rows:
        data = np.ascontiguousarray(data.T)
    if data.ndim != 2 or data.shape[0] != n_rows:
        raise ValueError(f"binned_spikes shape inconsistent with timestamps: {data.shape} vs {n_rows}: {path}")
    return data


def _single_nwb(directory: Path) -> Path:
    files = sorted(directory.glob("*.nwb"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one NWB in {directory}, found {len(files)}")
    return files[0]


def _day_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"required data directory not found: {root}")
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def discover_sweep_inputs(
    data_root: str | Path,
    tasks: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
) -> list[tuple[str, str, str | None, str, Path]]:
    """按数据根目录布局发现会话。

    返回 (task, level, horizon, session_key, nwb_path)；train = public heldin，
    query = 各级 heldout。
    """
    root = Path(data_root)
    want_tasks = tuple(tasks) if tasks else TASKS
    want_levels = set(levels) if levels else {"train", "easy", "normal", "hard"}
    rows: list[tuple[str, str, str | None, str, Path]] = []
    for task in want_tasks:
        if "train" in want_levels:
            for d in _day_dirs(root / "public" / task / "easy" / "heldin"):
                rows.append((task, "train", None, d.name, _single_nwb(d)))
        if "easy" in want_levels:
            for d in _day_dirs(root / "private" / task / "easy" / "heldout"):
                rows.append((task, "easy", None, d.name, _single_nwb(d)))
        for level in ("normal", "hard"):
            if level not in want_levels:
                continue
            for horizon in _day_dirs(root / "private" / task / level / "heldout"):
                for d in _day_dirs(horizon):
                    rows.append((task, level, horizon.name, d.name, _single_nwb(d)))
    return rows


def _jsonable(v: float | int) -> float | None:
    return None if isinstance(v, float) and not np.isfinite(v) else float(v)


def _session_row(m: Any) -> dict[str, Any]:
    return {
        "task": m.task, "level": m.level, "horizon": m.horizon, "session_key": m.session_key,
        "gap_days": _jsonable(m.gap_days), "cos_raw": _jsonable(m.cos_raw),
        "cos_centered": _jsonable(m.cos_centered), "norm_ratio": _jsonable(m.norm_ratio),
    }


def _attach_channel_summary(slot: Any, prefix: str, fr: np.ndarray, centroid: np.ndarray) -> None:
    cols = {
        f"{prefix}_fr": fr,
        f"{prefix}_centroid_fr": centroid,
    }
    new = xr.Dataset({k: (("channel_id",), v) for k, v in cols.items()}, coords={"channel_id": np.arange(fr.shape[0])})
    if slot.channel_summary is None:
        slot.channel_summary = new
    else:
        slot.channel_summary = slot.channel_summary.assign({k: new[k] for k in new.data_vars})


def _slot_fr(context: Any, key: str) -> np.ndarray:
    """会话发放率向量的输入矩阵：query 侧取根容器 `binned_spikes` 的计数矩阵，support 侧取记录的 `electrophysiology` 信号。"""
    slot = context.ecephys[key]
    x = np.asarray(session_counts_matrix(context, key))
    if x.ndim == 2 and x.shape[1] != slot.channel_count and x.shape[0] == slot.channel_count:
        x = x.T
    if x.ndim != 2:
        raise ValueError(f"ecephys[{key!r}] counts matrix must be (T, C), got {x.shape}")
    return session_fr(x)


@register_as_linxi_processor(stage=PROCESS_STAGES.ANALYZE)
class LinceDriftAnalysis(DefaultProcessor):
    """跨天漂移指标算子。

    Parameters：``source`` = context（读 ecephys 槽）| sessions（直传清单：task/level/
    session_key 必填，horizon 可选，x 为 2-D ndarray 或 nwb_path）| sweep（``data_root``
    必须显式传入，tasks/levels 筛选）。context 态：``target_keys`` 默认
    ["query"]；质心 ``centroid_key``（键或键列表）与 ``centroid``（显式向量）二选一；
    ``session_key``/``train_session_keys`` 目录名算 gap_days，缺任一侧 NaN+告警；
    ``result_prefix`` 为 channel_summary 列名前缀，默认 drift。
    """

    def __init__(
        self,
        source: Literal["context", "sessions", "sweep"] = "context",
        sessions: list[dict[str, Any]] | None = None,
        data_root: str | None = None,
        tasks: list[str] | None = None,
        levels: list[str] | None = None,
        target_keys: list[str] | None = None,
        centroid_key: str | list[str] | None = None,
        centroid: list[float] | None = None,
        session_key: str | None = None,
        train_session_keys: list[str] | None = None,
        result_prefix: str = "drift",
        name: str | None = None,
    ):
        super().__init__(name)
        self.source = source
        self.sessions = sessions
        self.data_root = data_root
        self.tasks = tasks
        self.levels = levels
        self.target_keys = list(target_keys or ["query"])
        self.centroid_key = centroid_key
        self.centroid = centroid
        self.session_key = session_key
        self.train_session_keys = train_session_keys
        self.result_prefix = result_prefix
        self.result: DriftAnalysisResult | None = None

    def _process(self, context: Any) -> Any:
        if self.source == "sessions":
            self._run_batch(context, self._inputs_from_sessions())
        elif self.source == "sweep":
            self._run_batch(context, self._inputs_from_sweep())
        else:
            self._run_context(context)
        return context

    def _inputs_from_sessions(self) -> list[DriftSessionInput]:
        if not self.sessions:
            raise ValueError("LinceDriftAnalysis(source='sessions') requires a non-empty `sessions` list")
        inputs: list[DriftSessionInput] = []
        for spec in self.sessions:
            x = spec.get("x")
            if x is None:
                path = spec.get("nwb_path")
                if path is None:
                    raise ValueError(f"session entry needs `x` or `nwb_path`: {spec.get('session_key')!r}")
                x = read_binned_spikes(path)
            inputs.append(
                DriftSessionInput(
                    task=spec["task"], level=spec["level"], session_key=spec["session_key"],
                    x=np.asarray(x), horizon=spec.get("horizon"),
                )
            )
        return inputs

    def _inputs_from_sweep(self) -> list[DriftSessionInput]:
        if self.data_root is None:
            raise ValueError(
                "LinceDriftAnalysis(source='sweep') requires an explicit `data_root` processor param "
                "(the challenge `challenge_data` directory)."
            )
        rows = discover_sweep_inputs(self.data_root, self.tasks, self.levels)
        return [
            DriftSessionInput(task=t, level=l, horizon=h, session_key=k, x=read_binned_spikes(p))
            for t, l, h, k, p in rows
        ]

    def _run_batch(self, context: Any, inputs: list[DriftSessionInput]) -> None:
        result = compute_drift_metrics(inputs)
        self.result = result
        summary = {f: _jsonable(getattr(result, f)) for f in SUMMARY_FIELDS}
        for m in result.sessions:
            logger.info(f"[drift] {self.source} session: " + json.dumps(_session_row(m), ensure_ascii=False))
        logger.info(f"[drift] {self.source} summary: " + json.dumps(summary, ensure_ascii=False))

    def _context_centroid(self, context: Any) -> np.ndarray:
        if self.centroid is not None:
            return np.asarray(self.centroid, dtype=np.float64).reshape(-1)
        keys = [self.centroid_key] if isinstance(self.centroid_key, str) else list(self.centroid_key or [])
        if not keys:
            raise ValueError("LinceDriftAnalysis(source='context') requires `centroid_key` or `centroid`")
        missing = [k for k in keys if context.ecephys.get(k) is None]
        if missing:
            raise ValueError(f"centroid_key not found in context.ecephys: {missing} (have {sorted(context.ecephys)})")
        return centroid_from_frs([_slot_fr(context, k) for k in keys])

    def _context_gap_days(self) -> float:
        if not self.session_key or not self.train_session_keys:
            logger.warning("[drift] context 模式缺 session_key/train_session_keys，gap_days=NaN")
            return float("nan")
        target = parse_session_date(self.session_key)
        return float((target - max(parse_session_date(k) for k in self.train_session_keys)).days)

    def _run_context(self, context: Any) -> None:
        centroid = self._context_centroid(context)
        gap_days = self._context_gap_days()
        targets: dict[str, Any] = {}
        for key in self.target_keys:
            slot = context.ecephys.get(key)
            if slot is None:
                raise ValueError(f"target_key {key!r} not found in context.ecephys (have {sorted(context.ecephys)})")
            fr = _slot_fr(context, key)
            metrics = {
                "cos_raw": cosine(fr, centroid),
                "cos_centered": cosine(fr - fr.mean(), centroid - centroid.mean()),
                "norm_ratio": float(np.linalg.norm(fr)) / float(np.linalg.norm(centroid))
                if np.linalg.norm(centroid) else float("nan"),
            }
            _attach_channel_summary(slot, self.result_prefix, fr, centroid)
            targets[key] = {"gap_days": _jsonable(gap_days), **{k: _jsonable(v) for k, v in metrics.items()}}
        context.put_metric(TARGETS_KEY, targets)
        logger.info("[drift] context summary: " + json.dumps(targets, ensure_ascii=False))
