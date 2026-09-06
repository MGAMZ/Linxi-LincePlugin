"""赛道 baseline 解码推理算子：预置权重只读加载，adapt + predict + 官方口径 session 评分。

两种数据来源（``source`` 参数）：

* ``context``：消费任务 1 §6 的布局约定 —— ``context.ecephys["query"]`` /
  ``context.ecephys["support"]``（X→electrophysiology，Y→auxiliary_channels
  的 cursor_vel_x / cursor_vel_y，trial 表→events，hard 级缺 support 键即 (0, C)
  空 support 语义）。逐 bin 预测写回 query 记录的
  auxiliary_channels["cursor_vel_pred_x" / "cursor_vel_pred_y"]，session 结果 JSON
  合并进 ``context.notes["lince_baseline_decode"]`` 并以 logger.info 输出机器可读行。
* ``sweep``：按官方评测协议从 ``data_root`` 自动发现并重跑全部（或筛选的）session，
  官方 loader 与模型类均在运行时经 ``baseline_code_path`` sys.path 导入，插件不复制官方源码。

权重目录默认指向数据根预置权重 ``<data_root 上级>/challenge_code/Participant/{WF,GRU}``，
只读加载；每个 session 结束后对权重文件做 sha256 复核，任何改动即报错。
GRU 需要 torch（可选依赖组 ``[gru]``）及其 CUDA 环境；WF 仅 CPU/numpy。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from linshu_format.core import TimeSeries
from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor

from . import decode_engine

QUERY_KEY = "query"
SUPPORT_KEY = "support"
VELOCITY_X = "cursor_vel_x"
VELOCITY_Y = "cursor_vel_y"
PREDICTED_X = "cursor_vel_pred_x"
PREDICTED_Y = "cursor_vel_pred_y"
NOTES_KEY = "lince_baseline_decode"


def _bin_timestamps(record: Any, side: str) -> np.ndarray:
    signal = record.electrophysiology
    if signal is None:
        raise ValueError(f"ecephys[{side!r}] has no electrophysiology signal")
    n_bins = int(signal.data.shape[0])
    if signal.timestamps is not None and len(signal.timestamps) == n_bins:
        return np.asarray(signal.timestamps, dtype=np.float64)
    for key in (VELOCITY_X, VELOCITY_Y):
        aux = record.auxiliary_channels.get(key)
        if aux is not None and aux.timestamps is not None and len(aux.timestamps) == n_bins:
            return np.asarray(aux.timestamps, dtype=np.float64)
    if signal.starting_time is not None and signal.rate:
        return np.asarray(signal.starting_time, dtype=np.float64) + (
            np.arange(n_bins, dtype=np.float64) / float(signal.rate)
        )
    raise ValueError(
        f"ecephys[{side!r}] carries no per-bin time base (timestamps or starting_time+rate); "
        "trial masking cannot be reconstructed"
    )


def _session_arrays(context: Any, side: str):
    record = context.ecephys[side]
    x = np.asarray(record.electrophysiology.data, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ecephys[{side!r}] electrophysiology must be (T, C), got {x.shape}")
    timestamps = _bin_timestamps(record, side)
    aux = record.auxiliary_channels
    missing = [key for key in (VELOCITY_X, VELOCITY_Y) if key not in aux]
    if missing:
        raise ValueError(
            f"ecephys[{side!r}].auxiliary_channels missing {missing}; "
            f"available: {sorted(aux)}"
        )
    y = np.column_stack(
        [np.asarray(aux[VELOCITY_X].data).reshape(-1), np.asarray(aux[VELOCITY_Y].data).reshape(-1)]
    ).astype(np.float32)
    if not (len(x) == len(y) == len(timestamps)):
        raise ValueError(
            f"ecephys[{side!r}] time dimension mismatch: X={len(x)}, Y={len(y)}, "
            f"timestamps={len(timestamps)}"
        )
    events = record.events
    if events is None:
        raise ValueError(f"ecephys[{side!r}] has no events trial table")
    table = events.table
    required = ("start_time", "stop_time", "trial_id")
    absent = [column for column in required if column not in table.data_vars]
    if absent:
        raise ValueError(
            f"ecephys[{side!r}].events.table missing columns {absent}; required: {list(required)}"
        )
    trial_ids = np.full(len(timestamps), -1, dtype=np.int64)
    starts = np.asarray(table["start_time"].values, dtype=np.float64)
    stops = np.asarray(table["stop_time"].values, dtype=np.float64)
    ids = np.asarray(table["trial_id"].values, dtype=np.int64)
    for start, stop, trial_id in zip(starts, stops, ids):
        trial_ids[(timestamps >= start) & (timestamps <= stop)] = int(trial_id)
    return x, y, trial_ids, timestamps


@register_as_linxi_processor(stage=PROCESS_STAGES.POSTPROCESS)
class BaselineDecodeInfer(DefaultProcessor):
    PROCESSOR_NAME = "BaselineDecodeInfer"  # 注册名 = PROCESSOR_NAME（缺省回退类名，registry.py:32-33）；显式声明以稳定 YAML 契约

    def __init__(
        self,
        *,
        model: str = "wf",
        source: str = "context",
        baseline_code_path: str | None = None,
        data_root: str | None = None,
        weights_dir: str = "",
        task: str = "",
        level: str = "",
        horizon: str = "",
        session_key: str = "",
        strict_trial_counts: bool = True,
        name: str | None = None,
    ):
        super().__init__(name)
        if source not in ("context", "sweep"):
            raise ValueError(f"source must be 'context' or 'sweep', got {source!r}")
        self.model = model
        self.source = source
        self.baseline_code_path = baseline_code_path
        self.data_root = data_root
        self.weights_dir = weights_dir
        self.task = task
        self.level = level
        self.horizon = horizon
        self.session_key = session_key
        self.strict_trial_counts = strict_trial_counts
        self.session_results: list[dict[str, Any]] = []
        self.summary: dict[str, Any] | None = None

    def _weights(self) -> str:
        if self.weights_dir:
            return self.weights_dir
        if self.data_root is None:
            raise ValueError(
                "BaselineDecodeInfer requires an explicit `data_root` processor param when `weights_dir` "
                "is empty (preset weights resolve to `<data_root 上级>/challenge_code/Participant/...`); "
                "the plugin ships no machine-local default."
            )
        return decode_engine.default_weights_dir(self.data_root, self.model)

    def _run_sweep(self) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        out = decode_engine.run_sweep(
            model=self.model,
            baseline_code_path=self.baseline_code_path,
            data_root=self.data_root,
            weights_dir=self.weights_dir,
            task=self.task,
            level=self.level,
            horizon=self.horizon,
            session_key=self.session_key,
            strict_trial_counts=self.strict_trial_counts,
        )
        records = [
            row
            for levels in out["task_results"].values()
            for rows in levels.values()
            for row in rows
        ]
        return records, out["summary"]

    def _run_context(self, context: Any) -> list[dict[str, Any]]:
        if not self.task or not self.level:
            raise ValueError("source='context' requires non-empty task and level params")
        if self.baseline_code_path is None:
            raise ValueError(
                "BaselineDecodeInfer requires an explicit `baseline_code_path` processor param (the "
                "challenge `challenge_code` directory containing Platform/ and Participant/); "
                "the plugin ships no machine-local default."
            )
        weights = self._weights()
        base_manifest = decode_engine.weights_manifest(weights, self.model)
        if not base_manifest:
            raise FileNotFoundError(f"no preset weights found below: {weights}")

        def weights_check() -> None:
            if decode_engine.weights_manifest(weights, self.model) != base_manifest:
                raise RuntimeError(f"preset weights changed during evaluation below: {weights}")

        x_query, y_query, query_trial_ids, query_timestamps = _session_arrays(context, QUERY_KEY)
        if self.strict_trial_counts:
            found = decode_engine.count_trials(query_trial_ids)
            expect = decode_engine.LEVEL_TRIALS[self.level][1]
            if found != expect:
                raise ValueError(f"ecephys['{QUERY_KEY}'] expects {expect} trials, found {found}")
        if SUPPORT_KEY in context.ecephys:
            x_support, y_support, support_trial_ids, _ = _session_arrays(context, SUPPORT_KEY)
            if self.strict_trial_counts:
                found = decode_engine.count_trials(support_trial_ids)
                expect = decode_engine.LEVEL_TRIALS[self.level][0]
                if found != expect:
                    raise ValueError(
                        f"ecephys['{SUPPORT_KEY}'] expects {expect} trials, found {found}"
                    )
            overlap = sorted(
                set(np.unique(support_trial_ids[support_trial_ids >= 0]).tolist())
                & set(np.unique(query_trial_ids[query_trial_ids >= 0]).tolist())
            )
            if overlap:
                raise ValueError(f"Support/query trial_id overlap: {overlap[:10]}")
            x_support, y_support = decode_engine.trial_only(x_support, y_support,
                                                            support_trial_ids)
        else:
            x_support = np.empty((0, x_query.shape[1]), dtype=np.float32)
            y_support = np.empty((0, 2), dtype=np.float32)
        query_mask = query_trial_ids >= 0
        x_query, y_query = decode_engine.trial_only(x_query, y_query, query_trial_ids)

        result, prediction = decode_engine.run_session(
            task_name=self.task,
            level=self.level,
            horizon=self.horizon or None,
            session_key=self.session_key or "context",
            x_query=x_query,
            y_query=y_query,
            x_support=x_support,
            y_support=y_support,
            submission_factory=lambda: decode_engine.make_submission(
                self.model, self.baseline_code_path, weights
            ),
            weights_check=weights_check,
        )

        velocity_unit = context.ecephys[QUERY_KEY].auxiliary_channels[VELOCITY_X].unit
        for key, column in ((PREDICTED_X, prediction[:, 0]), (PREDICTED_Y, prediction[:, 1])):
            context.ecephys[QUERY_KEY].auxiliary_channels[key] = TimeSeries(
                name=key,
                data=np.ascontiguousarray(column, dtype=np.float32),
                timestamps=query_timestamps[query_mask],
                unit=velocity_unit,
            )
        return [result]

    def _record_notes(self, context: Any, records: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {"model": self.model, "source": self.source, "sessions": records}
        if self.summary is not None:
            payload["aggregate"] = {
                key: self.summary[key]
                for key in ("final_score", "task_scores", "level_scores")
            }
        merged: dict[str, Any] = {}
        if context.notes:
            try:
                parsed = json.loads(context.notes)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                merged = parsed
            else:
                merged = {"prior_notes": context.notes}
        merged[NOTES_KEY] = payload
        context.notes = json.dumps(merged, ensure_ascii=False)
        for row in records:
            context_note = {
                key: row[key]
                for key in ("session_id", "r2_x", "r2_y", "r2_mean_raw", "r2_mean",
                            "session_score", "latency_per_bin_ms")
            }
            context_note["model"] = self.model
            logger.info(f"[BaselineDecodeInfer] {json.dumps(context_note, ensure_ascii=False)}")

    def _process(self, context: Any) -> Any:
        if self.source == "sweep":
            records, self.summary = self._run_sweep()
        else:
            records = self._run_context(context)
        self.session_results = records
        self._record_notes(context, records)
        return context
