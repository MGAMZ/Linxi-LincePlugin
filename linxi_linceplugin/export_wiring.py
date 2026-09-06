"""临策赛道 LinshuFile 导出接线（EXPORT 阶段）：把流水线中的记录与评测结果写出为 ``.ls`` 产物。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor
from linxi.processor.export import WriteLinshuFile
from linshu_format.core import TimeIntervals

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["LinceExportError", "LinceWriteLinshuFile"]

_NUMERIC_KINDS = frozenset("biuf")


class LinceExportError(RuntimeError):
    """导出接线的前置条件或产物断言失败。"""


def _classify_object_column(arr: np.ndarray) -> str:
    """把 object dtype 列分为 ``str``（原样保留）/ ``ragged``（补齐）/ ``keep`` / ``opaque``（不可表达）。"""
    if arr.size == 0:
        return "keep"
    items = list(arr)
    if all(isinstance(x, str) for x in items):
        return "str"
    if all(isinstance(x, np.ndarray) for x in items):
        ndims = {x.ndim for x in items}
        if ndims == {1} or ndims == {2}:
            if ndims == {1} or len({x.shape[1] for x in items}) == 1:
                if all(x.dtype.kind in _NUMERIC_KINDS for x in items):
                    return "ragged"
    return "opaque"


def _pad_ragged(name: str, col: np.ndarray, primary: str) -> tuple[xr.DataArray, xr.DataArray]:
    items = [np.asarray(x, dtype=np.float64) for x in col]
    max_len = max(x.shape[0] for x in items)
    if items[0].ndim == 1:
        padded = np.full((len(items), max_len), np.nan)
        dims = (primary, f"{name}_bin")
        for i, x in enumerate(items):
            padded[i, : x.shape[0]] = x
    else:
        padded = np.full((len(items), max_len, items[0].shape[1]), np.nan)
        dims = (primary, f"{name}_bin", f"{name}_coord")
        for i, x in enumerate(items):
            padded[i, : x.shape[0], :] = x
    lengths = np.array([x.shape[0] for x in items], dtype=np.int64)
    pad_da = xr.DataArray(padded, dims=dims, name=name)
    vlen_da = xr.DataArray(lengths, dims=(primary,), name=f"{name}_valid_len")
    return pad_da, vlen_da


@register_as_linxi_processor(stage=PROCESS_STAGES.EXPORT)
class LinceWriteLinshuFile(DefaultProcessor):
    """把解码评测结构接入上游 ``WriteLinshuFile`` 并写出 ``.ls``（Zarr 目录 store）。

    Example YAML configuration::

        - stage: export
          processor_name: "LinceWriteLinshuFile"
          params:
            output_path: "/path/to/output/linshu_e2e/sub-VM23_ses-MA-CO-20231227-01.ls"
            ecephys_key: "query"
            session_scalar_cols: ["session_r2_mean", "session_score"]
            on_lossy: "warn"
            overwrite: true

    Parameters
    ----------
    output_path / chunk_frames / write_workers / unit / overwrite / skip_fields /
    skip_raw_signals :
        原样透传给组合的上游 ``WriteLinshuFile``；``unit`` 保持上游固定 ``"volts"`` 语义
        （赛题 binned_spikes 实为计数）。
    ecephys_key : str
        信号槽键，临策双记录布局默认 ``"query"``。
    session_scalar_cols : list[str] | None
        trials 表中按 trial 广播的 session 级标量列名；逐列做常量校验并按有损策略上报。
    on_lossy : {"warn", "fail"}
        有损策略：``"warn"`` 挪位保留/丢弃均报结构化 WARNING；``"fail"`` 一律失败。
    require_eval_cols : bool
        True 时要求 trials 表存在结构列以外的评测列，防止解码结果缺失时空壳导出。
    structural_cols : list[str] | None
        视为非评测的结构列集合，默认 ``start_time``/``stop_time``/``trial_id``。
    """

    def __init__(
        self,
        output_path: str | None = None,
        ecephys_key: str = "query",
        chunk_frames: int = 30000,
        write_workers: int = 8,
        unit: str = "volts",
        overwrite: bool = True,
        skip_fields: list[str] | None = None,
        skip_raw_signals: bool = False,
        session_scalar_cols: list[str] | None = None,
        on_lossy: Literal["warn", "fail"] = "warn",
        require_eval_cols: bool = True,
        structural_cols: list[str] | None = None,
    ):
        super().__init__()
        if on_lossy not in ("warn", "fail"):
            raise ValueError(f"on_lossy 必须是 'warn' 或 'fail'，收到 {on_lossy!r}")
        self.ecephys_key = ecephys_key
        self._on_lossy: Literal["warn", "fail"] = on_lossy
        self._require_eval_cols = require_eval_cols
        self._structural = set(structural_cols or ("start_time", "stop_time", "trial_id"))
        self._session_cols = list(session_scalar_cols or ())
        self._writer = WriteLinshuFile(
            output_path=output_path,
            ecephys_key=ecephys_key,
            chunk_frames=chunk_frames,
            write_workers=write_workers,
            unit=unit,
            overwrite=overwrite,
            skip_fields=skip_fields,
            skip_raw_signals=skip_raw_signals,
        )

    def _process(self, context: LinxiContext) -> LinxiContext:
        self._check_signal_slots(context)
        self._normalize_trials(context)
        context = self._writer(context)
        self._check_product(context)
        return context

    def _check_signal_slots(self, context: LinxiContext) -> None:
        if context.recording is None:
            raise LinceExportError(
                "context.recording 为 None：上游 WriteLinshuFile 对该情形只告警跳过写出，"
                "本接线层将其转为失败。请确认 load 算子已构建 recording 槽。"
            )
        slot = context.ecephys.get(self.ecephys_key)
        if slot is None:
            raise LinceExportError(
                f"context.ecephys 缺信号槽 {self.ecephys_key!r}（writer 只放空占位不放信号内容），"
                f"现有键={sorted(context.ecephys)}"
            )
        if slot.electrophysiology is None and not slot.auxiliary_channels:
            raise LinceExportError(
                f"ecephys[{self.ecephys_key!r}] 为空占位（electrophysiology=None 且 auxiliary_channels 空），"
                "落盘只会得到 _linshu_type 空壳；信号槽须由 load 算子预填"
            )

    def _normalize_trials(self, context: LinxiContext) -> None:
        trials = context.trials
        if trials is None:
            if self._require_eval_cols:
                raise LinceExportError("context.trials 为 None：无解码评测试次表可导出；如仅需元数据导出请显式 require_eval_cols=False")
            return

        table = trials.table
        start = table["start_time"]
        if start.ndim != 1:
            raise LinceExportError(f"trials.table.start_time 必须为一维，实际 dims={start.dims}")
        primary = start.dims[0]
        n = int(start.sizes[primary])

        if self._require_eval_cols and not set(table.data_vars) - self._structural:
            raise LinceExportError("trials 表仅含结构列，未合入任何解码评测列（require_eval_cols=False 可放开）")

        for col in self._session_cols:
            if col not in table.data_vars:
                raise LinceExportError(f"session_scalar_cols 引用不存在的列 {col!r}（现有列={sorted(table.data_vars)}）")
            vals = np.asarray(table[col].values)
            if vals.ndim != 1 or vals.size != n:
                raise LinceExportError(f"session 级标量列 {col!r} 必须是长度 {n} 的一维广播列，实际 shape={vals.shape}")
            if not bool((vals == vals[0]).all()):
                raise LinceExportError(f"列 {col!r} 声明为 session 级标量但逐 trial 取值不一致，广播假设被破坏")

        new_vars: dict[str, xr.DataArray] = {}
        dropped: list[str] = []
        changed = False
        for name, var in table.data_vars.items():
            if primary in var.dims and var.sizes[primary] != n:
                raise LinceExportError(f"列 {name!r} 在主维度 {primary!r} 上长度 {var.sizes[primary]} != trial 数 {n}")
            if var.dtype != object:
                new_vars[name] = var
                continue
            kind = _classify_object_column(np.asarray(var.values))
            if kind in ("str", "keep"):
                new_vars[name] = var
            elif kind == "ragged":
                pad_da, vlen_da = _pad_ragged(name, np.asarray(var.values), primary)
                new_vars[name] = pad_da
                new_vars[vlen_da.name] = vlen_da
                changed = True
            else:
                dropped.append(name)
                changed = True

        relocated = [f"{col}={np.asarray(table[col].values)[:1].tolist()[0]!r}" for col in self._session_cols]
        self._apply_lossy_policy(dropped=dropped, relocated=relocated)

        if changed:
            context.trials = TimeIntervals(name=trials.name, description=trials.description, table=xr.Dataset(new_vars, attrs=table.attrs))

    def _apply_lossy_policy(self, *, dropped: list[str], relocated: list[str]) -> None:
        if dropped:
            msg = (
                "dropped 无法用 LinshuFile 表达的字段"
                f"（对象列元素非字符串/非一致形状数值数组）: {sorted(dropped)}"
            )
            if self._on_lossy == "fail":
                raise LinceExportError(msg)
            logger.warning(f"LinceWriteLinshuFile[lossy=warn] {msg}")
        if relocated:
            msg = f"relocated session 级标量以 trials 广播列保留（LinshuFile 根级无指标容器）: {', '.join(relocated)}"
            if self._on_lossy == "fail":
                raise LinceExportError(msg)
            logger.warning(f"LinceWriteLinshuFile[lossy=warn] {msg}")

    def _check_product(self, context: LinxiContext) -> None:
        raw = context.run_state.output_path
        if raw is None:
            raise LinceExportError("上游 WriteLinshuFile 未写出 .ls 产物路径（命中静默跳过路径），按失败处理")
        out = Path(raw)
        if not out.is_dir() or not any((out / m).exists() for m in ("zarr.json", ".zgroup", ".zmetadata")):
            raise LinceExportError(f"导出产物不是带格式标记的 Zarr 目录 store: {out}")
        n_files = sum(1 for p in out.rglob("*") if p.is_file())
        if n_files <= 2:
            raise LinceExportError(f"导出 store 仅有 {n_files} 个文件，疑似空壳: {out}")
