from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor
from linxi.processor.export import WriteLinshuFile
from linshu_format.core import TimeIntervals

from ..memory_state_decode.load_session import UNITS_LEVELS_ATTR, flatten_units_channel_coord
from ._trials import LinceExportError, lossy_policy_messages, normalize_trials

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["LinceExportError", "LinceWriteLinshuFile"]

_NOTES_EXPORT_KEY = "lince_dpa_export"


@register_as_linxi_processor(stage=PROCESS_STAGES.EXPORT)
class LinceWriteLinshuFile(DefaultProcessor):
    """把解码评测结构接入上游 `WriteLinshuFile` 并写出 `.ls`。

    Example YAML configuration::

        - stage: export
          processor_name: "LinceWriteLinshuFile"
          params:
            output_path: "/path/to/output/MA-CO-20231227-01.ls"
            ecephys_key: "query"
            session_scalar_cols: ["session_r2_mean", "session_score"]
            on_lossy: "warn"
            overwrite: true

    Args
    ----
    output_path / overwrite / skip_fields / skip_raw_signals
        透传上游 `WriteLinshuFile` 的同名参数。
    ecephys_key
        导出的 ecephys 记录键。
    session_scalar_cols
        trials 表中按 trial 广播的 session 级标量列名。
    on_lossy
        有损处理策略，取 `warn` 或 `fail`。
    require_eval_cols
        为 True 时要求 trials 表含结构列以外的评测列。
    structural_cols
        视为非评测的结构列集合。
    """

    def __init__(
        self,
        output_path: str | None = None,
        ecephys_key: str = "query",
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
            overwrite=overwrite,
            skip_fields=skip_fields,
            skip_raw_signals=skip_raw_signals,
        )

    def _process(self, context: LinxiContext) -> LinxiContext:
        self._check_signal_slots(context)
        self._normalize_trials(context)
        restore = self._flatten_units_coords(context)
        try:
            context = self._writer(context)
        finally:
            for holder, field, original in restore:
                setattr(holder, field, original)
        self._check_product(context)
        return context

    def _flatten_units_coords(self, context: LinxiContext) -> list[tuple[Any, str, xr.DataArray]]:
        targets: list[tuple[Any, str, str, xr.DataArray]] = []
        spikes = context.binned_spikes
        if spikes is not None:
            targets.append((spikes, "counts", "binned_spikes.counts", spikes.counts))
        for key, slot in context.ecephys.items():
            signal = slot.electrophysiology
            if signal is not None:
                targets.append((signal, "data", f"ecephys/{key}/electrophysiology.data", signal.data))

        originals: list[tuple[Any, str, xr.DataArray]] = []
        records: list[dict[str, Any]] = []
        for holder, field, path, matrix in targets:
            flat = flatten_units_channel_coord(matrix)
            if flat is None:
                continue
            originals.append((holder, field, matrix))
            records.append({"field": path, "levels": list(flat.attrs[UNITS_LEVELS_ATTR]), "shape": [int(s) for s in matrix.shape]})
            setattr(holder, field, flat)
        if originals:
            self._register_coord_flatten(context, records)
        return originals

    @staticmethod
    def _register_coord_flatten(context: LinxiContext, records: list[dict[str, Any]]) -> None:
        logger.info(f"LinceWriteLinshuFile[dpa] units MultiIndex 坐标展开为 level 列: {records}")
        merged: dict[str, Any]
        if context.notes:
            try:
                merged = json.loads(context.notes)
                if not isinstance(merged, dict):
                    merged = {"lince_notes_previous": context.notes}
            except ValueError:
                merged = {"lince_notes_previous": context.notes}
        else:
            merged = {}
        merged[_NOTES_EXPORT_KEY] = {"units_coord_flatten": records}
        context.notes = json.dumps(merged, ensure_ascii=False, sort_keys=True)

    def _check_signal_slots(self, context: LinxiContext) -> None:
        if context.recording is None:
            raise LinceExportError(
                "context.recording 为 None：该情形上游 WriteLinshuFile 仅告警跳过写出，此处按失败处理。"
                "请确认 load 算子已构建信号记录。"
            )
        slot = context.ecephys.get(self.ecephys_key)
        if slot is None:
            raise LinceExportError(
                f"context.ecephys 缺信号槽 {self.ecephys_key!r}，现有键={sorted(context.ecephys)}"
            )
        if slot.electrophysiology is None and not slot.auxiliary_channels:
            raise LinceExportError(
                f"ecephys[{self.ecephys_key!r}] 为空占位，产物将只剩 _linshu_type 类型标记。"
                "信号槽须由 load 算子预填"
            )

    def _normalize_trials(self, context: LinxiContext) -> None:
        trials = context.trials
        if trials is None:
            if self._require_eval_cols:
                raise LinceExportError("context.trials 为 None：无解码评测试次表可导出。如仅需元数据导出请显式 require_eval_cols=False")
            return

        new_vars, changed, dropped, relocated = normalize_trials(
            trials.table,
            structural=self._structural,
            session_cols=self._session_cols,
            require_eval=self._require_eval_cols,
        )
        for message in lossy_policy_messages(dropped, relocated):
            if self._on_lossy == "fail":
                raise LinceExportError(message)
            logger.warning(f"LinceWriteLinshuFile[lossy=warn] {message}")

        if changed:
            context.trials = TimeIntervals(name=trials.name, description=trials.description,
                                           table=xr.Dataset(new_vars, attrs=trials.table.attrs))

    def _check_product(self, context: LinxiContext) -> None:
        raw = context.run_state.output_path
        if raw is None:
            raise LinceExportError("上游 WriteLinshuFile 命中静默跳过路径，未写出 .ls 产物路径，按失败处理")
        out = Path(raw)
        if not out.is_dir() or not any((out / m).exists() for m in ("zarr.json", ".zgroup", ".zmetadata")):
            raise LinceExportError(f"导出产物不是带格式标记的 Zarr 目录 store: {out}")
        n_files = sum(1 for p in out.rglob("*") if p.is_file())
        if n_files <= 2:
            raise LinceExportError(f"导出 store 仅有 {n_files} 个文件，疑似空壳: {out}")
