"""`context.ieeg` 连续电位记录的 LinshuFile 导出。"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr
from linxi.logger import logger
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import TimeIntervals

from ._trials import LinceExportError, lossy_policy_messages, normalize_trials

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["LinceWriteIeegLinshuFile"]


@register_as_linxi_processor(stage=PROCESS_STAGES.EXPORT)
class LinceWriteIeegLinshuFile(DefaultProcessor):
    """把 `context.ieeg[recording_key]` 记录与试次表评测列写出为 `.ls`。

    Example YAML configuration::

        - stage: export
          processor_name: "LinceWriteIeegLinshuFile"
          params:
            output_path: "/path/to/output/P01_20240905-single-MA.ls"
            recording_key: "query"
            on_lossy: "warn"
            overwrite: true

    Args
    ----
    output_path
        `.ls` store 路径，必须显式传入。
    recording_key
        `context.ieeg` 槽键。
    overwrite
        目标已存在时是否替换。
    skip_fields
        按叶字段名跳过写入的集合。
    require_eval_cols
        为 True 时要求 trials 表含结构列以外的评测列。
    on_lossy
        有损处理策略，取 `warn` 或 `fail`。
    structural_cols
        视为非评测的结构列集合。
    """

    def __init__(
        self,
        output_path: str | None = None,
        recording_key: str = "query",
        overwrite: bool = True,
        skip_fields: list[str] | None = None,
        require_eval_cols: bool = True,
        on_lossy: Literal["warn", "fail"] = "warn",
        structural_cols: list[str] | None = None,
    ):
        super().__init__()
        if on_lossy not in ("warn", "fail"):
            raise ValueError(f"on_lossy 必须是 'warn' 或 'fail'，收到 {on_lossy!r}")
        self.output_path = output_path
        self.recording_key = recording_key
        self._overwrite = overwrite
        self._skip_fields = skip_fields
        self._on_lossy: Literal["warn", "fail"] = on_lossy
        self._require_eval_cols = require_eval_cols
        self._structural = set(structural_cols or ("start_time", "stop_time", "trial_id"))

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.output_path is None:
            raise LinceExportError(
                "LinceWriteIeegLinshuFile requires an explicit `output_path` processor param")
        slot = context.ieeg.get(self.recording_key)
        if slot is None:
            raise LinceExportError(f"context.ieeg 缺信号槽 {self.recording_key!r}，现有键为 {sorted(context.ieeg)}")
        if slot.electrophysiology is None:
            raise LinceExportError(
                f"ieeg[{self.recording_key!r}].electrophysiology 为空：信号槽须由 load 算子预填")
        if context.trials is None and self._require_eval_cols:
            raise LinceExportError("context.trials 为 None：无评测试次表可导出。如仅需信号导出请显式 require_eval_cols=False")

        self._normalize_trials(context)
        out = Path(self.output_path)
        self._write(context, out)
        context.run_state.output_path = str(out)
        self._check_product(out)
        logger.info(f"LinceWriteIeegLinshuFile exported .ls: {out}")
        return context

    def _normalize_trials(self, context: LinxiContext) -> None:
        trials = context.trials
        if trials is None:
            return
        new_vars, changed, dropped, relocated = normalize_trials(
            trials.table,
            structural=self._structural,
            session_cols=[],
            require_eval=self._require_eval_cols,
        )
        for message in lossy_policy_messages(dropped, relocated):
            if self._on_lossy == "fail":
                raise LinceExportError(message)
            logger.warning(f"LinceWriteIeegLinshuFile[lossy=warn] {message}")
        if changed:
            intervals = TimeIntervals(name=trials.name, description=trials.description,
                                      table=xr.Dataset(new_vars, attrs=trials.table.attrs))
            context.trials = intervals
            context.ieeg[self.recording_key].events = intervals

    def _write(self, context: LinxiContext, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if not self._overwrite:
                raise FileExistsError(f"Output .ls file already exists: {out}")
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
            logger.warning(f"Removed existing .ls output: {out}")
        tmp = out.parent / f"{out.name}.tmp-{uuid.uuid4().hex[:8]}"
        try:
            context.write(str(tmp), exclude_fields=set(self._skip_fields) if self._skip_fields else None)
            os.replace(tmp, out)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _check_product(self, out: Path) -> None:
        if not out.is_dir() or not any((out / m).exists() for m in ("zarr.json", ".zgroup", ".zmetadata")):
            raise LinceExportError(f"导出产物不是带格式标记的 Zarr 目录 store: {out}")
        n_files = sum(1 for p in out.rglob("*") if p.is_file())
        if n_files <= 2:
            raise LinceExportError(f"导出 store 仅有 {n_files} 个文件，疑似空壳: {out}")
