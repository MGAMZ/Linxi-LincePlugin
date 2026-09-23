"""临策赛道评测侧车导出算子。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor

from ._trials import LinceExportError

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["EVAL_SCHEMA_KEYS", "ExportEvalMetrics"]

EVAL_SCHEMA_KEYS = ("session_id", "tier", "span", "metrics", "drift_targets")
_SIDECAR_SUFFIX = ".eval.json"


def strictify(value: Any, path: str = "$", conversions: list[str] | None = None) -> Any:
    """递归转换值为 JSON 严格可序列化形态。"""
    if conversions is None:
        conversions = []
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        conversions.append(path)
        return None
    if isinstance(value, dict):
        return {key: strictify(item, f"{path}.{key}", conversions) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strictify(item, f"{path}[{index}]", conversions) for index, item in enumerate(value)]
    raise TypeError(
        f"context.metrics 含非 JSON 原生对象: {path} 处为 {type(value).__name__}，不静默丢弃"
    )


@register_as_linxi_processor(stage=PROCESS_STAGES.EXPORT)
class ExportEvalMetrics(DefaultProcessor):
    """把 `context.metrics` 全部键写出为 `.ls` store 同级的 eval JSON 侧车。

    须在写出 `.ls` 的导出算子之后同段执行。

    Example YAML configuration::

        - stage: export
          processor_name: "ExportEvalMetrics"
          params: {}
    """

    PROCESSOR_NAME = "ExportEvalMetrics"

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def _process(self, context: LinxiContext) -> LinxiContext:
        raw = context.run_state.output_path
        if raw is None:
            raise LinceExportError(
                "context.run_state.output_path 为 None：无法从 .ls 产物路径派生侧车命名根，"
                "请确认同段已有 WriteLinshuFile 系算子写出成功"
            )
        store = Path(raw)
        sidecar = store.with_name(store.stem + _SIDECAR_SUFFIX)
        bag = dict(context.metrics)
        payload = {key: bag.get(key) for key in EVAL_SCHEMA_KEYS}
        payload.update({key: value for key, value in bag.items() if key not in payload})
        conversions: list[str] = []
        payload = strictify(payload, conversions=conversions)
        if conversions:
            logger.info(f"[ExportEvalMetrics] NaN/±Inf→null 转换清单: {conversions}")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, allow_nan=False)
        logger.info(f"[ExportEvalMetrics] eval sidecar 写出: {sidecar}")
        return context
