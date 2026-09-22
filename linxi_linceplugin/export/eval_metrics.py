"""临策赛道评测侧车导出算子（EXPORT 阶段）：把 `context.metrics` 暂存袋序列化为与 `.ls` 产物同目录同命名根的 `<数据文件名>.eval.json`。

序列化政策：NaN 与 ±Inf 一律转换为 null（缺失语义）并打印一条转换路径清单日志；schema 键缺失落 null；非 JSON 原生对象（ndarray 等）即时抛 `TypeError`；写出使用 `json.dump(..., allow_nan=False)`，产物保证为严格 JSON。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor

from .linshufile import LinceExportError

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["EVAL_SCHEMA_KEYS", "ExportEvalMetrics"]

EVAL_SCHEMA_KEYS = ("session_id", "tier", "span", "metrics", "drift_targets")
_SIDECAR_SUFFIX = ".eval.json"


def strictify(value: Any, path: str = "$", conversions: list[str] | None = None) -> Any:
    """递归产出 JSON-strict 值：非有限浮点（NaN/±Inf）转 None 并把路径记入 `conversions`；非 JSON 原生类型抛 `TypeError`。"""
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
        f"context.metrics 暂存袋含非 JSON 原生对象: {path} 处为 {type(value).__name__}，"
        "拒绝静默丢键（Fast Fail）"
    )


@register_as_linxi_processor(stage=PROCESS_STAGES.EXPORT)
class ExportEvalMetrics(DefaultProcessor):
    """把评测暂存袋整袋写出为 `.ls` store 的同级 eval JSON 侧车。

    侧车路径在运行时从上游 `WriteLinshuFile` 写定的 `context.run_state.output_path` 派生：`<store 父目录>/<store 命名根>.eval.json`（如 `MA-CO-20231227-01.ls` → `MA-CO-20231227-01.eval.json`）。本算子须在写出 `.ls` 的导出算子之后同段执行。

    袋内容即侧车结构：解码算子填 `session_id` / `tier` / `span` / `metrics`，漂移算子填 `drift_targets`；schema 之外的袋键一并写出。

    Example YAML configuration::

        - stage: export
          processor_name: "ExportEvalMetrics"
          params: {}

    Parameters
    ----------
    name : str | None
        处理器实例名（引擎惯例）。
    """

    PROCESSOR_NAME = "ExportEvalMetrics"  # 显式声明注册名

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
