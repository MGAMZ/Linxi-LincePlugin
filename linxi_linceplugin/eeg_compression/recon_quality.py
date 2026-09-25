"""Compress2BCI 重建质量算子：原始与重建对比的 PRD/CR 指标与判题格式合规检查。"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor

from .compliance import check_flex, check_ieeg, check_np
from .load_np_stream import _resolve_bin_meta
from .scoring import dir_bytes, flex_metrics, ieeg_metrics, np_metrics

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["Compress2BciReconQuality"]


def _payload_bytes(path: str) -> int:
    """单文件或目录内全部文件的字节合计，作 CR 的分子与分母口径。"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    return dir_bytes(path)


def _resolve_single_bin(input_path: str, which: str) -> str:
    """定位重建 *.bin：接受 bin 文件路径或目录（目录内 *.bin 须唯一，不要求配套 .meta）。"""
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".bin"):
            raise ValueError(f"{which} 应为 SpikeGLX *.bin 文件: {input_path}")
        return input_path
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"{which} 不存在: {input_path}")
    bins = sorted(str(p) for p in Path(input_path).rglob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"目录内未找到 *.bin: {input_path}")
    if len(bins) > 1:
        raise ValueError(f"目录内 *.bin 不唯一: {bins}")
    return bins[0]


def _require_rec(path: str, which: str) -> None:
    """flex 两侧必须是现存 *.rec 文件，形态不符即抛参数错误。"""
    if not os.path.isfile(path) or not path.lower().endswith(".rec"):
        raise ValueError(f"flex 子赛道 {which} 应指向现存 *.rec 文件: {path}")


def _np_quality(original_path: str, rebuilt_path: str,
                compressed_path: str | None) -> dict:
    """np 子赛道：PRD 走全流 int16 口径，合规检查含压缩目录的 meta 检索。"""
    if compressed_path is None:
        raise ValueError("np 子赛道必须提供 compressed_path，判题门要求 .ap.meta 随压缩结果保存")
    orig_bin, orig_meta = _resolve_bin_meta(original_path)
    decomp_bin = _resolve_single_bin(rebuilt_path, "rebuilt_path")
    return {
        "subtrack": "np",
        "original": orig_bin,
        "rebuilt": decomp_bin,
        "compressed": compressed_path,
        "cr": _payload_bytes(orig_bin) / dir_bytes(compressed_path),
        "metrics": asdict(np_metrics(orig_bin, orig_meta, decomp_bin)),
        "compliance": check_np(orig_bin, orig_meta, compressed_path, rebuilt_path),
    }


def _ieeg_quality(original_path: str, rebuilt_path: str,
                  compressed_path: str | None) -> dict:
    """ieeg 子赛道：神经通道按物理单位进 PRD，合规检查含 BIDS 伴生元数据比对。"""
    cr = None if compressed_path is None else _payload_bytes(original_path) / dir_bytes(compressed_path)
    return {
        "subtrack": "ieeg",
        "original": original_path,
        "rebuilt": rebuilt_path,
        "compressed": compressed_path,
        "cr": cr,
        "metrics": asdict(ieeg_metrics(original_path, rebuilt_path)),
        "compliance": check_ieeg(original_path, rebuilt_path),
    }


def _flex_quality(original_path: str, rebuilt_path: str,
                  compressed_path: str | None) -> dict:
    """flex 子赛道：帧内 int16 神经段进 PRD，合规检查含 XML 头与帧几何比对。"""
    _require_rec(original_path, "original_path")
    _require_rec(rebuilt_path, "rebuilt_path")
    cr = None if compressed_path is None else os.path.getsize(original_path) / dir_bytes(compressed_path)
    return {
        "subtrack": "flex",
        "original": original_path,
        "rebuilt": rebuilt_path,
        "compressed": compressed_path,
        "cr": cr,
        "metrics": asdict(flex_metrics(original_path, rebuilt_path)),
        "compliance": check_flex(original_path, rebuilt_path),
    }


_HANDLERS = {"np": _np_quality, "ieeg": _ieeg_quality, "flex": _flex_quality}


@register_as_linxi_processor(stage=PROCESS_STAGES.QUALITY)
class Compress2BciReconQuality(DefaultProcessor):
    """按子赛道对比原始与重建产物，输出指标与判题合规结果到 `context.metrics` 的 QUALITY 算子。"""

    def __init__(
        self,
        subtrack: str | None = None,
        original_path: str | None = None,
        rebuilt_path: str | None = None,
        compressed_path: str | None = None,
        metrics_key: str | None = None,
        name: str | None = None,
    ):
        super().__init__(name)
        self.subtrack = subtrack
        self.original_path = original_path
        self.rebuilt_path = rebuilt_path
        self.compressed_path = compressed_path
        self.metrics_key = metrics_key

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.original_path is None or self.rebuilt_path is None:
            raise ValueError(
                "Compress2BciReconQuality requires explicit `original_path` and "
                "`rebuilt_path` processor params."
            )
        handler = _HANDLERS.get(self.subtrack)
        if handler is None:
            raise ValueError(
                f"subtrack 必须为 {sorted(_HANDLERS)} 之一，实际为 {self.subtrack!r}"
            )
        result = handler(self.original_path, self.rebuilt_path, self.compressed_path)
        key = self.metrics_key or f"recon_quality_{self.subtrack}"
        context.metrics[key] = result
        return context
