"""高保真脑电数据压缩赛道判题格式合规检查器。"""

from __future__ import annotations

import fnmatch
import os
from typing import Callable

import pyedflib

from .scoring import (
    DEFAULT_BLOCK_BYTES,
    _FLEX_FRAME_BYTES,
    _NP_CHANNEL_KEY,
    _bytes_identical,
    _flex_data_start,
    _flex_frames,
    _meta_value,
    _resolve_edf,
)

__all__ = ["check_np", "check_ieeg", "check_flex"]

# BIDS-iEEG 文件组构成：赛题 readme.md L101-111 文件清单树，与训练集
# sub-UCLA17 目录实物枚举一致（1 个 EDF + 6 个伴生元数据）。
IEEG_FILE_KINDS: tuple[str, ...] = (
    "*_scans.tsv",
    "*_ieeg.edf",
    "*_ieeg.json",
    "*_channels.tsv",
    "*_channels.json",
    "*_electrodes.tsv",
    "*_coordsystem.json",
)


def _gate(name: str, status: str, reason: str) -> dict:
    return {"gate": name, "status": status, "reason": reason}


def _run_gate(name: str, probe: Callable[[], tuple[str, str]]) -> dict:
    """单 gate 边界：probe 返回 (status, reason)，逃逸异常转为 error 并附异常摘要。"""
    try:
        status, reason = probe()
    except Exception as exc:
        return _gate(name, "error", f"{type(exc).__name__}: {exc}")
    return _gate(name, status, reason)


def _result(subtrack: str, gates: list[dict]) -> dict:
    return {"subtrack": subtrack, "gates": gates}


def _candidates(root: str, preferred_name: str, suffix: str) -> list[str]:
    """目录树内检索目标文件：与 preferred_name 同名者优先，否则收集全部指定后缀文件。"""
    exact: list[str] = []
    suffixed: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(dirpath, fname)
            if fname == preferred_name:
                exact.append(path)
            elif fname.lower().endswith(suffix):
                suffixed.append(path)
    return sorted(exact) if exact else sorted(suffixed)


def _single(candidates: list[str], where: str) -> tuple[str | None, str]:
    if not candidates:
        return None, f"{where} 内未检索到目标文件"
    if len(candidates) > 1:
        return None, f"{where} 内目标文件不唯一: {candidates}"
    return candidates[0], ""


def check_np(orig_bin: str, orig_meta: str, comp_dir: str, decomp_dir: str,
             block: int = DEFAULT_BLOCK_BYTES) -> dict:
    """np 子赛道判题格式检查：重建 bin 存在、字节大小一致、行宽对齐、meta 随压缩保存并逐位恢复。

    Args
    ----
    orig_bin :
        原始 *.ap.bin。
    orig_meta :
        原始 *.ap.meta，提供行宽键 nSavedChans。
    comp_dir :
        压缩结果目录。
    decomp_dir :
        解压结果目录，或重建 bin 的单文件路径（单文件形态检索其同级目录）。
    block :
        逐位比对的单块字节上限。
    """
    if not os.path.isfile(orig_bin):
        raise FileNotFoundError(f"原始 bin 不存在: {orig_bin}")
    if not os.path.isfile(orig_meta):
        raise FileNotFoundError(f"原始 meta 不存在: {orig_meta}")

    bin_name = os.path.basename(orig_bin)
    meta_name = os.path.basename(orig_meta)
    orig_size = os.path.getsize(orig_bin)
    decomp_note = ""
    if not (os.path.isdir(decomp_dir) or os.path.isfile(decomp_dir)):
        decomp_note = f"解压输出路径不存在: {decomp_dir}"
    comp_note = "" if os.path.isdir(comp_dir) else f"压缩结果目录不存在: {comp_dir}"

    if decomp_note:
        bin_candidates: list[str] = []
        decomp_meta_candidates: list[str] = []
    elif os.path.isfile(decomp_dir):
        as_meta = decomp_dir.lower().endswith(".meta")
        bin_candidates = [] if as_meta else [decomp_dir]
        if as_meta:
            decomp_meta_candidates = [decomp_dir]
        else:
            decomp_meta_candidates = _candidates(os.path.dirname(decomp_dir), meta_name, ".meta")
    else:
        bin_candidates = _candidates(decomp_dir, bin_name, ".bin")
        decomp_meta_candidates = _candidates(decomp_dir, meta_name, ".meta")
    comp_meta_candidates = [] if comp_note else _candidates(comp_dir, meta_name, ".meta")

    def gate_bin_exists() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        path, note = _single(bin_candidates, f"解压输出 {decomp_dir}")
        if path is None:
            return "fail", f"{note}（重建 *.ap.bin 缺失，检索名 {bin_name} / 后缀 .bin）"
        return "pass", f"重建 bin 定位: {path}"

    def gate_bin_size() -> tuple[str, str]:
        path, note = _single(bin_candidates, f"解压输出 {decomp_dir}")
        if path is None:
            return "fail", f"重建 bin 未唯一定位，无法比对大小: {note}"
        rebuilt = os.path.getsize(path)
        if rebuilt != orig_size:
            return "fail", f"重建 bin 字节数 {rebuilt} 与原始 {orig_size} 不一致"
        return "pass", f"重建 bin 字节数与原始一致: {orig_size}"

    def gate_bin_aligned() -> tuple[str, str]:
        n_channels = int(_meta_value(orig_meta, _NP_CHANNEL_KEY))
        row_bytes = n_channels * 2  # int16 行宽，本数据集 nSavedChans=385 即 770 字节
        path, note = _single(bin_candidates, f"解压输出 {decomp_dir}")
        if path is None:
            return "fail", f"重建 bin 未唯一定位，无法检查行宽对齐: {note}"
        rebuilt = os.path.getsize(path)
        if rebuilt % row_bytes:
            return "fail", (f"重建 bin {rebuilt} 字节不是行宽 {row_bytes} "
                            f"(nSavedChans={n_channels} x int16 2 字节) 的整数倍")
        return "pass", f"重建 bin 按行宽 {row_bytes} 整除对齐，共 {rebuilt // row_bytes} 行"

    def gate_meta_in_comp() -> tuple[str, str]:
        if comp_note:
            return "fail", comp_note
        path, note = _single(comp_meta_candidates, f"压缩结果目录 {comp_dir}")
        if path is None:
            return "fail", (f"{note}；readme 要求 .ap.meta 随压缩结果保存"
                            f"（检索名 {meta_name} / 后缀 .meta）")
        return "pass", f"压缩结果内的 meta: {path}"

    def gate_meta_restored() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        path, note = _single(decomp_meta_candidates, f"解压输出 {decomp_dir}")
        if path is None:
            return "fail", f"{note}（恢复的 *.ap.meta 缺失，检索名 {meta_name} / 后缀 .meta）"
        size = os.path.getsize(path)
        orig_meta_size = os.path.getsize(orig_meta)
        if size != orig_meta_size:
            return "fail", (f"恢复 meta 字节数 {size} 与原始 {orig_meta_size} 不一致: {path}")
        if not _bytes_identical(path, orig_meta, size, block):
            return "fail", f"恢复 meta 与原始逐位不一致: {path} vs {orig_meta}"
        return "pass", f"恢复 meta 与原始逐位一致: {path}"

    return _result("np", [
        _run_gate("np_rebuilt_bin", gate_bin_exists),
        _run_gate("np_bin_size_identical", gate_bin_size),
        _run_gate("np_bin_int16_aligned", gate_bin_aligned),
        _run_gate("np_meta_in_comp", gate_meta_in_comp),
        _run_gate("np_meta_restored_identical", gate_meta_restored),
    ])


def check_ieeg(orig_dir_or_edf: str, decomp_dir: str,
               block: int = DEFAULT_BLOCK_BYTES) -> dict:
    """ieeg 子赛道判题格式检查：重建 EDF 可被 pyedflib 打开，BIDS 伴生元数据齐全且逐位一致。

    Args
    ----
    orig_dir_or_edf :
        原始 BIDS 受试者目录（伴生元数据基准来源），或单个 EDF 文件路径。
    decomp_dir :
        重建结果目录。
    block :
        逐位比对的单块字节上限。
    """
    orig_root = os.path.abspath(orig_dir_or_edf)
    if not os.path.exists(orig_root):
        raise FileNotFoundError(f"原始侧不存在: {orig_dir_or_edf}")
    decomp_root = os.path.abspath(decomp_dir)
    decomp_note = "" if os.path.isdir(decomp_root) else f"重建目录不存在: {decomp_dir}"

    sidecar_map: dict[str, str] = {}
    if os.path.isdir(orig_root):
        for dirpath, _dirs, files in os.walk(orig_root):
            for fname in files:
                if fname.lower().endswith(".edf"):
                    continue
                if not any(fnmatch.fnmatch(fname, pat) for pat in IEEG_FILE_KINDS):
                    continue
                if fname in sidecar_map:
                    raise ValueError(f"原始目录内伴生元数据重名: {fname}")
                sidecar_map[fname] = os.path.join(dirpath, fname)

    def gate_edf_readable() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        decomp_edf = _resolve_edf(decomp_root, "decomp_dir")
        with pyedflib.EdfReader(decomp_edf) as reader:
            n_signals = int(reader.signals_in_file)
            n_records = int(reader.datarecords_in_file)
        if n_signals <= 0:
            return "fail", f"重建 EDF 无信号通道: {decomp_edf}"
        return "pass", (f"重建 EDF 可由 pyedflib 打开: {os.path.basename(decomp_edf)}，"
                        f"{n_signals} 通道 x {n_records} 记录")

    def locate_sidecar(name: str) -> tuple[str | None, str]:
        suffix = os.path.splitext(name)[1].lower()
        return _single(_candidates(decomp_root, name, suffix), f"重建目录 {decomp_dir}")

    def gate_bids_complete() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        if os.path.isfile(orig_root):
            return "fail", "原始侧为单个 EDF 文件，无法建立 BIDS 伴生元数据基准"
        missing = []
        for name in sidecar_map:
            path, _note = locate_sidecar(name)
            if path is None:
                missing.append(name)
        if missing:
            return "fail", (f"重建目录缺少伴生元数据 {missing}"
                            f"（文件清单见 readme.md L101-111）")
        return "pass", f"BIDS 文件组齐备：EDF + {len(sidecar_map)} 个伴生元数据全部定位"

    def gate_bids_identical() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        if os.path.isfile(orig_root):
            return "fail", "原始侧为单个 EDF 文件，无法建立 BIDS 伴生元数据基准"
        diffs = []
        for name, orig_path in sidecar_map.items():
            path, note = locate_sidecar(name)
            if path is None:
                diffs.append(f"{name}: {note}")
                continue
            size = os.path.getsize(orig_path)
            if os.path.getsize(path) != size or not _bytes_identical(path, orig_path, size, block):
                diffs.append(f"{name}: 逐位不一致")
        if diffs:
            return "fail", "伴生元数据一致性问题: " + "; ".join(diffs)
        return "pass", f"{len(sidecar_map)} 个伴生元数据与原始逐位一致"

    return _result("ieeg", [
        _run_gate("ieeg_edf_readable", gate_edf_readable),
        _run_gate("ieeg_bids_complete", gate_bids_complete),
        _run_gate("ieeg_bids_identical", gate_bids_identical),
    ])


def check_flex(orig_rec: str, decomp_rec: str,
               block: int = DEFAULT_BLOCK_BYTES) -> dict:
    """flex 子赛道判题格式检查：XML 头逐位一致、重建帧数一致、每帧 2070 字节对齐。

    Args
    ----
    orig_rec :
        原始 .rec。
    decomp_rec :
        重建 .rec。
    block :
        逐位比对的单块字节上限。
    """
    if not os.path.isfile(orig_rec):
        raise FileNotFoundError(f"原始 .rec 不存在: {orig_rec}")
    orig_start = _flex_data_start(orig_rec)
    orig_frames = _flex_frames(orig_rec, orig_start)
    decomp_note = "" if os.path.isfile(decomp_rec) else f"重建 .rec 不存在: {decomp_rec}"

    def decomp_geometry() -> tuple[int, int]:
        start = _flex_data_start(decomp_rec)
        return start, os.path.getsize(decomp_rec) - start

    def gate_header() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        try:
            start = _flex_data_start(decomp_rec)
        except ValueError as exc:
            return "fail", f"重建 .rec 的 XML 头起点无法推导: {exc}"
        if start != orig_start:
            return "fail", f"XML 头长度不一致: 原始 {orig_start} 字节, 重建 {start} 字节"
        if not _bytes_identical(orig_rec, decomp_rec, start, block):
            return "fail", f"XML 头前 {start} 字节存在逐位差异"
        return "pass", f"XML 头逐位一致（{start} 字节）"

    def gate_alignment() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        try:
            _start, payload = decomp_geometry()
        except ValueError as exc:
            return "fail", f"重建 .rec 数据区无法界定: {exc}"
        remainder = payload % _FLEX_FRAME_BYTES
        if remainder:
            return "fail", (f"重建数据区 {payload} 字节不是帧长 {_FLEX_FRAME_BYTES} "
                            f"的整数倍（余 {remainder} 字节）")
        return "pass", f"重建数据区 {payload} 字节按帧长 {_FLEX_FRAME_BYTES} 整除对齐"

    def gate_frames() -> tuple[str, str]:
        if decomp_note:
            return "fail", decomp_note
        try:
            _start, payload = decomp_geometry()
        except ValueError as exc:
            return "fail", f"重建 .rec 数据区无法界定: {exc}"
        if payload % _FLEX_FRAME_BYTES:
            return "fail", "重建数据区存在非整帧余数，帧数无定义（见 flex_frame_alignment）"
        n_frames = payload // _FLEX_FRAME_BYTES
        if n_frames != orig_frames:
            return "fail", f"重建帧数 {n_frames} 与原始 {orig_frames} 不一致"
        return "pass", f"帧数一致: {n_frames}"

    return _result("flex", [
        _run_gate("flex_header_identical", gate_header),
        _run_gate("flex_frame_alignment", gate_alignment),
        _run_gate("flex_frame_count_equal", gate_frames),
    ])
