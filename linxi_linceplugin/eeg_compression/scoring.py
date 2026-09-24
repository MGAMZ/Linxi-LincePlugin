"""高保真脑电数据压缩赛道按赛题 readme 口径的评分核心。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pyedflib

__all__ = [
    "DEFAULT_BLOCK_BYTES",
    "Score", "NpMetrics", "IeegMetrics", "FlexMetrics",
    "dir_bytes", "prd_streaming", "speed_mbytes_per_s",
    "subtrack_score", "total_score",
    "np_metrics", "ieeg_metrics", "flex_metrics",
]

DEFAULT_BLOCK_BYTES = 64 * 1024 * 1024

# 计分公式常数，出处为赛题 readme「性能评价方法」小节
_CR_BASELINE = 25.0
_SPEED_BASELINE = 100.0
_PRD_SHIFT = 0.1
_ZERO_CR = 5.0
_ZERO_PRD_MAIN = 1.0         # readme L57 判零线（主口径）
_ZERO_PRD_SENSITIVITY = 0.1  # 参赛者 README 判零线（敏感性口径）

_TOTAL_WEIGHTS = {"np": 0.4, "ieeg": 0.3, "flex": 0.3}

# np 子赛道 .meta 给出列数的键，实测值 385（384 通道 + 1 同步通道）
_NP_CHANNEL_KEY = "nSavedChans"

# flex 帧结构实测口径：帧 = 22 字节前缀（1 marker + 17 设备字节 + 4 字节时间戳）+ 1024 路 int16
_FLEX_FRAME_BYTES = 2070
_FLEX_PREFIX_BYTES = 22
_FLEX_END_TAG = b"</Configuration>"
_FLEX_MAX_HEADER_BYTES = 16 * 1024 * 1024

# EDF 标注通道识别口径与官方 ieeg_validate.py 一致：标签去空格后等于 EDFAnnotations
_EDF_ANNOTATION_LABEL = "EDFAnnotations"


def dir_bytes(path: str) -> int:
    """统计目录内全部文件（含子目录与隐藏文件）的字节之和，作为 CR 的分母口径。

    Args
    ----
    path :
        压缩结果目录。
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"目录不存在: {path}")
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def _sqrt_ratio(squared_error: float, squared_original: float) -> float:
    if squared_original == 0:
        raise ValueError("原始侧平方和为 0，PRD 无定义")
    return float(np.sqrt(squared_error / squared_original))


def prd_streaming(a_path: str, b_path: str, dtype: np.dtype | str,
                  block: int = DEFAULT_BLOCK_BYTES) -> float:
    """逐元素对齐地流式计算 PRD = sqrt(sum((x-y)^2) / sum(x^2))，返回开方后的小数比值。

    乘 100 的百分数换算由调用方在展示时自行进行。x 侧平方和为 0 时抛 ValueError。

    Args
    ----
    a_path :
        原始侧二进制文件。
    b_path :
        重建侧二进制文件。
    dtype :
        两侧共同的元素类型。
    block :
        单次读取的字节上限。
    """
    item = np.dtype(dtype)
    size_a = os.path.getsize(a_path)
    size_b = os.path.getsize(b_path)
    if size_a != size_b:
        raise ValueError(f"两侧字节数不一致: {a_path}({size_a}) vs {b_path}({size_b})")
    if size_a % item.itemsize:
        raise ValueError(f"文件字节数不是 {item} 元素宽度的整数倍: {a_path}({size_a})")
    if int(block) <= 0:
        raise ValueError(f"block 必须为正: {block}")

    samples_per_block = max(1, int(block) // item.itemsize)
    read_bytes = samples_per_block * item.itemsize
    squared_error = 0.0
    squared_original = 0.0
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        while True:
            original = np.frombuffer(fa.read(read_bytes), dtype=item)
            rebuilt = np.frombuffer(fb.read(read_bytes), dtype=item)
            if original.size == 0:
                break
            if original.size != rebuilt.size:
                raise ValueError(f"两侧文件结尾位置不一致: {a_path} vs {b_path}")
            x = original.astype(np.float64)
            y = rebuilt.astype(np.float64)
            diff = x - y
            squared_error += float(diff @ diff)
            squared_original += float(x @ x)
    return _sqrt_ratio(squared_error, squared_original)


def speed_mbytes_per_s(nbytes: int, seconds: float) -> float:
    """按 1e6 字节每秒的十进制口径换算吞吐。

    Args
    ----
    nbytes :
        处理量字节数。
    seconds :
        耗时秒数。
    """
    return nbytes / seconds / 1e6


@dataclass(frozen=True, slots=True)
class Score:
    """子赛道得分：total 为主口径，total_sensitivity 为 PRD 超 10% 判零的敏感性口径。"""

    cr: float
    prd: float
    s_comp: float
    s_decomp: float
    total: float
    total_sensitivity: float


def subtrack_score(cr: float, prd: float, s_comp: float, s_decomp: float) -> Score:
    """按 readme 分段公式计算子赛道得分，主口径在 PRD>100% 或 CR<5 时判零。

    Args
    ----
    cr :
        压缩比。
    prd :
        开方小数比值的 PRD，0.1 即 10%。
    s_comp :
        压缩吞吐，单位 MB/s。
    s_decomp :
        解压吞吐，单位 MB/s。
    """
    terms = (0.4 * cr / _CR_BASELINE
             + 0.4 * 0.2 / (prd + _PRD_SHIFT)
             + 0.1 * s_comp / _SPEED_BASELINE
             + 0.1 * s_decomp / _SPEED_BASELINE)
    total = 0.0 if (prd > _ZERO_PRD_MAIN or cr < _ZERO_CR) else terms
    total_sensitivity = 0.0 if (prd > _ZERO_PRD_SENSITIVITY or cr < _ZERO_CR) else terms
    return Score(cr=cr, prd=prd, s_comp=s_comp, s_decomp=s_decomp,
                 total=total, total_sensitivity=total_sensitivity)


def total_score(scores: dict[str, float]) -> float:
    """按 np 0.4 / ieeg 0.3 / flex 0.3 加权汇总子赛道得分。

    Args
    ----
    scores :
        子赛道名到得分的映射，键必须恰为 np / ieeg / flex。
    """
    missing = sorted(set(_TOTAL_WEIGHTS) - set(scores))
    unexpected = sorted(set(scores) - set(_TOTAL_WEIGHTS))
    if missing or unexpected:
        raise ValueError(f"子赛道键不符: 缺少 {missing}，多余 {unexpected}，"
                         f"应为 {sorted(_TOTAL_WEIGHTS)}")
    return sum(weight * scores[name] for name, weight in _TOTAL_WEIGHTS.items())


@dataclass(frozen=True, slots=True)
class NpMetrics:
    """np 子赛道装载结果，PRD 为全流口径。"""

    prd: float
    n_channels: int
    n_samples_per_channel: int


def _meta_value(meta_path: str, key: str) -> str:
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            name, sep, value = line.partition("=")
            if sep and name.strip() == key:
                return value.strip()
    raise ValueError(f"{meta_path} 缺少键 {key}")


def np_metrics(orig_bin: str, orig_meta: str, decomp_bin: str,
               block: int = DEFAULT_BLOCK_BYTES) -> NpMetrics:
    """np 子赛道装载：两侧按 .meta 列数的 int16 行流式对齐，全流进 PRD。

    第 385 列为同步通道，赛题口径不做排除，全流参与 PRD。

    Args
    ----
    orig_bin :
        原始 .ap.bin。
    orig_meta :
        原始 .ap.meta，提供列数键 nSavedChans。
    decomp_bin :
        重建 .ap.bin。
    block :
        单次读取的字节上限。
    """
    n_channels = int(_meta_value(orig_meta, _NP_CHANNEL_KEY))
    row_bytes = n_channels * np.dtype("<i2").itemsize
    size = os.path.getsize(orig_bin)
    if size % row_bytes:
        raise ValueError(f"{orig_bin} 字节数 {size} 不是 {n_channels} 列 int16 行宽 "
                         f"{row_bytes} 的整数倍")
    prd = prd_streaming(orig_bin, decomp_bin, np.dtype("<i2"), block)
    return NpMetrics(prd=prd, n_channels=n_channels, n_samples_per_channel=size // row_bytes)


def _resolve_edf(input_path: str, option_name: str) -> str:
    """接受单个 EDF 路径，或恰好含一个 EDF 的目录，与官方 resolve_edf 同口径。"""
    source = os.path.abspath(input_path)
    if os.path.isfile(source):
        return source
    if not os.path.isdir(source):
        raise FileNotFoundError(f"{option_name} 不存在: {input_path}")
    found = []
    for root, _dirs, files in os.walk(source):
        found.extend(os.path.join(root, name) for name in files
                     if name.lower().endswith(".edf"))
    if not found:
        raise FileNotFoundError(f"{option_name} 之下未找到 EDF: {input_path}")
    if len(found) != 1:
        listing = "\n  ".join(os.path.relpath(path, source) for path in sorted(found))
        raise ValueError(f"{option_name} 必须恰好含一个 EDF，实际 {len(found)} 个:\n  {listing}")
    return found[0]


def _gain_offset(header: dict) -> tuple[float, float]:
    pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
    dmin, dmax = int(header["digital_min"]), int(header["digital_max"])
    span = dmax - dmin
    gain = (pmax - pmin) / span if span != 0 else 1.0
    return gain, pmin - gain * dmin


def _edf_layout(reader: pyedflib.EdfReader) -> tuple[int, int, list[dict]]:
    """解析 EDF 数据区几何：记录数、单记录字节数与逐通道布局表。"""
    n_records = int(reader.datarecords_in_file)
    signals = []
    total_samples = 0
    for index in range(reader.signals_in_file):
        header = reader.getSignalHeader(index)
        samples = int(reader.samples_in_datarecord(index))
        gain, offset = _gain_offset(header)
        signals.append({
            "index": index,
            "label": header["label"],
            "samples_per_record": samples,
            "physical_min": float(header["physical_min"]),
            "physical_max": float(header["physical_max"]),
            "digital_min": int(header["digital_min"]),
            "digital_max": int(header["digital_max"]),
            "is_annotation": header["label"].replace(" ", "") == _EDF_ANNOTATION_LABEL,
            "gain": gain,
            "offset": offset,
        })
        total_samples += samples
    bytes_per_record = total_samples * np.dtype("<i2").itemsize
    return n_records, bytes_per_record, signals


def _validate_edf_layout(n_orig: int, n_decomp: int,
                         orig_signals: list[dict], decomp_signals: list[dict]) -> None:
    """布局不一致时拒绝计算 PRD，字段集与官方 _validate_layout 一致。"""
    if n_orig != n_decomp:
        raise ValueError(f"记录数不一致: original={n_orig}, reconstructed={n_decomp}")
    if len(orig_signals) != len(decomp_signals):
        raise ValueError(f"通道数不一致: original={len(orig_signals)}, "
                         f"reconstructed={len(decomp_signals)}")
    compare_fields = ("label", "samples_per_record", "physical_min", "physical_max",
                      "digital_min", "digital_max")
    for orig, decomp in zip(orig_signals, decomp_signals):
        for field in compare_fields:
            if orig[field] != decomp[field]:
                raise ValueError(f"通道 {orig['index']} ({orig['label']!r}) 的 {field} "
                                 f"不一致: {orig[field]!r} vs {decomp[field]!r}")


def _bytes_identical(path_a: str, path_b: str, length: int, block: int) -> bool:
    """逐块比较两文件从 0 开始的 length 字节。"""
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        remaining = length
        while remaining > 0:
            chunk = min(block, remaining)
            if fa.read(chunk) != fb.read(chunk):
                return False
            remaining -= chunk
    return True


@dataclass(frozen=True, slots=True)
class IeegMetrics:
    """ieeg 子赛道装载结果，PRD 为物理单位且仅含神经通道。"""

    prd: float
    n_records: int
    n_neural_channels: int
    n_annotation_channels: int
    header_identical: bool
    annotation_identical: bool


def ieeg_metrics(orig_dir_or_edf: str, decomp_dir: str,
                 block: int = DEFAULT_BLOCK_BYTES) -> IeegMetrics:
    """ieeg 子赛道装载：神经通道按物理单位进 PRD，标注通道与 EDF 头逐位比对。

    标注通道不进 PRD；无标注通道时 annotation_identical 记 True。PRD 与官方
    ieeg_validate.py 的 sqrt(squared_error / squared_original) 同值，校准公式
    与其 compute_gain_offset 一致。

    Args
    ----
    orig_dir_or_edf :
        原始 EDF 文件，或恰好含一个 EDF 的 BIDS 目录。
    decomp_dir :
        重建结果目录（或重建 EDF 文件），恰好含一个 EDF。
    block :
        单次读取的字节上限。
    """
    orig_path = _resolve_edf(orig_dir_or_edf, "orig_dir_or_edf")
    decomp_path = _resolve_edf(decomp_dir, "decomp_dir")

    with pyedflib.EdfReader(orig_path) as orig_reader, \
            pyedflib.EdfReader(decomp_path) as decomp_reader:
        n_orig, bpr_orig, orig_signals = _edf_layout(orig_reader)
        n_decomp, bpr_decomp, decomp_signals = _edf_layout(decomp_reader)
        _validate_edf_layout(n_orig, n_decomp, orig_signals, decomp_signals)

        orig_header_bytes = os.path.getsize(orig_path) - n_orig * bpr_orig
        decomp_header_bytes = os.path.getsize(decomp_path) - n_decomp * bpr_decomp
        header_identical = (orig_header_bytes == decomp_header_bytes
                            and _bytes_identical(orig_path, decomp_path,
                                                 orig_header_bytes, block))

        records_per_block = max(1, int(block) // bpr_orig)
        neural = [s for s in orig_signals if not s["is_annotation"]]
        annotations = [s for s in orig_signals if s["is_annotation"]]
        squared_error = 0.0
        squared_original = 0.0
        annotation_identical = True
        samples_capacity = records_per_block * max(
            (s["samples_per_record"] for s in orig_signals), default=1)
        buf_orig = np.zeros(samples_capacity, dtype=np.int32)
        buf_decomp = np.zeros(samples_capacity, dtype=np.int32)
        for first_record in range(0, n_orig, records_per_block):
            count = min(records_per_block, n_orig - first_record)
            for signal in neural + annotations:
                channel = signal["index"]
                per_channel = count * signal["samples_per_record"]
                start = first_record * signal["samples_per_record"]
                view_orig = buf_orig[:per_channel]
                view_decomp = buf_decomp[:per_channel]
                orig_reader.read_digital_signal(channel, start, per_channel, view_orig)
                decomp_reader.read_digital_signal(channel, start, per_channel, view_decomp)
                if signal["is_annotation"]:
                    annotation_identical &= bool(np.array_equal(view_orig, view_decomp))
                    continue
                gain, offset = signal["gain"], signal["offset"]
                x = view_orig.astype(np.float64) * gain + offset
                y = view_decomp.astype(np.float64) * gain + offset
                diff = x - y
                squared_error += float(diff @ diff)
                squared_original += float(x @ x)

        prd = _sqrt_ratio(squared_error, squared_original)
        return IeegMetrics(
            prd=prd,
            n_records=n_orig,
            n_neural_channels=len(neural),
            n_annotation_channels=len(annotations),
            header_identical=header_identical,
            annotation_identical=annotation_identical,
        )


def _flex_data_start(path: str) -> int:
    """定位 XML 头结束后的数据起点，闭合标签后紧邻的 CR/LF 归入头部。"""
    with open(path, "rb") as fh:
        data = bytearray()
        while len(data) < _FLEX_MAX_HEADER_BYTES:
            chunk = fh.read(min(1024 * 1024, _FLEX_MAX_HEADER_BYTES - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            pos = data.find(_FLEX_END_TAG)
            if pos >= 0:
                end = pos + len(_FLEX_END_TAG)
                while len(data) < end + 2:
                    extra = fh.read(2)
                    if not extra:
                        break
                    data.extend(extra)
                while end < len(data) and data[end] in (10, 13):
                    end += 1
                return end
    raise ValueError(f"{path} 的前 {_FLEX_MAX_HEADER_BYTES} 字节内找不到 "
                     f"{_FLEX_END_TAG!r}")


def _flex_frames(path: str, data_start: int) -> int:
    size = os.path.getsize(path)
    payload = size - data_start
    if payload < 0 or payload % _FLEX_FRAME_BYTES:
        raise ValueError(f"{path} 数据区 {payload} 字节不是帧长 {_FLEX_FRAME_BYTES} 的整数倍")
    return payload // _FLEX_FRAME_BYTES


@dataclass(frozen=True, slots=True)
class FlexMetrics:
    """flex 子赛道装载结果，PRD 为 1024 列神经段 int16 口径。"""

    prd: float
    n_frames: int
    xml_header_identical: bool
    frame_prefix_identical: bool


def flex_metrics(orig_rec: str, decomp_rec: str,
                 block: int = DEFAULT_BLOCK_BYTES) -> FlexMetrics:
    """flex 子赛道装载：帧内 1024 列 int16 神经段进 PRD，XML 头与帧前缀逐位比对。

    帧结构实测为 2070 = 22 字节前缀 + 1024 x int16，前缀不进 PRD。

    Args
    ----
    orig_rec :
        原始 .rec。
    decomp_rec :
        重建 .rec。
    block :
        单次读取的字节上限。
    """
    orig_header_bytes = _flex_data_start(orig_rec)
    decomp_header_bytes = _flex_data_start(decomp_rec)
    xml_header_identical = (orig_header_bytes == decomp_header_bytes
                            and _bytes_identical(orig_rec, decomp_rec,
                                                 orig_header_bytes, block))
    n_frames = _flex_frames(orig_rec, orig_header_bytes)
    if n_frames != _flex_frames(decomp_rec, decomp_header_bytes):
        raise ValueError(f"{orig_rec} 与 {decomp_rec} 帧数不一致")

    frames_per_block = max(1, int(block) // _FLEX_FRAME_BYTES)
    read_bytes = frames_per_block * _FLEX_FRAME_BYTES
    neural_dtype = np.dtype("<i2")
    squared_error = 0.0
    squared_original = 0.0
    frame_prefix_identical = True
    with open(orig_rec, "rb") as fa, open(decomp_rec, "rb") as fb:
        fa.seek(orig_header_bytes)
        fb.seek(decomp_header_bytes)
        while True:
            raw_orig = np.frombuffer(fa.read(read_bytes), dtype=np.uint8)
            raw_decomp = np.frombuffer(fb.read(read_bytes), dtype=np.uint8)
            if raw_orig.size == 0:
                break
            if raw_orig.size != raw_decomp.size:
                raise ValueError(f"两侧数据区结尾位置不一致: {orig_rec} vs {decomp_rec}")
            block_orig = raw_orig.reshape(-1, _FLEX_FRAME_BYTES)
            block_decomp = raw_decomp.reshape(-1, _FLEX_FRAME_BYTES)
            frame_prefix_identical &= bool(
                np.array_equal(block_orig[:, :_FLEX_PREFIX_BYTES],
                               block_decomp[:, :_FLEX_PREFIX_BYTES]))
            x = np.ascontiguousarray(block_orig[:, _FLEX_PREFIX_BYTES:]).view(neural_dtype)
            y = np.ascontiguousarray(block_decomp[:, _FLEX_PREFIX_BYTES:]).view(neural_dtype)
            xv = x.astype(np.float64).reshape(-1)
            yv = y.astype(np.float64).reshape(-1)
            diff = xv - yv
            squared_error += float(diff @ diff)
            squared_original += float(xv @ xv)
        prd = _sqrt_ratio(squared_error, squared_original)
    return FlexMetrics(prd=prd, n_frames=n_frames, xml_header_identical=xml_header_identical,
                       frame_prefix_identical=frame_prefix_identical)
