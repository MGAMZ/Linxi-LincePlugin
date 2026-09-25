"""Compress2BCI np 子赛道 SpikeGLX .ap.bin/.ap.meta 文件对投影进临析内部表示的 LOAD 阶段载入算子。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import EcephysRecording, ElectricalSignalSeries, ElectrodeTable

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["LoadCompress2BciNpStream", "NpStreamArrays", "read_np_meta", "read_np_stream"]

_SIGNAL_NAME = "ap_raw"

# 载入必需的 meta 键：列数、采样率、逐带保存通道数（nAP,nLF,nSY）
_REQUIRED_META_KEYS = ("nSavedChans", "imSampRate", "snsApLfSy")


def read_np_meta(meta_path: str) -> dict[str, str]:
    """逐行解析 SpikeGLX .meta 的 key=value 对，~ 前缀长行为扩展表，整行跳过。"""
    meta: dict[str, str] = {}
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("~"):
                continue
            name, sep, value = line.partition("=")
            if sep:
                meta[name.strip()] = value.strip()
    return meta


def _require_key(meta: dict[str, str], key: str, meta_path: str) -> str:
    if key not in meta:
        raise ValueError(f"{meta_path} 缺少必需键 {key}")
    return meta[key]


def _resolve_bin_meta(input_path: str) -> tuple[str, str]:
    """定位 .ap.bin 与配套 .ap.meta，接受 bin 文件路径或含 bin 的目录路径。"""
    if os.path.isfile(input_path):
        bin_path = input_path
        if not bin_path.lower().endswith(".bin"):
            raise ValueError(f"输入文件应为 SpikeGLX *.ap.bin: {input_path}")
        meta_path = bin_path[: -len(".bin")] + ".meta"
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"输入 bin 缺少配套 .meta: {meta_path} (bin: {bin_path})")
        return bin_path, meta_path
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    bins = sorted(str(p) for p in Path(input_path).rglob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"目录内未找到 *.bin: {input_path}")
    if len(bins) > 1:
        raise ValueError(f"目录内 *.bin 不唯一: {bins}")
    bin_path = bins[0]
    meta_path = bin_path[: -len(".bin")] + ".meta"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"目录 {input_path} 内的 bin 缺少配套 .meta: {meta_path}")
    return bin_path, meta_path


@dataclass(frozen=True, slots=True)
class NpStreamArrays:
    """单个 np 子赛道文件对的载入结果，信号为 int16 原始码值，未做电压换算。"""

    signal: np.ndarray
    channel_labels: tuple[str, ...]
    sync_channel_indices: tuple[int, ...]
    sampling_frequency: float
    meta: dict[str, str]
    session_identifier: str


def read_np_stream(input_path: str) -> NpStreamArrays:
    """读取 np 文件对：meta 提供通道数/采样率/同步通道标记，bin 以只读 memmap 载入。"""
    bin_path, meta_path = _resolve_bin_meta(input_path)
    meta = read_np_meta(meta_path)
    for key in _REQUIRED_META_KEYS:
        _require_key(meta, key, meta_path)
    n_channels = int(meta["nSavedChans"])
    rate = float(meta["imSampRate"])
    n_ap, n_lf, n_sy = (int(x) for x in meta["snsApLfSy"].split(","))

    labels = tuple([f"AP{i}" for i in range(n_ap)]
                   + [f"LF{i}" for i in range(n_lf)]
                   + [f"SY{i}" for i in range(n_sy)])
    if len(labels) != n_channels:
        raise ValueError(
            f"{meta_path}: nSavedChans={n_channels} 与 snsApLfSy 各带之和 {len(labels)} 不一致，"
            f"子集选择保存（保存流非按带连续）不支持自动命名")

    size = os.path.getsize(bin_path)
    declared = meta.get("fileSizeBytes")
    if declared is not None and int(declared) != size:
        raise ValueError(f"{meta_path}: fileSizeBytes={declared} 与 {bin_path} 实际字节数 {size} 不一致")
    row_bytes = n_channels * np.dtype("<i2").itemsize
    if size % row_bytes:
        raise ValueError(f"{bin_path}: 字节数 {size} 不是行宽 {row_bytes} "
                         f"(nSavedChans={n_channels} x int16) 的整数倍")

    # snsApLfSy=384,0,1 与 snsSaveChanSubset=0:383,768 互证：保存流按 AP 带后接同步带，
    # 末位 n_sy 个通道为同步通道（本数据集第 385 通道，源索引 768）。
    signal = np.memmap(bin_path, dtype="<i2", mode="r", shape=(size // row_bytes, n_channels))
    return NpStreamArrays(
        signal=signal,
        channel_labels=labels,
        sync_channel_indices=tuple(range(n_channels - n_sy, n_channels)),
        sampling_frequency=rate,
        meta=meta,
        session_identifier=Path(bin_path).stem,
    )


def _np_recording(arrays: NpStreamArrays) -> EcephysRecording:
    is_sync = np.zeros(len(arrays.channel_labels), dtype=bool)
    is_sync[list(arrays.sync_channel_indices)] = True
    channel_table = pd.DataFrame(
        {"is_sync": is_sync}, index=pd.Index(arrays.channel_labels, name="channel_name"))
    series = ElectricalSignalSeries(
        name=_SIGNAL_NAME,
        data=xr.DataArray(arrays.signal, dims=("time", "channel"),
                         coords={"channel": list(arrays.channel_labels)}),
        rate=arrays.sampling_frequency,
        description="SpikeGLX AP 原始 int16 码值，meta imCalibrated=false，未做电压换算",
    )
    return EcephysRecording(
        sampling_frequency=arrays.sampling_frequency,
        channel_count=len(arrays.channel_labels),
        hardware_info=f"SpikeGLX {arrays.meta.get('typeThis', 'imec')} "
                      f"sn={arrays.meta.get('imDatPrb_sn', '')}",
        electrophysiology=series,
        electrodes=ElectrodeTable(name="np_channels",
                                  table=xr.Dataset.from_dataframe(channel_table)),
    )


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadCompress2BciNpStream(DefaultProcessor):
    """把一个 Compress2BCI np 子赛道的 SpikeGLX .ap.bin/.ap.meta 文件对载入 `context.ecephys` 的 LOAD 算子。"""

    def __init__(
        self,
        input_path: str | None = None,
        recording_key: str = "query",
        name: str | None = None,
    ):
        super().__init__(name)
        self.input_path = input_path
        self.recording_key = recording_key

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.input_path is None:
            raise ValueError(
                "LoadCompress2BciNpStream requires an explicit `input_path` processor param; "
                "accepts a *.ap.bin file or a directory containing exactly one *.bin with its *.meta."
            )
        arrays = read_np_stream(self.input_path)
        context.ecephys[self.recording_key] = _np_recording(arrays)
        context.session = arrays.session_identifier
        return context
