"""Compress2BCI flex 子赛道 Trodes .rec 投影进临析内部表示的 LOAD 阶段载入算子。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linxi_trodes._rec_header import parse_rec_header
from linshu_format.core import EcephysRecording, ElectricalSignalSeries, TimeSeries

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["LoadCompress2BciTrodesRec", "TrodesRecArrays", "read_trodes_rec"]

_SIGNAL_NAME = "rec_raw"
_PREFIX_SERIES_NAME = "frame_prefix"
_TIMESTAMP_BYTES = 4


@dataclass(frozen=True, slots=True)
class TrodesRecArrays:
    """单个 Trodes .rec 的载入结果，神经段与帧前缀均为只读 memmap，未做电压换算。"""

    neural: np.ndarray
    frame_prefix: np.ndarray
    sampling_frequency: float
    n_frames: int
    binary_offset: int
    packet_size: int
    metadata: dict[str, str]
    session_identifier: str


def read_trodes_rec(input_path: str) -> TrodesRecArrays:
    """读取 Trodes .rec：头部口径复用 linxi_trodes.parse_rec_header，帧区 memmap 解码。

    注意 parse_rec_header 返回的 header_size 是帧内设备字节前缀长度（本数据集 17，
    官方口径 device_bytes），与 XML 头长度无关；XML 头起于文件偏移 0、止于
    binary_offset（本数据集 657928），帧区几何一律以 binary_offset 为准。
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    header = parse_rec_header(input_path)
    prefix_bytes = header.packet_size - 2 * header.num_channels
    # 帧前缀 = 1 标记字节 + header_size 设备字节 + 4 字节 uint32 时间戳
    if prefix_bytes != 1 + header.header_size + _TIMESTAMP_BYTES:
        raise ValueError(f"{input_path}: 帧结构不自洽，packet_size={header.packet_size} 推得前缀 "
                         f"{prefix_bytes} 字节，与 1+{header.header_size}+{_TIMESTAMP_BYTES} 不符")

    size = os.path.getsize(input_path)
    payload = size - header.binary_offset
    if payload <= 0:
        raise ValueError(f"{input_path}: binary_offset={header.binary_offset} 后无帧区数据")
    if payload % header.packet_size:
        raise ValueError(f"{input_path}: 帧区 {payload} 字节不是 packet_size={header.packet_size} "
                         f"的整数倍")
    n_frames = payload // header.packet_size
    if (header.binary_offset + prefix_bytes) % 2:
        raise ValueError(f"{input_path}: 神经段起点 {header.binary_offset + prefix_bytes} "
                         f"未按 int16 对齐")

    neural = np.memmap(input_path, dtype="<i2", mode="r",
                       offset=header.binary_offset + prefix_bytes,
                       shape=(n_frames, header.num_channels))
    frame_prefix = np.memmap(input_path, dtype=np.uint8, mode="r",
                             offset=header.binary_offset, shape=(n_frames, prefix_bytes))
    return TrodesRecArrays(
        neural=neural,
        frame_prefix=frame_prefix,
        sampling_frequency=header.sampling_rate,
        n_frames=n_frames,
        binary_offset=header.binary_offset,
        packet_size=header.packet_size,
        metadata=dict(header.metadata),
        session_identifier=Path(input_path).stem,
    )


def _rec_recording(arrays: TrodesRecArrays) -> EcephysRecording:
    hardware = "; ".join(f"{k}={v}" for k, v in sorted(arrays.metadata.items())
                         if v and v != "-1")
    series = ElectricalSignalSeries(
        name=_SIGNAL_NAME,
        data=xr.DataArray(arrays.neural, dims=("time", "channel")),
        rate=arrays.sampling_frequency,
        description=(f"Trodes .rec 原始 int16 码值，帧几何 {arrays.packet_size} 字节 = "
                     f"{arrays.frame_prefix.shape[1]} 前缀 + {arrays.neural.shape[1]} x int16，"
                     f"XML 头 {arrays.binary_offset} 字节"),
    )
    prefix = TimeSeries(
        name=_PREFIX_SERIES_NAME,
        data=xr.DataArray(arrays.frame_prefix, dims=("time", "prefix_byte")),
        rate=arrays.sampling_frequency,
        description="帧前缀原始字节（1 标记 + 17 设备 + 4 时间戳），不透明保留",
    )
    return EcephysRecording(
        sampling_frequency=arrays.sampling_frequency,
        channel_count=arrays.neural.shape[1],
        hardware_info=hardware or None,
        electrophysiology=series,
        auxiliary_channels={_PREFIX_SERIES_NAME: prefix},
    )


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadCompress2BciTrodesRec(DefaultProcessor):
    """把一个 Compress2BCI flex 子赛道的 Trodes .rec 载入 `context.ecephys` 的 LOAD 算子。"""

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
                "LoadCompress2BciTrodesRec requires an explicit `input_path` processor param; "
                "the Trodes *.rec file must contain its XML configuration header."
            )
        arrays = read_trodes_rec(self.input_path)
        context.ecephys[self.recording_key] = _rec_recording(arrays)
        context.session = arrays.session_identifier
        return context
