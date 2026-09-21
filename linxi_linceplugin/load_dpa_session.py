"""DPA 赛题 NWB → 临析内部表示的载入算子（LOAD 阶段）。

读取 units-based DPA session 的发放率矩阵、units 表与 trial 表。query 侧（链上主记录）把速率矩阵写入 LinshuFile 根容器 `binned_spikes`（counts 的 channel 坐标列为完整 units 表），单元表与逐 spike 发放序列写入根容器 `neurons`，试次表同步写顶层 `context.trials` 与 `context.recording`；support 侧（校准记录）整体写入 `context.ecephys[recording_key]`，units 表挂在主信号数据的 channel 坐标列。session 标识取 NWB 文件名主干。本 schema 无 acquisition/electrodes，行为序列与试次时间投影均无对应物，不产出。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import spikeinterface.core as sc
import xarray as xr
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import BinnedSpikes, EcephysRecording, EcephysSortResult, NeuronInfo, TimeIntervals, TimeSeries, Unit

from .dpa_nwb import DpaSessionArrays, read_dpa_nwb

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

_FR_RATE = 1.0
_TIME_REFERENCE = "delay_concat_bins"
_TRIAL_INDEX_KEY = "trial_index"
_BIN_INDEX_KEY = "bin_index"
_SIGNAL_NAME = "firing_rate_1000ms"
_SPIKE_AXIS_NOTE = (
    "spike_samples 为延迟拼接轴上的秒值（=trial_slices[trial, 0] + 延迟内相对秒），sampling_frequency=1.0 Hz 使 sample 数值即秒"
)


def _plain(value: object) -> Any:
    return value.item() if isinstance(value, np.generic) else value


_COORD_RENAMES = {"channel": "unit_channel"}
# 导出侧多级坐标展开的层级序属性名（列化落盘的 units 表自描述，读回经 `units_frame_from_disk` 还原）。
UNITS_LEVELS_ATTR = "lince_units_levels"


def _channel_coord(units: pd.DataFrame) -> xr.DataArray:
    """units 表整体挂为矩阵 channel 维的多级坐标（level 0 = id，与数据列序位置对应）。

    Kilosort 列 `channel` 与坐标所挂的维度同名，xarray 拒绝该多级索引，坐标层改名为
    `unit_channel`，`dpa_units` 读回时还原列名。
    """
    index = pd.MultiIndex.from_frame(units)
    index = index.set_names([_COORD_RENAMES.get(n, n) for n in index.names])
    return xr.DataArray(index, dims=("channel",))


def _dpa_neuron_info(arrays: DpaSessionArrays) -> NeuronInfo:
    """units 表与逐单元 spike 序列物化为契约的神经元容器；spike 行块边界由 spike_times_index 给出。"""
    rows = arrays.units.to_dict("records")
    starts = arrays.trial_slices[:, 0]
    bounds = np.concatenate(([0], arrays.spike_ends))
    units: list[Unit] = []
    for i, row in enumerate(rows):
        pairs = arrays.spike_pairs[bounds[i]: bounds[i + 1]]
        spike_samples = starts[pairs[:, 0].astype(np.int64)] + pairs[:, 1]
        units.append(
            Unit(
                id=int(row["id"]),
                spike_samples=spike_samples,
                properties={k: _plain(v) for k, v in row.items() if k != "id"},
                quality=str(row["KS_Label"]),
                channel_id=int(row["channel"]),
            )
        )
    return NeuronInfo(
        sort_result=EcephysSortResult(units=units, sampling_frequency=_FR_RATE),
        other_global_properties={"spike_time_axis": _SPIKE_AXIS_NOTE},
    )


def _binned_spikes(arrays: DpaSessionArrays) -> BinnedSpikes:
    """构建 query 侧速率矩阵的根容器 `binned_spikes`。

    源数据无逐行真实时间戳（trial 延迟窗拼接后仅存索引列），`time` 置空、行→时间语义由
    `time_reference` 声明；1000 ms bin 的发放率（Hz）在数值上等于该 bin 的尖峰计数，落入计数容器。
    """
    counts = xr.DataArray(arrays.fr, dims=("time", "channel"), coords={"channel": _channel_coord(arrays.units)})
    return BinnedSpikes(
        counts=counts,
        time=None,
        bin_sec=1.0 / _FR_RATE,
        bin_samples=1,
        sampling_frequency=_FR_RATE,
        time_reference=_TIME_REFERENCE,
        dtype=str(arrays.fr.dtype),
        source_pipeline=arrays.fr_description,
        source_sorter=None,
        source_recording_ref=arrays.session_identifier,
    )


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadLinceDpaSession(DefaultProcessor):
    """把一个 DPA 赛题 session 的 NWB 投影进临析内部表示的 LOAD 算子。

    Parameters
    ----------
    input_path:
        session NWB 文件路径。必须显式传入；本算子刻意不消费 ``run_state.input_path``，以免多 load 步骤相互污染。
    recording_key:
        写入 `context.ecephys` 的键。query = 链上主记录（速率矩阵与 units 坐标写入根容器 `binned_spikes`，units 与 spike 序列写入 `neurons`，同时填 `context.recording` 与顶层 `context.trials`）；support = 校准记录（整体只写入该键，units 表挂主信号 channel 坐标）。
    """

    def __init__(
        self,
        input_path: str | None = None,
        recording_key: Literal["query", "support"] = "query",
        name: str | None = None,
    ):
        super().__init__(name)
        self.input_path = input_path
        self.recording_key = recording_key

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.input_path is None:
            raise ValueError(
                "LoadLinceDpaSession requires an explicit `input_path` processor param; "
                "context.run_state.input_path is neither read nor written."
            )
        arrays = read_dpa_nwb(self.input_path)
        recording = sc.NumpyRecording(arrays.fr, sampling_frequency=_FR_RATE)
        intervals = TimeIntervals(name="trials", table=arrays.trials)
        query_side = self.recording_key == "query"

        if query_side:
            ecephys = EcephysRecording(sampling_frequency=_FR_RATE, channel_count=int(arrays.fr.shape[1]))
        else:
            ecephys = EcephysRecording.from_spikeinterface_recording(recording, name=_SIGNAL_NAME, unit=arrays.fr_unit)
            signal = ecephys.electrophysiology
            ecephys.electrophysiology.data = signal.data.assign_coords({"channel": _channel_coord(arrays.units)})
        ecephys.events = intervals
        ecephys.auxiliary_channels[_TRIAL_INDEX_KEY] = TimeSeries(name=_TRIAL_INDEX_KEY, data=arrays.trial_index, rate=_FR_RATE)
        ecephys.auxiliary_channels[_BIN_INDEX_KEY] = TimeSeries(name=_BIN_INDEX_KEY, data=arrays.bin_index, rate=_FR_RATE)

        context.ecephys[self.recording_key] = ecephys
        if query_side:
            context.binned_spikes = _binned_spikes(arrays)
            context.neurons = _dpa_neuron_info(arrays)
            context.recording = recording
            context.trials = intervals
        context.session = Path(self.input_path).stem
        return context


def flatten_units_channel_coord(matrix: xr.DataArray) -> xr.DataArray | None:
    """矩阵 channel 坐标为 units 表 MultiIndex 时 reset_index 展开为逐 level 坐标列并返回副本，非该形态返回 None。

    xarray→zarr 编码层拒序列化 MultiIndex 坐标（``NotImplementedError: variable 'channel' is a MultiIndex,
    which cannot yet be serialized``），导出前须列化；层级序记入 `UNITS_LEVELS_ATTR` 属性供读回。
    """
    index = matrix.indexes.get("channel") if "channel" in matrix.dims else None
    if not isinstance(index, pd.MultiIndex):
        return None
    flat = matrix.reset_index("channel", drop=False)
    return flat.assign_attrs({**matrix.attrs, UNITS_LEVELS_ATTR: list(index.names)})


def units_frame(matrix: xr.DataArray) -> pd.DataFrame:
    """units 矩阵的 channel 坐标 → units 表，列集与列序同 `read_dpa_nwb` 的 units 帧。

    接受两形态：内存 MultiIndex（载入态，`_channel_coord` 产出）与落盘逐 level 坐标列
    （`flatten_units_channel_coord` 输出经 `from_zarr` 读回）。
    """
    index = matrix.indexes.get("channel") if "channel" in matrix.dims else None
    if isinstance(index, pd.MultiIndex):
        return index.to_frame(index=False).rename(columns={v: k for k, v in _COORD_RENAMES.items()})
    levels = matrix.attrs[UNITS_LEVELS_ATTR]
    frame = pd.DataFrame({name: np.asarray(matrix.coords[name].values) for name in levels})
    return frame.rename(columns={v: k for k, v in _COORD_RENAMES.items()})


def dpa_units(context: Any, side: str) -> pd.DataFrame:
    """逐侧 units 表（列集与列序同 `read_dpa_nwb` 的 units 帧）：query 侧取 `binned_spikes` 的 channel 坐标，support 侧取记录主信号的 channel 坐标。"""
    if side == "query":
        spikes = context.binned_spikes
        if spikes is None:
            raise ValueError("context.binned_spikes 为 None：query 侧 units 表应由 LoadLinceDpaSession(recording_key='query') 产出")
        coord = spikes.counts.coords["channel"]
    else:
        slot = context.ecephys.get(side)
        signal = slot.electrophysiology if slot is not None else None
        if signal is None:
            raise ValueError(f"ecephys[{side!r}] 无主信号：support 侧 units 表应由 LoadLinceDpaSession(recording_key={side!r}) 产出")
        coord = signal.data.coords["channel"]
    index = coord.to_index()
    if not isinstance(index, pd.MultiIndex):
        raise ValueError(f"{side!r} 侧矩阵的 channel 坐标不是 DPA units 多级坐标（index.name={index.name!r}）")
    return index.to_frame(index=False).rename(columns={v: k for k, v in _COORD_RENAMES.items()})


__all__ = [
    "DpaSessionArrays",
    "LoadLinceDpaSession",
    "UNITS_LEVELS_ATTR",
    "dpa_units",
    "flatten_units_channel_coord",
    "read_dpa_nwb",
    "units_frame",
]
