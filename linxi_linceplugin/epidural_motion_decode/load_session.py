"""NEO 赛题 BDF session 到临析内部表示的 LOAD 阶段载入算子。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import ElectricalSignalSeries, TimeIntervals, iEEGRecording

from ._bdf import EpiSessionArrays, read_epi_session

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

_SIGNAL_NAME = "epidural_ieeg"
_SIGNAL_UNIT = "V"
_POWER_LINE_FREQUENCY = 50.0


def epi_signal(context: LinxiContext, side: str) -> xr.DataArray:
    """取 `ieeg[side]` 主信号矩阵，形状 (time, channel)。"""
    slot = context.ieeg.get(side)
    if slot is None or slot.electrophysiology is None:
        raise ValueError(f"ieeg[{side!r}] 无主信号：应由 LoadLinceEpiSession(recording_key={side!r}) 产出")
    return slot.electrophysiology.data


def _epi_recording(arrays: EpiSessionArrays) -> iEEGRecording:
    data = xr.DataArray(
        arrays.signal,
        dims=("time", "channel"),
        coords={"channel": list(arrays.channel_labels)},
    )
    series = ElectricalSignalSeries(
        name=_SIGNAL_NAME,
        data=data,
        rate=arrays.sampling_frequency,
        unit=_SIGNAL_UNIT,
    )
    return iEEGRecording(
        sampling_frequency=arrays.sampling_frequency,
        power_line_frequency=_POWER_LINE_FREQUENCY,
        channel_count=int(arrays.signal.shape[1]),
        electrode_placement="epidural",
        electrophysiology=series,
        events=TimeIntervals(name="trials", table=arrays.trials),
    )


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadLinceEpiSession(DefaultProcessor):
    """把一个含 BDF 三件套的 NEO 赛题 session 目录投影进临析内部表示的 LOAD 算子。"""

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
                "LoadLinceEpiSession requires an explicit `input_path` processor param; "
                "context.run_state.input_path is neither read nor written."
            )
        arrays = read_epi_session(self.input_path)
        recording = _epi_recording(arrays)
        context.ieeg[self.recording_key] = recording
        if self.recording_key == "query":
            context.trials = recording.events
        context.session = arrays.session_identifier
        return context


__all__ = ["EpiSessionArrays", "LoadLinceEpiSession", "epi_signal", "read_epi_session"]
