"""NEO 赛题 BDF session 到临析内部表示的载入算子（LOAD 阶段）。

信号与试次表写入 `context.ieeg[recording_key]`（契约的 iEEGRecording 容器，electrode_placement 标注 epidural）；query 侧同时填顶层 `context.trials`。发布文件头部不合规的读取绕行在 `_bdf` 层处理，本算子不做数据修复。
"""

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
    """逐侧信号矩阵 (time, channel)：取 `ieeg[side]` 主信号。"""
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
    """把一个 NEO 赛题 session 目录（BDF 三件套）投影进临析内部表示的 LOAD 算子。

    Parameters
    ----------
    input_path:
        session 目录路径（目录名以 ``-single-MA`` 或 ``-dual-MA`` 结尾，范式由名称判定）。必须显式传入。
    recording_key:
        写入 `context.ieeg` 的键。query = 链上主记录（同时填顶层 `context.trials`）；support = 辅助记录（只写槽）。
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
