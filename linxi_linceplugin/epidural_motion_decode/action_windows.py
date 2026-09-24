"""按官方评测契约截取逐试次动作窗口的算子"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import TimeIntervals

from .load_session import epi_signal

if TYPE_CHECKING:
    import xarray as xr
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["EpiExtractActionWindows", "extract_trial_windows"]

_WINDOW_COLUMN = "window_signal"


def trials_frame(table: "xr.Dataset") -> pd.DataFrame:
    """试次表 Dataset 转 DataFrame。"""
    return pd.DataFrame({name: np.asarray(table[name].values) for name in table.data_vars})


def extended_intervals(base: TimeIntervals, frame: "pd.DataFrame") -> TimeIntervals:
    return TimeIntervals(name=base.name, description=base.description, table=frame)


def extract_trial_windows(signal: np.ndarray, trials: pd.DataFrame, sampling_frequency: float,
                          window_start: float, window_duration: float) -> tuple[np.ndarray, ...]:
    """逐试次截取 (window_samples, channel) 窗口，依次返回窗口堆叠、动作 trigger 起点采样点与窗口起止采样点。"""
    onset_samples = np.empty(len(trials), dtype=np.int64)
    windows = np.empty((len(trials), int(round(window_duration * sampling_frequency)),
                        signal.shape[1]), dtype=np.float32)
    begins = np.empty(len(trials), dtype=np.int64)
    for i, start_time in enumerate(trials["start_time"]):
        onset = int(round(start_time * sampling_frequency))
        begin = onset + int(round(window_start * sampling_frequency))
        end = begin + windows.shape[1]
        if begin < 0 or end > signal.shape[0]:
            raise ValueError(f"trial {trials['trial_id'].iloc[i]} 窗口越出信号范围: [{begin}, {end}) / {signal.shape[0]}")
        onset_samples[i] = onset
        begins[i] = begin
        windows[i] = signal[begin:end, :]
    return windows, onset_samples, begins, begins + windows.shape[1]


@register_as_linxi_processor(stage=PROCESS_STAGES.PREPROCESS)
class EpiExtractActionWindows(DefaultProcessor):
    """按官方评测契约把记录试次表扩展出评测窗口列的 PREPROCESS 算子。

    Parameters
    ----------
    recording_key:
        读写的 `context.ieeg` 槽键。
    window_start / window_duration:
        窗口相对动作开始 trigger 的起点与时长，单位为秒。
    """

    def __init__(
        self,
        recording_key: str = "query",
        window_start: float = 0.2,
        window_duration: float = 2.0,
        name: str | None = None,
    ):
        super().__init__(name)
        self.recording_key = recording_key
        self.window_start = window_start
        self.window_duration = window_duration

    def _process(self, context: LinxiContext) -> LinxiContext:
        slot = context.ieeg.get(self.recording_key)
        if slot is None or slot.events is None:
            raise ValueError(f"ieeg[{self.recording_key!r}] 或其 events 为空：应由 LoadLinceEpiSession 先载入")
        matrix = epi_signal(context, self.recording_key)
        signal = np.asarray(matrix.values, dtype=np.float64)
        frame = trials_frame(slot.events.table)
        sampling_frequency = float(slot.sampling_frequency)
        windows, onsets, begins, stops = extract_trial_windows(
            signal, frame, sampling_frequency, self.window_start, self.window_duration)

        frame["window_start_sample"] = begins
        frame["window_stop_sample"] = stops
        frame["sample_id"] = [f"{context.session}_onset_{s}" for s in onsets]
        frame[_WINDOW_COLUMN] = [windows[i] for i in range(len(frame))]

        intervals = extended_intervals(slot.events, frame)
        slot.events = intervals
        if self.recording_key == "query":
            context.trials = intervals
        return context
