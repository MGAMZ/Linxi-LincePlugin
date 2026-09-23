"""读取赛题 NWB 至临析内部表示的载入算子"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import spikeinterface.core as sc
import xarray as xr
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import BinnedSpikes, EcephysRecording, TimeIntervals, TimeSeries
from linshu_format.core.time_axis import authoritative_axis, make_time_axis
from pynwb import NWBHDF5IO

if TYPE_CHECKING:
    import pandas as pd
    from linxi.fabric.linxi_context import LinxiContext

_SPIKE_COUNTS_UNIT = "spike counts"
_SESSION_TIME_REFERENCE = "session_start_epoch"
# 元素依次为 acquisition 名、分量轴后缀、是否行为序列。
_AUXILIARY_SOURCES: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("cursor_vel", ("x", "y"), True),
    ("cursor_pos", ("x", "y"), True),
    ("eval_mask", (), False),
)


def _source_keys(name: str, axes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{name}_{axis}" for axis in axes) if axes else (name,)


_BEHAVIOR_KEYS = frozenset(
    key for name, axes, is_behavior in _AUXILIARY_SOURCES if is_behavior for key in _source_keys(name, axes)
)


@dataclass(frozen=True, slots=True)
class LinceSessionArrays:
    """单个赛题 NWB 的只读物化结果。"""

    neural: np.ndarray
    timestamps: np.ndarray
    trials: "pd.DataFrame"
    aux: dict[str, tuple[np.ndarray, np.ndarray, str | None]]
    session_start_time: datetime
    source_identifier: str
    counts_provenance: str | None


def _require_acquisition(nwbfile: object, name: str, path: Path) -> object:
    series = nwbfile.acquisition.get(name)  # pyright: ignore[reportAttributeAccessIssue]
    if series is None:
        raise ValueError(f"NWB missing acquisition/{name}: {path}")
    return series


def _read_time_series(series: object, label: str, path: Path) -> tuple[np.ndarray, np.ndarray, str | None]:
    data = np.asarray(series.data[:]).reshape(-1)  # pyright: ignore[reportAttributeAccessIssue]
    timestamps = np.asarray(series.timestamps[:], dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
    raw_unit = getattr(series, "unit", None)
    unit = None if raw_unit is None else str(raw_unit)
    if len(data) != len(timestamps):
        raise ValueError(f"NWB {label} length mismatch in {path}: data={len(data)}, timestamps={len(timestamps)}")
    return data, timestamps, unit


def read_lince_nwb(nwb_path: str | Path) -> LinceSessionArrays:
    """只读物化一个赛题 session 的 NWB 数据。"""
    path = Path(nwb_path)
    if not path.is_file():
        raise FileNotFoundError(f"Lince session NWB file not found: {path}")
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        spikes = _require_acquisition(nwbfile, "binned_spikes", path)
        neural = np.asarray(spikes.data[:])  # pyright: ignore[reportAttributeAccessIssue]
        timestamps = np.asarray(spikes.timestamps[:], dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
        if neural.ndim == 2 and neural.shape[0] != len(timestamps) and neural.shape[1] == len(timestamps):
            neural = np.ascontiguousarray(neural.T)
        if neural.ndim != 2:
            raise ValueError(f"NWB binned_spikes must be 2-D (T, C), got shape {neural.shape}: {path}")
        raw_provenance = getattr(spikes, "description", None)
        counts_provenance = None if raw_provenance is None else str(raw_provenance)
        session_start_time = nwbfile.session_start_time  # pyright: ignore[reportAttributeAccessIssue]
        if not isinstance(session_start_time, datetime):
            raise ValueError(f"NWB session_start_time must be a real datetime: {path} (got {session_start_time!r})")
        source_identifier = str(nwbfile.identifier)  # pyright: ignore[reportAttributeAccessIssue]

        aux: dict[str, tuple[np.ndarray, np.ndarray, str | None]] = {}
        for name, axes, _is_behavior in _AUXILIARY_SOURCES:
            container = _require_acquisition(nwbfile, name, path)
            if axes:
                for axis, key in zip(axes, _source_keys(name, axes)):
                    child = container.time_series.get(axis)  # pyright: ignore[reportAttributeAccessIssue]
                    if child is None:
                        raise ValueError(f"NWB missing acquisition/{name}/{axis}: {path}")
                    aux[key] = _read_time_series(child, f"acquisition/{name}/{axis}", path)
            else:
                aux[name] = _read_time_series(container, f"acquisition/{name}", path)

        if nwbfile.trials is None:  # pyright: ignore[reportAttributeAccessIssue]
            raise ValueError(f"NWB missing intervals/trials: {path}")
        trials = nwbfile.trials.to_dataframe()  # pyright: ignore[reportAttributeAccessIssue]
    required = {"start_time", "stop_time", "trial_id"}
    if not required.issubset(trials.columns):
        raise ValueError(f"NWB trials table must contain {sorted(required)}: {path}")
    trials = trials.copy()
    trials["start_time"] = trials["start_time"].astype(np.float64)
    trials["stop_time"] = trials["stop_time"].astype(np.float64)
    trials["trial_id"] = trials["trial_id"].astype(np.int64)

    n_bins = len(timestamps)
    if len(neural) != n_bins:
        raise ValueError(f"NWB time dimension mismatch in {path}: binned_spikes rows={len(neural)}, timestamps={n_bins}")
    for key, (data, child_ts, _) in aux.items():
        if len(data) != n_bins or len(child_ts) != n_bins:
            raise ValueError(f"NWB time dimension mismatch in {path}: {key}={len(data)}/{len(child_ts)}, binned_spikes={n_bins}")
    for key in ("cursor_vel_x", "cursor_vel_y"):
        if not np.isfinite(aux[key][0]).all():
            raise ValueError(f"NWB contains NaN or Inf in {key}: {path}")
    return LinceSessionArrays(
        neural=neural,
        timestamps=timestamps,
        trials=trials,
        aux=aux,
        session_start_time=session_start_time,
        source_identifier=source_identifier,
        counts_provenance=counts_provenance,
    )


def estimate_bin_rate(timestamps: np.ndarray) -> float:
    """从相邻 bin 间隔的众数估计名义采样率。

    时间戳在 trial 边界存在约 1.7 s 间隙，不能按首尾跨度均摊估计。
    """
    if len(timestamps) < 2:
        raise ValueError(f"Cannot estimate bin rate from {len(timestamps)} timestamps")
    deltas = np.round(np.diff(timestamps), 6)
    values, counts = np.unique(deltas, return_counts=True)
    mode = float(values[counts.argmax()])
    if mode <= 0.0:
        raise ValueError("Timestamps are not strictly increasing; cannot estimate bin rate")
    return 1.0 / mode


def project_bins_to_trials(timestamps: np.ndarray, trials: "pd.DataFrame") -> np.ndarray:
    """bin→trial 投影。"""
    trial_ids = np.full(len(timestamps), -1, dtype=np.int64)
    for row in trials.itertuples(index=False):
        mask = (timestamps >= row.start_time) & (timestamps <= row.stop_time)
        trial_ids[mask] = int(row.trial_id)
    return trial_ids


def session_counts_matrix(context: Any, side: str) -> xr.DataArray:
    """取指定数据侧的计数矩阵，形状 (time, channel)。"""
    if side == "query":
        spikes = context.binned_spikes
        if spikes is None:
            raise ValueError("context.binned_spikes 为 None：query 侧计数矩阵应由 LoadLinceSession(recording_key='query') 产出")
        return spikes.counts
    signal = context.ecephys[side].electrophysiology
    if signal is None:
        raise ValueError(f"ecephys[{side!r}] has no electrophysiology signal")
    return signal.data


def session_axis(context: Any, side: str) -> np.ndarray:
    """取计数行时间的权威时间轴，单位秒。"""
    matrix_source: Any = context.binned_spikes if side == "query" else context.ecephys[side].electrophysiology
    if matrix_source is None:
        raise ValueError(f"{side!r} 侧计数时间轴不可得：对应容器尚未产出")
    axis, _provenance = authoritative_axis(matrix_source)
    return axis


def session_behavior(context: Any, side: str) -> dict[str, TimeSeries]:
    """取指定数据侧的行为序列，即光标速度与位置。"""
    if side == "query":
        behavior = context.behavior_recording
        if not behavior:
            raise ValueError("context.behavior_recording 为空：query 侧行为序列应由 LoadLinceSession(recording_key='query') 产出")
        return behavior
    return context.ecephys[side].auxiliary_channels


def _binned_spikes_container(arrays: LinceSessionArrays, channel_ids: np.ndarray, rate: float) -> BinnedSpikes:
    """构建 query 侧计数矩阵根容器 `binned_spikes`。"""
    spec = make_time_axis(arrays.timestamps, n_rows=int(arrays.neural.shape[0]), nominal_rate=rate)
    time = spec.timestamps if spec.timestamps is not None else np.asarray(arrays.timestamps, dtype=np.float64)
    bin_sec = 1.0 / rate
    return BinnedSpikes(
        counts=xr.DataArray(arrays.neural, dims=("time", "channel"), coords={"channel": channel_ids}),
        time=time,
        bin_sec=bin_sec,
        bin_samples=int(round(bin_sec * rate)),
        sampling_frequency=rate,
        time_reference="bin_center",
        dtype=str(arrays.neural.dtype),
        source_pipeline=arrays.counts_provenance,
        source_sorter=None,
        source_recording_ref=arrays.source_identifier,
    )


def _session_label(path: Path, data_root: str) -> str:
    directory = path.resolve().parent
    try:
        return directory.relative_to(Path(data_root).resolve()).as_posix()
    except ValueError:
        return directory.name


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadLinceSession(DefaultProcessor):
    """把一个赛题 session 的 NWB 投影进临析内部表示的 LOAD 算子。"""

    def __init__(
        self,
        input_path: str | None = None,
        recording_key: Literal["query", "support"] = "query",
        data_root: str | None = None,
        name: str | None = None,
    ):
        super().__init__(name)
        self.input_path = input_path
        self.recording_key = recording_key
        self.data_root = data_root

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.input_path is None:
            raise ValueError(
                "LoadLinceSession requires an explicit `input_path` processor param; "
                "context.run_state.input_path is neither read nor written."
            )
        if self.data_root is None:
            raise ValueError(
                "LoadLinceSession requires an explicit `data_root` processor param; the session "
                "label is derived from the NWB directory relative to it."
            )
        arrays = read_lince_nwb(self.input_path)
        rate = estimate_bin_rate(arrays.timestamps)
        recording = sc.NumpyRecording(arrays.neural, sampling_frequency=rate)
        recording.set_times(arrays.timestamps, segment_index=0, with_warning=False)
        intervals = TimeIntervals(name="trials", table=arrays.trials)
        query_side = self.recording_key == "query"

        if query_side:
            ecephys = EcephysRecording(
                sampling_frequency=rate,
                channel_count=int(arrays.neural.shape[1]),
                session_start_time=arrays.session_start_time,
                time_reference=_SESSION_TIME_REFERENCE,
            )
            behavior: dict[str, TimeSeries] = {}
        else:
            ecephys = EcephysRecording.from_spikeinterface_recording(recording, name="binned_spikes", unit=_SPIKE_COUNTS_UNIT)
            ecephys.session_start_time = arrays.session_start_time
            ecephys.time_reference = _SESSION_TIME_REFERENCE
        ecephys.events = intervals
        for key, (data, child_ts, unit) in arrays.aux.items():
            series = TimeSeries(name=key, data=data, timestamps=child_ts, rate=rate, unit=unit)
            if query_side and key in _BEHAVIOR_KEYS:
                behavior[key] = series
            else:
                ecephys.auxiliary_channels[key] = series

        context.ecephys[self.recording_key] = ecephys
        if query_side:
            context.binned_spikes = _binned_spikes_container(arrays, np.asarray(recording.get_channel_ids()), rate)
            context.behavior_recording = behavior
            context.recording = recording
            context.trials = intervals
        context.session = _session_label(Path(self.input_path), self.data_root)
        return context


__all__ = [
    "LinceSessionArrays",
    "LoadLinceSession",
    "estimate_bin_rate",
    "project_bins_to_trials",
    "read_lince_nwb",
    "session_axis",
    "session_behavior",
    "session_counts_matrix",
]
