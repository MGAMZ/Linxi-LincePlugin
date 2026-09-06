"""赛题 NWB → 临析内部表示的载入算子。

赛题神经输入是 ``acquisition/binned_spikes``（uint8 计数普通 TimeSeries），文件内没有
ElectricalSeries / electrodes / units 表，上游 ``LoadNWB`` 对这类文件不可用；本算子自研
pynwb 只读载入，把三域投影到 LinshuFile 声明字段：

- X (T,512) uint8 → ``ecephys[recording_key].electrophysiology``（真实绝对时间戳写入
  ``TimeSeries.timestamps``，不伪造均匀网格）＋ spikeinterface 活槽 ``context.recording``（仅 query 侧）；
- cursor_vel x/y、cursor_pos x/y、eval_mask → ``auxiliary_channels``，dtype 与单位保真；
- intervals/trials → 该记录的 ``events``（query 侧同步写顶层 ``context.trials``），
  start/stop 保留原始绝对 epoch 秒，trial_id 规整为 int64。

``input_path`` 必须逐步骤显式传参：本算子不读也不写 ``context.run_state.input_path``，
同一 pipeline 内 query / support 两个 load 步骤因此互不污染。

session 标识由 NWB 所在目录相对 ``data_root`` 的路径派生（赛题文件 ``nwbfile.session_id=None``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import spikeinterface.core as sc
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import EcephysRecording, TimeIntervals, TimeSeries
from pynwb import NWBHDF5IO

if TYPE_CHECKING:
    import pandas as pd
    from linxi.fabric.linxi_context import LinxiContext

_SPIKE_COUNTS_UNIT = "spike counts"
_AUXILIARY_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cursor_vel", ("x", "y")),
    ("cursor_pos", ("x", "y")),
    ("eval_mask", ()),
)


@dataclass(frozen=True, slots=True)
class LinceSessionArrays:
    """单个赛题 NWB 的只读物化结果，全部数值已脱离 HDF5 句柄。"""

    neural: np.ndarray
    timestamps: np.ndarray
    trials: "pd.DataFrame"
    aux: dict[str, tuple[np.ndarray, np.ndarray, str | None]]


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
    """按平台 loader（data.py:158-203）同一口径只读物化一个赛题 session。"""
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

        aux: dict[str, tuple[np.ndarray, np.ndarray, str | None]] = {}
        for name, axes in _AUXILIARY_SOURCES:
            container = _require_acquisition(nwbfile, name, path)
            if axes:
                for axis in axes:
                    child = container.time_series.get(axis)  # pyright: ignore[reportAttributeAccessIssue]
                    if child is None:
                        raise ValueError(f"NWB missing acquisition/{name}/{axis}: {path}")
                    key = f"{name}_{axis}"
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
    return LinceSessionArrays(neural=neural, timestamps=timestamps, trials=trials, aux=aux)


def estimate_bin_rate(timestamps: np.ndarray) -> float:
    """从相邻 bin 间隔的众数估计名义采样率（trial 内 20 ms → 50 Hz）。

    绝对时间戳在 trial 边界存在 ~1.7 s 间隙，不能用首尾跨度均摊估计。
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
    """bin→trial 投影，与平台 loader（data.py:181-188）逐位同语义：闭区间、按表序覆盖、表外为 -1。"""
    trial_ids = np.full(len(timestamps), -1, dtype=np.int64)
    for row in trials.itertuples(index=False):
        mask = (timestamps >= row.start_time) & (timestamps <= row.stop_time)
        trial_ids[mask] = int(row.trial_id)
    return trial_ids


def _session_label(path: Path, data_root: str) -> str:
    directory = path.resolve().parent
    try:
        return directory.relative_to(Path(data_root).resolve()).as_posix()
    except ValueError:
        return directory.name


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadLinceSession(DefaultProcessor):
    """把一个赛题 session 的 NWB 投影进临析内部表示的 LOAD 算子。

    Parameters
    ----------
    input_path:
        session NWB 文件路径。必须显式传入；本算子刻意不消费 ``run_state.input_path``，以免多 load 步骤相互污染。
    recording_key:
        写入 ``context.ecephys`` 的键。query = 评测侧（同时填 ``context.recording`` 与顶层 ``context.trials``）；support = 校准侧（只写自己的键）。
    data_root:
        数据根，用于派生 session 标识。必须显式传入；本仓库不内置任何机器本地数据根默认值。
    """

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
                "context.run_state.input_path is neither read nor written so that multiple "
                "load steps in one pipeline stay independent."
            )
        if self.data_root is None:
            raise ValueError(
                "LoadLinceSession requires an explicit `data_root` processor param; the session "
                "label is derived from the NWB directory relative to it, and the plugin ships "
                "no machine-local default."
            )
        arrays = read_lince_nwb(self.input_path)
        rate = estimate_bin_rate(arrays.timestamps)
        recording = sc.NumpyRecording(arrays.neural, sampling_frequency=rate, t_starts=[float(arrays.timestamps[0])])
        intervals = TimeIntervals(name="trials", table=arrays.trials)

        ecephys = EcephysRecording.from_spikeinterface_recording(recording, name="binned_spikes", unit=_SPIKE_COUNTS_UNIT)
        ecephys.electrophysiology.timestamps = arrays.timestamps
        ecephys.electrophysiology.starting_time = None
        ecephys.events = intervals
        for key, (data, child_ts, unit) in arrays.aux.items():
            ecephys.auxiliary_channels[key] = TimeSeries(name=key, data=data, timestamps=child_ts, unit=unit)

        context.ecephys[self.recording_key] = ecephys
        if self.recording_key == "query":
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
]
