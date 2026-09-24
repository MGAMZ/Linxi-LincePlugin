"""精细手部运动解码 session trial npz 投影进临析内部表示的 LOAD 阶段载入算子。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import spikeinterface.core as sc
import xarray as xr
from linxi.logger import logger
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import BinnedSpikes, EcephysRecording, TimeIntervals, TimeSeries
from linshu_format.core.time_axis import make_time_axis

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

_PARTITION_NAMES = ("heldin", "heldout")
_CHANNEL_COUNT = 256
_N_KEYPOINTS = 21
_KEYPOINT_AXES = 3
_REQUIRED_NPZ_KEYS = ("tx", "sbp", "condition", "Ton_frame_index", "reward_frame_index", "frame_timestamps_ns")
_SOURCE_PIPELINE = "lince-finehand-tx-recount"
_SIGNAL_NAME = "tx_recount"
_SPIKE_COUNTS_UNIT = "spike counts"
_SESSION_TIME_REFERENCE = "session_start_epoch"
_BIN_TIME_REFERENCE = "bin_end"
_KEYPOINT_SERIES_NAME = "hand_keypoints"
_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class FineHandSessionArrays:
    """单个赛题 session 全部 trial npz 与任务表的只读物化结果。"""

    counts: np.ndarray
    sbp: np.ndarray
    dt_sec: np.ndarray
    time_sec: np.ndarray
    keypoint: np.ndarray | None
    keypoint_names: np.ndarray | None
    trials: pd.DataFrame
    taskinfos: pd.DataFrame
    session_id: str


def _partition_dir(root: Path) -> Path:
    found = [root / name for name in _PARTITION_NAMES if (root / name).is_dir()]
    if len(found) != 1:
        raise ValueError(f"{root}: 须恰好包含 heldin/ 或 heldout/ 一个分区子目录，实际找到 {[p.name for p in found]}")
    return found[0]


def _scalar_int(value: object, what: str, path: Path) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{path}: {what} 必须是整数标量，实际 dtype={array.dtype}，ndim={array.ndim}")
    return int(array)


@dataclass(frozen=True, slots=True)
class _TrialArrays:
    """单个 trial npz 的物化结果。"""

    tx: np.ndarray
    sbp: np.ndarray
    timestamps: np.ndarray
    condition: str
    ton: int
    reward: int
    keypoint: np.ndarray | None


def _load_trial(path: Path) -> _TrialArrays:
    with np.load(path) as data:
        missing = [key for key in _REQUIRED_NPZ_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path}: npz 缺字段 {missing}")
        return _TrialArrays(
            tx=np.asarray(data["tx"]),
            sbp=np.asarray(data["sbp"]),
            timestamps=np.asarray(data["frame_timestamps_ns"]),
            condition=str(data["condition"].item()),
            ton=_scalar_int(data["Ton_frame_index"], "Ton_frame_index", path),
            reward=_scalar_int(data["reward_frame_index"], "reward_frame_index", path),
            keypoint=np.asarray(data["keypoint"]) if "keypoint" in data else None,
        )


def _frame_durations_ns(timestamps: np.ndarray, path: Path) -> np.ndarray:
    """逐帧时长（int64 ns）：第 t 帧取与前一帧的实算差，首帧沿用第二帧的间隔。"""
    if timestamps.shape[0] < 2:
        raise ValueError(f"{path}: frame_timestamps_ns 不足 2 帧，无法逐帧实算时长")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: frame_timestamps_ns 非严格递增")
    durations = np.empty(timestamps.shape[0], dtype=np.int64)
    durations[1:] = np.diff(timestamps)
    durations[0] = durations[1]
    return durations


def _check_finite(values: np.ndarray, field: str, path: Path) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: {field} 含非有限值")


def read_finehand_session(session_dir: str | Path) -> FineHandSessionArrays:
    """读取一个赛题 session 的 taskinfos.csv 与全部 trial npz。"""
    root = Path(session_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"赛题 session 目录不存在: {root}")
    partition = _partition_dir(root)
    csv_path = partition / "taskinfos.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"{partition}: 缺 taskinfos.csv")
    taskinfos = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_columns = ["trial_index", "Ton_frame_index", "reward_frame_index", "n_frames"]
    missing_columns = [column for column in required_columns if column not in taskinfos.columns]
    if missing_columns:
        raise ValueError(f"{csv_path}: taskinfos.csv 缺列 {missing_columns}")
    trial_indices = taskinfos["trial_index"].tolist()
    if trial_indices != list(range(1, len(taskinfos) + 1)):
        raise ValueError(f"{csv_path}: trial_index 必须自 1 连续，实际 {trial_indices[:5]}... 共 {len(trial_indices)} 行")
    npz_stems = sorted(p.stem for p in partition.glob("trial_*.npz"))
    expected_stems = sorted(f"trial_{index:03d}" for index in trial_indices)
    if npz_stems != expected_stems:
        raise ValueError(f"{partition}: trial npz 文件集与 taskinfos 行集不一致: npz={npz_stems[:5]}...，csv 期望 {expected_stems[:5]}...")

    counts_parts: list[np.ndarray] = []
    sbp_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    keypoint_parts: list[np.ndarray] = []
    trial_rows: list[dict[str, object]] = []
    for row in taskinfos.itertuples(index=False):
        path = partition / f"trial_{int(row.trial_index):03d}.npz"
        trial = _load_trial(path)
        tx, sbp, timestamps, keypoint = trial.tx, trial.sbp, trial.timestamps, trial.keypoint
        n_frames = int(row.n_frames)
        if tx.ndim != 2 or tx.shape != (n_frames, _CHANNEL_COUNT):
            raise ValueError(f"{path}: tx 形状 {tx.shape} 与 n_frames={n_frames}、通道数={_CHANNEL_COUNT} 不符")
        if sbp.shape != (n_frames, _CHANNEL_COUNT):
            raise ValueError(f"{path}: sbp 形状 {sbp.shape} 与 n_frames={n_frames}、通道数={_CHANNEL_COUNT} 不符")
        if timestamps.ndim != 1 or timestamps.shape != (n_frames,) or timestamps.dtype.kind != "i":
            raise ValueError(f"{path}: frame_timestamps_ns 必须是 int dtype 一维数组且长度为 {n_frames}，实际 dtype={timestamps.dtype}，shape={timestamps.shape}")
        if keypoint is not None and keypoint.shape != (n_frames, _N_KEYPOINTS * _KEYPOINT_AXES):
            raise ValueError(f"{path}: keypoint 形状 {keypoint.shape} 与 ({n_frames}, {_N_KEYPOINTS * _KEYPOINT_AXES}) 不符")
        _check_finite(tx, "tx", path)
        _check_finite(sbp, "sbp", path)
        if keypoint is not None:
            _check_finite(keypoint, "keypoint", path)
        if int(row.Ton_frame_index) != trial.ton or int(row.reward_frame_index) != trial.reward:
            raise ValueError(f"{csv_path}: trial {row.trial_index} 的事件索引与 npz 不一致: csv=({row.Ton_frame_index}, {row.reward_frame_index})，npz=({trial.ton}, {trial.reward})")
        if not 0 <= trial.ton < n_frames or not 0 <= trial.reward < n_frames:
            raise ValueError(f"{path}: 事件索引越出 [0, {n_frames}) 帧范围")

        durations_ns = _frame_durations_ns(timestamps.astype(np.int64, copy=False), path)
        dt_sec = durations_ns.astype(np.float64) / _NS_PER_SECOND
        counts_parts.append((tx * dt_sec[:, None]).astype(np.float32))
        sbp_parts.append(sbp)
        dt_parts.append(dt_sec)
        time_parts.append(timestamps.astype(np.float64) / _NS_PER_SECOND)
        if keypoint is not None:
            keypoint_parts.append(keypoint.reshape(n_frames, _N_KEYPOINTS, _KEYPOINT_AXES))
        trial_rows.append(
            {
                "start_time": float(time_parts[-1][0]),
                "stop_time": float(time_parts[-1][-1]),
                "trial_index": int(row.trial_index),
                "Ton_frame_index": trial.ton,
                "reward_frame_index": trial.reward,
                "condition": trial.condition,
                "n_frames": n_frames,
            }
        )
    if 0 < len(keypoint_parts) < len(trial_rows):
        raise ValueError(f"{partition}: keypoint 字段仅出现于 {len(keypoint_parts)}/{len(trial_rows)} 个 trial，形态不一致")

    keypoint: np.ndarray | None = None
    keypoint_names: np.ndarray | None = None
    if len(keypoint_parts) == len(trial_rows):
        keypoint = np.concatenate(keypoint_parts, axis=0)
        names_path = root / "keypoint_names.npy"
        if not names_path.is_file():
            raise FileNotFoundError(f"{root}: trial 含 keypoint 但缺会话根文件 keypoint_names.npy")
        keypoint_names = np.load(names_path)
        if keypoint_names.ndim != 1 or keypoint_names.shape[0] != _N_KEYPOINTS:
            raise ValueError(f"{names_path}: keypoint 名单必须是 {_N_KEYPOINTS} 名一维数组，实际 shape={keypoint_names.shape}")

    return FineHandSessionArrays(
        counts=np.concatenate(counts_parts, axis=0),
        sbp=np.concatenate(sbp_parts, axis=0),
        dt_sec=np.concatenate(dt_parts, axis=0),
        time_sec=np.concatenate(time_parts, axis=0),
        keypoint=keypoint,
        keypoint_names=keypoint_names,
        trials=pd.DataFrame(trial_rows),
        taskinfos=taskinfos,
        session_id=root.name,
    )


def _binned_spikes(arrays: FineHandSessionArrays, median_dt: float, rate: float) -> BinnedSpikes:
    """TX 反算计数写根容器 `binned_spikes`：counts[t] = tx[t] × dt[t]。"""
    return BinnedSpikes(
        counts=xr.DataArray(arrays.counts, dims=("time", "channel"), coords={"channel": np.arange(_CHANNEL_COUNT)}),
        time=arrays.time_sec,
        bin_sec=median_dt,
        bin_samples=int(round(median_dt * rate)),
        sampling_frequency=rate,
        time_reference=_BIN_TIME_REFERENCE,
        dtype=str(arrays.counts.dtype),
        source_pipeline=_SOURCE_PIPELINE,
        source_sorter=None,
        source_recording_ref=arrays.session_id,
    )


def _keypoint_behavior(arrays: FineHandSessionArrays, rate: float) -> dict[str, TimeSeries]:
    """关键点序列写 behavior_recording 单条多维 TimeSeries，通道名取 keypoint_names.npy。"""
    if arrays.keypoint is None:
        logger.warning(f"{arrays.session_id}: trial npz 无 keypoint 字段，behavior_recording 关键点槽留空")
        return {}
    assert arrays.keypoint_names is not None
    n_rows = int(arrays.counts.shape[0])
    axis = make_time_axis(arrays.time_sec, n_rows=n_rows, nominal_rate=rate)
    data = xr.DataArray(
        arrays.keypoint,
        dims=("time", "keypoint", "xyz"),
        coords={"keypoint": arrays.keypoint_names, "xyz": ("xyz", ["x", "y", "z"])},
    )
    return {
        _KEYPOINT_SERIES_NAME: TimeSeries(
            name=_KEYPOINT_SERIES_NAME,
            description="21 关键点 × xyz 逐帧序列，关键点顺序取 keypoint_names.npy，坐标单位赛题数据未声明",
            data=data,
            timestamps=axis.timestamps,
            starting_time=axis.starting_time,
            rate=axis.rate,
        )
    }


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadLinceFineHandSession(DefaultProcessor):
    """把一个精细手部运动解码 session 的 trial npz 投影进临析内部表示的 LOAD 算子。"""

    def __init__(
        self,
        session_dir: str | None = None,
        recording_key: Literal["query", "support"] = "query",
        sidecar_dir: str | None = None,
        name: str | None = None,
    ):
        super().__init__(name)
        self.session_dir = session_dir
        self.recording_key = recording_key
        self.sidecar_dir = sidecar_dir

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.session_dir is None:
            raise ValueError(
                "LoadLinceFineHandSession requires an explicit `session_dir` processor param; "
                "the session directory must contain a heldin/ or heldout/ partition."
            )
        arrays = read_finehand_session(self.session_dir)
        median_dt = float(np.median(arrays.dt_sec))
        rate = 1.0 / median_dt
        recording = sc.NumpyRecording(arrays.counts, sampling_frequency=rate)
        recording.set_times(arrays.time_sec, segment_index=0, with_warning=False)
        intervals = TimeIntervals(name="trials", table=arrays.trials)
        query_side = self.recording_key == "query"

        if query_side:
            ecephys = EcephysRecording(
                sampling_frequency=rate,
                channel_count=_CHANNEL_COUNT,
                time_reference=_SESSION_TIME_REFERENCE,
            )
            context.binned_spikes = _binned_spikes(arrays, median_dt, rate)
            context.behavior_recording = _keypoint_behavior(arrays, rate)
            context.recording = recording
            context.trials = intervals
        else:
            ecephys = EcephysRecording.from_spikeinterface_recording(recording, name=_SIGNAL_NAME, unit=_SPIKE_COUNTS_UNIT)
        ecephys.events = intervals
        context.ecephys[self.recording_key] = ecephys
        context.session = arrays.session_id
        self._write_sidecars(arrays)
        return context

    def _write_sidecars(self, arrays: FineHandSessionArrays) -> None:
        """SBP 全矩阵与 44 列任务信息表的伴随导出。"""
        if self.sidecar_dir is None:
            logger.info(f"{arrays.session_id}: 未设 sidecar_dir，SBP 伴随 npy 与 44 列伴随 CSV 本次不导出")
            return
        out = Path(self.sidecar_dir)
        out.mkdir(parents=True, exist_ok=True)
        sbp_path = out / f"{arrays.session_id}_sbp.npy"
        np.save(sbp_path, arrays.sbp)
        csv_path = out / f"{arrays.session_id}_taskinfos_full.csv"
        arrays.taskinfos.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"{arrays.session_id}: 伴随导出写出 {sbp_path} 与 {csv_path}")


__all__ = ["FineHandSessionArrays", "LoadLinceFineHandSession", "read_finehand_session"]
