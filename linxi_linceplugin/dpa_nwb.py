"""DPA 赛题 NWB 的读取与校验层（h5py 只读）。

读取发放率矩阵、units 表与试次表，任一结构契约不符时报错并附路径与双方值。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd

Role = Literal["train", "eval", "eval-1", "eval-2"]

NWB_RE = re.compile(
    r"^sub-(?P<subject>m\d+)_ses-(?P<date>\d{8})_task-DPA-(?P<role>train|eval|eval-1|eval-2)\.nwb$"
)
BASE_COLS = frozenset({"start_time", "stop_time", "delay_duration", "well_trained"})
LABEL_COLS = frozenset({"sample_cue", "test_cue", "is_correct", "lick_response"})
LABELLED_ROLES = frozenset({"train", "eval-1"})
FR_SERIES_PATH = "processing/ecephys/Firing_rate_1000ms"
FR_PATH = f"{FR_SERIES_PATH}/data"
TRIALS_PATH = "intervals/trials"
UNITS_PATH = "units"
SPIKE_DATASETS = frozenset({"spike_times", "spike_times_index"})
NWB_SPEC_VERSION = "2.9.0"
FR_DATASET_UNIT = "Hz"
_ROLE_BY_TEXT: dict[str, Role] = {"train": "train", "eval": "eval", "eval-1": "eval-1", "eval-2": "eval-2"}


def _text(v: object) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _integral(v: np.ndarray, what: str, path: Path) -> np.ndarray:
    """把整值 float 数组 cast 为 int64。存在非有限或非整值元素时抛 ValueError。"""
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{path}: {what} 含非有限值")
    bad = v[np.not_equal(v, np.round(v))]
    if bad.size:
        raise ValueError(f"{path}: {what} 存在非整值: {np.unique(bad)[:5]}")
    return np.round(v).astype(np.int64)


def _dataset_column(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[:]
    if values.dtype.kind in "SO":
        return np.array([_text(v) for v in values])
    return values


def parse_session_name(path: Path) -> tuple[str, str, Role]:
    """从文件名解析 (subject, session_date, role)。不符合 sub-*_ses-*_task-DPA-*.nwb 约定时抛 ValueError。"""
    m = NWB_RE.match(path.name)
    if m is None:
        raise ValueError(f"文件名不符合 sub-*_ses-*_task-DPA-*.nwb 约定: {path}")
    return m["subject"], m["date"], _ROLE_BY_TEXT[m["role"]]


def check_role_columns(path: Path, role: Role, columns: frozenset[str]) -> None:
    """文件名角色与 trials 实际列集交叉校验：train/eval-1 要求标签四列在位，eval/eval-2 要求全部缺失。"""
    expected = BASE_COLS | LABEL_COLS if role in LABELLED_ROLES else BASE_COLS
    if columns == expected:
        return
    missing = sorted(expected - columns)
    unexpected = sorted(columns - expected)
    detail = ""
    if missing:
        detail += f"；缺失列 {missing}"
    if unexpected:
        detail += f"；多余列 {unexpected}"
    raise ValueError(f"{path}: 文件名角色 {role!r} 与 trials 列集不符，期望列 {sorted(expected)}，实际列 {sorted(columns)}{detail}")


@dataclass(frozen=True, slots=True)
class DpaSessionArrays:
    """单个 DPA session NWB 的只读物化结果，全部数值已脱离 HDF5 句柄。"""

    subject: str
    session_date: str
    role: Role
    units: pd.DataFrame
    fr: np.ndarray
    trials: pd.DataFrame
    delay_per_trial: np.ndarray
    trial_slices: np.ndarray
    trial_index: np.ndarray
    bin_index: np.ndarray
    spike_pairs: np.ndarray
    spike_ends: np.ndarray
    fr_unit: str
    fr_description: str
    session_identifier: str


def read_dpa_nwb(nwb_path: str | Path) -> DpaSessionArrays:
    """以 h5py 只读物化一个 DPA session NWB，全契约校验后返回 DpaSessionArrays。"""
    p = Path(nwb_path)
    if not p.is_file():
        raise FileNotFoundError(f"DPA session NWB file not found: {p}")
    subject, session_date, role = parse_session_name(p)
    with h5py.File(p, "r") as f:
        spec_version = _text(f.attrs["nwb_version"])
        if spec_version != NWB_SPEC_VERSION:
            raise ValueError(f"{p}: nwb_version {spec_version!r} != {NWB_SPEC_VERSION!r}")
        file_subject = _text(f["general/subject/subject_id"][()])
        if file_subject != subject:
            raise ValueError(f"{p}: subject_id {file_subject!r} 与文件名解析 {subject!r} 不符")
        session_identifier = _text(f["general/session_id"][()])
        expected_session_id = f"sub-{subject}_ses-{session_date}_task-DPA"
        if session_identifier != expected_session_id:
            raise ValueError(f"{p}: session_id {session_identifier!r} 与文件名期望 {expected_session_id!r} 不符")

        trials = f[TRIALS_PATH]
        cols = tuple(_text(c) for c in trials.attrs["colnames"])
        check_role_columns(p, role, frozenset(cols))
        delay_per_trial = _integral(trials["delay_duration"][:], "delay_duration", p)
        trials_df = pd.DataFrame({c: _dataset_column(trials[c]) for c in cols}, index=trials["id"][:])
        _require_arange(trials_df.index.to_numpy(), "trials id", p)

        units = f[UNITS_PATH]
        units_df = pd.DataFrame(
            {name: _dataset_column(units[name])
             for name in sorted(units.keys())
             if isinstance(units[name], h5py.Dataset) and name not in SPIKE_DATASETS}
        )
        _require_arange(units_df["id"].to_numpy(), "units id", p)
        spike_pairs = units["spike_times"][:]
        spike_ends = units["spike_times_index"][:].astype(np.int64)
        n_spikes = _integral(units["n_spikes"][:], "n_spikes", p)
        _check_spike_counts(spike_pairs, spike_ends, n_spikes, p)
        _check_spike_windows(spike_pairs, delay_per_trial, p)

        fr_series = f[FR_SERIES_PATH]
        fr_data = fr_series["data"]
        fr_unit = _text(fr_data.attrs["unit"])
        if fr_unit != FR_DATASET_UNIT:
            raise ValueError(f"{p}: FR dataset unit {fr_unit!r} != {FR_DATASET_UNIT!r}")
        fr_description = _text(fr_series.attrs["description"])
        fr_full = fr_data[:]
        n_units = len(units_df)
        if fr_full.shape[1] != n_units + 2:
            raise ValueError(f"{p}: FR 列数 {fr_full.shape[1]} ≠ units 行数 {n_units} + 2 索引列")
        delay_sum = int(delay_per_trial.sum())
        if fr_full.shape[0] != delay_sum:
            raise ValueError(f"{p}: FR 行数 {fr_full.shape[0]} ≠ Σdelay_duration {delay_sum}")
        trial_index = fr_full[:, 0].astype(np.int64)
        bin_index = fr_full[:, 1].astype(np.int64)
        expected_trial = np.repeat(np.arange(len(delay_per_trial), dtype=np.int64), delay_per_trial)
        expected_bin = np.concatenate([np.arange(1, int(d) + 1, dtype=np.int64) for d in delay_per_trial])
        if not np.array_equal(trial_index, expected_trial):
            raise ValueError(f"{p}: FR col0 与 0-based trial 主序（逐 trial 恰占 delay 行）不符")
        if not np.array_equal(bin_index, expected_bin):
            raise ValueError(f"{p}: FR col1 与 trial 内 1-based bin 序号不符")
        fr = fr_full[:, 2:]

    ends = np.cumsum(delay_per_trial)
    trial_slices = np.stack((ends - delay_per_trial, ends), axis=1).astype(np.int64)
    return DpaSessionArrays(
        subject=subject, session_date=session_date, role=role,
        units=units_df, fr=fr, trials=trials_df,
        delay_per_trial=delay_per_trial, trial_slices=trial_slices,
        trial_index=trial_index, bin_index=bin_index,
        spike_pairs=spike_pairs, spike_ends=spike_ends,
        fr_unit=fr_unit, fr_description=fr_description, session_identifier=session_identifier,
    )


def _require_arange(ids: np.ndarray, what: str, path: Path) -> None:
    expected = np.arange(len(ids), dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f"{path}: {what} 不是恒等 arange({len(ids)})，位置↔标识对应被破坏")


def _check_spike_counts(pairs: np.ndarray, ends: np.ndarray, n_spikes: np.ndarray, path: Path) -> None:
    if not np.array_equal(np.cumsum(n_spikes, dtype=np.int64), ends):
        raise ValueError(f"{path}: spike_times_index 与 n_spikes 累计和不一致")
    if int(ends[-1]) != len(pairs):
        raise ValueError(f"{path}: spike_times_index 末元素 {int(ends[-1])} != spike_times 行数 {len(pairs)}")


def _check_spike_windows(pairs: np.ndarray, delays: np.ndarray, path: Path) -> None:
    if len(pairs) == 0:
        return
    trials_of_spike = _integral(pairs[:, 0], "spike_times col0", path)
    if trials_of_spike.min() < 0 or trials_of_spike.max() >= len(delays):
        raise ValueError(f"{path}: spike_times col0 出现越界 trial 号 [{int(trials_of_spike.min())}, {int(trials_of_spike.max())}] / n_trials={len(delays)}")
    times = pairs[:, 1]
    if (times < 0).any() or (times >= delays[trials_of_spike]).any():
        raise ValueError(f"{path}: spike_times col1 存在落于所属 trial 延迟窗 [0, delay_duration) 之外的值")
