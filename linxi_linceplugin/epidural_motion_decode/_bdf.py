"""NEO 赛题 BDF 三件套（data.bdf / evt.bdf / recordInformation.json）读取与试次表构建。

发布文件的头部不符合 EDF+/BDF+ 规范（保留区位移），严格校验打不开时改读内存修复副本；试次表由成对的开始/结束 trigger 构建，Trigger 编码按范式（single / dual）区分语义。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

__all__ = ["EpiSessionArrays", "TRIGGER_TO_LABEL", "build_trials", "read_epi_session"]

_BDF_DATETIME_RE = re.compile(rb"\d{2}\.\d{2}\.\d{2}\d{2}\.\d{2}\.\d{2}")
_START_CODES = (1, 3, 5, 7)
TRIGGER_TO_LABEL = {
    "single": {1: 0, 3: 2, 5: 1, 7: 3},
    "dual": {1: 4, 3: 5, 5: 6, 7: 7},
}


@dataclass(frozen=True, slots=True)
class EpiSessionArrays:
    """单个 NEO session 目录的只读物化结果。"""

    signal: np.ndarray
    channel_labels: tuple[str, ...]
    sampling_frequency: float
    trials: pd.DataFrame
    paradigm: str
    session_identifier: str


def _sanitize_bdf_header(header: bytes) -> bytes:
    match = _BDF_DATETIME_RE.search(header[100:184])
    if match is None:
        raise ValueError("BDF 头部找不到日期时间字段，无法修复")
    shift = 168 - (100 + match.start())
    if not 0 < shift <= 8:
        raise ValueError(f"BDF 头部位移量异常 (shift={shift})，无法修复")
    return header[:88 - shift] + b" " * shift + header[88 - shift:256 - shift]


def _open_reader(path: str | Path) -> tuple[pyedflib.EdfReader, str | None]:
    """返回 (reader, 临时副本路径)：规范合规文件直开，否则打开头部修复副本。"""
    path = str(path)
    try:
        return pyedflib.EdfReader(path), None
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(suffix=".bdf")
    os.close(fd)
    shutil.copy2(path, tmp)
    with open(tmp, "r+b") as f:
        head = f.read(256)
        f.seek(0)
        f.write(_sanitize_bdf_header(head))
    try:
        return pyedflib.EdfReader(tmp), tmp
    except OSError:
        os.remove(tmp)
        raise


def _read_signals(path: str | Path) -> tuple[np.ndarray, list[str], float]:
    reader, tmp = _open_reader(path)
    try:
        n_ch = reader.signals_in_file
        data = np.stack([reader.readSignal(i) for i in range(n_ch)], axis=-1)
        labels = list(reader.getSignalLabels())
        rate = float(reader.getSampleFrequency(0))
    finally:
        reader.close()
        if tmp:
            os.remove(tmp)
    return data, labels, rate


def _read_event_samples(path: str | Path, begin_ms: float, rate: float) -> tuple[np.ndarray, np.ndarray]:
    reader, tmp = _open_reader(path)
    try:
        onsets, _durations, contents = reader.readAnnotations()
    finally:
        reader.close()
        if tmp:
            os.remove(tmp)
    codes = np.asarray([int(c) for c in contents], dtype=np.int64)
    samples = np.asarray((np.asarray(onsets) - begin_ms * 1e-3) * rate, dtype=np.int64)
    return samples, codes


def build_trials(start_samples: np.ndarray, codes: np.ndarray, paradigm: str,
                 sampling_frequency: float) -> pd.DataFrame:
    """成对 trigger（开始码 c、结束码 c+1，按时间序对齐）构建试次表，标签按范式编码。"""
    mapping = TRIGGER_TO_LABEL[paradigm]
    rows: list[dict[str, object]] = []
    for start_code in _START_CODES:
        starts = np.sort(start_samples[codes == start_code])
        ends = np.sort(start_samples[codes == start_code + 1])
        if len(starts) != len(ends):
            raise ValueError(
                f"trigger {start_code}/{start_code + 1} 数量不配对: {len(starts)}/{len(ends)}")
        if len(starts) != 20:
            raise ValueError(f"trigger {start_code} 应为 20 个试次，实际 {len(starts)}")
        if np.any(ends <= starts):
            raise ValueError(f"trigger {start_code} 存在结束早于开始的试次")
        for start, end in zip(starts, ends):
            rows.append({
                "start_time": float(start) / sampling_frequency,
                "stop_time": float(end) / sampling_frequency,
                "trigger": int(start_code),
                "label": int(mapping[start_code]),
            })
    trials = pd.DataFrame(rows).sort_values("start_time", kind="stable").reset_index(drop=True)
    trials.insert(0, "trial_id", np.arange(len(trials), dtype=np.int64))
    return trials


def read_epi_session(session_dir: str | Path) -> EpiSessionArrays:
    """物化一个 session 目录：信号 (T, C) 伏特、通道标签、采样率与试次表。"""
    session_dir = Path(session_dir)
    name = session_dir.name
    if name.endswith("-single-MA"):
        paradigm = "single"
    elif name.endswith("-dual-MA"):
        paradigm = "dual"
    else:
        raise ValueError(f"session 目录名应以 -single-MA 或 -dual-MA 结尾: {name}")
    record_info = json.loads((session_dir / "recordInformation.json").read_text(encoding="utf-8"))
    begin_ms = float(record_info["DataFileInformations"][0]["BeginTimeStamp"])
    signal, labels, rate = _read_signals(session_dir / "data.bdf")
    signal = signal * 1e-6
    declared = float(record_info["SampleRate"])
    if declared != rate:
        raise ValueError(f"recordInformation SampleRate={declared} 与 BDF 头部 {rate} 不一致: {name}")
    start_samples, codes = _read_event_samples(session_dir / "evt.bdf", begin_ms, rate)
    if np.any(start_samples < 0) or np.any(start_samples >= signal.shape[0]):
        raise ValueError(f"事件采样点越出数据范围 (n_samples={signal.shape[0]}): {name}")
    trials = build_trials(start_samples, codes, paradigm, rate)
    return EpiSessionArrays(
        signal=signal,
        channel_labels=tuple(labels),
        sampling_frequency=rate,
        trials=trials,
        paradigm=paradigm,
        session_identifier=name,
    )
