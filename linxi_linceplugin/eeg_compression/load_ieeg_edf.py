"""Compress2BCI ieeg 子赛道 BIDS EDF（含七类伴生元数据）投影进临析内部表示的 LOAD 阶段载入算子。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyedflib
import xarray as xr
from linxi.logger import logger
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor
from linshu_format.core import ElectrodeTable, ElectricalSignalSeries, TimeSeries, iEEGRecording

from .scoring import _edf_layout, _resolve_edf

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["LoadCompress2BciIeegEdf", "IeegEdfArrays", "read_ieeg_edf"]

_SIGNAL_NAME = "ieeg_raw"

# BIDS-iEEG 六类伴生元数据（第七类为 EDF 本体，见 compliance.IEEG_FILE_KINDS）
_SIDECAR_KINDS = ("_channels.tsv", "_channels.json", "_electrodes.tsv",
                  "_coordsystem.json", "_ieeg.json", "_scans.tsv")

# EDF 物理维度到 TimeSeries.unit 全名的映射，BIDS iEEG 惯例为微伏
_EDF_DIMENSION_UNITS = {"uV": "microvolts", "µV": "microvolts", "mV": "millivolts", "V": "volts"}


def _collect_sidecars(root: str) -> dict[str, str]:
    """目录树内按 BIDS 后缀检索伴生元数据，每类至多一个，多余即歧义。"""
    found: dict[str, list[str]] = {kind: [] for kind in _SIDECAR_KINDS}
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            for kind in _SIDECAR_KINDS:
                if fname.endswith(kind):
                    found[kind].append(os.path.join(dirpath, fname))
    result: dict[str, str] = {}
    for kind, paths in found.items():
        if len(paths) > 1:
            raise ValueError(f"目录 {root} 内 {kind} 不唯一: {sorted(paths)}")
        if paths:
            result[kind] = paths[0]
    return result


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _session_label(root: str, edf_path: str) -> str:
    """BIDS 会话标识：root 到 EDF 的目录链中的 sub-*/ses-* 两级（root 自身命名合规时计入）；
    无 BIDS 目录时取 EDF 父目录名。"""
    bids = [d for d in Path(os.path.relpath(os.path.dirname(edf_path), root)).parts
            if d.startswith(("sub-", "ses-"))]
    root_name = Path(root).name
    if root_name.startswith(("sub-", "ses-")):
        bids = [root_name, *bids]
    if bids:
        return "/".join(bids)
    return Path(edf_path).parent.name


@dataclass(frozen=True, slots=True)
class IeegEdfArrays:
    """单个 ieeg EDF 的载入结果，信号为 int16 原始码值，物理换算参数在 gain/offset。"""

    signal: np.ndarray
    channel_labels: tuple[str, ...]
    sampling_frequency: float
    gain: float
    offset: float
    unit: str
    channels_table: pd.DataFrame | None
    ieeg_json: dict[str, object]
    coordsystem_json: dict[str, str]
    session_identifier: str
    annotation_signals: dict[str, np.ndarray] = field(default_factory=dict)
    acquisition_time: datetime | None = None


def read_ieeg_edf(input_path: str) -> IeegEdfArrays:
    """读取 BIDS 受试者目录或单个 EDF：神经通道进主信号矩阵，七类元数据读取注入。

    标注通道（label 去空格后等于 EDFAnnotations）不进主矩阵，逐通道存入
    annotation_signals；本数据集 27 名均无标注通道。
    """
    edf_path = _resolve_edf(input_path, "input_path")
    root = input_path if os.path.isdir(input_path) else os.path.dirname(edf_path)
    sidecars = _collect_sidecars(os.path.abspath(root))

    with pyedflib.EdfReader(edf_path) as reader:
        n_records, _bpr, signals = _edf_layout(reader)
        neural = [s for s in signals if not s["is_annotation"]]
        annotated = [s for s in signals if s["is_annotation"]]
        if not neural:
            raise ValueError(f"EDF 无神经通道: {edf_path}")
        sprs = {s["samples_per_record"] for s in neural}
        if len(sprs) != 1:
            raise ValueError(f"神经通道逐记录采样数不一致 {sorted(sprs)}，不支持非均匀 EDF: {edf_path}")
        params = {(s["gain"], s["offset"]) for s in neural}
        if len(params) != 1:
            raise ValueError(f"神经通道换算参数不一致 ({len(params)} 种)，无法用单一 "
                             f"conversion 表示: {edf_path}")
        gain, offset = next(iter(params))
        dimensions = {reader.getSignalHeader(s["index"])["dimension"] for s in neural}
        if len(dimensions) != 1:
            raise ValueError(f"神经通道物理维度不一致: {sorted(dimensions)}")
        dimension = next(iter(dimensions))
        if dimension not in _EDF_DIMENSION_UNITS:
            raise ValueError(f"EDF 物理维度 {dimension!r} 不在映射表 {sorted(_EDF_DIMENSION_UNITS)}")
        labels = tuple(str(s["label"]) for s in neural)
        rates = {float(reader.getSampleFrequency(s["index"])) for s in neural}
        if len(rates) != 1:
            raise ValueError(f"神经通道采样率不一致: {sorted(rates)}")
        rate = next(iter(rates))
        n_samples = n_records * next(iter(sprs))
        signal = np.empty((n_samples, len(neural)), dtype=np.int16)
        buf = np.empty(n_samples, dtype=np.int32)
        for col, sig in enumerate(neural):
            reader.read_digital_signal(sig["index"], 0, n_samples, buf)
            int16_info = np.iinfo(np.int16)
            if buf.min() < int16_info.min or buf.max() > int16_info.max:
                raise ValueError(f"通道 {sig['label']!r} 码值越出 int16 范围: {edf_path}")
            signal[:, col] = buf
        annotation_signals: dict[str, np.ndarray] = {}
        for sig in annotated:
            n_ann = n_records * sig["samples_per_record"]
            ann_buf = np.zeros(n_ann, dtype=np.int32)
            reader.read_digital_signal(sig["index"], 0, n_ann, ann_buf)
            annotation_signals[str(sig["label"])] = ann_buf.astype(np.int16)

    channels_table: pd.DataFrame | None = None
    channels_path = sidecars.get("_channels.tsv")
    if channels_path is not None:
        channels_table = _read_tsv(channels_path).set_index("name", drop=False)
        if list(channels_table["name"]) != list(labels):
            raise ValueError(f"{channels_path}: channels.tsv 通道名单与 EDF 标签逐位不一致 "
                             f"(tsv {len(channels_table)} 行, EDF {len(labels)} 通道)")

    ieeg_json: dict[str, object] = {}
    ieeg_meta_path = sidecars.get("_ieeg.json")
    if ieeg_meta_path is not None:
        with open(ieeg_meta_path, encoding="utf-8") as fh:
            ieeg_json = json.load(fh)
        declared = ieeg_json.get("SamplingFrequency")
        if isinstance(declared, (int, float)) and float(declared) != rate:
            raise ValueError(f"{ieeg_meta_path}: SamplingFrequency={declared} 与 EDF 采样率 {rate} 不一致")

    coordsystem_json: dict[str, str] = {}
    coords_path = sidecars.get("_coordsystem.json")
    if coords_path is not None:
        with open(coords_path, encoding="utf-8") as fh:
            coordsystem_json = json.load(fh)

    acquisition_time: datetime | None = None
    scans_path = sidecars.get("_scans.tsv")
    if scans_path is not None:
        scans = _read_tsv(scans_path)
        row: pd.Series | None = None
        if len(scans) == 1:
            row = scans.iloc[0]
        elif "filename" in scans.columns:
            hits = scans[scans["filename"].str.endswith(os.path.basename(edf_path))]
            if len(hits) == 1:
                row = hits.iloc[0]
            else:
                logger.warning(f"{scans_path}: 命中本 EDF 的行数为 {len(hits)}，acquisition_time 不注入")
        if row is not None:
            acq = str(row["acq_time"])
            if acq not in ("n/a", ""):
                acquisition_time = datetime.fromisoformat(acq.replace("Z", "+00:00"))

    electrodes_path = sidecars.get("_electrodes.tsv")
    if electrodes_path is not None and channels_table is not None:
        electrodes = _read_tsv(electrodes_path).set_index("name", drop=False)
        for axis in ("x", "y", "z"):
            if axis in electrodes.columns:
                values = pd.to_numeric(electrodes[axis], errors="coerce")
                coords = pd.Series({name: values.get(name, np.nan) for name in channels_table["name"]})
                channels_table[f"electrode_{axis}_mm"] = coords.values

    return IeegEdfArrays(
        signal=signal,
        channel_labels=labels,
        sampling_frequency=rate,
        gain=gain,
        offset=offset,
        unit=_EDF_DIMENSION_UNITS[dimension],
        channels_table=channels_table,
        ieeg_json=ieeg_json,
        coordsystem_json=coordsystem_json,
        session_identifier=_session_label(os.path.abspath(root), edf_path),
        annotation_signals=annotation_signals,
        acquisition_time=acquisition_time,
    )


def _opt_str(value: object) -> str | None:
    text = str(value).strip()
    return None if text in ("", "n/a", "N/A", "nan", "None") else text


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _ieeg_recording(arrays: IeegEdfArrays) -> iEEGRecording:
    series = ElectricalSignalSeries(
        name=_SIGNAL_NAME,
        data=xr.DataArray(arrays.signal, dims=("time", "channel"),
                         coords={"channel": list(arrays.channel_labels)}),
        rate=arrays.sampling_frequency,
        unit=arrays.unit,
        conversion_scale_factor=arrays.gain,
        conversion_offset=arrays.offset,
        description="EDF 原始 int16 码值，物理值 = 码值 x conversion_scale_factor + conversion_offset",
    )
    electrodes: ElectrodeTable | None = None
    if arrays.channels_table is not None:
        electrodes = ElectrodeTable(
            name="ieeg_channels",
            table=xr.Dataset.from_dataframe(arrays.channels_table.set_index("name")))
    recording = iEEGRecording(
        sampling_frequency=arrays.sampling_frequency,
        power_line_frequency=_num(arrays.ieeg_json.get("PowerLineFrequency")),
        channel_count=len(arrays.channel_labels),
        device=_opt_str(arrays.ieeg_json.get("Manufacturer")),
        reference=_opt_str(arrays.ieeg_json.get("iEEGReference")),
        coordinates_reference=_opt_str(arrays.coordsystem_json.get("iEEGCoordinateSystem")),
        electrophysiology=series,
        electrodes=electrodes,
    )
    for label, codes in arrays.annotation_signals.items():
        recording.auxiliary_channels[label] = TimeSeries(
            name=label, data=xr.DataArray(codes, dims=("time",)),
            unit=None, description="EDF 标注通道原始码值，无损保留，不进 PRD")
    return recording


@register_as_linxi_processor(stage=PROCESS_STAGES.LOAD)
class LoadCompress2BciIeegEdf(DefaultProcessor):
    """把一个 Compress2BCI ieeg 子赛道的 BIDS 受试者目录或单个 EDF 载入 `context.ieeg` 的 LOAD 算子。"""

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
                "LoadCompress2BciIeegEdf requires an explicit `input_path` processor param; "
                "accepts a BIDS subject directory or a single *.edf path."
            )
        arrays = read_ieeg_edf(self.input_path)
        context.ieeg[self.recording_key] = _ieeg_recording(arrays)
        context.session = arrays.session_identifier
        if arrays.acquisition_time is not None:
            context.acquisition_time = arrays.acquisition_time
        task_name = _opt_str(arrays.ieeg_json.get("TaskName"))
        if task_name is not None:
            context.task_name = task_name
        return context
