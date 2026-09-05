"""跨天漂移指标纯计算内核：`运动跨天解码/scripts/explore_data.py` 口径 1:1 移植。

指标族（计划 lince-plugin-linshu-phase2 行 152-158）：cos_raw、|norm_ratio-1|、
cos_centered、pearson_gap_cos。聚合口径与脚本逐式对齐（引文注明 explore_data.py 行号）；
不移植报告渲染、两两 pair_decay、MAD 离群与速度统计（非本任务指标族）。

纯 numpy 计算，无 IO、无落盘。退化输入策略（设计选择，见 task-9 证据 §5）：
零 bin → fr 全 NaN（脚本侧 np.mean 空切片同为 NaN，仅免去 RuntimeWarning）；
零通道 → 余弦零范数守卫 NaN（explore_data.py:98-99）；任务缺 train 会话 →
该任务质心不可得，logger.warning 一次并令该任务余弦类指标为 NaN（脚本会
np.vstack([]) 崩溃，算子按"受控空 + 告警"取代，属退化面而非口径面）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
from linxi.logger import logger

__all__ = [
    "DriftAnalysisResult",
    "DriftSessionInput",
    "DriftSessionMetrics",
    "centroid_from_frs",
    "compute_drift_metrics",
    "cosine",
    "parse_session_date",
    "pearson",
    "session_fr",
]

DATE_RE = re.compile(r"MA-[A-Z]{2}-(\d{8})-\d{2}")  # explore_data.py:29
TRAIN_LEVEL = "train"  # explore_data.py:148：public heldin 以 level="train" 入册
HOLDOUT_LEVELS = ("easy", "normal", "hard")  # explore_data.py:242/289

_NAN = float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """explore_data.py:95-100 原样移植：零范数守卫返回 NaN。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return _NAN
    return float(np.dot(a, b) / (na * nb))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """explore_data.py:103-113 原样移植：n<2 或常数序列返回 NaN。"""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.size < 2:
        return _NAN
    xd = xs - xs.mean()
    yd = ys - ys.mean()
    den = float(np.sqrt((xd * xd).sum() * (yd * yd).sum()))
    if den == 0.0:
        return _NAN
    return float((xd * yd).sum() / den)


def parse_session_date(session_key: str) -> date:
    """explore_data.py:87-92 原样移植：目录名缺 8 位日期串即 ValueError。"""
    match = DATE_RE.search(session_key)
    if match is None:
        raise ValueError(f"session 目录名缺少 8 位日期串: {session_key}")
    ymd = match.group(1)
    return date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))


def session_fr(x: np.ndarray) -> np.ndarray:
    """通道发放率向量：X 沿时间维取均值（explore_data.py:120）。"""
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"drift input X must be 2-D (T, C), got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.full(arr.shape[1], np.nan, dtype=np.float64)
    return arr.mean(axis=0).astype(np.float64)


def centroid_from_frs(frs: Sequence[np.ndarray]) -> np.ndarray:
    """任务质心 = train 会话 fr 按通道逐维均值（explore_data.py:160）。"""
    return np.mean(np.vstack(list(frs)), axis=0)


@dataclass(frozen=True, slots=True)
class DriftSessionInput:
    """一个 session 的漂移计算输入：原始 (T, C) 发放矩阵 + 目录名派生的分级元信息。"""

    task: str
    level: str
    session_key: str
    x: np.ndarray
    horizon: str | None = None


@dataclass(frozen=True, slots=True)
class DriftSessionMetrics:
    """逐 session 漂移画像（explore_data.py SessionRecord 的漂移指标子集）。"""

    task: str
    level: str
    horizon: str | None
    session_key: str
    session_date: date
    gap_days: float
    n_bins: int
    fr: np.ndarray
    fr_l2: float
    cos_raw: float
    cos_centered: float
    norm_ratio: float


@dataclass(frozen=True, slots=True)
class DriftAnalysisResult:
    """批次结果：逐 session 表 + 质心 + 黄金口径四族聚合（hard 12-session 均值为脚本 build_report 口径）。"""

    sessions: tuple[DriftSessionMetrics, ...]
    centroids: dict[str, np.ndarray]
    cos_raw_mean_hard: float
    one_minus_cos_raw_mean_hard: float  # 去增益残差 reorder_hard（explore_data.py:314）
    cos_centered_mean_hard: float
    norm_ratio_mean_hard: float
    norm_ratio_dev_hard: float  # |mean(norm_ratio_hard)-1|（explore_data.py:313/315 gain_hard）
    pearson_gap_cos: float  # query 全体 gap_days×cos_raw 相关（explore_data.py:289-293 decay_corr）


def _gmean(rows: Sequence[DriftSessionMetrics], attr: str) -> float:
    """explore_data.py:304-306 gmean：NaN 先滤、均值后出；空集 NaN。"""
    vals = [getattr(r, attr) for r in rows if np.isfinite(getattr(r, attr))]
    return float(np.mean(vals)) if vals else _NAN


def compute_drift_metrics(inputs: Sequence[DriftSessionInput]) -> DriftAnalysisResult:
    """按 explore_data.py 的加载→质心→逐 session→分级聚合顺序计算全套漂移指标。"""
    tasks = sorted({i.task for i in inputs})

    frs: list[np.ndarray] = [session_fr(i.x) for i in inputs]
    dates: list[date] = [parse_session_date(i.session_key) for i in inputs]

    centroids: dict[str, np.ndarray] = {}
    train_last: dict[str, date] = {}
    for task in tasks:
        train_frs = [f for f, i in zip(frs, inputs) if i.task == task and i.level == TRAIN_LEVEL]
        if not train_frs:
            logger.warning(
                f"[drift] task {task!r} 无 level='train' 会话，质心不可得；该任务余弦类指标置 NaN"
            )
            continue
        centroids[task] = centroid_from_frs(train_frs)
        train_last[task] = max(d for d, i in zip(dates, inputs) if i.task == task and i.level == TRAIN_LEVEL)

    sessions: list[DriftSessionMetrics] = []
    for i, fr, d in zip(inputs, frs, dates):
        cent = centroids.get(i.task)
        if cent is None:
            cos_raw = cos_centered = norm_ratio = _NAN
        else:
            cos_raw = cosine(fr, cent)  # explore_data.py:164
            cos_centered = cosine(fr - fr.mean(), cent - cent.mean())  # :165 先去直流再取余弦
            c_norm = float(np.linalg.norm(cent))  # :166-167 质心零范数 → NaN
            norm_ratio = float(np.linalg.norm(fr)) / c_norm if c_norm else _NAN
        gap_days = (d - train_last[i.task]).days if i.task in train_last else _NAN  # :168
        sessions.append(
            DriftSessionMetrics(
                task=i.task, level=i.level, horizon=i.horizon, session_key=i.session_key,
                session_date=d, gap_days=float(gap_days), n_bins=int(np.asarray(i.x).shape[0]),
                fr=fr, fr_l2=float(np.linalg.norm(fr)),
                cos_raw=cos_raw, cos_centered=cos_centered, norm_ratio=norm_ratio,
            )
        )

    # gap×cos 相关：全体 query session（easy/normal/hard），cos_raw 非有限的成对剔除
    # （explore_data.py:289-293：先滤 isfinite(cos_raw)，gap 与 cos 同序过滤）
    heldout = [m for m in sessions if m.level in HOLDOUT_LEVELS]
    cos_list = [m.cos_raw for m in heldout if np.isfinite(m.cos_raw)]
    gap_for_cos = [m.gap_days for m in heldout if np.isfinite(m.cos_raw)]
    pearson_gap_cos = pearson(gap_for_cos, cos_list)

    hard = [m for m in sessions if m.level == "hard"]  # :300，跨两任务合并的 hard 12-session 均值
    cos_raw_mean_hard = _gmean(hard, "cos_raw")
    norm_ratio_mean_hard = _gmean(hard, "norm_ratio")
    return DriftAnalysisResult(
        sessions=tuple(sessions),
        centroids=centroids,
        cos_raw_mean_hard=cos_raw_mean_hard,
        one_minus_cos_raw_mean_hard=1.0 - cos_raw_mean_hard,  # :314
        cos_centered_mean_hard=_gmean(hard, "cos_centered"),  # :311
        norm_ratio_mean_hard=norm_ratio_mean_hard,
        norm_ratio_dev_hard=abs(norm_ratio_mean_hard - 1.0),  # :315
        pearson_gap_cos=pearson_gap_cos,
    )
