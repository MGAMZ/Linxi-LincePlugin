"""跨天漂移指标纯计算内核。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
from linxi.logger import logger

__all__ = [
    "DEFAULT_DATE_PATTERN",
    "DriftAnalysisResult",
    "DriftSessionInput",
    "DriftSessionMetrics",
    "DriftVocabulary",
    "centroid_from_frs",
    "compute_drift_metrics",
    "cosine",
    "parse_session_date",
    "pearson",
    "session_fr",
]

DEFAULT_DATE_PATTERN = r"MA-[A-Z]{2}-(\d{8})-\d{2}"
DATE_RE = re.compile(DEFAULT_DATE_PATTERN)
TRAIN_LEVEL = "train"
HOLDOUT_LEVELS = ("easy", "normal", "hard")
AGG_LEVEL = "hard"

_NAN = float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return _NAN
    return float(np.dot(a, b) / (na * nb))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """皮尔逊相关系数。"""
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


def parse_session_date(session_key: str, date_pattern: str = DEFAULT_DATE_PATTERN) -> date:
    """从 session 目录名解析日期。"""
    regex = DATE_RE if date_pattern == DEFAULT_DATE_PATTERN else re.compile(date_pattern)
    match = regex.search(session_key)
    if match is None:
        raise ValueError(f"session 目录名缺少 8 位日期串: {session_key}")
    ymd = match.group(1)
    return date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))


def session_fr(x: np.ndarray) -> np.ndarray:
    """通道发放率向量。"""
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"drift input X must be 2-D (T, C), got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.full(arr.shape[1], np.nan, dtype=np.float64)
    return arr.mean(axis=0).astype(np.float64)


def centroid_from_frs(frs: Sequence[np.ndarray]) -> np.ndarray:
    """发放率质心。"""
    return np.mean(np.vstack(list(frs)), axis=0)


@dataclass(frozen=True, slots=True)
class DriftSessionInput:
    """一个 session 的漂移计算输入。"""

    task: str
    level: str
    session_key: str
    x: np.ndarray
    horizon: str | None = None


@dataclass(frozen=True, slots=True)
class DriftVocabulary:
    """会话分级词表。"""

    date_pattern: str = DEFAULT_DATE_PATTERN
    train_level: str = TRAIN_LEVEL
    holdout_levels: tuple[str, ...] = HOLDOUT_LEVELS
    agg_level: str = AGG_LEVEL


@dataclass(frozen=True, slots=True)
class DriftSessionMetrics:
    """单个 session 的漂移指标。"""

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
    """批次漂移结果。"""

    sessions: tuple[DriftSessionMetrics, ...]
    centroids: dict[str, np.ndarray]
    cos_raw_mean_hard: float
    one_minus_cos_raw_mean_hard: float
    cos_centered_mean_hard: float
    norm_ratio_mean_hard: float
    norm_ratio_dev_hard: float
    pearson_gap_cos: float


def _gmean(rows: Sequence[DriftSessionMetrics], attr: str) -> float:
    vals = [getattr(r, attr) for r in rows if np.isfinite(getattr(r, attr))]
    return float(np.mean(vals)) if vals else _NAN


def compute_drift_metrics(
    inputs: Sequence[DriftSessionInput],
    vocabulary: DriftVocabulary = DriftVocabulary(),
) -> DriftAnalysisResult:
    """计算全套跨天漂移指标与批次聚合。"""
    tasks = sorted({i.task for i in inputs})

    frs: list[np.ndarray] = [session_fr(i.x) for i in inputs]
    dates: list[date] = [parse_session_date(i.session_key, vocabulary.date_pattern) for i in inputs]

    centroids: dict[str, np.ndarray] = {}
    train_last: dict[str, date] = {}
    for task in tasks:
        train_frs = [f for f, i in zip(frs, inputs) if i.task == task and i.level == vocabulary.train_level]
        if not train_frs:
            logger.warning(
                f"[drift] task {task!r} 无 level={vocabulary.train_level!r} 会话，质心不可得，该任务余弦类指标置 NaN"
            )
            continue
        centroids[task] = centroid_from_frs(train_frs)
        train_last[task] = max(d for d, i in zip(dates, inputs) if i.task == task and i.level == vocabulary.train_level)

    sessions: list[DriftSessionMetrics] = []
    for i, fr, d in zip(inputs, frs, dates):
        cent = centroids.get(i.task)
        if cent is None:
            cos_raw = cos_centered = norm_ratio = _NAN
        else:
            cos_raw = cosine(fr, cent)
            cos_centered = cosine(fr - fr.mean(), cent - cent.mean())
            c_norm = float(np.linalg.norm(cent))
            norm_ratio = float(np.linalg.norm(fr)) / c_norm if c_norm else _NAN
        gap_days = (d - train_last[i.task]).days if i.task in train_last else _NAN
        sessions.append(
            DriftSessionMetrics(
                task=i.task, level=i.level, horizon=i.horizon, session_key=i.session_key,
                session_date=d, gap_days=float(gap_days), n_bins=int(np.asarray(i.x).shape[0]),
                fr=fr, fr_l2=float(np.linalg.norm(fr)),
                cos_raw=cos_raw, cos_centered=cos_centered, norm_ratio=norm_ratio,
            )
        )

    heldout = [m for m in sessions if m.level in vocabulary.holdout_levels]
    cos_list = [m.cos_raw for m in heldout if np.isfinite(m.cos_raw)]
    gap_for_cos = [m.gap_days for m in heldout if np.isfinite(m.cos_raw)]
    pearson_gap_cos = pearson(gap_for_cos, cos_list)

    agg = [m for m in sessions if m.level == vocabulary.agg_level]
    cos_raw_mean_hard = _gmean(agg, "cos_raw")
    norm_ratio_mean_hard = _gmean(agg, "norm_ratio")
    return DriftAnalysisResult(
        sessions=tuple(sessions),
        centroids=centroids,
        cos_raw_mean_hard=cos_raw_mean_hard,
        one_minus_cos_raw_mean_hard=1.0 - cos_raw_mean_hard,
        cos_centered_mean_hard=_gmean(agg, "cos_centered"),
        norm_ratio_mean_hard=norm_ratio_mean_hard,
        norm_ratio_dev_hard=abs(norm_ratio_mean_hard - 1.0),
        pearson_gap_cos=pearson_gap_cos,
    )
