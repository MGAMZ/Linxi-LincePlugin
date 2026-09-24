"""赛道评分公式：逐 session R² 与延迟分。

官方两版赛包均未公布计分权重，本模块的 `R2_WEIGHT`、`LATENCY_WEIGHT` 为内部约定，
产出的 `session_score` 是插件内部指标，不应与官方平台分数互相解释。
"""

from __future__ import annotations

from typing import Any

import numpy as np

BIN_SIZE_MS = 20.0
R2_WEIGHT = 0.95
LATENCY_WEIGHT = 0.05
LEVELS = ("easy", "hard", "normal")
TASKS = ("MA_CO", "MA_RT")


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} != {y_pred.shape}")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_latency_score(
    latency_per_bin_ms: float, bin_size_ms: float = BIN_SIZE_MS
) -> float:
    return float(np.clip(1.0 - latency_per_bin_ms / bin_size_ms, 0.0, 1.0))


def score_session(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    total_latency_ms: float,
    session_id: str,
    bin_size_ms: float = BIN_SIZE_MS,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.ndim != 2 or y_true.shape[1] != 2 or y_pred.shape != y_true.shape:
        raise ValueError(f"Expected matching (T, 2) arrays, got {y_true.shape}, {y_pred.shape}")
    if len(y_true) == 0 or not np.isfinite(y_pred).all():
        raise ValueError("Prediction is empty or contains NaN/Inf")

    r2_x = compute_r2(y_true[:, 0], y_pred[:, 0])
    r2_y = compute_r2(y_true[:, 1], y_pred[:, 1])
    r2_mean_raw = (r2_x + r2_y) / 2.0
    r2_mean = max(r2_mean_raw, 0.0)
    latency_per_bin_ms = float(total_latency_ms) / len(y_true)
    latency_score = compute_latency_score(latency_per_bin_ms, bin_size_ms)
    final_score = R2_WEIGHT * r2_mean + LATENCY_WEIGHT * latency_score
    return {
        "session_id": session_id,
        "n_bins": len(y_true),
        "r2_x": float(r2_x),
        "r2_y": float(r2_y),
        "r2_mean_raw": float(r2_mean_raw),
        "r2_mean": float(r2_mean),
        "total_latency_ms": float(total_latency_ms),
        "latency_per_bin_ms": latency_per_bin_ms,
        "latency_score": latency_score,
        "session_score": float(final_score),
    }
