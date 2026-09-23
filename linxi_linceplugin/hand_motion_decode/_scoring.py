"""赛道评分公式：逐 session R²、延迟分与分级聚合。"""

from __future__ import annotations

from typing import Any

import numpy as np

BIN_SIZE_MS = 20.0
R2_WEIGHT = 0.95
LATENCY_WEIGHT = 0.05
LEVEL_WEIGHTS = {"easy": 0.25, "normal": 0.45, "hard": 0.30}
EVALUATION_ORDER = ("easy", "hard", "normal")
LEVELS = ("easy", "hard", "normal")
TASKS = ("MA_CO", "MA_RT")
LEVEL_TRIALS = {"easy": (100, 25), "hard": (0, 25), "normal": (20, 25)}


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


def aggregate_scores(
    task_results: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    task_scores: dict[str, float] = {}
    level_scores_by_task: dict[str, dict[str, float]] = {}
    for task_name in TASKS:
        if task_name not in task_results:
            raise ValueError(f"Missing task result: {task_name}")
        level_scores: dict[str, float] = {}
        for level in LEVELS:
            sessions = task_results[task_name].get(level, [])
            if not sessions:
                raise ValueError(f"No scores for {task_name}/{level}")
            level_scores[level] = float(np.mean([row["session_score"] for row in sessions]))
        task_scores[task_name] = sum(
            level_scores[level] * LEVEL_WEIGHTS[level] for level in LEVELS
        )
        level_scores_by_task[task_name] = level_scores

    final_score = float(np.mean([task_scores[task] for task in TASKS]))
    return {
        "final_score": final_score,
        "task_scores": task_scores,
        "level_scores": level_scores_by_task,
        "task_aggregation": "uniform_mean",
        "level_weights": dict(LEVEL_WEIGHTS),
    }
