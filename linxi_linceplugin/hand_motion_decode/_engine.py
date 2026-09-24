"""赛道 baseline 解码协议引擎。

调用约定遵循官方提交接口：reset 与 predict 全档位执行，adapt 仅在 normal 档且收到
带标签校准数据时执行，Easy 与 Hard 档不接收校准样本。
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ._scoring import LEVELS, TASKS, score_session

_KIND_DIRS = {"wf": "WF", "gru": "GRU"}
_WEIGHT_SUFFIX = {"WF": "pkl", "GRU": "pt"}


def resolve_kind(model: str) -> str:
    kind = _KIND_DIRS.get(model.strip().lower()) if isinstance(model, str) else None
    if kind is None:
        raise ValueError(f"model must be one of {sorted(_KIND_DIRS)}, got {model!r}")
    return kind


def default_weights_dir(data_root: str, model: str) -> str:
    kind = resolve_kind(model)
    return str(Path(data_root).resolve().parent / "challenge_code" / "Participant" / kind)


def session_id_of(task_name: str, level: str, horizon: str | None, session_key: str) -> str:
    parts = [task_name, level]
    if horizon:
        parts.append(horizon)
    parts.append(session_key)
    return "/".join(parts)


def weights_manifest(weights_dir: str, model: str) -> dict[str, str]:
    suffix = _WEIGHT_SUFFIX[resolve_kind(model)]
    root = Path(weights_dir)
    digest: dict[str, str] = {}
    for path in sorted(root.rglob(f"model.{suffix}")):
        digest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def make_submission(model: str, baseline_code_path: str, weights_dir: str) -> Any:
    kind = resolve_kind(model)
    participant_root = Path(baseline_code_path) / "Participant"
    if not participant_root.is_dir():
        raise FileNotFoundError(
            f"challenge Participant tree not found under baseline_code_path: {participant_root}"
        )
    model_py = participant_root / kind / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(f"baseline model source not found: {model_py}")
    suffix = _WEIGHT_SUFFIX[kind]
    root = Path(weights_dir)
    missing = [
        str(root / task / f"model.{suffix}")
        for task in TASKS
        if not (root / task / f"model.{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"preset weights missing: {missing}")
    spec = importlib.util.spec_from_file_location(
        f"lince_baseline_submission_{uuid4().hex}", model_py
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline model source: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SubmissionModel(model_dir=root)


def trial_only(neural: np.ndarray, velocity: np.ndarray, trial_ids: np.ndarray):
    mask = trial_ids >= 0
    if not np.any(mask):
        raise ValueError("NWB contains no bins assigned to a trial")
    return neural[mask], velocity[mask]


def count_trials(trial_ids: np.ndarray) -> int:
    trial_ids = np.asarray(trial_ids)
    return int(np.unique(trial_ids[trial_ids >= 0]).size)


def _validate_prediction(prediction: Any, n_bins: int) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != (n_bins, 2):
        raise ValueError(f"predict() must return shape {(n_bins, 2)}, got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("predict() returned NaN or Inf")
    return prediction


def _synchronize_device() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_session(
    *,
    task_name: str,
    level: str,
    horizon: str | None,
    session_key: str,
    x_query: np.ndarray,
    y_query: np.ndarray,
    x_support: np.ndarray,
    y_support: np.ndarray,
    submission_factory: Callable[[], Any],
    support_trials: int,
    query_trials: int,
    weights_check: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVELS)}")
    session_id = session_id_of(task_name, level, horizon, session_key)
    model = submission_factory()
    model.reset(task_name=task_name, level=level)

    has_support = x_support.shape[0] > 0
    adapt_seconds = 0.0
    if level == "normal":
        if not has_support:
            raise ValueError("level 'normal' requires labeled support samples for adapt()")
        adapt_start = time.perf_counter()
        model.adapt(x_support, y_support, task_name=task_name, level=level)
        adapt_seconds = time.perf_counter() - adapt_start
        if weights_check is not None:
            weights_check()
    elif has_support:
        raise ValueError(f"level {level!r} must not receive support samples")

    gc.collect()
    _synchronize_device()
    start_ns = time.perf_counter_ns()
    prediction = model.predict(x_query, task_name=task_name, level=level)
    _synchronize_device()
    total_latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
    prediction = _validate_prediction(prediction, len(x_query))
    if weights_check is not None:
        weights_check()

    result = score_session(
        y_query, prediction, total_latency_ms=total_latency_ms, session_id=session_id
    )
    result.update({
        "task_name": task_name,
        "level": level,
        "horizon": horizon,
        "session_key": session_key,
        "support_trials": support_trials,
        "query_trials": query_trials,
        "adapt_seconds": float(adapt_seconds),
    })
    return result, prediction
