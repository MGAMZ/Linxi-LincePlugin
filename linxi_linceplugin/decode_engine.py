"""赛道 baseline 解码协议引擎：以预置权重执行 adapt/predict 与逐 session 评分。"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .decode_scoring import (
    EVALUATION_ORDER,
    LEVELS,
    LEVEL_TRIALS,
    TASKS,
    aggregate_scores,
    compute_latency_score,
    compute_r2,
    score_session,
)

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


def load_official_loader(baseline_code_path: str):
    root = Path(baseline_code_path)
    if not (root / "Platform").is_dir():
        raise FileNotFoundError(f"challenge baseline code tree not found: {root}")
    parent = str(root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(f"{root.name}.Platform.Platform_Implementation.data")


def weights_manifest(weights_dir: str, model: str) -> dict[str, str]:
    suffix = _WEIGHT_SUFFIX[resolve_kind(model)]
    root = Path(weights_dir)
    digest: dict[str, str] = {}
    for path in sorted(root.rglob(f"model.{suffix}")):
        digest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def make_submission(model: str, baseline_code_path: str, weights_dir: str) -> Any:
    kind = resolve_kind(model)
    model_py = Path(baseline_code_path) / "Participant" / kind / "model.py"
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
    weights_check: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    session_id = session_id_of(task_name, level, horizon, session_key)
    model = submission_factory()
    model.reset(task_name=task_name, level=level, session_id=session_id)

    adapt_start = time.perf_counter()
    model.adapt(x_support, y_support, task_name=task_name, level=level, session_id=session_id)
    adapt_seconds = time.perf_counter() - adapt_start
    if weights_check is not None:
        weights_check()

    gc.collect()
    _synchronize_device()
    start_ns = time.perf_counter_ns()
    prediction = model.predict(x_query, task_name=task_name, level=level, session_id=session_id)
    _synchronize_device()
    total_latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
    prediction = _validate_prediction(prediction, len(x_query))
    if weights_check is not None:
        weights_check()

    result = score_session(
        y_query, prediction, total_latency_ms=total_latency_ms, session_id=session_id
    )
    support_trials, query_trials = LEVEL_TRIALS[level]
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


def _load_pair(data, spec, strict_trial_counts: bool):
    query = data.load_nwb_session(spec.query_path)
    if strict_trial_counts:
        data.validate_trial_count(query, data.LEVEL_CONFIGS[spec.level].query_trials,
                                  spec.query_path)
    x_query, y_query = trial_only(query.neural, query.velocity, query.trial_ids)
    if spec.support_path is None:
        return x_query, y_query, None, None
    support = data.load_nwb_session(spec.support_path)
    if strict_trial_counts:
        data.validate_trial_count(support, data.LEVEL_CONFIGS[spec.level].support_trials,
                                  spec.support_path)
    x_support, y_support = trial_only(support.neural, support.velocity, support.trial_ids)
    overlap = sorted(
        set(np.unique(support.trial_ids[support.trial_ids >= 0]).tolist())
        & set(np.unique(query.trial_ids[query.trial_ids >= 0]).tolist())
    )
    if overlap:
        raise ValueError(
            f"Support/query trial_id overlap for {spec.session_id}: {overlap[:10]}"
        )
    return x_query, y_query, x_support, y_support


def run_sweep(
    *,
    model: str,
    baseline_code_path: str | None = None,
    data_root: str | None = None,
    weights_dir: str = "",
    task: str = "",
    level: str = "",
    horizon: str = "",
    session_key: str = "",
    strict_trial_counts: bool = True,
) -> dict[str, Any]:
    kind = resolve_kind(model)
    if task and task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
    if level and level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVELS)}")
    if baseline_code_path is None:
        raise ValueError(
            "run_sweep requires an explicit `baseline_code_path` argument (the challenge `challenge_code` "
            "directory containing Platform/ and Participant/)."
        )
    if data_root is None:
        raise ValueError(
            "run_sweep requires an explicit `data_root` argument (the challenge `challenge_data` directory)."
        )
    weights = weights_dir or default_weights_dir(data_root, model)
    data = load_official_loader(baseline_code_path)

    tasks = (task,) if task else TASKS
    discovered = {name: data.discover_task_sessions(data_root, name) for name in tasks}
    if not task:
        for name in tasks:
            data.validate_session_layout(discovered[name], strict_counts=True)

    base_manifest = weights_manifest(weights, model)
    if not base_manifest:
        raise FileNotFoundError(f"no preset {_WEIGHT_SUFFIX[kind]} weights below: {weights}")

    def weights_check() -> None:
        if weights_manifest(weights, model) != base_manifest:
            raise RuntimeError(f"preset weights changed during evaluation below: {weights}")

    def factory() -> Any:
        return make_submission(model, baseline_code_path, weights)

    factory()

    task_results: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {lv: [] for lv in LEVELS} for name in tasks
    }
    execution_log: list[str] = []
    for lv in EVALUATION_ORDER:
        if level and lv != level:
            continue
        for name in tasks:
            for spec in discovered[name][lv]:
                if horizon and spec.horizon != horizon:
                    continue
                if session_key and spec.session_key != session_key:
                    continue
                x_query, y_query, x_support, y_support = _load_pair(
                    data, spec, strict_trial_counts
                )
                if x_support is None:
                    x_support = np.empty((0, x_query.shape[1]), dtype=np.float32)
                    y_support = np.empty((0, 2), dtype=np.float32)
                result, _ = run_session(
                    task_name=name, level=lv, horizon=spec.horizon,
                    session_key=spec.session_key, x_query=x_query, y_query=y_query,
                    x_support=x_support, y_support=y_support, submission_factory=factory,
                    weights_check=weights_check,
                )
                task_results[name][lv].append(result)
                execution_log.append(spec.session_id)

    if not execution_log:
        raise ValueError(
            f"no sessions matched task={task!r} level={level!r} "
            f"horizon={horizon!r} session_key={session_key!r}"
        )

    summary: dict[str, Any] | None = None
    if not (task or level or horizon or session_key):
        summary = aggregate_scores(task_results)
    return {
        "task_results": task_results,
        "execution_log": execution_log,
        "evaluation_order": list(EVALUATION_ORDER),
        "weights_dir": weights,
        "summary": summary,
    }
