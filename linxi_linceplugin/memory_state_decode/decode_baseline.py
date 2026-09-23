"""DPA 赛道 baseline 分类解码的 POSTPROCESS 阶段算子。"""

from __future__ import annotations

import hashlib
import importlib
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from linxi.logger import logger
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor

from ._nwb import LABELLED_ROLES, parse_session_name

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["EVAL_METRIC_FIELDS", "DpaBaselineInfer", "load_track_pipeline", "track_source_manifest"]

EVAL_METRIC_FIELDS = (
    "split", "subject", "session_date", "mem_acc", "corr_acc",
    "session_score", "n_trials", "method",
)
_PRED_COLUMNS = ("mem_pred_lbl", "corr_pred_lbl")
_TRAIN_GLOB = "*_task-DPA-train.nwb"
_TRAIN_FILE_COUNT = 4
_TRACK_SOURCE_DIRS = ("lince_memory", "scripts")


@dataclass(frozen=True, slots=True)
class TrackPipeline:
    """赛道仓库管线与评分模块的运行时指针。"""

    train_matrix: Callable[[list[Path]], tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]]
    fit_cv: Callable[[np.ndarray, np.ndarray], tuple[Any, dict[str, Any]]]
    proxy_submission: Callable[[dict[str, Any], Path, Path], tuple[Path, dict[str, Any]]]
    read_submission: Callable[[Path], Any]
    score_sessions: Callable[..., Any]
    load_session: Callable[[Path], Any]
    unit_space_features: Callable[[Any], Any]
    cv_grid: Callable[[np.ndarray, np.ndarray], tuple[list[dict[str, Any]], dict[str, Any]]]
    c1_submission: Callable[[Path, Any], Any]
    data_root: Path


def load_track_pipeline(track_root: str) -> TrackPipeline:
    """装载赛道仓库的数值管线与评分模块。"""
    root = Path(track_root)
    absent = [rel for rel in ("paths.py", "lince_memory", "scripts") if not (root / rel).exists()]
    if absent:
        raise FileNotFoundError(
            f"track_root {root} 缺赛道仓库组件 {absent}，须指向记忆状态跨个体跨天解码仓库根目录"
        )
    for entry in (str(root.resolve()), str((root / "scripts").resolve())):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    scorer = importlib.import_module("lince_memory.scorer")
    proxy = importlib.import_module("eval_c3_proxy")
    paths = importlib.import_module("paths")
    features = importlib.import_module("lince_memory.features")
    data = importlib.import_module("lince_memory.data")
    repro = importlib.import_module("repro_baseline_c1")
    subs = importlib.import_module("make_submissions")
    return TrackPipeline(
        train_matrix=proxy.train_matrix,
        fit_cv=proxy.fit_cv,
        proxy_submission=proxy.proxy_submission,
        read_submission=scorer.read_submission,
        score_sessions=scorer.score_sessions,
        load_session=data.load_session,
        unit_space_features=features.unit_space_features,
        cv_grid=repro.cv_grid,
        c1_submission=subs.c1_submission,
        data_root=Path(paths.DATA_ROOT),
    )


def track_source_manifest(track_root: str) -> dict[str, str]:
    """DPA 赛道仓库源文件的 sha256 清单。"""
    root = Path(track_root)
    files = [root / "paths.py", *(p for d in _TRACK_SOURCE_DIRS for p in sorted((root / d).glob("*.py")))]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@dataclass(frozen=True, slots=True)
class InferOutcome:
    """单个评测文件的推理结果。"""

    mem_pred: np.ndarray
    corr_pred: np.ndarray
    metrics: dict[str, Any]
    extra_bag: dict[str, Any] = field(default_factory=dict)


def _run_c3_baseline(track: TrackPipeline, eval_file: Path, source_check: Callable[[], None]) -> InferOutcome:
    """challenge3 跨个体基线实现。"""
    train_files = sorted((track.data_root / "challenge3").glob(_TRAIN_GLOB))
    if len(train_files) != _TRAIN_FILE_COUNT:
        raise FileNotFoundError(
            f"c3 train 应为 {_TRAIN_FILE_COUNT} 个 NWB，实际 {len(train_files)}：{track.data_root / 'challenge3'}"
        )
    x, mem_y, corr_y, _ = track.train_matrix(train_files)
    models: dict[str, Any] = {}
    for tag, y in (("mem", mem_y), ("corr", corr_y)):
        model, detail = track.fit_cv(x, y)
        models[tag] = model
        logger.info(f"[DpaBaselineInfer] {tag}: best C={detail['best_C']:g} CV={detail['cv_mean']:.4f} ± {detail['cv_std']:.4f}")
    source_check()

    subject, session_date, role = parse_session_name(eval_file)
    with tempfile.TemporaryDirectory(prefix="lince_dpa_infer_") as tmp:
        sub_path, _ = track.proxy_submission(models, eval_file, Path(tmp))
        source_check()
        sub = track.read_submission(sub_path)
        if role in LABELLED_ROLES:
            session = track.score_sessions(((sub_path, eval_file),)).sessions[0]
            metrics: dict[str, Any] = {
                "mem_acc": session.mem_accuracy, "corr_acc": session.corr_accuracy,
                "session_score": session.session_score,
            }
        else:
            metrics = {"mem_acc": None, "corr_acc": None, "session_score": None}
    n_trials = int(sub.mem_pred_lbl.shape[0])
    metrics.update({
        "split": role, "subject": subject, "session_date": session_date,
        "n_trials": n_trials, "method": "baseline",
    })
    return InferOutcome(mem_pred=sub.mem_pred_lbl, corr_pred=sub.corr_pred_lbl, metrics=metrics)


def _run_c1_baseline(track: TrackPipeline, eval_file: Path, source_check: Callable[[], None]) -> InferOutcome:
    """challenge1 同天解码基线实现。"""
    subject, session_date, role = parse_session_name(eval_file)
    if role != "train":
        raise ValueError(f"c1 同天基线要求带标签 train 帧，实际 role={role!r}：{eval_file}")
    frame = track.load_session(eval_file)
    fs = track.unit_space_features(frame)
    source_check()
    cv_detail: dict[str, Any] = {}
    for tag, y in (("mem", fs.mem_lbl), ("corr", fs.corr_lbl)):
        grid, best = track.cv_grid(fs.X, y)
        cv_detail[tag] = {"best_C": best["C"], "cv_mean": best["cv_mean"], "cv_std": best["cv_std"], "cv_grid": grid}
        logger.info(f"[DpaBaselineInfer] c1 {tag}: best C={best['C']:g} CV={best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    source_check()
    sub = track.c1_submission(eval_file, frame)
    source_check()

    n = int(frame.delay_per_trial.shape[0])
    order = np.asarray(sub.trial_indices)
    if order.shape[0] != n or not np.array_equal(np.sort(order), np.arange(n, dtype=np.int64)):
        raise ValueError(f"c1 预测 trial_indices 非 0..{n - 1} 置换：{order.shape} / unique={int(np.unique(order).shape[0])}")
    preds: dict[str, np.ndarray] = {}
    for tag, values in (("mem", sub.mem_pred_lbl), ("corr", sub.corr_pred_lbl)):
        column = np.empty(n, dtype=np.int64)
        column[order] = np.asarray(values, dtype=np.int64)
        preds[tag] = column

    metrics: dict[str, Any] = {
        "mem_acc": cv_detail["mem"]["cv_mean"], "corr_acc": cv_detail["corr"]["cv_mean"],
        "session_score": 0.5 * (cv_detail["mem"]["cv_mean"] + cv_detail["corr"]["cv_mean"]),
    }
    metrics.update({
        "split": role, "subject": subject, "session_date": session_date,
        "n_trials": n, "method": "baseline",
    })
    extra = {"lince_dpa_c1_cv": {
        "file": eval_file.name, "n_samples": int(fs.X.shape[0]), "n_features": int(fs.X.shape[1]),
        "sampling_seed": int(fs.sampling_seed), **cv_detail,
    }}
    return InferOutcome(mem_pred=preds["mem"], corr_pred=preds["corr"], metrics=metrics, extra_bag=extra)


def _run_baseline(track: TrackPipeline, eval_file: Path, source_check: Callable[[], None]) -> InferOutcome:
    """baseline 管线按评测文件所在子挑战目录路由。"""
    match eval_file.parent.name:
        case "challenge1":
            return _run_c1_baseline(track, eval_file, source_check)
        case "challenge3":
            return _run_c3_baseline(track, eval_file, source_check)
        case other:
            raise ValueError(f"评测文件父目录 {other!r} 非 challenge1/challenge3，baseline 无法路由：{eval_file}")


_METHODS: dict[str, Callable[[TrackPipeline, Path, Callable[[], None]], InferOutcome]] = {
    "baseline": _run_baseline,
}


@register_as_linxi_processor(stage=PROCESS_STAGES.POSTPROCESS)
class DpaBaselineInfer(DefaultProcessor):
    """对 DPA 评测文件执行赛道 baseline 分类解码。"""

    PROCESSOR_NAME = "DpaBaselineInfer"

    def __init__(
        self,
        *,
        track_root: str,
        eval_path: str,
        method: str = "baseline",
        name: str | None = None,
    ):
        super().__init__(name)
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")
        self.track_root = track_root
        self.eval_path = eval_path
        self.method = method
        self.last_metrics: dict[str, Any] | None = None

    def _attach_predictions(self, context: LinxiContext, outcome: InferOutcome) -> None:
        trials = context.trials
        if trials is None:
            raise ValueError("context.trials 为 None：预测列须挂在 LoadLinceDpaSession(query) 产出的试次表上")
        table = trials.table
        primary = str(table["start_time"].dims[0])
        n = int(table.sizes[primary])
        for name, arr in zip(_PRED_COLUMNS, (outcome.mem_pred, outcome.corr_pred)):
            if arr.shape[0] != n:
                raise ValueError(f"预测行数 {arr.shape[0]} ≠ 试次表行数 {n}")
            table[name] = xr.DataArray(arr.astype(np.int64), dims=(primary,))

    def _process(self, context: LinxiContext) -> LinxiContext:
        eval_file = Path(self.eval_path)
        if context.session != eval_file.stem:
            raise ValueError(
                f"eval_path 命名主干 {eval_file.stem!r} 与上游载入的 context.session {context.session!r} 不符"
            )
        base_manifest = track_source_manifest(self.track_root)

        def source_check() -> None:
            if track_source_manifest(self.track_root) != base_manifest:
                raise RuntimeError(f"赛道数值路径源码在评测期间发生变化：{self.track_root}")

        track = load_track_pipeline(self.track_root)
        outcome = _METHODS[self.method](track, eval_file, source_check)
        self._attach_predictions(context, outcome)
        context.put_metric("session_id", context.session)
        context.put_metric("metrics", {key: outcome.metrics[key] for key in EVAL_METRIC_FIELDS})
        for key, value in outcome.extra_bag.items():
            context.put_metric(key, value)
        self.last_metrics = dict(outcome.metrics)
        logger.info(f"[DpaBaselineInfer] {context.session}: {outcome.metrics}")
        return context
