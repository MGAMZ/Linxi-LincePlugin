"""NEO 赛题 baseline 解码推理的 POSTPROCESS 阶段算子：PSD 特征 + 收缩线性 LDA 统一 8 分类。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import detrend, filtfilt, iirnotch, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor

from ._bdf import read_epi_session
from .action_windows import extract_trial_windows, extended_intervals, trials_frame

if TYPE_CHECKING:
    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["EpiBaselineInfer", "psd_features"]

_WELCH_NPERSEG = 500
_BAND_EDGES = np.arange(2.0, 151.0, 4.0)
_NOTCH_FREQS = (50.0, 100.0, 150.0)
_WINDOW_START = 0.2
_WINDOW_DURATION = 2.0


def psd_features(windows: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """(N, T, C) 窗口数组 → (N, channel × bins) log PSD 特征矩阵。"""
    x = detrend(np.transpose(windows, (0, 2, 1)).astype(np.float64), axis=-1)
    for f0 in _NOTCH_FREQS:
        b, a = iirnotch(f0 / (sampling_frequency / 2.0), Q=30.0)
        x = filtfilt(b, a, x, axis=-1)
    frequencies, psd = welch(x, fs=sampling_frequency, nperseg=_WELCH_NPERSEG, axis=-1)
    bands = [np.log10(psd[..., (frequencies >= lo) & (frequencies < hi)].mean(axis=-1) + 1e-20)
             for lo, hi in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:])]
    return np.stack(bands, axis=-1).reshape(len(windows), -1)


def _make_lda8() -> object:
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.125] * 8),
    )


def _train_dataset(train_root: Path) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for session_dir in sorted(p for p in train_root.iterdir() if (p / "data.bdf").is_file()):
        arrays = read_epi_session(session_dir)
        windows, _onsets, _begins, _stops = extract_trial_windows(
            arrays.signal, arrays.trials, arrays.sampling_frequency, _WINDOW_START, _WINDOW_DURATION)
        features.append(psd_features(windows, arrays.sampling_frequency))
        labels.append(arrays.trials["label"].to_numpy())
    return np.concatenate(features), np.concatenate(labels)


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> dict[str, float]:
    per_class = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        per_class[str(c)] = 0.0 if denom == 0 else float(2 * tp / denom)
    return per_class


@register_as_linxi_processor(stage=PROCESS_STAGES.POSTPROCESS)
class EpiBaselineInfer(DefaultProcessor):
    """对 query 试次表的评测窗口列执行统一 8 分类 baseline 推理，产出 `pred_label` 与评测载荷。

    Parameters
    ----------
    train_root:
        训练 session 集合根目录，其下每个含 `data.bdf` 的子目录为一个 session。
    recording_key:
        读写的 `context.ieeg` 槽键。
    """

    def __init__(self, train_root: str | None = None, recording_key: str = "query",
                 name: str | None = None):
        super().__init__(name)
        self.train_root = train_root
        self.recording_key = recording_key

    def _process(self, context: LinxiContext) -> LinxiContext:
        if self.train_root is None:
            raise ValueError("EpiBaselineInfer requires an explicit `train_root` processor param")
        slot = context.ieeg.get(self.recording_key)
        if slot is None or slot.events is None:
            raise ValueError(f"ieeg[{self.recording_key!r}] 或其 events 为空：应由 LoadLinceEpiSession 先载入")
        table = slot.events.table
        if "window_signal" not in table.data_vars:
            raise ValueError(
                f"ieeg[{self.recording_key!r}] 试次表缺 window_signal 列：应先执行 EpiExtractActionWindows")
        frame = trials_frame(table)
        x_train, y_train = _train_dataset(Path(self.train_root))
        model = _make_lda8().fit(x_train, y_train)
        windows = np.stack(list(frame["window_signal"]))
        pred = model.predict(psd_features(windows, float(slot.sampling_frequency)))

        frame["pred_label"] = np.asarray(pred, dtype=np.int64)
        intervals = extended_intervals(slot.events, frame)
        slot.events = intervals
        if self.recording_key == "query":
            context.trials = intervals

        labels = frame["label"].to_numpy()
        classes = sorted(int(c) for c in np.unique(labels))
        per_class = _class_f1(labels, np.asarray(pred), classes)
        context.metrics.update({
            "session_id": context.session,
            "tier": "single" if str(context.session).endswith("-single-MA") else "dual",
            "metrics": {
                "macro_f1": float(np.mean(list(per_class.values()))),
                "per_class_f1": per_class,
                "n_eval_trials": int(len(frame)),
                "n_train_trials": int(len(y_train)),
                "classes": classes,
                "method": "psd_lda8_official_protocol",
            },
        })
        return context
