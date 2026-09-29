"""精细手部赛道 GRU 推理与四分量评分算子"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from linxi.logger import logger
from linxi.processor import PROCESS_STAGES, DefaultProcessor, register_as_linxi_processor

from .load_session import FineHandSessionArrays, _partition_dir, read_finehand_session

if TYPE_CHECKING:
    import torch

    from linxi.fabric.linxi_context import LinxiContext

__all__ = ["EVAL_METRIC_FIELDS", "FineHandGruInfer"]

EVAL_METRIC_FIELDS = ("s_wrist", "s_pose", "s_motion", "s_bone", "final_score")
_CHUNK_FRAMES = 512
_INPUT_SIZE = 512
_OUTPUT_DIM = 63
_SIDECAR_SUFFIX = ".eval.json"


def _build_gru_model(input_size: int, hidden_size: int) -> torch.nn.Module:
    """构建与官方提交 runtime 结构一致的单层因果 GRU 解码模型，torch 在此处延迟导入。"""
    import torch
    from torch import nn

    class SimpleGRUModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, _OUTPUT_DIM)

        def forward(self, values: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            values = torch.tanh(self.input_projection(values))
            values, hidden = self.gru(values, hidden)
            return self.output(values), hidden

    return SimpleGRUModel(input_size, hidden_size)


def _resolve_device(requested: str) -> torch.device:
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' 但当前环境 CUDA 不可用，请改用 device='cpu'")
    return torch.device(requested)


def _load_model(model_dir: str, session_id: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    import torch

    path = Path(model_dir) / session_id / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{session_id}: 官方 checkpoint 不存在，期望路径 {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["session_id"] != session_id:
        raise ValueError(f"{path}: checkpoint session_id={checkpoint['session_id']!r} 与目标 session {session_id!r} 不符")
    if checkpoint["input_size"] != _INPUT_SIZE:
        raise ValueError(f"{path}: session {session_id} 的 input_size={checkpoint['input_size']}，512 通道输入契约不符")
    model = _build_gru_model(int(checkpoint["input_size"]), int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _load_scoring(track_root: str) -> ModuleType:
    root = Path(track_root)
    if not (root / "lince_finehand" / "scoring.py").is_file():
        raise FileNotFoundError(
            f"track_root {root} 缺 lince_finehand/scoring.py，须指向精细手部运动解码赛道仓库根目录"
        )
    entry = str(root.resolve())
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return importlib.import_module("lince_finehand.scoring")


@register_as_linxi_processor(stage=PROCESS_STAGES.POSTPROCESS)
class FineHandGruInfer(DefaultProcessor):
    """用官方预置 checkpoint 对单个 session 执行 GRU 推理，四分量评分写 metrics 袋并导出评测结果文件。"""

    PROCESSOR_NAME = "FineHandGruInfer"

    def __init__(
        self,
        *,
        session_dir: str,
        model_dir: str,
        track_root: str,
        output_dir: str,
        device: str = "cpu",
        num_threads: int | None = 4,
        name: str | None = None,
    ):
        super().__init__(name)
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device 必须是 cpu/cuda/auto 之一，实际 {device!r}")
        self.session_dir = session_dir
        self.model_dir = model_dir
        self.track_root = track_root
        self.output_dir = output_dir
        self.device = device
        self.num_threads = num_threads
        self.last_metrics: dict[str, float] | None = None

    def _ensure_output_dir(self) -> Path:
        out = Path(self.output_dir)
        resolved = out.resolve()
        for read_only in (Path(self.session_dir), Path(self.model_dir)):
            root = read_only.resolve()
            if resolved == root or root in resolved.parents:
                raise ValueError(f"output_dir {resolved} 位于只读输入根 {root} 之内，拒绝写入")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _predict_trials(
        self,
        model: torch.nn.Module,
        checkpoint: dict[str, Any],
        arrays: FineHandSessionArrays,
        device: torch.device,
    ) -> list[np.ndarray]:
        import torch

        tx_rate = np.ascontiguousarray(arrays.counts / arrays.dt_sec[:, None], dtype=np.float32)
        mean = checkpoint["feature_mean"].numpy()
        std = checkpoint["feature_std"].numpy()
        target_std = checkpoint["target_std"].numpy()
        target_mean = checkpoint["target_mean"].numpy()
        frame_counts = [int(n) for n in arrays.trials["n_frames"]]
        offsets = np.cumsum([0, *frame_counts])
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for index, n_frames in enumerate(frame_counts):
                lo, hi = int(offsets[index]), int(offsets[index + 1])
                features = np.concatenate((tx_rate[lo:hi], arrays.sbp[lo:hi]), axis=1)
                values = (features - mean) / std
                hidden = None
                chunks: list[np.ndarray] = []
                for start in range(0, values.shape[0], _CHUNK_FRAMES):
                    batch = torch.from_numpy(values[start : start + _CHUNK_FRAMES]).unsqueeze(0).to(device)
                    output, hidden = model(batch, hidden)
                    chunks.append(output.squeeze(0).cpu().numpy())
                prediction = np.concatenate(chunks) * target_std + target_mean
                if not np.isfinite(prediction).all():
                    raise ValueError(f"{arrays.session_id}: trial {index + 1} 预测含 NaN/Inf")
                predictions.append(prediction.astype(np.float32))
        return predictions

    def _write_pred_h5(self, path: Path, predictions: list[np.ndarray], arrays: FineHandSessionArrays, names: tuple[str, ...]) -> None:
        with h5py.File(path, "w") as handle:
            handle.attrs["model_type"] = "simple_gru"
            handle.attrs["dataset_version"] = "Data2BCI_version3"
            handle.create_dataset("predicted_trajectory", data=np.vstack(predictions))
            handle.create_dataset("trial_lengths", data=np.asarray([p.shape[0] for p in predictions], dtype=np.int64))
            handle.create_dataset("trial_indices", data=np.asarray(arrays.trials["trial_index"].tolist(), dtype=np.int64))
            handle.create_dataset("keypoint_names", data=np.asarray(names, dtype=h5py.string_dtype("utf-8")))

    def _process(self, context: LinxiContext) -> LinxiContext:
        output_root = self._ensure_output_dir()
        arrays = read_finehand_session(self.session_dir)
        if context.session != arrays.session_id:
            raise ValueError(
                f"session_dir 主干 {arrays.session_id!r} 与上游载入的 context.session {context.session!r} 不符"
            )
        if arrays.keypoint is None:
            raise ValueError(f"{arrays.session_id}: trial npz 无 keypoint 真值，无法评分")
        if self.num_threads is not None:
            import torch

            torch.set_num_threads(self.num_threads)
        scoring = _load_scoring(self.track_root)
        device = _resolve_device(self.device)
        model, checkpoint = _load_model(self.model_dir, arrays.session_id, device)
        names = tuple(str(item) for item in np.asarray(arrays.keypoint_names).reshape(-1))
        if names != tuple(checkpoint["keypoint_names"]):
            raise ValueError(f"{arrays.session_id}: 模型与数据 keypoint 名单不一致")

        predictions = self._predict_trials(model, checkpoint, arrays, device)
        pred_path = output_root / f"{arrays.session_id}_pred.h5"
        self._write_pred_h5(pred_path, predictions, arrays, names)
        score = scoring.score_session(pred_path, _partition_dir(Path(self.session_dir)))
        metrics = {field: float(getattr(score, field)) for field in EVAL_METRIC_FIELDS}
        non_finite = [field for field, value in metrics.items() if not math.isfinite(value)]
        if non_finite:
            raise RuntimeError(f"{arrays.session_id}: 评分字段非有限值 {non_finite}")

        context.put_metric("session_id", arrays.session_id)
        context.put_metric("metrics", metrics)
        eval_path = output_root / f"{self.name}{_SIDECAR_SUFFIX}"
        with eval_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, allow_nan=False)
        for produced in (pred_path, eval_path):
            if not produced.is_file():
                raise RuntimeError(f"{arrays.session_id}: 产物未落盘 {produced}")
        self.last_metrics = metrics
        logger.info(
            f"[FineHandGruInfer] {arrays.session_id}: device={device} num_threads={self.num_threads} "
            f"输入重建口径=counts 逐帧除以 int64 ns 域反算 dt；分数={json.dumps(metrics, ensure_ascii=False)}"
        )
        logger.info(f"[FineHandGruInfer] 产物写出: {pred_path} 与 {eval_path}")
        return context
