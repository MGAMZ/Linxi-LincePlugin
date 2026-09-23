"""临策 NEO 赛道 e2e 示例驱动：替换 YAML 路径占位符后运行固定的单动作与双动作 session 流水线，逐样本与基线复现 golden 比对。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from linxi.fabric.linxi_context import LinxiContext
from linxi.fabric.metadata_injection import apply_pipeline_metadata
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner
import numpy as np

MACRO_TOLERANCE = 1e-12
CHAIN_CONFIGS = {
    "single": "e2e_epi_P01_20240905-single-MA.yaml",
    "dual": "e2e_epi_P01_20241112-dual-MA.yaml",
}
_PATH_TOKENS = ("__DATA_ROOT__", "__OUTPUT_ROOT__")


def _resolve_config(config_path: Path, roots: dict[str, str], work_dir: Path) -> Path:
    text = config_path.read_text(encoding="utf-8")
    for token, root in zip(_PATH_TOKENS, (roots["data"], roots["output"]), strict=True):
        text = text.replace(token, root)
    leftover = next((t for t in _PATH_TOKENS if t in text), None)
    if leftover is not None:
        raise ValueError(f"config {config_path} still contains unresolved path token {leftover!r}")
    out_path = work_dir / f"{config_path.stem}.resolved.yaml"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _run_chain(config_path: Path, roots: dict[str, str], work_dir: Path) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_config(config_path, roots, work_dir)
    pipeline_def = PipelineDefinition.from_file(resolved)
    context = LinxiContext(file_create_date=datetime.now())
    apply_pipeline_metadata(context, pipeline_def)
    context.run_state.work_dir = work_dir
    final = TaskRunner(pipeline_def).run(context, allow_errors=False)[-1]
    store = Path(final.run_state.output_path)
    sidecar = json.loads(store.with_name(store.stem + ".eval.json").read_text(encoding="utf-8"))
    return store, sidecar


def _load_golden(path: Path) -> dict[str, tuple[int, int]]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r["sample_id"]: (int(r["y_true"]), int(r["pred_official"])) for r in rows}


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    scores = []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def _parity(store: Path, sidecar: dict[str, Any], golden: dict[str, tuple[int, int]]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok: list[bool] = []
    back = LinxiContext.from_zarr(str(store))
    slot = back.ieeg["query"]
    table = slot.events.table
    ids = [str(s) for s in table["sample_id"].values]
    pred = np.asarray(table["pred_label"].values)
    labels = np.asarray(table["label"].values)

    missing = [i for i in ids if i not in golden]
    ok.append(not missing)
    lines.append(f"  golden 覆盖: {len(ids) - len(missing)}/{len(ids)} 样本可对齐"
                 + (f"，缺失 {missing[:3]}" if missing else ""))
    gold_pred = np.asarray([golden[i][1] for i in ids if i in golden])
    gold_true = np.asarray([golden[i][0] for i in ids if i in golden])
    ours = np.asarray([p for p, i in zip(pred, ids) if i in golden])
    mismatch = int(np.sum(ours != gold_pred))
    ok.append(mismatch == 0)
    lines.append(f"  逐样本 pred_label vs pred_official: 不一致 {mismatch} / {len(ours)}")
    label_mismatch = int(np.sum(labels != gold_true))
    ok.append(label_mismatch == 0)
    lines.append(f"  真值 label vs golden y_true: 不一致 {label_mismatch} / {len(labels)}")

    classes = sorted(int(c) for c in np.unique(labels))
    recomputed = _macro_f1(labels, pred, classes)
    delta = abs(sidecar["metrics"]["macro_f1"] - recomputed)
    ok.append(delta <= MACRO_TOLERANCE)
    lines.append(f"  侧车 macro_f1={sidecar['metrics']['macro_f1']!r} 重算={recomputed!r} |Δ|={delta:.3e}")

    sig = slot.electrophysiology.data.values
    ok.append(sig.shape[1] == 8 and float(slot.sampling_frequency) == 1000.0)
    lines.append(f"  信号形状 {sig.shape} 采样率 {slot.sampling_frequency} 单位 {slot.electrophysiology.unit}")
    window = table["window_signal"].values
    ok.append(window.shape == (len(ids), 2000, 8))
    lines.append(f"  window_signal 形状 {window.shape}，契约为 (N, 2000, 8)，提交端转置为 (N, 8, 2000)")
    return all(ok), lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="heldin 数据根，含各 session 目录")
    parser.add_argument("--output-root", required=True, help="产物根目录")
    parser.add_argument("--golden", required=True, help="基线复现 golden CSV 路径")
    parser.add_argument("--chain", choices=(*CHAIN_CONFIGS, "both"), default="both")
    args = parser.parse_args()

    roots = {"data": str(Path(args.data_root).resolve()), "output": str(Path(args.output_root).resolve())}
    golden = _load_golden(Path(args.golden))
    chains = list(CHAIN_CONFIGS) if args.chain == "both" else [args.chain]
    failures = 0
    for name in chains:
        config = Path(__file__).resolve().parent / CHAIN_CONFIGS[name]
        work_dir = Path(roots["output"]) / "_work" / name
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"[e2e] chain={name} config={config.name}")
        store, sidecar = _run_chain(config, roots, work_dir)
        ok, lines = _parity(store, sidecar, golden)
        print("\n".join(lines))
        print(f"[e2e] chain={name} -> {'PARITY PASS' if ok else 'PARITY FAIL'}")
        failures += int(not ok)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
