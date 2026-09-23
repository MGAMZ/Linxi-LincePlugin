"""临策 DPA e2e 示例驱动"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# linxi 须先于其余第三方依赖导入，进程以 OpenBLAS 单线程计算，保证数值可重现。
from linxi.fabric.linxi_context import LinxiContext
from linxi.fabric.metadata_injection import apply_pipeline_metadata
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner
import numpy as np

PARITY_TOLERANCE = 1e-6
IDENTITY_CFG = "identity/pooled_cvC"
HEAD_REFIT_CFG = "head_refit/support4fold_oof"
SUBSTITUTE_CANDIDATES = (IDENTITY_CFG, "coral/aligned_cvC", "session_ensemble/sum_decision_sharedC")
PROXIES = ("sub-m090_ses-20210527_task-DPA-eval-1.nwb", "sub-m095_ses-20210709_task-DPA-eval-1.nwb")
TAGS = ("mem", "corr")
_GT_COLUMN = {"mem": "sample_cue", "corr": "is_correct"}
_PRED_COLUMN = {"mem": "mem_pred_lbl", "corr": "corr_pred_lbl"}
_CHAIN_CONFIGS = {"c1": "e2e_dpa_sub-m091_ses-20210608.yaml", "m090eval": "e2e_dpa_sub-m090_ses-20210527.yaml"}
_PATH_TOKENS = ("__DATA_ROOT__", "__TRACK_ROOT__", "__OUTPUT_ROOT__")


def _resolve_config(config_path: Path, roots: dict[str, str], work_dir: Path) -> Path:
    """把 YAML 占位符替换为命令行根路径，写入工作目录下的临时配置。"""
    text = config_path.read_text(encoding="utf-8")
    for token, root in zip(_PATH_TOKENS, (roots["data"], roots["track"], roots["output"]), strict=True):
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


def _delta_line(name: str, ours: float, gold: float, ok: list[bool]) -> str:
    delta = abs(ours - gold)
    ok.append(delta <= PARITY_TOLERANCE)
    return f"  {name}: run={ours!r} golden={gold!r} |Δ|={delta:.3e}"


def _c1_parity(sidecar: dict[str, Any], golden: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    ok: list[bool] = []
    lines: list[str] = []
    metrics = sidecar["metrics"]
    cv_bag = sidecar["lince_dpa_c1_cv"]
    for tag in TAGS:
        lines.append(_delta_line(f"{tag}_acc", metrics[f"{tag}_acc"], golden[tag]["cv_mean"], ok))
        lines.append(_delta_line(f"{tag}_cv_std CV明细", cv_bag[tag]["cv_std"], golden[tag]["cv_std"], ok))
        for key in ("best_C", "cv_mean"):
            ok.append(cv_bag[tag][key] == golden[tag][key])
            lines.append(f"  CV明细 {tag}.{key}: run={cv_bag[tag][key]!r} golden={golden[tag][key]!r}")
    for row_run, row_gold in zip(cv_bag["mem"]["cv_grid"], golden["mem"]["cv_grid"], strict=True):
        ok.append(row_run == row_gold)
    recomposed = 0.5 * (golden["mem"]["cv_mean"] + golden["corr"]["cv_mean"])
    lines.append(_delta_line("session_score 按golden重合成", metrics["session_score"], recomposed, ok))
    for key in ("n_samples", "n_features", "sampling_seed"):
        ok.append(cv_bag[key] == golden[key])
        lines.append(f"  CV明细 {key}: run={cv_bag[key]!r} golden={golden[key]!r}")
    manifest = {"golden_field_source": "baseline_c1_cv.json unit_space CV", "executed_methods": {t: "identity/baseline 取自 c1 同天 CV" for t in TAGS}, "chain_substitute": {t: "不触发" for t in TAGS}}
    return all(ok), lines, manifest


def _executed_configs(selection: dict[str, Any], results: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    executed: dict[str, str] = {}
    substitute: dict[str, str] = {}
    for tag in TAGS:
        cfg = selection["selections"][tag]["selected_config"]
        if cfg == IDENTITY_CFG:
            executed[tag] = IDENTITY_CFG
            substitute[tag] = "不触发"
            continue
        if cfg != HEAD_REFIT_CFG:
            raise ValueError(f"{tag} 选定法 {cfg} 非恒等且非 HeadRefit，选定法无链上解码通道，无法对账")
        means = {c: sum(r["accuracy"] for r in results["rows"] if r["tag"] == tag and r["config"] == c and r["proxy"] in PROXIES) / len(PROXIES) for c in SUBSTITUTE_CANDIDATES}
        winner = max(SUBSTITUTE_CANDIDATES, key=lambda c: means[c])
        executed[tag] = winner
        substitute[tag] = f"HeadRefit→{winner}"
        if winner != IDENTITY_CFG:
            raise ValueError(f"{tag} 代跑法 {winner} 无链上解码通道，无法对账，chain_substitute 为 {substitute[tag]}")
    return executed, substitute


def _golden_row(results: dict[str, Any], tag: str, config: str) -> dict[str, Any]:
    rows = [r for r in results["rows"] if r["tag"] == tag and r["config"] == config and r["proxy"] == PROXIES[0]]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one golden row for {tag}/{config}/{PROXIES[0]}, found {len(rows)}")
    return rows[0]


def _flips_against_store(trials: Any, tag: str, correct_vector: list[bool]) -> int:
    match = np.equal(trials[_GT_COLUMN[tag]].values, trials[_PRED_COLUMN[tag]].values)
    return int(np.count_nonzero(match != np.asarray(correct_vector, dtype=bool)))


def _m090_parity(sidecar: dict[str, Any], results: dict[str, Any], selection: dict[str, Any], store: Path) -> tuple[bool, list[str], dict[str, Any]]:
    ok: list[bool] = []
    lines: list[str] = []
    executed, substitute = _executed_configs(selection, results)
    metrics = sidecar["metrics"]
    trials = LinxiContext.from_zarr(str(store)).trials.table
    flips: dict[str, int] = {}
    for tag in TAGS:
        row = _golden_row(results, tag, executed[tag])
        lines.append(_delta_line(f"{tag}_acc", metrics[f"{tag}_acc"], row["accuracy"], ok))
        ok.append(int(metrics["n_trials"]) == int(row["n_trials"]))
        lines.append(f"  {tag}.n_trials: run={metrics['n_trials']!r} golden={row['n_trials']!r}")
        flips[tag] = _flips_against_store(trials, tag, row["correct_vector"])
        ok.append(flips[tag] == 0)
        lines.append(f"  {tag} 预测翻转 trial 数: {flips[tag]}，判据为 0，golden correct_vector {sum(row['correct_vector'])}/{len(row['correct_vector'])} 正确")
    recomposed = 0.5 * sum(_golden_row(results, tag, executed[tag])["accuracy"] for tag in TAGS)
    lines.append(_delta_line("session_score 按golden重合成", metrics["session_score"], recomposed, ok))
    manifest = {"golden_field_source": "methods_results.json m090 eval-1 行，逐标签实际执行法", "executed_methods": executed, "chain_substitute": substitute, "flips": flips}
    return all(ok), lines, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chain", required=True, choices=sorted(_CHAIN_CONFIGS), help="选链：c1 = sub-m091_ses-20210608 同天基线，m090eval = sub-m090_ses-20210527 评测双输入")
    parser.add_argument("-c", "--config", type=Path, default=None, help="链 YAML，缺省按 --chain 选内置示例")
    parser.add_argument("-o", "--work-dir", type=Path, default=None, help="TaskRunner 工作目录，缺省用临时目录")
    parser.add_argument("--data-root", required=True, help="赛题 challenge_data 根目录，替换 __DATA_ROOT__")
    parser.add_argument("--track-root", required=True, help="赛道仓库根目录，替换 __TRACK_ROOT__")
    parser.add_argument("--output-root", required=True, help="产物输出根目录，替换 __OUTPUT_ROOT__")
    parser.add_argument("--golden", type=Path, default=None, help="c1 链=baseline_c1_cv.json，m090eval 链=methods_results.json，给出则执行逐链 golden parity")
    parser.add_argument("--selection", type=Path, default=None, help="method_selection.json，m090eval 链 golden parity 必填")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="lince_dpa_e2e_work_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    roots = {"data": args.data_root, "track": args.track_root, "output": str(output_root)}
    config_path = args.config or Path(__file__).with_name(_CHAIN_CONFIGS[args.chain])

    store, sidecar = _run_chain(config_path, roots, work_dir)
    print(f"[e2e] chain: {args.chain}")
    print(f"[e2e] store: {store}")
    print(f"[e2e] eval_sidecar: {store.with_name(store.stem + '.eval.json')}")
    print(f"[e2e] metrics_row: {json.dumps(sidecar['metrics'], ensure_ascii=False)}")

    if args.golden is None:
        return 0
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    if args.chain == "c1":
        ok, lines, manifest = _c1_parity(sidecar, golden)
    else:
        if args.selection is None:
            parser.error("m090eval 链 golden parity 需要 --selection，即 method_selection.json")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        ok, lines, manifest = _m090_parity(sidecar, golden, selection, store)
    print(f"[e2e] golden parity, per-field abs <= {PARITY_TOLERANCE:g}, 单线程标准口径:")
    print("\n".join(lines))
    manifest_path = output_root / f"{args.chain}.e2e_manifest.json"
    manifest_path.write_text(json.dumps({"chain": args.chain, "golden": str(args.golden), **manifest}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[e2e] manifest: {manifest_path}")
    print(f"[e2e] PARITY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
