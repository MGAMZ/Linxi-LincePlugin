"""临策 e2e WF 示例驱动：替换 YAML 路径占位符后执行流水线，可选与黄金结果对账。

用法与占位符说明见 examples/README.md。
"""

from __future__ import annotations

import numpy as np  # noqa: F401  # 必须先于 linxi：锁定与官方 harness 一致的 BLAS 线程配置

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from linxi.fabric.linxi_context import LinxiContext
from linxi.fabric.metadata_injection import apply_pipeline_metadata
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner

PARITY_TOLERANCE = 1e-6
PARITY_FIELDS = ("r2_x", "r2_y", "r2_mean_raw")
DEFAULT_CONFIG = Path(__file__).with_name("e2e_wf_MA-CO-20231227-01.yaml")
_PATH_TOKENS = ("__DATA_ROOT__", "__CODE_ROOT__", "__OUTPUT_ROOT__")


def _golden_row(golden_path: Path, session_id: str) -> dict[str, float]:
    results = json.loads(golden_path.read_text(encoding="utf-8"))
    task, level = session_id.split("/")[:2]
    rows = [r for r in results["task_results"][task][level] if r["session_id"] == session_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one golden row for {session_id!r}, found {len(rows)}")
    return rows[0]


def _parity_report(row: dict[str, float], gold: dict[str, float]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    for field in PARITY_FIELDS:
        delta = abs(row[field] - gold[field])
        ok &= delta <= PARITY_TOLERANCE
        lines.append(f"  {field}: run={row[field]!r} golden={gold[field]!r} |Δ|={delta:.3e}")
    recomposed = 0.95 * max(gold["r2_mean_raw"], 0.0) + 0.05 * gold["latency_score"]
    delta_score = abs(recomposed - gold["session_score"])
    ok &= delta_score <= PARITY_TOLERANCE
    lines.append(
        f"  session_score(黄金延迟重合成): ours={recomposed!r} golden={gold['session_score']!r} |Δ|={delta_score:.3e}"
    )
    lines.append(f"  session_score(本机直测) 仅展示，延迟分量硬件相关: run={row['session_score']!r}")
    return ok, lines


def _resolve_config(config_path: Path, data_root: str, code_root: str, output_root: str, work_dir: Path) -> Path:
    """把 YAML 占位符替换为命令行根路径，写入工作目录下的临时配置；仓库 YAML 不被改动。"""
    text = config_path.read_text(encoding="utf-8")
    for token, root in zip(_PATH_TOKENS, (data_root, code_root, output_root)):
        text = text.replace(token, root)
    leftover = next((t for t in _PATH_TOKENS if t in text), None)
    if leftover is not None:
        raise ValueError(f"config {config_path} still contains unresolved path token {leftover!r}")
    out_path = work_dir / f"{config_path.stem}.resolved.yaml"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--work-dir", type=Path, default=None, help="TaskRunner 工作目录；缺省用临时目录")
    parser.add_argument("--data-root", required=True, help="赛题 challenge_data 根目录，替换 __DATA_ROOT__")
    parser.add_argument("--code-root", required=True, help="赛道 challenge_code 目录，替换 __CODE_ROOT__")
    parser.add_argument("--output-root", required=True, help="产物输出根目录，替换 __OUTPUT_ROOT__")
    parser.add_argument("--golden", type=Path, default=None, help="wf_challenge_results.json 路径；给出则执行 ≤1e-6 对账")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="lince_e2e_work_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = _resolve_config(args.config, args.data_root, args.code_root, args.output_root, work_dir)

    pipeline_def = PipelineDefinition.from_file(resolved_config)
    context = LinxiContext(file_create_date=datetime.now())
    apply_pipeline_metadata(context, pipeline_def)
    context.run_state.work_dir = work_dir

    results = TaskRunner(pipeline_def).run(context, allow_errors=False)
    final = results[-1]

    notes = json.loads(final.notes)
    row = notes["lince_baseline_decode"]["sessions"][0]
    print(f"[e2e] store: {final.run_state.output_path}")
    print(f"[e2e] session_score_row: {json.dumps(row, ensure_ascii=False)}")
    print(f"[e2e] drift_targets: {json.dumps(notes['lince_drift_analysis']['targets'], ensure_ascii=False)}")

    if args.golden is None:
        return 0
    gold = _golden_row(args.golden, row["session_id"])
    ok, lines = _parity_report(row, gold)
    print("[e2e] golden parity (tolerance 1e-6):")
    print("\n".join(lines))
    print(f"[e2e] PARITY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
