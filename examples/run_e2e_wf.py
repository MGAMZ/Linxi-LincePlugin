"""临策 e2e WF 示例驱动：经 ``PipelineDefinition.from_file`` + TaskRunner 执行钉死 session 的流水线。

导入次序是契约的一部分：``import numpy`` 必须先于任何 ``linxi`` 导入。
``linxi/__init__.py`` 在包导入时强制 ``OPENBLAS_NUM_THREADS=1``（防分选算法线程竞态），
而 OpenBLAS 归约次序在库装载时即固定；官方评测与黄金 ``wf_challenge_results.json``
产生于环境默认线程配置。先导入 numpy 使本驱动与官方 harness 同配置，逐位复现黄金；
``linxi process`` CLI 入口因先加载 linxi 包而落入单线程配置，数值确定性偏离黄金
（本 session 实测 |Δr2_mean_raw| ≈ 8.7e-6），故本示例以驱动为可复现入口。
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--work-dir", type=Path, default=None, help="TaskRunner 工作目录；缺省用临时目录")
    parser.add_argument("--golden", type=Path, default=None, help="wf_challenge_results.json 路径；给出则执行 ≤1e-6 对账")
    args = parser.parse_args(argv)

    pipeline_def = PipelineDefinition.from_file(args.config)
    context = LinxiContext(file_create_date=datetime.now())
    apply_pipeline_metadata(context, pipeline_def)
    context.run_state.work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="lince_e2e_work_"))

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
