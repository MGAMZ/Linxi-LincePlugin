"""临策高保真脑电数据压缩 e2e np 冒烟驱动"""
# allow: SIZE_OK — e2e 驱动按 examples 成例保持单文件自包含。
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
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

# 分支未合并安装时 linxi_linceplugin 可能解析到安装位置而非本仓，插入仓库根保证导入落在当前分支代码。
_PLUGIN_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[0] != _PLUGIN_ROOT:
    sys.path.insert(0, _PLUGIN_ROOT)

DEFAULT_CONFIG = Path(__file__).with_name("e2e_eeg_compression.yaml")
_PATH_TOKENS = ("__TRACK_ROOT__", "__OUTPUT_ROOT__")
EVAL_NAME = "np_smoke.eval.json"
METRICS_KEY = "recon_quality_np"
EXPECTED_SESSION = "np_smoke.ap"
ANCHOR_RELPATH = Path("output/csv/baseline_np_smoke_meta_fixed.json")
GATE_NAMES = ("np_rebuilt_bin", "np_bin_size_identical", "np_bin_int16_aligned",
              "np_meta_in_comp", "np_meta_restored_identical")
# 排除集五项：file_create_date、notes 内时间衍生字段、history 记录的时间戳与耗时、from-value 目录名、环境信息键。
ROOT_ATTR_EXCLUDE = ("file_create_date", "python_environments")
HISTORY_ATTR_EXCLUDE = ("timestamp", "duration")
NOTE_TIME_KEYS = ("total_latency_ms", "latency_per_bin_ms", "latency_score", "session_score", "adapt_seconds")
_FROM_VALUE_RE = re.compile(r"from-value-[0-9a-f]{6,}")
_HISTORY_ATTR_RE = re.compile(r"history/\d+/\.zattrs")


def _resolve_config(config_path: Path, roots: dict[str, str], work_dir: Path) -> Path:
    """把 YAML 占位符替换为命令行根路径，写入工作目录下的临时配置。"""
    text = config_path.read_text(encoding="utf-8")
    for token, root in zip(_PATH_TOKENS, (roots["track"], roots["output"]), strict=True):
        text = text.replace(token, root)
    leftover = next((t for t in _PATH_TOKENS if t in text), None)
    if leftover is not None:
        raise ValueError(f"config {config_path} still contains unresolved path token {leftover!r}")
    out_path = work_dir / f"{config_path.stem}.resolved.yaml"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _run_chain(resolved: Path, work_dir: Path) -> tuple[Path, Path, bytes]:
    """执行一次链，返回 .ls store 路径、指标表文件路径与其字节。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    pipeline_def = PipelineDefinition.from_file(resolved)
    context = LinxiContext(file_create_date=datetime.now())
    apply_pipeline_metadata(context, pipeline_def)
    context.run_state.work_dir = work_dir
    final = TaskRunner(pipeline_def).run(context, allow_errors=False)[-1]
    store = Path(final.run_state.output_path)
    if not store.is_dir():
        raise RuntimeError(f"导出产物不存在: {store}")
    eval_path = store.with_name(store.stem + ".eval.json")
    if not eval_path.is_file():
        raise RuntimeError(f"指标表文件不存在: {eval_path}")
    return store, eval_path, eval_path.read_bytes()


def _strip_dict(obj: Any, drop: tuple[str, ...], hits: list[str], where: str) -> Any:
    if isinstance(obj, dict):
        kept: dict[Any, Any] = {}
        for key, value in obj.items():
            if key in drop:
                hits.append(f"{where}:{key}")
            else:
                kept[key] = _strip_dict(value, drop, hits, where)
        return kept
    if isinstance(obj, list):
        return [_strip_dict(v, drop, hits, where) for v in obj]
    return obj


def _canon_attrs(raw: bytes, drop: tuple[str, ...], where: str, hits: list[str]) -> bytes:
    obj = json.loads(raw)
    return json.dumps(_strip_dict(obj, drop, hits, where), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _canon_zmetadata(raw: bytes, hits: list[str]) -> bytes:
    doc = json.loads(raw)
    meta = doc.get("metadata", {})
    if isinstance(meta.get(".zattrs"), dict):
        meta[".zattrs"] = _strip_dict(meta[".zattrs"], ROOT_ATTR_EXCLUDE, hits, ".zmetadata:root")
    return json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _store_digests(store: Path) -> tuple[dict[str, str], list[str]]:
    """遍历 store 成员，排除集字段先剥离、from-value 目录名先规范化，再逐成员取 sha256。"""
    digests: dict[str, str] = {}
    hits: list[str] = []
    for path in sorted(store.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(store).as_posix()
        raw = path.read_bytes()
        if name == ".zattrs":
            raw = _canon_attrs(raw, ROOT_ATTR_EXCLUDE, ".zattrs", hits)
        elif name == ".zmetadata":
            raw = _canon_zmetadata(raw, hits)
        elif _HISTORY_ATTR_RE.fullmatch(name):
            raw = _canon_attrs(raw, HISTORY_ATTR_EXCLUDE, name, hits)
        norm = _FROM_VALUE_RE.sub("from-value-<TOKEN>", name)
        if norm != name:
            hits.append(f"{name} → {norm}")
        if norm in digests:
            raise RuntimeError(f"from-value 规范化后成员名冲突: {norm}")
        digests[norm] = hashlib.sha256(raw).hexdigest()
    return digests, hits


def _tokenize(text: str, roots: dict[str, str]) -> str:
    """把指标表文本中的命令行根路径还原为占位符，令黄金指纹跨机器可迁移。"""
    return text.replace(roots["track"], "__TRACK_ROOT__").replace(roots["output"], "__OUTPUT_ROOT__")


def _flatten(obj: Any, prefix: str = "$") -> dict[str, Any]:
    if isinstance(obj, dict):
        return {k: v for item in obj.items() for k, v in _flatten(item[1], f"{prefix}.{item[0]}").items()}
    if isinstance(obj, list):
        return {k: v for i, item in enumerate(obj) for k, v in _flatten(item, f"{prefix}[{i}]").items()}
    return {prefix: obj}


def _anchor_check(metrics_doc: dict[str, Any], track_root: Path) -> list[str]:
    """指标对赛道冒烟基准 JSON 的 cr/prd 逐位等值门与五合规门全过门。"""
    fails: list[str] = []
    anchor_path = track_root / ANCHOR_RELPATH
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    quality = metrics_doc.get(METRICS_KEY)
    if not isinstance(quality, dict):
        return [f"指标表缺 {METRICS_KEY} 键"]
    for field, ours, gold in (("cr", quality.get("cr"), anchor["cr"]),
                              ("metrics.prd", quality.get("metrics", {}).get("prd"), anchor["prd"])):
        mark = "OK" if ours == gold else "FAIL"
        print(f"[anchor] {field}: ours={ours!r} anchor={gold!r} 逐位一致: {mark}")
        if ours != gold:
            fails.append(f"{field} 与冒烟基准 {anchor_path.name} 不一致")
    gates = {g["gate"]: g["status"] for g in quality.get("compliance", {}).get("gates", [])}
    bad = [name for name in GATE_NAMES if gates.get(name) != "pass"]
    print(f"[anchor] 合规五门: {json.dumps(gates, ensure_ascii=False)}")
    if bad:
        fails.append(f"合规门未全过: {bad}")
    return fails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--work-dir", type=Path, default=None, help="驱动工作目录，缺省用临时目录")
    parser.add_argument("--track-root", required=True, help="高保真脑电数据压缩赛道仓库根目录，替换 __TRACK_ROOT__")
    parser.add_argument("--output-root", type=Path, required=True, help="产物输出根目录，替换 __OUTPUT_ROOT__")
    parser.add_argument("--golden", type=Path, default=None, help="指纹留存/比对文件：不存在则写入本轮指纹，存在则逐字段比对")
    args = parser.parse_args(argv)

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="lince_eeg_e2e_work_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    roots = {"track": args.track_root, "output": str(output_root)}
    resolved = _resolve_config(args.config, roots, work_dir)
    print(f"[e2e] resolved config: {resolved}")

    store1, eval_path1, eval1 = _run_chain(resolved, work_dir / "run1")
    snap_dir = work_dir / "run1_snapshot"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_store = snap_dir / store1.name
    shutil.copytree(store1, snap_store)
    shutil.copy2(eval_path1, snap_dir / EVAL_NAME)
    store2, eval_path2, eval2 = _run_chain(resolved, work_dir / "run2")
    print(f"[e2e] store: {store2}")
    print(f"[e2e] 指标表文件: {eval_path2}")

    fails: list[str] = []
    if store1 != store2 or eval_path1 != eval_path2:
        fails.append(f"两轮导出目标不一致: {store1} vs {store2}")

    # 层 1：指标表两轮逐字节一致
    sha1, sha2 = hashlib.sha256(eval1).hexdigest(), hashlib.sha256(eval2).hexdigest()
    print(f"[recon] {EVAL_NAME} sha256: run1={sha1} run2={sha2}")
    if eval1 != eval2:
        fails.append("指标表两轮逐字节不一致")

    # 层 2：.ls 排除集外成员两轮逐字节一致；session 标签符合夹具
    digests1, hits1 = _store_digests(snap_store)
    digests2, hits2 = _store_digests(store2)
    print(f"[recon] 字节比对范围: run1 {len(digests1)} 成员 / run2 {len(digests2)} 成员")
    print(f"[recon] 排除集命中: {sorted(set(hits1) | set(hits2)) or '无'}")
    only1, only2 = sorted(set(digests1) - set(digests2)), sorted(set(digests2) - set(digests1))
    differ = sorted(k for k in set(digests1) & set(digests2) if digests1[k] != digests2[k])
    byte_ok = not (only1 or only2 or differ)
    if not byte_ok:
        fails.append(f".ls 字节层不一致: only_run1={only1} only_run2={only2} differ={differ}")
    print(f"[recon] 排除集外逐字节一致: {len(set(digests1) & set(digests2)) - len(differ)}/{len(digests1)} 成员 {'OK' if byte_ok else 'FAIL'}")
    attrs2 = json.loads((store2 / ".zattrs").read_text(encoding="utf-8"))
    if attrs2.get("session") != EXPECTED_SESSION:
        fails.append(f"store session={attrs2.get('session')!r} != {EXPECTED_SESSION!r}")

    # 层 3：指标对赛道冒烟基准锚点与合规五门
    fails.extend(_anchor_check(json.loads(eval2.decode("utf-8")), Path(args.track_root)))

    # 层 4：golden 指纹留存/比对（占位符化指标表，逐字段）
    if args.golden is not None:
        tokenized = _tokenize(eval2.decode("utf-8"), roots)
        table = json.loads(tokenized)
        fingerprint = {"fingerprint_version": 1, "chain": "eeg_compression_np_smoke",
                       "eval_sha256_tokenized": hashlib.sha256(tokenized.encode("utf-8")).hexdigest(),
                       "metrics_table": table}
        if args.golden.exists():
            gold = json.loads(args.golden.read_text(encoding="utf-8"))
            cur_flat, gold_flat = _flatten(fingerprint["metrics_table"]), _flatten(gold["metrics_table"])
            absent = object()
            diffs = sorted(p for p in set(cur_flat) | set(gold_flat)
                           if cur_flat.get(p, absent) != gold_flat.get(p, absent))
            golden_ok = not diffs
            for p in diffs[:10]:
                print(f"[golden] 差异字段 {p}: run={cur_flat.get(p)!r} golden={gold_flat.get(p)!r}")
            if not golden_ok:
                fails.append(f"GOLDEN 比对不一致字段数: {len(diffs)}")
            print(f"[golden] 与 {args.golden} 逐字段比对: {'PASS' if golden_ok else 'FAIL'}（{len(cur_flat)} 字段）")
        else:
            args.golden.parent.mkdir(parents=True, exist_ok=True)
            args.golden.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[golden] 指纹留存: {args.golden}（{len(_flatten(table))} 字段）")

    for line in fails:
        print(f"[FAIL] {line}")
    print(f"E2E EEG COMPRESSION {'PASS' if not fails else 'FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
