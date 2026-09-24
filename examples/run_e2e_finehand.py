"""临策精细手部 e2e MINI 驱动：同一输入连跑两次链，对账 .ls 与评测结果文件的自身决定论。

对账口径参照运动跨天解码赛道 linshufile_validation.md §1 的排除集先例，不设逐位 PARITY 门。
"""
# allow: SIZE_OK — e2e 驱动按 examples 成例单文件自包含，计划限定本笔提交只含链配置与驱动两个文件。
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import numpy as np
import xarray as xr
import zarr
from pydantic import BaseModel

DEFAULT_CONFIG = Path(__file__).with_name("e2e_finehand_mini.yaml")
_PATH_TOKENS = ("__DATA_ROOT__", "__TRACK_ROOT__", "__OUTPUT_ROOT__")
MINI_SESSION = "2026060501"
EVAL_NAME = "FineHandGruInfer.eval.json"
EVAL_KEYS = ["s_wrist", "s_pose", "s_motion", "s_bone", "final_score"]
PRESET_SCORES = {"s_wrist": 34.68660729162529, "s_pose": 35.42921990421653, "s_motion": 4.483756368734944, "s_bone": 34.98987258629803, "final_score": 29.010539312180555}
PRESET_TOLERANCE = 1e-3
SAMPLE_SEED = 20260924
# 排除集五项：file_create_date、notes 内时间衍生字段、history 记录的时间戳与耗时、from-value 目录名、环境信息键。
# 环境信息键当前上游拼写为 python_environments（LinshuFormat 9daaa89 修正了母本报告时期的 python_environmenets 拼写），两拼写一并排除。
ROOT_ATTR_EXCLUDE = ("file_create_date", "python_environments", "python_environmenets")
HISTORY_ATTR_EXCLUDE = ("timestamp", "duration")
NOTE_TIME_KEYS = ("total_latency_ms", "latency_per_bin_ms", "latency_score", "session_score", "adapt_seconds")
_FROM_VALUE_RE = re.compile(r"from-value-[0-9a-f]{6,}")
_HISTORY_ATTR_RE = re.compile(r"history/\d+/\.zattrs")


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


def _run_chain(resolved: Path, work_dir: Path) -> Path:
    """执行一次链并返回 .ls store 路径。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    pipeline_def = PipelineDefinition.from_file(resolved)
    context = LinxiContext(file_create_date=datetime.now())
    apply_pipeline_metadata(context, pipeline_def)
    context.run_state.work_dir = work_dir
    final = TaskRunner(pipeline_def).run(context, allow_errors=False)[-1]
    store = Path(final.run_state.output_path)
    if not store.is_dir():
        raise RuntimeError(f"导出产物不存在: {store}")
    return store


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
    for path in store.rglob("*"):
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


def _array_eq(a: Any, b: Any) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def _eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, BaseModel):
        return type(a) is type(b) and all(_eq(getattr(a, f), getattr(b, f)) for f in type(a).model_fields)
    if isinstance(a, xr.Dataset):
        return isinstance(b, xr.Dataset) and set(a.data_vars) == set(b.data_vars) and all(_eq(a[k], b[k]) for k in a.data_vars)
    if isinstance(a, xr.DataArray):
        return isinstance(b, xr.DataArray) and a.dims == b.dims and a.shape == b.shape and _array_eq(a.data, b.data)
    if isinstance(a, (np.ndarray, zarr.Array)):
        return isinstance(b, (np.ndarray, zarr.Array)) and _array_eq(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return bool(a == b)


def _history_equal(records_a: list[Any], records_b: list[Any]) -> bool:
    if len(records_a) != len(records_b):
        return False
    if not records_a:
        return True
    keep = tuple(f for f in type(records_a[0]).model_fields if f not in HISTORY_ATTR_EXCLUDE)
    return all(_eq(getattr(x, f), getattr(y, f)) for x, y in zip(records_a, records_b, strict=True) for f in keep)


def _notes_canon(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return _strip_dict(parsed, NOTE_TIME_KEYS, [], "notes")


def _compare_models(ca: LinxiContext, cb: LinxiContext) -> tuple[int, list[str]]:
    compared = 0
    mismatches: list[str] = []
    for name in type(ca).model_fields:
        if name in ROOT_ATTR_EXCLUDE:
            continue
        va, vb = getattr(ca, name), getattr(cb, name)
        if name == "history":
            ok = _history_equal(va, vb)
        elif name == "notes" and isinstance(va, str):
            ok = _eq(_notes_canon(va), _notes_canon(vb))
        else:
            ok = _eq(va, vb)
        compared += 1
        if not ok:
            mismatches.append(name)
    return compared, mismatches


def _recount_source(session_dir: Path) -> np.ndarray:
    """按载入算子同一规则从源 npz 重算 counts：int64 ns 域逐帧时长、首帧沿用第二帧间隔、tx×dt 后 cast float32。"""
    partitions = [session_dir / p for p in ("heldin", "heldout") if (session_dir / p).is_dir()]
    if len(partitions) != 1:
        raise RuntimeError(f"{session_dir}: 须恰好含 heldin/ 或 heldout/ 一个分区目录")
    parts = []
    for path in sorted(partitions[0].glob("trial_*.npz")):
        with np.load(path) as data:
            tx = np.asarray(data["tx"])
            ts = np.asarray(data["frame_timestamps_ns"], dtype=np.int64)
        dt = np.empty(ts.shape[0], dtype=np.int64)
        dt[1:] = np.diff(ts)
        dt[0] = dt[1]
        parts.append((tx * (dt.astype(np.float64) / 1_000_000_000)[:, None]).astype(np.float32))
    return np.concatenate(parts)


def _sample_indices(n_frames: int) -> list[int]:
    rng = np.random.default_rng(SAMPLE_SEED)
    extra = rng.choice(n_frames, size=100, replace=False)
    return sorted(set(range(min(100, n_frames))) | {int(i) for i in extra})


def _eval_form_check(raw: bytes) -> tuple[dict[str, float], list[str]]:
    problems: list[str] = []
    if raw.endswith(b"\n"):
        problems.append("eval.json 含尾随换行")
    doc = json.loads(raw)
    if list(doc) != EVAL_KEYS:
        problems.append(f"eval.json 键序 {list(doc)} != {EVAL_KEYS}")
    if not all(isinstance(v, float) and math.isfinite(v) for v in doc.values()):
        problems.append("eval.json 含非有限或非浮点值")
    return {k: float(v) for k, v in doc.items()}, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--work-dir", type=Path, default=None, help="驱动工作目录，缺省用临时目录")
    parser.add_argument("--data-root", required=True, help="赛题数据根目录，替换 __DATA_ROOT__")
    parser.add_argument("--track-root", required=True, help="精细手部赛道仓库根目录，替换 __TRACK_ROOT__")
    parser.add_argument("--output-root", type=Path, required=True, help="产物输出根目录，替换 __OUTPUT_ROOT__")
    parser.add_argument("--golden", type=Path, default=None, help="指纹留存/比对文件：不存在则写入本轮指纹，存在则逐成员比对")
    args = parser.parse_args(argv)

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="lince_finehand_e2e_work_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    roots = {"data": args.data_root, "track": args.track_root, "output": str(output_root)}
    resolved = _resolve_config(args.config, roots, work_dir)
    print(f"[e2e] resolved config: {resolved}")

    store1 = _run_chain(resolved, work_dir / "run1")
    snap_dir = work_dir / "run1_snapshot"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_store = snap_dir / store1.name
    shutil.copytree(store1, snap_store)
    eval1 = (output_root / EVAL_NAME).read_bytes()
    shutil.copy2(output_root / EVAL_NAME, snap_dir / EVAL_NAME)
    store2 = _run_chain(resolved, work_dir / "run2")
    eval2 = (output_root / EVAL_NAME).read_bytes()
    print(f"[e2e] store: {store2}")
    print(f"[e2e] 评测结果文件: {output_root / EVAL_NAME}")

    fails: list[str] = []
    if store1 != store2:
        fails.append(f"两轮导出目标不一致: {store1} vs {store2}")

    # 层 1a：.eval.json 逐字节一致 + 形态
    sha1, sha2 = hashlib.sha256(eval1).hexdigest(), hashlib.sha256(eval2).hexdigest()
    print(f"[recon] eval.json sha256: run1={sha1} run2={sha2}")
    if eval1 != eval2:
        fails.append("eval.json 两轮逐字节不一致")
    scores, form_problems = _eval_form_check(eval2)
    fails.extend(form_problems)
    print(f"[recon] eval.json 形态（五键固定键序、无尾随换行、值有限）: {'OK' if not form_problems else form_problems}; 分数={json.dumps(scores, ensure_ascii=False)}")

    # 层 1b：pipeline 根 attrs 两轮逐字节一致（母本先例中该键差异源于 output_path 重定向的实验变量，本链两轮共用同一参数集）
    attrs1 = json.loads((snap_store / ".zattrs").read_text(encoding="utf-8"))
    attrs2 = json.loads((store2 / ".zattrs").read_text(encoding="utf-8"))
    pipeline_ok = attrs1.get("pipeline") is not None and attrs1["pipeline"] == attrs2.get("pipeline")
    if not pipeline_ok:
        fails.append("pipeline 根 attrs 两轮不一致或缺席")
    print(f"[recon] pipeline 键两轮逐字节一致: {'OK' if pipeline_ok else 'FAIL'}")
    for tag, attrs in (("run1", attrs1), ("run2", attrs2)):
        if attrs.get("session") != MINI_SESSION:
            fails.append(f"{tag} store session={attrs.get('session')!r} != {MINI_SESSION!r}")

    # 层 2a：.ls 排除集外成员逐字节
    digests1, hits1 = _store_digests(snap_store)
    digests2, hits2 = _store_digests(store2)
    print(f"[recon] 字节比对范围: run1 {len(digests1)} 成员 / run2 {len(digests2)} 成员")
    print(f"[recon] 排除集命中: {sorted(set(hits1) | set(hits2)) or '无'}")
    only1, only2 = sorted(set(digests1) - set(digests2)), sorted(set(digests2) - set(digests1))
    differ = sorted(k for k in set(digests1) & set(digests2) if digests1[k] != digests2[k])
    byte_ok = not (only1 or only2 or differ or store1 != store2)
    if not byte_ok:
        fails.append(f".ls 字节层不一致: only_run1={only1} only_run2={only2} differ={differ}")
    print(f"[recon] 排除集外逐字节一致: {len(set(digests1) & set(digests2)) - len(differ)}/{len(digests1)} 成员 {'OK' if byte_ok else 'FAIL'}")

    # 层 2b：from_zarr 读回逐字段值级一致
    ctx1 = LinxiContext.from_zarr(str(snap_store))
    ctx2 = LinxiContext.from_zarr(str(store2))
    compared, field_bad = _compare_models(ctx1, ctx2)
    value_ok = not field_bad
    if field_bad:
        fails.append(f".ls 值级不一致字段: {field_bad}")
    print(f"[recon] from_zarr 值级一致: {compared - len(field_bad)}/{compared} 声明字段 {'OK' if value_ok else 'FAIL'}")

    # 层 3：读回 binned_spikes counts 与源 npz 重算抽样一致
    matrix = ctx2.binned_spikes.counts if ctx2.binned_spikes is not None else None
    if matrix is None:
        fails.append("读回 binned_spikes 缺席（链未产出根容器）")
        print("[recon] 读回抽样: FAIL（binned_spikes 缺席）")
    else:
        readback = np.asarray(matrix.data if isinstance(matrix, xr.DataArray) else matrix)
        expected = _recount_source(Path(args.data_root) / "Data_heldin" / MINI_SESSION)
        if readback.shape != expected.shape or readback.dtype != np.float32:
            fails.append(f"读回 counts 形状/类型不符: {readback.shape}/{readback.dtype} vs {expected.shape}/float32")
            print("[recon] 读回抽样: FAIL（形状/类型不符）")
        else:
            idx = np.asarray(_sample_indices(expected.shape[0]))
            hit = bool(np.array_equal(readback[idx], expected[idx]))
            if not hit:
                bad = int(np.count_nonzero((readback[idx] != expected[idx]).any(axis=1)))
                fails.append(f"读回 counts 抽样不一致: {bad}/{len(idx)} 帧")
            print(f"[recon] 读回抽样 {len(idx)} 帧（前 100 + seed={SAMPLE_SEED} 随机 100）逐位一致: {'OK' if hit else 'FAIL'}")

    # 层 1c：分数对任务 8 预置权重锚点（记录性门，防错权重/错 session 灾难性漂移）
    if set(scores) == set(EVAL_KEYS):
        anchor_delta = max(abs(scores[k] - PRESET_SCORES[k]) for k in EVAL_KEYS)
        print(f"[recon] 分数对预置权重锚点最大差: {anchor_delta:.3e}（门槛 {PRESET_TOLERANCE:g}）")
        if anchor_delta > PRESET_TOLERANCE:
            fails.append(f"分数对预置权重锚点偏差超 {PRESET_TOLERANCE:g}: {anchor_delta:.3e}")

    # golden 指纹留存/比对
    if args.golden is not None:
        fingerprint = {"fingerprint_version": 1, "session": MINI_SESSION, "eval_sha256": sha2, "eval": scores, "store_members": dict(sorted(digests2.items()))}
        if args.golden.exists():
            gold = json.loads(args.golden.read_text(encoding="utf-8"))
            common = set(fingerprint["store_members"]) & set(gold["store_members"])
            diff = sorted(set(fingerprint["store_members"]) ^ set(gold["store_members"])) + sorted(k for k in common if fingerprint["store_members"][k] != gold["store_members"][k])
            golden_ok = not diff and fingerprint["eval_sha256"] == gold["eval_sha256"]
            if not golden_ok:
                fails.append(f"GOLDEN 比对不一致: {diff[:10]}")
            print(f"[golden] 与 {args.golden} 比对: {'PASS' if golden_ok else 'FAIL'}")
        else:
            args.golden.parent.mkdir(parents=True, exist_ok=True)
            args.golden.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[golden] 指纹留存: {args.golden}")

    for line in fails:
        print(f"[FAIL] {line}")
    print(f"E2E MINI {'PASS' if not fails else 'FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
