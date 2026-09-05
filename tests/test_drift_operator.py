"""漂移算子测试：注册、context 投影、批次/sweep 模式、黄金 hard-12 现场对账。"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("linxi")

from _drift_helpers import (  # noqa: E402
    DATA_ROOT,
    REPORT_GOLDEN_4DP,
    is_nan as _is,
    load_explore_module as _explore,
    load_neural as _load_neural,
)
from linxi.processor import PROCESS_STAGES  # noqa: E402
from linxi.processor.registry import (  # noqa: E402
    get_processor_registration_name,
    get_registered_processors,
)

requires_data = pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"data root absent: {DATA_ROOT}")


@pytest.fixture(scope="module")
def drift_op():
    return importlib.import_module("linxi_linceplugin.drift_analysis")


@pytest.fixture(scope="module")
def drift_core():
    return importlib.import_module("linxi_linceplugin._drift_core")


def _make_context():
    from linxi.fabric.linxi_context import LinxiContext

    return LinxiContext(file_create_date=datetime.now())


def _record_for(x: np.ndarray, name: str):
    import spikeinterface.core as sc
    from linshu_format.core import EcephysRecording

    return EcephysRecording.from_spikeinterface_recording(
        sc.NumpyRecording(x.astype(np.float32), sampling_frequency=50.0), name=name
    )


def test_drift_operator_registered_at_analyze_stage(drift_op):
    names = [
        get_processor_registration_name(cls)
        for cls in get_registered_processors(PROCESS_STAGES.ANALYZE)
    ]
    assert "LinceDriftAnalysis" in names


def test_drift_operator_context_mode_projects_slots(drift_op):
    explore_module = _explore()
    rng = np.random.default_rng(11)
    support = rng.integers(0, 5, size=(300, 512), dtype=np.uint8)
    query = rng.integers(0, 5, size=(200, 512), dtype=np.uint8)

    context = _make_context()
    context.ecephys["support"] = _record_for(support, "support")
    context.ecephys["query"] = _record_for(query, "binned_spikes")

    op = drift_op.LinceDriftAnalysis(
        source="context",
        target_keys=["query"],
        centroid_key=["support"],
        session_key="MA-CO-20240305-01",
        train_session_keys=["MA-CO-20240117-01"],
    )
    out = op(context)

    slot = out.ecephys["query"]
    assert slot.channel_summary is not None
    ds = slot.channel_summary
    for col in ("drift_fr", "drift_centroid_fr", "drift_cos_raw", "drift_cos_centered", "drift_norm_ratio", "drift_gap_days"):
        assert col in ds.data_vars, col
        assert ds[col].sizes["channel_id"] == 512

    q_fr = query.mean(axis=0).astype(np.float64)
    s_fr = support.mean(axis=0).astype(np.float64)
    # 槽内存 float32 迹线（NumpyRecording 物化），容差按计划 1e-6 档而非 float64 位级
    assert np.allclose(ds["drift_fr"].values, q_fr, atol=1e-6)
    assert np.allclose(ds["drift_centroid_fr"].values, s_fr, atol=1e-6)
    assert float(ds["drift_cos_raw"].values[0]) == pytest.approx(explore_module.cosine(q_fr, s_fr), abs=1e-6)
    assert float(ds["drift_gap_days"].values[0]) == pytest.approx((datetime(2024, 3, 5) - datetime(2024, 1, 17)).days, abs=0)

    notes = json.loads(out.notes)
    payload = notes["lince_drift_analysis"]
    assert payload["mode"] == "context"
    row = payload["targets"]["query"]
    assert row["cos_raw"] == pytest.approx(explore_module.cosine(q_fr, s_fr), abs=1e-6)
    assert row["gap_days"] == 48

    # 无质心来源时受控报错（Fast Fail，不静默产出 NaN 画像）
    context_bare = _make_context()
    context_bare.ecephys["query"] = _record_for(query, "binned_spikes")
    with pytest.raises(ValueError, match="centroid"):
        drift_op.LinceDriftAnalysis(source="context", target_keys=["query"])(context_bare)

    # 历史 notes 非 JSON 时不丢信息（prior_notes 承接，同任务 8 模式）
    context2 = _make_context()
    context2.notes = "human memo"
    context2.ecephys["query"] = _record_for(query, "binned_spikes")
    context2.ecephys["support"] = _record_for(support, "support")
    out2 = drift_op.LinceDriftAnalysis(source="context", centroid_key=["support"])(context2)
    notes2 = json.loads(out2.notes)
    assert notes2["prior_notes"] == "human memo"
    assert "lince_drift_analysis" in notes2


def test_drift_operator_batch_mode_result_and_notes(drift_op):
    rng = np.random.default_rng(13)
    train_x = rng.integers(0, 5, size=(120, 512), dtype=np.uint8)
    hard_x = rng.integers(0, 5, size=(80, 512), dtype=np.uint8)
    sessions = [
        {"task": "MA_CO", "level": "train", "session_key": "MA-CO-20240117-01", "x": train_x},
        {"task": "MA_CO", "level": "hard", "horizon": "week", "session_key": "MA-CO-20240124-01", "x": hard_x},
    ]
    context = _make_context()
    op = drift_op.LinceDriftAnalysis(source="sessions", sessions=sessions)
    out = op(context)

    assert op.result is not None and len(op.result.sessions) == 2
    assert not _is(op.result.cos_raw_mean_hard)
    payload = json.loads(out.notes)["lince_drift_analysis"]
    assert payload["mode"] == "sessions"
    assert payload["summary"]["cos_raw_mean_hard"] == pytest.approx(op.result.cos_raw_mean_hard, abs=1e-12)
    assert {row["session_key"] for row in payload["sessions"]} == {"MA-CO-20240117-01", "MA-CO-20240124-01"}

    with pytest.raises(ValueError, match="nwb_path|x"):
        drift_op.LinceDriftAnalysis(source="sessions", sessions=[{"task": "t", "level": "train", "session_key": "MA-CO-20240101-01"}])(_make_context())


@requires_data
def test_drift_operator_sweep_mode_matches_direct_read_count(drift_op):
    """sweep 源目录发现与 collect_records 同构：全 2 任务 = 8 train + 38 query = 46 会话。"""
    context = _make_context()
    op = drift_op.LinceDriftAnalysis(source="sweep", levels=["train"])
    out = op(context)
    assert op.result is not None and len(op.result.sessions) == 8
    assert all(m.level == "train" for m in op.result.sessions)
    assert {m.task for m in op.result.sessions} == {"MA_CO", "MA_RT"}
    payload = json.loads(out.notes)["lince_drift_analysis"]
    assert payload["mode"] == "sweep" and payload["n_sessions"] == 8


@requires_data
def test_drift_golden_hard12_matches_live_script_unrounded(drift_core):
    """原始值基准法（计划行 156）：脚本现场产出（未舍入）vs 算子输出，逐项 ≤1e-6。

    同时钉报告 4 位舍入值（data_exploration.md，第二道目视核对）。
    """
    explore_module = _explore()
    records = explore_module.collect_records()
    explore_module.attach_centroid_metrics(records)
    heldout = [r for r in records if r.level in ("easy", "normal", "hard")]

    def gmean(rows, attr):
        vals = [getattr(r, attr) for r in rows if np.isfinite(getattr(r, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    hard = [r for r in records if r.level == "hard"]
    base = {
        "cos_raw_mean_hard": gmean(hard, "cos_raw"),
        "cos_centered_mean_hard": gmean(hard, "cos_centered"),
        "norm_ratio_dev_hard": abs(gmean(hard, "norm_ratio") - 1.0),
        "pearson_gap_cos": explore_module.pearson(
            [float(r.gap_days) for r in heldout if np.isfinite(r.cos_raw)],
            [r.cos_raw for r in heldout if np.isfinite(r.cos_raw)],
        ),
    }

    inputs = [
        drift_core.DriftSessionInput(
            task=r.task, level=r.level, horizon=r.horizon, session_key=r.session_key, x=_load_neural(r.path),
        )
        for r in records
    ]
    result = drift_core.compute_drift_metrics(inputs)

    for name, want in base.items():
        got = getattr(result, name)
        assert abs(got - want) <= 1e-6, f"{name}: script={want!r} operator={got!r}"
        nd = 3 if name == "pearson_gap_cos" else 4
        assert f"{got:.{nd}f}" == REPORT_GOLDEN_4DP[name], name

    op_rows = {(m.task, m.level, m.session_key): m for m in result.sessions}
    assert len(op_rows) == len(records)
    for r in records:
        m = op_rows[(r.task, r.level, r.session_key)]
        for attr in ("cos_raw", "cos_centered", "norm_ratio"):
            assert abs(getattr(m, attr) - getattr(r, attr)) <= 1e-6, f"{r.session_id}.{attr}"
        assert m.gap_days == r.gap_days, r.session_id
