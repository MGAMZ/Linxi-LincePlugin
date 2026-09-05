from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("linxi")
pytest.importorskip("sklearn")

from linxi.processor import PROCESS_STAGES  # noqa: E402
from linxi.processor.registry import get_processor_registration_name, get_registered_processors  # noqa: E402

TRACK_ROOT = Path("/mnt/f/mgam_repos/Lince/运动跨天解码")
DATASET_ROOT = Path("/mnt/f/mgam_datasets/Lince/运动跨天解码")
DATA_ROOT = DATASET_ROOT / "challenge_data"
GOLDEN_DIR = TRACK_ROOT / "output" / "csv"

needs_dataset = pytest.mark.skipif(
    not (DATA_ROOT / "public").is_dir(), reason="challenge data root not present"
)
needs_golden = pytest.mark.skipif(
    not (GOLDEN_DIR / "wf_challenge_results.json").is_file(),
    reason="golden result jsons not present",
)

EASY_SESSION = "MA_CO/easy/MA-CO-20231225-01"


def _golden(model: str) -> dict:
    return json.loads((GOLDEN_DIR / f"{model}_challenge_results.json").read_text(encoding="utf-8"))


def _golden_session(model: str, session_id: str) -> dict:
    data = _golden(model)
    task, level, *rest = session_id.split("/")
    key = rest[-1]
    for row in data["task_results"][task][level]:
        if row["session_key"] == key:
            return row
    raise KeyError(session_id)


def _sweep_operator(**params):
    from linxi_linceplugin.decode_baseline import BaselineDecodeInfer

    params.setdefault("source", "sweep")
    return BaselineDecodeInfer(**params)


def _make_context():
    from linxi.fabric.linxi_context import LinxiContext

    return LinxiContext(file_create_date=datetime.now())


# ---------------------------------------------------------------- registration

def test_decode_processor_registration_under_postprocess():
    import linxi_linceplugin.decode_baseline  # noqa: F401  triggers registration

    names = [
        get_processor_registration_name(cls)
        for cls in get_registered_processors(PROCESS_STAGES.POSTPROCESS)
    ]
    assert "BaselineDecodeInfer" in names


# ---------------------------------------------------------------- scoring core

def test_decode_score_session_keeps_negative_r2_raw_and_clips_score():
    from linxi_linceplugin.decode_engine import score_session

    rng = np.random.default_rng(3)
    y = rng.normal(size=(50, 2)).astype(np.float32)
    pred = -y * 4.0
    row = score_session(y, pred, total_latency_ms=10.0, session_id="s")
    assert row["r2_mean_raw"] < 0.0
    assert row["r2_mean"] == 0.0
    assert abs(row["session_score"] - 0.05 * row["latency_score"]) < 1e-15


def test_decode_score_session_latency_clip_bounds():
    from linxi_linceplugin.decode_engine import compute_latency_score

    assert compute_latency_score(0.0) == 1.0
    assert compute_latency_score(20.0) == 0.0
    assert compute_latency_score(100.0) == 0.0
    assert abs(compute_latency_score(1.0) - 0.95) < 1e-15


def test_decode_aggregate_weights_match_official_formula():
    from linxi_linceplugin.decode_engine import aggregate_scores

    def rows(score):
        return [{"session_score": score} for _ in range(2)]

    task_results = {
        "MA_CO": {"easy": rows(1.0), "normal": rows(0.5), "hard": rows(0.0)},
        "MA_RT": {"easy": rows(0.0), "normal": rows(0.0), "hard": rows(1.0)},
    }
    summary = aggregate_scores(task_results)
    co = 0.25 * 1.0 + 0.45 * 0.5 + 0.30 * 0.0
    rt = 0.25 * 0.0 + 0.45 * 0.0 + 0.30 * 1.0
    assert abs(summary["task_scores"]["MA_CO"] - co) < 1e-15
    assert abs(summary["task_scores"]["MA_RT"] - rt) < 1e-15
    assert abs(summary["final_score"] - (co + rt) / 2.0) < 1e-15


# ---------------------------------------------------------------- failure QA

def test_decode_wrong_weights_dir_raises_controlled_error_with_path(tmp_path):
    op = _sweep_operator(
        model="wf", task="MA_CO", level="easy", session_key="MA-CO-20231225-01",
        weights_dir=str(tmp_path / "nowhere"),
    )
    with pytest.raises(FileNotFoundError, match="nowhere"):
        op(_make_context())


def test_decode_unknown_model_kind_rejects():
    op = _sweep_operator(model="lstm")
    with pytest.raises(ValueError, match="model"):
        op(_make_context())


# ---------------------------------------------------------------- golden seams

@needs_dataset
@needs_golden
def test_decode_wf_sweep_single_session_matches_golden_r2():
    op = _sweep_operator(model="wf", task="MA_CO", level="easy",
                         session_key="MA-CO-20231225-01")
    op(_make_context())
    (row,) = op.session_results
    gold = _golden_session("wf", EASY_SESSION)
    assert row["session_id"] == EASY_SESSION
    for field in ("r2_x", "r2_y", "r2_mean_raw"):
        assert abs(row[field] - gold[field]) <= 1e-6, (field, row[field], gold[field])


@needs_dataset
@needs_golden
def test_decode_gru_sweep_single_session_matches_golden_r2():
    pytest.importorskip("torch")
    op = _sweep_operator(model="gru", task="MA_CO", level="easy",
                         session_key="MA-CO-20231225-01")
    op(_make_context())
    (row,) = op.session_results
    gold = _golden_session("gru", EASY_SESSION)
    for field in ("r2_x", "r2_y", "r2_mean_raw"):
        assert abs(row[field] - gold[field]) <= 1e-5, (field, row[field], gold[field])


@needs_dataset
@needs_golden
def test_decode_wf_hard_empty_support_matches_golden():
    gold_hard = _golden("wf")["task_results"]["MA_CO"]["hard"][0]
    op = _sweep_operator(model="wf", task="MA_CO", level="hard",
                         session_key=gold_hard["session_key"], horizon=gold_hard["horizon"])
    op(_make_context())
    (row,) = op.session_results
    assert row["support_trials"] == 0
    for field in ("r2_x", "r2_y", "r2_mean_raw"):
        assert abs(row[field] - gold_hard[field]) <= 1e-6, (field, row[field], gold_hard[field])


@needs_dataset
@needs_golden
def test_decode_wf_sweep_final_matches_golden_with_golden_latency():
    data = _golden("wf")
    op = _sweep_operator(model="wf")
    context = op(_make_context())
    ours = {r["session_id"]: r for r in op.session_results}
    assert len(ours) == 38
    adjusted = {}
    for task, levels in data["task_results"].items():
        adjusted[task] = {}
        for level, rows in levels.items():
            adjusted[task][level] = []
            for gold in rows:
                row = dict(ours[gold["session_id"]])
                row["latency_score"] = gold["latency_score"]
                row["session_score"] = 0.95 * row["r2_mean"] + 0.05 * row["latency_score"]
                adjusted[task][level].append(row)
    from linxi_linceplugin.decode_engine import aggregate_scores

    summary = aggregate_scores(adjusted)
    assert abs(summary["final_score"] - data["final_score"]) <= 1e-5


# ---------------------------------------------------------------- context mode

@needs_dataset
@needs_golden
def test_decode_context_mode_matches_golden_and_writes_predictions():
    import linxi.fabric.linxi_context as ctx_mod
    import spikeinterface.core as si
    from linshu_format.core import EcephysRecording, TimeIntervals, TimeSeries
    from linxi_linceplugin.decode_baseline import BaselineDecodeInfer

    sys.path.insert(0, str(TRACK_ROOT))
    from challenge_code.Platform.Platform_Implementation.data import load_nwb_session

    task, level, key = EASY_SESSION.split("/")
    support_nwb = sorted((DATA_ROOT / "public" / task / level / "heldin" / key).glob("*.nwb"))[0]
    query_nwb = sorted((DATA_ROOT / "private" / task / level / "heldout" / key).glob("*.nwb"))[0]
    sup, qry = load_nwb_session(support_nwb), load_nwb_session(query_nwb)

    def record_for(name, d):
        rec = EcephysRecording.from_spikeinterface_recording(
            si.NumpyRecording(d.neural, sampling_frequency=50.0), name=name,
            unit="spike counts",
        )
        rec.electrophysiology.timestamps = d.timestamps
        rec.auxiliary_channels["cursor_vel_x"] = TimeSeries(
            name="cursor_vel_x", data=d.velocity[:, 0], timestamps=d.timestamps,
        )
        rec.auxiliary_channels["cursor_vel_y"] = TimeSeries(
            name="cursor_vel_y", data=d.velocity[:, 1], timestamps=d.timestamps,
        )
        ids = np.unique(d.trial_ids[d.trial_ids >= 0])
        starts, stops = [], []
        for tid in ids:
            sel = d.trial_ids == tid
            starts.append(d.timestamps[sel].min())
            stops.append(d.timestamps[sel].max())
        import pandas as pd

        rec.events = TimeIntervals(
            name=f"{name}_trials",
            table=pd.DataFrame(
                {"start_time": starts, "stop_time": stops, "trial_id": ids}
            ),
        )
        return rec

    context = ctx_mod.LinxiContext(file_create_date=datetime.now())
    context.ecephys["query"] = record_for("query", qry)
    context.ecephys["support"] = record_for("support", sup)

    op = BaselineDecodeInfer(model="wf", source="context", task=task, level=level,
                             session_key=key)
    context = op(context)
    (row,) = op.session_results
    gold = _golden_session("wf", EASY_SESSION)
    for field in ("r2_x", "r2_y", "r2_mean_raw"):
        assert abs(row[field] - gold[field]) <= 1e-6, (field, row[field], gold[field])

    q = context.ecephys["query"]
    n_trial_bins = int((qry.trial_ids >= 0).sum())
    for pred_key in ("cursor_vel_pred_x", "cursor_vel_pred_y"):
        series = q.auxiliary_channels[pred_key]
        assert series.data.shape == (n_trial_bins,)
        assert series.data.dtype == np.float32
    notes = json.loads(context.notes)["lince_baseline_decode"]
    assert notes["sessions"][0]["session_id"] == EASY_SESSION


# ---------------------------------------------------------------- read-only QA

@needs_dataset
def test_decode_never_mutates_weights(tmp_path):
    from linxi_linceplugin import decode_engine

    weights = DATASET_ROOT / "challenge_code" / "Participant" / "WF"

    def manifest():
        return {
            str(p.relative_to(weights)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(weights.rglob("model.pkl"))
        }

    before = manifest()
    mtimes = {p: p.stat().st_mtime_ns for p in weights.rglob("model.pkl")}
    op = _sweep_operator(model="wf", task="MA_CO", level="easy",
                         session_key="MA-CO-20231225-01")
    op(_make_context())
    assert manifest() == before
    assert {p: p.stat().st_mtime_ns for p in weights.rglob("model.pkl")} == mtimes
    assert decode_engine.default_weights_dir(str(DATA_ROOT), "wf") == str(
        DATASET_ROOT / "challenge_code" / "Participant" / "WF"
    )
