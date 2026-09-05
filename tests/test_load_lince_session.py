"""LoadLinceSession 单测：内部布局 / 双键约定 / run_state 防污染 / 平台口径对账 / 数据根只读。"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("linxi")
pytest.importorskip("pynwb")
pytest.importorskip("spikeinterface")

import linxi_linceplugin.load_lince_session as lls
from linxi.fabric.linxi_context import LinxiContext
from linxi.processor import PROCESS_STAGES
from linxi.processor.registry import bootstrap_builtin_processors, get_registered_processors

TRACK_REPO = Path("/mnt/f/mgam_repos/Lince/运动跨天解码")
DATA_ROOT = Path(lls.DEFAULT_DATA_ROOT)
CHALLENGE_ROOT = DATA_ROOT.parent
SESSION_KEY = "MA-CO-20231227-01"
GOLDEN_NWB = DATA_ROOT / "public" / "MA_CO" / "easy" / "heldin" / SESSION_KEY / f"{SESSION_KEY}.nwb"
PARTNER_NWB = DATA_ROOT / "private" / "MA_CO" / "easy" / "heldout" / SESSION_KEY / f"{SESSION_KEY}.nwb"

requires_challenge_data = pytest.mark.skipif(
    not (GOLDEN_NWB.is_file() and PARTNER_NWB.is_file() and TRACK_REPO.is_dir()),
    reason=f"challenge data or platform copy absent below {DATA_ROOT}",
)


def _new_context() -> LinxiContext:
    return LinxiContext(file_create_date=datetime.now(timezone.utc), python_environmenets={})


def _load(context: LinxiContext, path: Path, recording_key: str) -> LinxiContext:
    return lls.LoadLinceSession(input_path=str(path), recording_key=recording_key)(context)


def _manifest() -> str:
    result = subprocess.run(
        ["find", str(CHALLENGE_ROOT), "-printf", "%p|%T@|%s\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture(scope="module", autouse=True)
def _challenge_root_stays_untouched():
    if not CHALLENGE_ROOT.is_dir():
        yield
        return
    before = _manifest()
    yield
    after = _manifest()
    if after != before:
        diffs = [line for line, other in zip(before.splitlines(), after.splitlines()) if line != other]
        raise AssertionError(f"challenge data root was modified during load tests, first diffs: {diffs[:5]}")


def test_registration_name_unique_in_load_stage():
    bootstrap_builtin_processors()
    names = [cls.__name__ for cls in get_registered_processors(PROCESS_STAGES.LOAD)]
    assert names.count("LoadLinceSession") == 1
    assert "LoadNWB" in names


def test_estimate_bin_rate_ignores_trial_gaps():
    base = 1_703_635_200.0
    ts = np.concatenate([base + np.arange(50) * 0.02, base + 50 * 0.02 + 1.695 + np.arange(50) * 0.02])
    assert lls.estimate_bin_rate(ts) == pytest.approx(50.0, abs=1e-6)


def test_projection_semantics_outside_trials_is_minus_one():
    import pandas as pd

    ts = np.array([0.0, 1.0, 2.0, 3.0, 9.0])
    trials = pd.DataFrame({"start_time": [0.5, 1.5], "stop_time": [1.0, 2.0], "trial_id": np.array([7, 8], dtype=np.int64)})
    ids = lls.project_bins_to_trials(ts, trials)
    np.testing.assert_array_equal(ids, np.array([-1, 7, 8, -1, -1], dtype=np.int64))
    assert ids.dtype == np.int64


@requires_challenge_data
def test_load_golden_session_query_layout():
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(GOLDEN_NWB), "r") as io:
        nwbfile = io.read()
        raw_x = np.asarray(nwbfile.acquisition["binned_spikes"].data[:])
        raw_ts = np.asarray(nwbfile.acquisition["binned_spikes"].timestamps[:], dtype=np.float64)
        raw_vel_x = np.asarray(nwbfile.acquisition["cursor_vel"].time_series["x"].data[:])
        raw_mask = np.asarray(nwbfile.acquisition["eval_mask"].data[:])
        n_trials = len(nwbfile.trials.to_dataframe())

    ctx = _load(_new_context(), GOLDEN_NWB, "query")
    assert set(ctx.ecephys) == {"query"}
    ecephys = ctx.ecephys["query"]

    series = ecephys.electrophysiology
    np.testing.assert_array_equal(np.asarray(series.data), raw_x)
    assert series.data.dtype == np.uint8
    assert series.data.shape == (len(raw_ts), 512)
    np.testing.assert_array_equal(series.timestamps, raw_ts)
    assert series.starting_time is None
    assert series.rate == pytest.approx(50.0, abs=1e-6)
    assert series.unit == "spike counts"
    assert ecephys.channel_count == 512
    assert ecephys.sampling_frequency == pytest.approx(50.0, abs=1e-6)

    aux = ecephys.auxiliary_channels
    assert set(aux) == {"cursor_vel_x", "cursor_vel_y", "cursor_pos_x", "cursor_pos_y", "eval_mask"}
    np.testing.assert_array_equal(np.asarray(aux["cursor_vel_x"].data), raw_vel_x)
    assert aux["cursor_vel_x"].data.dtype == np.float32
    assert aux["cursor_vel_x"].unit == "AU/s"
    assert aux["cursor_vel_y"].data.dtype == np.float32
    assert aux["cursor_pos_x"].unit == "AU"
    assert aux["eval_mask"].data.dtype == bool
    np.testing.assert_array_equal(np.asarray(aux["eval_mask"].data), raw_mask)
    for key in ("cursor_vel_x", "cursor_vel_y", "cursor_pos_x", "cursor_pos_y", "eval_mask"):
        np.testing.assert_array_equal(aux[key].timestamps, raw_ts)

    assert ecephys.events is ctx.trials
    table = ctx.trials.table
    for column in ("start_time", "stop_time", "trial_id"):
        assert column in table.data_vars
    assert table["trial_id"].dtype == np.int64
    assert table["start_time"].dtype == np.float64
    assert table.sizes["event"] == n_trials
    assert table["start_time"].values.min() > 1e9, "trial 起止必须保留绝对 epoch 时间戳"

    assert ctx.recording.get_num_samples() == len(raw_ts)
    assert ctx.recording.get_num_channels() == 512
    assert ctx.recording.get_dtype() == np.uint8
    assert ctx.session == f"public/MA_CO/easy/heldin/{SESSION_KEY}"
    assert ctx.run_state.input_path is None


@requires_challenge_data
def test_query_and_support_two_loads_keep_layout_contract():
    ctx = _load(_new_context(), PARTNER_NWB, "query")
    query_samples = ctx.recording.get_num_samples()
    query_trials = ctx.trials.table.sizes["event"]
    ctx = _load(ctx, GOLDEN_NWB, "support")

    assert set(ctx.ecephys) == {"query", "support"}
    support = ctx.ecephys["support"]
    assert support.electrophysiology.data.dtype == np.uint8
    assert support.electrophysiology.data.shape == (len(support.electrophysiology.timestamps), 512)
    assert support.events.table.sizes["event"] == 100
    assert ctx.trials.table.sizes["event"] == query_trials
    assert ctx.recording.get_num_samples() == query_samples, "support 载入不得覆盖 query 单槽"

    query_df = ctx.ecephys["query"].events.table.to_dataframe()
    query_ids = set(lls.project_bins_to_trials(np.asarray(ctx.ecephys["query"].electrophysiology.timestamps), query_df).tolist()) - {-1}
    support_ids = set(lls.project_bins_to_trials(np.asarray(support.electrophysiology.timestamps), support.events.table.to_dataframe()).tolist()) - {-1}
    assert query_ids and support_ids
    assert not (query_ids & support_ids), "平台 eval.py:87-93 契约：query/support trial_id 零交集"


@requires_challenge_data
def test_explicit_input_path_never_inherits_run_state():
    arrays = lls.read_lince_nwb(GOLDEN_NWB)
    ctx = _new_context()
    ctx.run_state.input_path = str(PARTNER_NWB)
    loaded = _load(ctx, GOLDEN_NWB, "query")

    assert loaded.session == f"public/MA_CO/easy/heldin/{SESSION_KEY}"
    np.testing.assert_array_equal(np.asarray(loaded.ecephys["query"].electrophysiology.timestamps), arrays.timestamps)
    assert ctx.run_state.input_path == str(PARTNER_NWB), "算子不得回写 run_state.input_path"


@requires_challenge_data
def test_missing_path_error_contains_the_path():
    bogus = "/tmp/opencode/lince-task7-does-not-exist/none.nwb"
    with pytest.raises(FileNotFoundError, match=re.escape(bogus)):
        _load(_new_context(), Path(bogus), "query")


def test_none_input_path_fails_fast():
    with pytest.raises(ValueError, match="input_path"):
        lls.LoadLinceSession()(_new_context())


@requires_challenge_data
def test_trial_projection_parity_with_platform_loader():
    sys.dont_write_bytecode = True
    if str(TRACK_REPO) not in sys.path:
        sys.path.insert(0, str(TRACK_REPO))
    from challenge_code.Platform.Platform_Implementation import data as platform_data

    golden = platform_data.load_nwb_session(GOLDEN_NWB)
    ctx = _load(_new_context(), GOLDEN_NWB, "query")
    ecephys = ctx.ecephys["query"]
    timestamps = np.asarray(ecephys.electrophysiology.timestamps)

    np.testing.assert_array_equal(timestamps, golden.timestamps)
    np.testing.assert_array_equal(np.asarray(ecephys.electrophysiology.data, dtype=np.float32), golden.neural)
    velocity = np.column_stack(
        [
            np.asarray(ecephys.auxiliary_channels["cursor_vel_x"].data),
            np.asarray(ecephys.auxiliary_channels["cursor_vel_y"].data),
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(velocity, golden.velocity)

    trials_df = ecephys.events.table.to_dataframe()
    np.testing.assert_array_equal(lls.project_bins_to_trials(timestamps, trials_df), golden.trial_ids)
    assert platform_data.count_trials(golden.trial_ids) == int(trials_df["trial_id"].nunique()) == 100


@requires_challenge_data
def test_validate_data_parity_fields():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "challenge_code.Platform.Platform_Implementation.validate_data", "--data-root", str(DATA_ROOT)],
        cwd=str(TRACK_REPO),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    target = next((line for line in result.stdout.splitlines() if line.startswith(f"OK MA_CO/easy/{SESSION_KEY}:")), None)
    assert target is not None, f"validate_data stdout 未含 session {SESSION_KEY}"
    support_trials, query_trials, channels = (int(v) for v in re.fullmatch(rf"OK MA_CO/easy/{SESSION_KEY}: support=(\d+), query=(\d+), channels=(\d+)", target).groups())

    ctx = _load(_new_context(), PARTNER_NWB, "query")
    ctx = _load(ctx, GOLDEN_NWB, "support")
    my_support = ctx.ecephys["support"].events.table.sizes["event"]
    my_query = ctx.ecephys["query"].events.table.sizes["event"]
    my_channels = int(ctx.ecephys["query"].electrophysiology.data.shape[1])
    assert (my_support, my_query, my_channels) == (support_trials, query_trials, channels)
