"""漂移纯计算内核测试：与现场导入的 explore_data.py 具名函数逐式对账（输入 pynwb 直读）。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("linxi")

from _drift_helpers import (  # noqa: E402
    DATA_ROOT,
    heldin_sessions,
    heldout_sessions,
    is_nan as _is,
    load_explore_module as _explore,
    load_neural as _load_neural,
)

requires_data = pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"data root absent: {DATA_ROOT}")


@pytest.fixture(scope="module")
def drift_core():
    return importlib.import_module("linxi_linceplugin._drift_core")


def test_drift_core_functions_match_live_script(drift_core):
    explore_module = _explore()
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.normal(size=512)
        b = rng.normal(size=512)
        assert drift_core.cosine(a, b) == pytest.approx(explore_module.cosine(a, b), abs=0.0)
    # explore_data.py:98-99 零范数守卫 → NaN（非 0、非异常）
    z = np.zeros(8)
    assert _is(drift_core.cosine(z, np.ones(8))) and _is(explore_module.cosine(z, np.ones(8)))
    assert _is(drift_core.cosine(np.ones(8), z))

    xs = [1.0, 4.0, 9.0, 16.0]
    ys = [2.0, 3.0, 5.0, 7.0]
    assert drift_core.pearson(xs, ys) == pytest.approx(explore_module.pearson(xs, ys), abs=0.0)
    # explore_data.py:106-107 n<2 → NaN；:111-112 常数序列 den==0 → NaN
    assert _is(drift_core.pearson([1.0], [2.0])) and _is(explore_module.pearson([1.0], [2.0]))
    assert _is(drift_core.pearson([1.0, 1.0], [2.0, 3.0])) and _is(
        explore_module.pearson([1.0, 1.0], [2.0, 3.0])
    )

    # explore_data.py:87-92 目录名日期解析与缺失即报错
    assert drift_core.parse_session_date("MA-CO-20240117-01") == explore_module.parse_session_date(
        "MA-CO-20240117-01"
    )
    with pytest.raises(ValueError, match="8"):
        drift_core.parse_session_date("not-a-session")


def test_drift_session_fr_matches_script_mean_axis0(drift_core):
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    fr = drift_core.session_fr(x)
    assert fr.dtype == np.float64 and np.allclose(fr, x.astype(np.float64).mean(axis=0))
    with pytest.raises(ValueError, match="2-D"):
        drift_core.session_fr(np.zeros(4))


@requires_data
def test_drift_batch_full_chain_matches_script_formulas_direct_read(drift_core):
    """MA_CO：2 heldin(train) + 1 easy + 1 normal + 1 hard，全链路数值 vs 脚本具名函数手算。"""
    explore_module = _explore()
    task = "MA_CO"
    train_dirs = heldin_sessions(task)[:2]
    easy = heldout_sessions(task, "easy")[0]
    normal = heldout_sessions(task, "normal")[0]
    hard = heldout_sessions(task, "hard")[0]

    inputs = [
        drift_core.DriftSessionInput(task=task, level="train", session_key=key, x=_load_neural(path))
        for key, path in train_dirs
    ]
    for level, (key, horizon, path) in zip(("easy", "normal", "hard"), (easy, normal, hard)):
        inputs.append(
            drift_core.DriftSessionInput(task=task, level=level, horizon=horizon, session_key=key, x=_load_neural(path))
        )
    result = drift_core.compute_drift_metrics(inputs)

    # 脚本口径手算（explore_data.py:120/155-169）：质心=train fr 逐维均值
    train_frs = [_load_neural(p).mean(axis=0).astype(np.float64) for _, p in train_dirs]
    centroid = np.mean(np.vstack(train_frs), axis=0)
    train_last = max(explore_module.parse_session_date(k) for k, _ in train_dirs)

    # heldin 与 easy heldout 同日期目录名（同天两拆分），键必须含 level
    by_key = {(m.level, m.session_key): m for m in result.sessions}
    for (key, _), fr in zip(train_dirs, train_frs):
        m = by_key[("train", key)]
        assert m.cos_raw == pytest.approx(explore_module.cosine(fr, centroid), abs=1e-12)
        assert m.cos_centered == pytest.approx(
            explore_module.cosine(fr - fr.mean(), centroid - centroid.mean()), abs=1e-12
        )
        assert m.norm_ratio == pytest.approx(
            float(np.linalg.norm(fr)) / float(np.linalg.norm(centroid)), abs=1e-12
        )
        assert m.gap_days == (explore_module.parse_session_date(key) - train_last).days
    easy_fr = _load_neural(easy[2]).mean(axis=0).astype(np.float64)
    assert by_key[("easy", easy[0])].cos_raw == pytest.approx(explore_module.cosine(easy_fr, centroid), abs=1e-12)

    hard_m = by_key[("hard", hard[0])]
    assert result.cos_raw_mean_hard == pytest.approx(hard_m.cos_raw, abs=1e-12)
    assert result.norm_ratio_dev_hard == pytest.approx(abs(hard_m.norm_ratio - 1.0), abs=1e-12)
    heldout = [by_key[(lv, k)] for lv, (k, _, _) in zip(("easy", "normal", "hard"), (easy, normal, hard))]
    assert result.pearson_gap_cos == pytest.approx(
        explore_module.pearson([float(m.gap_days) for m in heldout], [m.cos_raw for m in heldout]),
        abs=1e-12,
    )
    assert len(result.centroids) == 1 and result.centroids[task].shape == (512,)


class _Capture:
    def __init__(self) -> None:
        self.records: list[str] = []

    def __call__(self, msg: str) -> None:
        self.records.append(msg)

    def texts(self) -> str:
        return "\n".join(self.records)


def test_drift_degenerate_inputs_no_crash_diagnostic_nan(drift_core, monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(drift_core.logger, "warning", cap)

    empty_channels = np.zeros((10, 0), dtype=np.uint8)
    empty_bins = np.zeros((0, 4), dtype=np.uint8)
    silent = np.zeros((10, 4), dtype=np.uint8)
    live_a = np.arange(1, 11, dtype=np.uint8)[:, None] * np.ones((1, 4), dtype=np.uint8)
    live_b = np.arange(2, 12, dtype=np.uint8)[:, None] * np.ones((1, 4), dtype=np.uint8)

    inputs = [
        # 无 train 会话的任务：受控空质心 → 该任务全部指标 NaN + WARNING（设计选择，见证据 §5）
        drift_core.DriftSessionInput(task="MA_RT", level="hard", session_key="MA-RT-20240201-01", x=live_b),
        drift_core.DriftSessionInput(task="MA_CO", level="train", session_key="MA-CO-20240101-01", x=live_a),
        drift_core.DriftSessionInput(task="MA_CO", level="easy", session_key="MA-CO-20240102-01", x=empty_channels),
        drift_core.DriftSessionInput(task="MA_CO", level="normal", session_key="MA-CO-20240103-01", x=empty_bins),
        # 单有效 heldout：pearson n<2 → NaN（explore_data.py:106-107）
        drift_core.DriftSessionInput(task="MA_CO", level="hard", session_key="MA-CO-20240104-01", x=live_b),
        # 全零质心任务：norm_ratio 的 c_norm 守卫（:166-167）+ 零范数余弦守卫（:98-99）
        drift_core.DriftSessionInput(task="MA_ZT", level="train", session_key="MA-ZT-20240101-01", x=silent),
        drift_core.DriftSessionInput(task="MA_ZT", level="easy", session_key="MA-ZT-20240102-01", x=live_a),
    ]
    result = drift_core.compute_drift_metrics(inputs)

    by_key = {m.session_key: m for m in result.sessions}
    assert _is(by_key["MA-CO-20240102-01"].cos_raw)  # 0 通道 → 模长 0 → cosine 守卫 NaN
    assert _is(by_key["MA-CO-20240103-01"].cos_raw)  # 0 bin → fr 全 NaN
    assert _is(by_key["MA-RT-20240201-01"].cos_raw)  # 无 train 质心
    assert _is(by_key["MA-ZT-20240102-01"].cos_raw)  # 零质心 → 余弦 NaN
    assert _is(by_key["MA-ZT-20240102-01"].norm_ratio)  # c_norm==0 守卫
    assert not _is(by_key["MA-CO-20240104-01"].cos_raw)  # 单 hard 会话自身仍有值
    assert _is(result.pearson_gap_cos)
    assert "MA_RT" in cap.texts()
