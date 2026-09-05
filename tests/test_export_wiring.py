"""任务 10 导出接线测试：LinceWriteLinshuFile 前置断言 / ragged 补齐 / 有损策略 / 回读对账 / 注册。

输入为合成 context（LinxiContext + NumpyRecording + ecephys["query"] 槽 + trials 表），
形态对齐 task-1 §6 键布局与 task-2 §3 结果形状（负 R²、horizon 字符串分级列、ragged 逐 bin 预测）。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("linxi")
pytest.importorskip("linshu_format")
pytest.importorskip("spikeinterface")

import spikeinterface.core as sc
from linxi.fabric.linxi_context import LinxiContext
from linxi.processor import PROCESS_STAGES
from linxi.processor.export import linshu_writer as upstream_writer
from linxi.processor.registry import get_processor_registration_name, get_registered_processors
from linshu_format.core import EcephysRecording, LinshuFile, ProcessingRecord, TimeIntervals

import linxi_linceplugin.export_wiring as export_wiring

LinceExportError = export_wiring.LinceExportError
LinceWriteLinshuFile = export_wiring.LinceWriteLinshuFile

_RAGGED_PRED = [
    np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0], [18.0, 19.0]]),
    np.array([[20.0, 21.0], [22.0, 23.0]]),
    np.array([[30.0, 31.0], [32.0, 33.0], [34.0, 35.0], [36.0, 37.0]]),
]
_PRED_LENGTHS = [3, 5, 2, 4]
_R2 = [0.84, -63.17, -0.0065, 2.5]
_HORIZON = ["week", "halfmonth", "month", "week"]


def _recording() -> sc.NumpyRecording:
    rng = np.random.default_rng(7)
    return sc.NumpyRecording(rng.standard_normal((200, 4)), sampling_frequency=50.0)


def _trials_table(extra: dict[str, Any] | None = None) -> xr.Dataset:
    obj_pred = np.empty(4, dtype=object)
    for i, elem in enumerate(_RAGGED_PRED):
        obj_pred[i] = elem
    horizon = np.empty(4, dtype=object)
    for i, h in enumerate(_HORIZON):
        horizon[i] = h
    table = xr.Dataset(
        {
            "start_time": ("event", np.arange(4, dtype=float) * 2.0),
            "stop_time": ("event", np.arange(4, dtype=float) * 2.0 + 0.5),
            "trial_id": ("event", np.arange(4, dtype=np.int64)),
            "r2_x": ("event", np.array(_R2)),
            "session_r2_mean": ("event", np.full(4, 0.42)),
            "horizon": ("event", horizon),
            "pred": ("event", obj_pred),
        }
    )
    if extra:
        table = table.assign(extra)
    return table


def _base_context(
    *,
    recording: bool = True,
    slot: str | None = "filled",
    table: xr.Dataset | None = None,
    key: str = "query",
) -> LinxiContext:
    ctx = LinxiContext(file_create_date=datetime.now(), session="S1")
    if recording:
        ctx.recording = _recording()
    if slot == "filled":
        assert ctx.recording is not None
        ctx.ecephys[key] = EcephysRecording.from_spikeinterface_recording(ctx.recording, name=key)
    elif slot == "placeholder":
        ctx.ecephys[key] = EcephysRecording()
    ctx.trials = TimeIntervals(name="trials", table=table if table is not None else _trials_table())
    return ctx


def _adapter(tmp_path: Path, **overrides: Any) -> LinceWriteLinshuFile:
    params: dict[str, Any] = {
        "output_path": str(tmp_path / "sub-S1.ls"),
        "ecephys_key": "query",
        "session_scalar_cols": ["session_r2_mean"],
    }
    params.update(overrides)
    return LinceWriteLinshuFile(**params)


def _expected_padded() -> np.ndarray:
    padded = np.full((4, 5, 2), np.nan)
    for i, elem in enumerate(_RAGGED_PRED):
        padded[i, : elem.shape[0], :] = elem
    return padded


def test_export_operator_registered_under_export_stage() -> None:
    names = [
        get_processor_registration_name(cls)
        for cls in get_registered_processors(PROCESS_STAGES.EXPORT)
    ]
    assert "LinceWriteLinshuFile" in names


def test_export_happy_writes_store_with_zarr_markers(tmp_path: Path) -> None:
    ctx = _base_context()
    out = tmp_path / "sub-S1.ls"
    result = _adapter(tmp_path)(ctx)

    assert result.run_state.output_path == str(out)
    assert out.is_dir()
    assert (out / "zarr.json").exists() or (out / ".zgroup").exists() or (out / ".zmetadata").exists()
    assert sum(1 for p in out.rglob("*") if p.is_file()) > 2


def test_export_readback_matches_written_values(tmp_path: Path) -> None:
    ctx = _base_context()
    # 写盘时刻只序列化此前步骤的 history；本算子与嵌套 writer 的记录都在物理写之后才追加
    ctx.history.append(ProcessingRecord(step_name="LoadLinceSession", timestamp=0.0, duration=0.0, data_modified=["trials"], metadata_modified=[]))
    out = tmp_path / "sub-S1.ls"
    _adapter(tmp_path)(ctx)

    back = LinxiContext.from_zarr(str(out))
    table = back.trials.table
    np.testing.assert_allclose(np.asarray(table["r2_x"].values), _R2)
    assert [str(v) for v in np.asarray(table["horizon"].values)] == _HORIZON
    padded = np.asarray(table["pred"].values, dtype=float)
    np.testing.assert_allclose(padded, _expected_padded(), equal_nan=True)
    assert tuple(table["pred"].dims) == ("event", "pred_bin", "pred_coord")
    np.testing.assert_array_equal(np.asarray(table["pred_valid_len"].values), np.array(_PRED_LENGTHS, dtype=np.int64))
    assert [r.step_name for r in back.history] == ["LoadLinceSession"]
    assert back.session == "S1"

    via_contract = LinshuFile.from_zarr(str(out))
    assert sorted(via_contract.trials.table.data_vars) == sorted(table.data_vars)
    assert not hasattr(via_contract, "history")  # G6: history 仅 LinxiContext 回读类可见
    # python_environmenets 为环境快照（348 包），跨 run/跨读法比对必须排除；此处比对剩余根级标量
    exclude = {"python_environmenets", "file_create_date", "history"}
    scalars_back = {k: v for k, v in back.model_dump(exclude=exclude).items() if not isinstance(v, dict | list | np.ndarray)}
    scalars_contract = {k: v for k, v in via_contract.model_dump(exclude=exclude).items() if not isinstance(v, dict | list | np.ndarray)}
    assert scalars_back == scalars_contract


def test_export_raises_without_recording_and_no_dir(tmp_path: Path) -> None:
    ctx = _base_context(recording=False, slot=None)
    out = tmp_path / "sub-S1.ls"
    with pytest.raises(LinceExportError, match="recording"):
        _adapter(tmp_path)(ctx)
    assert not out.exists()


def test_export_raises_on_missing_signal_slot(tmp_path: Path) -> None:
    ctx = _base_context(slot=None)
    with pytest.raises(LinceExportError, match="query"):
        _adapter(tmp_path)(ctx)


def test_export_raises_on_empty_placeholder_slot(tmp_path: Path) -> None:
    ctx = _base_context(slot="placeholder")
    with pytest.raises(LinceExportError, match="占位"):
        _adapter(tmp_path)(ctx)


def test_export_skip_raw_signals_still_enforces_slot_guard(tmp_path: Path) -> None:
    ctx = _base_context(slot="placeholder")
    with pytest.raises(LinceExportError, match="占位"):
        _adapter(tmp_path, skip_raw_signals=True)(ctx)


def test_export_raises_when_trials_missing_and_when_only_structural(tmp_path: Path) -> None:
    ctx_no_trials = _base_context()
    ctx_no_trials.trials = None
    with pytest.raises(LinceExportError, match="trials"):
        _adapter(tmp_path)(ctx_no_trials)

    structural_only = _trials_table()[["start_time", "stop_time", "trial_id"]]
    with pytest.raises(LinceExportError, match="结构列"):
        _adapter(tmp_path)(_base_context(table=structural_only))

    out = tmp_path / "sub-S1.ls"
    _adapter(tmp_path, require_eval_cols=False, session_scalar_cols=[])(_base_context(table=structural_only))
    assert out.is_dir()


def test_export_object_str_columns_pass_through_and_ragged_gets_convention(tmp_path: Path) -> None:
    out = tmp_path / "sub-S1.ls"
    _adapter(tmp_path)(_base_context())
    table = LinxiContext.from_zarr(str(out)).trials.table
    assert "pred" in table.data_vars and "pred_valid_len" in table.data_vars
    assert "horizon" in table.data_vars and table["horizon"].dtype == object


def test_export_session_scalar_relocation_warns_structurally(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _adapter(tmp_path)(_base_context())
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "LinceWriteLinshuFile" in r.getMessage()]
    assert any("session_r2_mean" in m and "G1" in m for m in msgs)
    assert any("relocated" in m for m in msgs)


def test_export_nonconstant_session_scalar_raises(tmp_path: Path) -> None:
    table = _trials_table({"session_r2_mean": ("event", np.array([0.42, 0.42, 0.11, 0.42]))})
    with pytest.raises(LinceExportError, match="session_r2_mean"):
        _adapter(tmp_path)(_base_context(table=table))


def test_export_fail_mode_refuses_relocation(tmp_path: Path) -> None:
    with pytest.raises(LinceExportError, match="session_r2_mean"):
        _adapter(tmp_path, on_lossy="fail")(_base_context())


def test_export_unrepresentable_column_dropped_with_warning_or_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    dicts = np.empty(4, dtype=object)
    for i in range(4):
        dicts[i] = {"a": i}
    table = _trials_table({"meta_dicts": ("event", dicts)})
    adapter = _adapter(tmp_path)
    assert table["meta_dicts"].dtype == object

    with caplog.at_level(logging.WARNING):
        _adapter(tmp_path)(_base_context(table=table))
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "LinceWriteLinshuFile" in r.getMessage()]
    assert any("meta_dicts" in m and "G2" in m for m in msgs)
    assert "meta_dicts" not in LinxiContext.from_zarr(str(tmp_path / "sub-S1.ls")).trials.table.data_vars
    del adapter

    with pytest.raises(LinceExportError, match="meta_dicts"):
        _adapter(tmp_path, on_lossy="fail")(_base_context(table=table))


def test_export_product_assert_when_upstream_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upstream_writer.WriteLinshuFile, "materialize", lambda self, context: False)
    ctx = _base_context()
    with pytest.raises(LinceExportError, match="未写出"):
        _adapter(tmp_path)(ctx)
