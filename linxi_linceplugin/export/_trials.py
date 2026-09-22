"""trials 表的 LinshuFile 表达性归一（各赛道导出接线共用）：object 列分类、补齐与有损清单产出。"""
from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["LinceExportError", "normalize_trials"]

_NUMERIC_KINDS = frozenset("biuf")


class LinceExportError(RuntimeError):
    """导出接线的前置条件或产物断言失败。"""


def _classify_object_column(arr: np.ndarray) -> str:
    """把 object dtype 列分为 ``str``（原样保留）/ ``ragged``（补齐）/ ``keep`` / ``opaque``（不可表达）。"""
    if arr.size == 0:
        return "keep"
    items = list(arr)
    if all(isinstance(x, str) for x in items):
        return "str"
    if all(isinstance(x, np.ndarray) for x in items):
        ndims = {x.ndim for x in items}
        if ndims == {1} or ndims == {2}:
            if ndims == {1} or len({x.shape[1] for x in items}) == 1:
                if all(x.dtype.kind in _NUMERIC_KINDS for x in items):
                    return "ragged"
    return "opaque"


def _pad_ragged(name: str, col: np.ndarray, primary: str) -> tuple[xr.DataArray, xr.DataArray]:
    """object 列内等宽 1/2 维数值数组补齐为 (primary, len[, width]) 规则数组，并产出有效长度列。"""
    items = [np.asarray(x, dtype=np.float64) for x in col]
    max_len = max(x.shape[0] for x in items)
    if items[0].ndim == 1:
        padded = np.full((len(items), max_len), np.nan)
        dims = (primary, f"{name}_bin")
        for i, x in enumerate(items):
            padded[i, : x.shape[0]] = x
    else:
        padded = np.full((len(items), max_len, items[0].shape[1]), np.nan)
        dims = (primary, f"{name}_bin", f"{name}_coord")
        for i, x in enumerate(items):
            padded[i, : x.shape[0], :] = x
    lengths = np.array([x.shape[0] for x in items], dtype=np.int64)
    pad_da = xr.DataArray(padded, dims=dims, name=name)
    vlen_da = xr.DataArray(lengths, dims=(primary,), name=f"{name}_valid_len")
    return pad_da, vlen_da


def normalize_trials(
    table: xr.Dataset,
    *,
    structural: set[str],
    session_cols: list[str],
    require_eval: bool,
) -> tuple[dict[str, xr.DataArray], bool, list[str], list[str]]:
    """把 trials 表规整为可序列化变量集：维度与列校验、object 列补齐、session 级标量列核验。

    返回 (新变量集, 是否发生改动, 丢弃列清单, 挪位保留列清单)；校验失败抛 `LinceExportError`，有损策略由调用方执行。
    """
    start = table["start_time"]
    if start.ndim != 1:
        raise LinceExportError(f"trials.table.start_time 必须为一维，实际 dims={start.dims}")
    primary = start.dims[0]
    n = int(start.sizes[primary])

    if require_eval and not set(table.data_vars) - structural:
        raise LinceExportError("trials 表仅含结构列，未合入任何解码评测列（require_eval_cols=False 可放开）")

    for col in session_cols:
        if col not in table.data_vars:
            raise LinceExportError(f"session_scalar_cols 引用不存在的列 {col!r}（现有列={sorted(table.data_vars)}）")
        vals = np.asarray(table[col].values)
        if vals.ndim != 1 or vals.size != n:
            raise LinceExportError(f"session 级标量列 {col!r} 必须是长度 {n} 的一维广播列，实际 shape={vals.shape}")
        if not bool((vals == vals[0]).all()):
            raise LinceExportError(f"列 {col!r} 声明为 session 级标量但逐 trial 取值不一致，广播假设被破坏")

    new_vars: dict[str, xr.DataArray] = {}
    dropped: list[str] = []
    changed = False
    for name, var in table.data_vars.items():
        if primary in var.dims and var.sizes[primary] != n:
            raise LinceExportError(f"列 {name!r} 在主维度 {primary!r} 上长度 {var.sizes[primary]} != trial 数 {n}")
        if var.dtype != object:
            new_vars[name] = var
            continue
        kind = _classify_object_column(np.asarray(var.values))
        if kind in ("str", "keep"):
            new_vars[name] = var
        elif kind == "ragged":
            pad_da, vlen_da = _pad_ragged(name, np.asarray(var.values), primary)
            new_vars[name] = pad_da
            new_vars[vlen_da.name] = vlen_da
            changed = True
        else:
            dropped.append(name)
            changed = True

    relocated = [f"{col}={np.asarray(table[col].values)[:1].tolist()[0]!r}" for col in session_cols]
    return new_vars, changed, dropped, relocated


def lossy_policy_messages(dropped: list[str], relocated: list[str]) -> list[str]:
    """有损操作（丢弃 / 挪位）的结构化告警文案。"""
    messages: list[str] = []
    if dropped:
        messages.append(
            "dropped 无法用 LinshuFile 表达的字段"
            f"（对象列元素非字符串/非一致形状数值数组）: {sorted(dropped)}"
        )
    if relocated:
        messages.append(f"relocated session 级标量以 trials 广播列保留（LinshuFile 根级无指标容器）: {', '.join(relocated)}")
    return messages
