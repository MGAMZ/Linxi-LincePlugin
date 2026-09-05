"""漂移测试共享件：pynwb 直读 + 赛道目录发现 + explore_data.py 现场导入。

测试数据输入独立性（计划行 153）：输入一律经 pynwb 直读（与 explore_data.py 自身读取
口径等价），不 import 任务 7 载入算子；基线 = 现场 import 赛道脚本函数（只 import，
不跑 main()，零写盘）。
"""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

CHALLENGE_ROOT = Path("/mnt/f/mgam_repos/Lince/运动跨天解码")
DATA_ROOT = Path("/mnt/f/mgam_datasets/Lince/运动跨天解码/challenge_data")
EXPLORE_SCRIPT = CHALLENGE_ROOT / "scripts" / "explore_data.py"

# 报告 data_exploration.md 的 4 位舍入值（第二道目视核对）。
REPORT_GOLDEN_4DP = {
    "cos_raw_mean_hard": "0.8165",
    "norm_ratio_dev_hard": "0.0731",
    "cos_centered_mean_hard": "0.7120",
    "pearson_gap_cos": "-0.829",
}


@lru_cache(maxsize=1)
def load_explore_module():
    if str(CHALLENGE_ROOT) not in sys.path:
        sys.path.insert(0, str(CHALLENGE_ROOT))
    spec = importlib.util.spec_from_file_location("lince_explore_data", EXPLORE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lince_explore_data"] = module
    spec.loader.exec_module(module)
    return module


def load_neural(path: Path) -> np.ndarray:
    """pynwb 直读 binned_spikes（转置纠正同平台 loader data.py:158-203）。"""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        series = io.read().acquisition["binned_spikes"]
        timestamps = np.asarray(series.timestamps[:], dtype=np.float64)
        data = np.asarray(series.data[:])
    if data.ndim == 2 and data.shape[0] != len(timestamps) and data.shape[1] == len(timestamps):
        data = np.ascontiguousarray(data.T)
    assert data.ndim == 2 and data.shape[0] == len(timestamps), f"bad binned_spikes shape {data.shape}: {path}"
    return data


def one_nwb(directory: Path) -> Path:
    files = sorted(directory.glob("*.nwb"))
    assert len(files) == 1, f"expected exactly one NWB in {directory}, got {len(files)}"
    return files[0]


def heldin_sessions(task: str) -> list[tuple[str, Path]]:
    root = DATA_ROOT / "public" / task / "easy" / "heldin"
    return [(d.name, one_nwb(d)) for d in sorted(p for p in root.iterdir() if p.is_dir())]


def heldout_sessions(task: str, level: str) -> list[tuple[str, str | None, Path]]:
    if level == "easy":
        root = DATA_ROOT / "private" / task / "easy" / "heldout"
        return [(d.name, None, one_nwb(d)) for d in sorted(p for p in root.iterdir() if p.is_dir())]
    rows: list[tuple[str, str | None, Path]] = []
    horizon_root = DATA_ROOT / "private" / task / level / "heldout"
    for horizon in sorted(p for p in horizon_root.iterdir() if p.is_dir()):
        for day in sorted(p for p in horizon.iterdir() if p.is_dir()):
            rows.append((day.name, horizon.name, one_nwb(day)))
    return rows


def is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)
