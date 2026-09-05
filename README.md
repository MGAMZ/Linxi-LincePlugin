# Linxi-LincePlugin

临策算法竞赛通用算子插件仓库。面向"运动跨天解码"赛道提供四个 Linxi 算子——赛题 NWB 载入、赛道 baseline 解码推理、跨天漂移指标、LinshuFile 导出接线——一条 pipeline YAML 即可从原始数据链到 `.ls` 产物。仓库同时保留模板示例算子 `ExamplePluginProcessor` 与 `example_linear_probe`，作为自定义算子开发的可运行样例。

## 安装前提

### 环境与网络要求

- Python >= 3.12（本 README 逐字复现基准：Linux x86_64、Python 3.13.15）。
- 网络可访问 PyPI，且可匿名读取 `git+https://gitee.com/lgl-tianqiong/TianSuo-Trodes.git`（Linxi 运行依赖 `Linxi-Trodes` 的公开直链）。
- 赛题材料就位：数据根（默认路径见各算子 `data_root` 参数，只读使用）与赛道 baseline 代码目录 `challenge_code`（含 `Platform/` 与 `Participant/{WF,GRU}` 预置权重）。

### Linxi 的获取（现状如实说明）

Linxi 未发布到 PyPI。其 `pyproject.toml` 声明依赖 `linshu-format @ git+https://gitee.com/lgl-tianqiong/dbci-format.git@BC/LinshuFile`，该直链仓库匿名访问需要账号凭据（pip 依赖解析阶段报 `could not read Username for 'https://gitee.com'` 而失败）。截至本文档撰写，Linxi 不存在公开可解析的直接安装路径；选手需经临枢团队分发渠道取得源码后，按下面的"源码中继"路线安装。

### 安装 Linxi 与 linshu-format（源码中继路线）

1. 取得两个源码目录：Linxi 源码树（克隆地址或离线源码包见竞赛公告或向临枢团队索取）；dBCI-format 源码树（临枢数据契约仓库，`BC/LinshuFile` 分支）。
2. 在目标 Python 环境内执行（两个变量指向第 1 步取得的目录）：

```bash
export LINSHU_SRC=/tmp/lince-src/dbci-format   # dBCI-format 源码目录（BC/LinshuFile 分支）
export LINXI_SRC=/tmp/lince-src/linxi          # Linxi 源码目录
python -m pip install "$LINSHU_SRC"
python -m pip install pyinstaller "numpy>=2.3.2" spikeinterface probeinterface nicegui pandas openpyxl matplotlib pynwb "hdmf==4.3.1" "neuroconv>=0.9.1" nwbinspector zstandard hdf5plugin numba deprecated ndx-events "Linxi-Trodes @ git+https://gitee.com/lgl-tianqiong/TianSuo-Trodes.git@BC/yank-name-to-linxi" mne remfile hdmf-zarr
python -m pip install --no-deps "$LINXI_SRC"
```

`--no-deps` 是必须的：Linxi 的依赖声明里含上述不可匿名解析的直链，其全部运行时依赖已由第二条命令装齐。

3. 校验：`python -c "import linxi"` 无报错；`python -m pip show linxi linshu-format` 版本应为 `0.12.0.dev0` 与 `0.7.0`。版本口径以源码提交为准：linxi @ 47f0a369、dBCI-format @ c7fb0156（0.7.0 系树，SPEC_VERSION 0.3.0）。上述三条 pip 命令在 Linux x86_64 + Python 3.13 环境逐条退出码 0。

### 安装本插件仓库

```bash
cd Linxi-LincePlugin
python -m pip install -e ".[dev]"
python -m pytest
```

`pip install -e ".[dev]"` 装插件本体并带出必选依赖 `scikit-learn` 与开发依赖 `pytest`。自检预期全绿；复现基准（本机，无 torch 环境）为 `46 passed, 1 skipped`，skip 项是 GRU 测试随 torch 缺失自动跳过，不算失败。

## 依赖说明

| 组 | 包 | 用途 |
|---|---|---|
| 必选（`dependencies`） | scikit-learn | WF baseline 解码（`BaselineDecodeInfer` 的 `model="wf"` 路径，纯 CPU） |
| 可选 extras `[gru]` | torch | GRU baseline 解码（`model="gru"`） |
| 可选 extras `[dev]` | pytest | 仓库自检 |

GRU 路径仅适用于带 GPU 与 CUDA 运行时的竞赛环境（conda 环境 `dbci`），`[gru]` extras 的裸 `torch` 不钉 CUDA 版本、面向 CPU 的默认 wheel 无法直接跑 GRU 推理；本 README 的复现覆盖为 CPU 路线，不含 GRU 数值。

## 算子与注册

Linxi 按 pipeline YAML 的 `linxi_plugin` 字段逐个 `importlib.import_module` 所列模块，导入即注册。本仓库包 `__init__` 只聚合模板示例算子，四个赛题算子必须按子模块路径列出：

```yaml
linxi_plugin:
  - linxi_linceplugin.load_lince_session
  - linxi_linceplugin.decode_baseline
  - linxi_linceplugin.drift_analysis
  - linxi_linceplugin.export_wiring
```

| stage | processor_name | 导入模块 | 一句话 |
|---|---|---|---|
| load | `LoadLinceSession` | `linxi_linceplugin.load_lince_session` | 单个赛题 session 的 NWB 投影进临析内部表示 |
| postprocess | `BaselineDecodeInfer` | `linxi_linceplugin.decode_baseline` | 预置权重 WF/GRU baseline 解码推理与官方口径评分 |
| analyze | `LinceDriftAnalysis` | `linxi_linceplugin.drift_analysis` | 跨天漂移指标（cos_raw、cos_centered、norm_ratio、gap-days 相关） |
| export | `LinceWriteLinshuFile` | `linxi_linceplugin.export_wiring` | 表示层接线上游 `WriteLinshuFile` 写出 `.ls` |

下表参数默认值即竞赛分发布局的绝对路径；本地布局不同时逐参数显式覆盖。

### LoadLinceSession（load）

赛题 NWB 无 ElectricalSeries / units / electrodes 表，上游 `LoadNWB` 对这类文件不可用，本算子自研 pynwb 只读载入。

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_path` | 无（必显式传） | session NWB 文件路径；不传即 `ValueError`。本算子不读也不写 `run_state.input_path`，同一 pipeline 的 query / support 两个 load 步骤互不污染 |
| `recording_key` | `"query"` | 写入 `context.ecephys` 的键；`"query"` 侧同时填 `context.recording` 与顶层 `context.trials` |
| `data_root` | `/mnt/f/mgam_datasets/Lince/运动跨天解码/challenge_data` | 仅用于派生 session 标识（NWB 父目录相对该根的路径；赛题文件 `session_id=None`） |
| `name` | 无 | 算子实例名 |

输入：单 session NWB（必需容器 `acquisition/binned_spikes`、`cursor_vel`、`cursor_pos`、`eval_mask`、`intervals/trials`，缺失或时间维不一致时报错并给出文件路径）。输出：`ecephys[recording_key]`——`electrophysiology` 为 uint8 (T,512) 计数（`unit="spike counts"`、真实绝对时间戳写入 `timestamps`，不伪造均匀网格）；`auxiliary_channels` 含 `cursor_vel_x/y`、`cursor_pos_x/y`、`eval_mask`（dtype 保真、各带自身时间戳）；`events` 为 trials 表（`start_time`/`stop_time`/`trial_id`，秒级绝对 epoch）。

### BaselineDecodeInfer（postprocess）

预置权重只读加载，官方协议 adapt + predict + 评分；官方模型类与平台 loader 运行时从 `baseline_code_path` 导入，权重文件每个 session 前后做 sha256 复核，任何改动即报错。

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | `"wf"` | `"wf"` 或 `"gru"`；`"gru"` 需 `[gru]` extras 与 GPU 环境 |
| `source` | `"context"` | `"context"` 消费链路上游载入槽；`"sweep"` 按官方协议从 `data_root` 自动发现并重跑 session |
| `baseline_code_path` | `/mnt/f/mgam_repos/Lince/运动跨天解码/challenge_code` | 赛道 Platform 与 Participant 代码根 |
| `data_root` | `/mnt/f/mgam_datasets/Lince/运动跨天解码/challenge_data` | sweep 发现根与预置权重推导锚 |
| `weights_dir` | `""`（空） | 空 = `<data_root 上级>/challenge_code/Participant/{WF,GRU}`；本机解析为 `/mnt/f/mgam_datasets/Lince/运动跨天解码/challenge_code/Participant/{WF,GRU}` |
| `task` | `""` | `MA_CO` / `MA_RT` / 空=全选；`source="context"` 时必须非空 |
| `level` | `""` | `easy` / `normal` / `hard` / 空=全选；`source="context"` 时必须非空 |
| `horizon` | `""` | normal / hard 的时程片筛选；空=全部 |
| `session_key` | `""` | session 目录名筛选；空=全部 |
| `strict_trial_counts` | `true` | 按级别校验 support/query trial 数（easy 100/25、normal 20/25、hard 0/25）并做 trial_id 零交集校验 |

context 模式输入要求：`ecephys["query"]` 的 `auxiliary_channels` 含 `cursor_vel_x/y`、`events` 表含三结构列、逐 bin 时间基可得；缺 `ecephys["support"]` 键按 hard 级空校准语义处理。输出：`ecephys["query"].auxiliary_channels` 追加 `cursor_vel_pred_x` / `cursor_vel_pred_y`（float32 TimeSeries，仅 trial 内 bin）；`context.notes["lince_baseline_decode"]`（JSON，逐 session `r2_x`/`r2_y`/`r2_mean_raw`/`session_score` 等；完整 sweep 另含 `aggregate.final_score`）；每个 session 一行机器可读日志 `[BaselineDecodeInfer] {json}`。

评分口径与官方一致：负 R² 原值保留于 `r2_mean_raw`，计分时按 `max(R², 0)` clip；`session_score = 0.95·r2_mean + 0.05·延迟分`，延迟分 `clip(1 - latency_per_bin_ms/20, 0, 1)`；任务分 = 0.25·easy + 0.45·normal + 0.30·hard，Final = 两任务均值。`session_score` 含硬件相关延迟分量，跨机器比对请用 `r2_x`/`r2_y`/`r2_mean_raw`。

### LinceDriftAnalysis（analyze）

指标口径 1:1 移植赛道数据探查脚本（`运动跨天解码/scripts/explore_data.py`）：cos_raw、\|norm_ratio-1\|、cos_centered、gap_days 与 cos 的 Pearson 相关。纯计算不落盘，产物由 pipeline 持久化；不读写 `run_state.input_path`，不向数据根写任何文件。

| 参数 | 默认 | 说明 |
|---|---|---|
| `source` | `"context"` | `"context"` 读 ecephys 槽 / `"sessions"` 直传会话清单 / `"sweep"` 按 `data_root` 目录发现 |
| `sessions` | 无 | `sessions` 模式必填：逐项 `task`/`level`/`session_key` 加 `x`（2-D 发放矩阵）或 `nwb_path` |
| `data_root` | `/mnt/f/mgam_datasets/Lince/运动跨天解码/challenge_data` | sweep 发现根 |
| `tasks` | 无 | sweep 任务筛选；默认 `MA_CO`、`MA_RT` 全选 |
| `levels` | 无 | sweep 级别筛选；默认 `train`/`easy`/`normal`/`hard` |
| `target_keys` | `["query"]` | context 模式被分析槽键 |
| `centroid_key` | 无 | 质心来源槽键（字符串或列表）；与 `centroid` 二选一，均缺报 `ValueError` |
| `centroid` | 无 | 显式质心向量（512 维） |
| `session_key` | 无 | 目标 session 目录名，参与 gap_days 计算 |
| `train_session_keys` | 无 | 训练 session 目录名列表；gap_days = 目标日期 − max(训练日期)。两者任一缺失则 gap_days=NaN 并告警 |
| `result_prefix` | `"drift"` | `channel_summary` 列名前缀 |

输出：`ecephys[k].channel_summary` 追加 `<prefix>_fr`、`<prefix>_centroid_fr`、`<prefix>_cos_raw`、`<prefix>_cos_centered`、`<prefix>_norm_ratio`、`<prefix>_gap_days` 六列（按通道坐标，合并不覆盖既有列）；`context.notes["lince_drift_analysis"]`；Python 直调时实例属性 `.result`。

### LinceWriteLinshuFile（export）

EXPORT 阶段薄适配：表示层整理后组合调用上游 `WriteLinshuFile` 落盘，不做自定义序列化。上游 writer 在 `context.recording` 缺失时只告警跳过写出（静默空产物），本接线将其连同空槽、空壳 trials 等情形转为 `LinceExportError` 显式失败，并在写后断言产物目录、Zarr 标记与文件数。

| 参数 | 默认 | 说明 |
|---|---|---|
| `output_path` | 无 | `.ls` 产物路径（Zarr 目录 store，非单文件）；不传则按上游惯例写入工作目录 |
| `ecephys_key` | `"query"` | 导出信号槽键；槽内容须由载入链预填，空占位即失败 |
| `chunk_frames` | `30000` | 透传上游 |
| `write_workers` | `8` | 透传上游 |
| `unit` | `"volts"` | 透传上游（赛题 `binned_spikes` 实为 spike 计数，单位类型缺口属数据契约层议题） |
| `overwrite` | `true` | 透传上游；覆盖已有产物时打 INFO `Removed existing .ls output`，属正常 |
| `skip_fields` | 无 | 透传上游 |
| `skip_raw_signals` | `false` | 透传上游；置 true 时空占位防线仍生效 |
| `session_scalar_cols` | 无 | trials 表按 trial 广播的 session 级标量列名声明，逐列常量校验 |
| `on_lossy` | `"warn"` | `"warn"` 或 `"fail"`：无法表达的字段挪位/丢弃报结构化 WARNING，或一律失败 |
| `require_eval_cols` | `true` | true 要求 trials 表含结构列以外的评测列。本仓库示例链的解码结果落位在 `auxiliary_channels`/`notes`/`channel_summary`（trials 恒为结构三列），须置 `false` 并把 `session_scalar_cols` 置空——示例 YAML 已如此配置；若把评测列合入 trials 表则保持默认，可防空壳导出 |
| `structural_cols` | `start_time`/`stop_time`/`trial_id` | 视为非评测的结构列集合 |

输入：query 记录（或 `require_eval_cols=false` 下的元数据链）；输出：`<output_path>` Zarr 目录 store。

## Pipeline 示例（端到端）

`examples/e2e_wf_MA-CO-20231227-01.yaml`：钉死 easy session `MA_CO/easy/MA-CO-20231227-01`（query = private heldout 25 trial，support = 同日 public heldin 100 trial），链路 load×2 → WF 解码 → 漂移指标 → `.ls` 导出；session ID、数据根、产物路径全部以字面值写死在 YAML 内。

在仓库根目录、装好本插件的环境执行：

```bash
python examples/run_e2e_wf.py --golden "/mnt/f/mgam_repos/Lince/运动跨天解码/output/csv/wf_challenge_results.json"
```

`--golden` 为官方 WF 基线结果 JSON（随竞赛仓库分发，按本地路径调整）；省略则只跑链路并打印 session 指标。预期退出码 0，末行：

```text
[e2e] PARITY PASS
```

且逐字段对账输出 `r2_x`/`r2_y`/`r2_mean_raw` 三项 `|Δ|=0.000e+00`（容差 1e-6，逐位复现基准为 numpy 2.5.2 默认线程配置；numpy/BLAS 构建不同时以容差判定为准）。产物：`/mnt/f/mgam_repos/Lince/运动跨天解码/output/linshu_e2e/MA-CO-20231227-01.ls`（由 YAML `output_path` 钉死）。

### 为什么可复现入口是驱动而非 `linxi process`

`linxi/__init__.py` 在包导入时无条件设置环境变量 `OPENBLAS_NUM_THREADS=1`（防分选算法线程竞态），而 OpenBLAS 的归约次序在库装载时即固定。控制台入口 `linxi process` 必然先导入 linxi、后导入 numpy，WF Ridge 点积落在单线程归约次序上，解码数值确定性偏离官方黄金约 1e-5 量级（本 session 实测 `|Δr2_mean_raw|` = 8.7e-6）；numpy 先装载时该开关失效。`examples/run_e2e_wf.py` 首行导入 numpy，与官方评测 harness 同配置，因此是逐位复现黄金的入口。

CLI 入口可用于链路连通性演示（数值即上述单线程口径，产物相同）：

```bash
linxi process -c examples/e2e_wf_MA-CO-20231227-01.yaml -o workdir --raise-errors
```

退出码 0。CLI 会先做一次验证 dry-run（导出落到临时目录，日志行标 `VAL`），随后正式运行（日志行标 `RUN`）才写入 YAML 钉死的 `output_path`；看到两条 `Successfully exported .ls` 属正常。

GRU 端到端链路在 dbci + GPU 环境运行（见"依赖说明"）。

## 输出说明

- `.ls` 是 LinshuFile：Zarr 目录型 store（目录，不是单文件）。当前依赖线（zarr 2.x）写出 v2 格式，标记文件为 `.zgroup`/`.zmetadata`/`.zattrs`；每次写盘出现一条"非规范格式"UserWarning 属正常提示，不是失败。
- 断言产物非空的正确口径：目录存在且含格式标记、`find <产物>.ls -type f | wc -l` > 2；`test -s` 对目录恒真，不可用于此目的。导出算子内部自带同样的写后断言，失败以 `LinceExportError` 非零退出。
- 示例产物 `MA-CO-20231227-01.ls` 为 216 文件、约 5.5 MB；顶层组 `ecephys/`（query 与 support 双记录）、`trials/`、`history/`。
- WF 示例链的结果落位：逐 bin 预测在 `ecephys["query"].auxiliary_channels["cursor_vel_pred_x"/"cursor_vel_pred_y"]`，session 得分在根节点 `notes["lince_baseline_decode"]`，漂移指标在 `ecephys["query"].channel_summary` 的 `drift_*` 六列与 `notes["lince_drift_analysis"]`，`trials` 表为结构三列。
- 回读用 `LinxiContext.from_zarr`（`history` 仅对该类可见）；每次回读打印约 20 行缺失可选字段 warning，属正常噪声；根 attrs 的 `python_environmenets` 是写出时自动快照的包清单，字段比对应排除。

## 自定义算子与更多开发说明

- 导入插件包/子模块时立即执行注册逻辑；不要在模块顶层写重型副作用代码；同名 processor 或 probe 注册时直接报错。
- 开发自定义算子的步骤（继承 `DefaultProcessor`、`register_as_linxi_processor(stage=...)`、probe 注册装饰器）见 `NEXTSTEPS.md`；YAML 加载入口为 `linxi process -c <yaml>` CLI 或 `PipelineDefinition.from_file`：

```python
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner

pipeline = PipelineDefinition.from_file("examples/e2e_wf_MA-CO-20231227-01.yaml")
runner = TaskRunner(pipeline)
```

## 目录结构

```text
Linxi-LincePlugin/
├── linxi_linceplugin/
│   ├── load_lince_session.py   # LOAD 赛题 NWB 载入
│   ├── decode_baseline.py      # POSTPROCESS baseline 解码推理
│   ├── drift_analysis.py       # ANALYZE 跨天漂移指标
│   ├── export_wiring.py        # EXPORT LinshuFile 导出接线
│   ├── processor.py / probe.py # 模板示例算子
│   └── __init__.py
├── examples/                   # e2e 示例 YAML、驱动与说明
├── tests/
├── .gitignore
├── NEXTSTEPS.md
├── pyproject.toml
└── README.md
```
