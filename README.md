# Linxi-LincePlugin

临策算法竞赛通用算子插件仓库。面向"运动跨天解码"赛道提供四个 Linxi 算子：赛题 NWB 载入、赛道 baseline 解码推理、跨天漂移指标、LinshuFile 导出。
一条 pipeline YAML 即可对原始数据进行处理并输出 `.ls` 格式的集成输出。

## 安装前提

### 环境与网络要求

- Python >= 3.12
- 网络可访问 PyPI，且可读取 `git+https://gitee.com/lgl-tianqiong/TianSuo-Trodes.git`
- 数据集与赛道 baseline 代码目录 `challenge_code`

### Linxi 的获取

Linxi 未发布到 PyPI。竞赛正式上线时，将提供公开可用的PyPI版本，通过 `pip install linxi` 安装。

### 安装 Linxi 与 linshu-format

1. `pip install linxi linshu-format` (待竞赛开赛时，这两个仓库已经上传至PyPI)
2. 校验：`python -c "import linxi"` 无报错

### 安装本插件仓库

```bash
cd Linxi-LincePlugin
python -m pip install -e ".[dev]"
python -m pytest
```

需要真实赛题数据参与的测试用例经环境变量 `LINCE_CHALLENGE_DATA`（`challenge_data` 根目录）与 `LINCE_TRACK_REPO`（赛道代码仓目录）启用；未设置时这些用例自动跳过。

## 依赖说明

| 组 | 包 | 用途 |
|---|---|---|
| 必选 | scikit-learn | WF baseline 解码（`BaselineDecodeInfer` 的 `model="wf"` 路径，纯 CPU） |
| 可选 extras `[gru]` | torch | GRU baseline 解码（`model="gru"`） |
| 可选 extras `[dev]` | pytest | 仓库自检 |

## 算子与注册

在配置文件中增加以下字段：

```yaml
linxi_plugin:
  - linxi_linceplugin.load_lince_session
  - linxi_linceplugin.decode_baseline
  - linxi_linceplugin.drift_analysis
  - linxi_linceplugin.export_wiring
```

| stage | processor_name | 导入模块 | 描述 |
|---|---|---|---|
| load | `LoadLinceSession` | `linxi_linceplugin.load_lince_session` | 单个赛题 session 的 NWB 投影进临析内部表示 |
| postprocess | `BaselineDecodeInfer` | `linxi_linceplugin.decode_baseline` | 预置权重 WF/GRU baseline 解码推理与官方口径评分 |
| analyze | `LinceDriftAnalysis` | `linxi_linceplugin.drift_analysis` | 跨天漂移指标（cos_raw、cos_centered、norm_ratio、gap-days 相关） |
| export | `LinceWriteLinshuFile` | `linxi_linceplugin.export_wiring` | 表示层接线上游 `WriteLinshuFile` 写出 `.ls` |

### LoadLinceSession

赛题 NWB 无 ElectricalSeries / units / electrodes 表，上游 `LoadNWB` 对这类文件不可用，本算子自定义 pynwb 只读载入。

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_path` | `None` | session NWB 文件路径 |
| `recording_key` | "query" | 区分数据为评测集（query）还是校准集（support）。 |
| `data_root` | 必填 | 数据根目录路径 |
| `name` | `None` | 算子实例名 |

### BaselineDecodeInfer

以官方预置权重执行 baseline 解码推理与评分。

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | "wf" | 解码模型：`"wf"` / `"gru"` |
| `source` | "context" | 数据来源：`"context"` 使用流水线中已载入的数据，`"sweep"` 按 `data_root` 扫描发现 |
| `baseline_code_path` | 必填 | 赛道 baseline 代码目录（含 `Platform/` 与 `Participant/`） |
| `data_root` | 必填 | `challenge_data` 根目录 |
| `weights_dir` | `""` | 预置权重目录 |
| `task` | `""` | 任务类型：`MA_CO` / `MA_RT`；`"context"` 模式必填 |
| `level` | `""` | 评测难度：`easy` / `normal` / `hard`；`"context"` 模式必填 |
| `horizon` | `""` | normal / hard 的时程片 |
| `session_key` | `""` | session 目录名 |
| `strict_trial_counts` | `true` | 是否按赛题规定校验 trial 数 |
| `name` | `None` | 算子实例名 |

### LinceDriftAnalysis

计算跨天漂移指标，口径与赛道数据探查脚本一致。

| 参数 | 默认 | 说明 |
|---|---|---|
| `source` | "context" | 数据来源：`"context"` 流水线已载入记录，`"sessions"` 显式会话清单，`"sweep"` 按 `data_root` 扫描发现 |
| `sessions` | `None` | 会话清单（`"sessions"` 模式必填），逐项含 `task`/`level`/`session_key` 与 (T,512) 计数矩阵或 NWB 路径 |
| `data_root` | `None` | `"sweep"` 模式必填：`challenge_data` 根目录 |
| `tasks` / `levels` | `None` | sweep 的任务 / 级别范围 |
| `target_keys` | `["query"]` | 被分析的记录键 |
| `centroid_key` / `centroid` | `None` | 质心来源（记录键或 512 维向量），至少提供其一 |
| `session_key` / `train_session_keys` | `None` | 目标 session 与训练 session 名（gap_days 用） |
| `result_prefix` | "drift" | 输出列名前缀 |
| `name` | `None` | 算子实例名 |

### LinceWriteLinshuFile

将流水线中的记录与结果导出为 LinshuFile `.ls` 产物。

| 参数 | 默认 | 说明 |
|---|---|---|
| `output_path` | `None` | `.ls` 产物路径（Zarr 目录，非单文件） |
| `ecephys_key` | "query" | 导出的记录键 |
| `unit` | "volts" | 信号单位声明 |
| `session_scalar_cols` | `None` | trials 表中按 trial 广播的 session 级标量列名 |
| `on_lossy` | "warn" | 字段无法表达时的策略：`"warn"` / `"fail"` |
| `require_eval_cols` | `true` | 是否要求 trials 表含评测列 |
| `structural_cols` | `None` | 非评测结构列集合（默认 `start_time`/`stop_time`/`trial_id`） |
| `chunk_frames` / `write_workers` / `overwrite` / `skip_fields` / `skip_raw_signals` | `30000` / `8` / `true` / `None` / `false` | 透传上游 `WriteLinshuFile` 的同名参数 |

## 端到端示例

`examples/e2e_wf_MA-CO-20231227-01.yaml`：easy 任务 `MA-CO-20231227-01` 的 WF 全链示例，链路为载入 → 解码推理 → 漂移指标 → 导出 `.ls`。示例内的数据路径是占位符，用于自有流水线时替换为实际路径。

在仓库根目录执行驱动脚本：

```bash
python examples/run_e2e_wf.py \
  --data-root <challenge_data 根目录> \
  --code-root <challenge_code 目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/wf_challenge_results.json>
```

## 开发自定义算子

开发步骤见 `NEXTSTEPS.md`。在 Python 中加载 YAML 并构建运行器，完整执行序列见 `examples/run_e2e_wf.py`：

```python
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner

pipeline = PipelineDefinition.from_file("<pipeline YAML 路径>")
runner = TaskRunner(pipeline)
```

## 目录结构

```text
Linxi-LincePlugin/
├── linxi_linceplugin/        # 算子源码（含四个赛题算子与模板示例算子）
├── examples/                 # 端到端示例 YAML 与驱动脚本
├── tests/                    # 测试
├── NEXTSTEPS.md
├── pyproject.toml
└── README.md
```
