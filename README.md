# Linxi-LincePlugin

临策算法竞赛通用算子插件仓库。面向"运动跨天解码"赛道提供五个 Linxi 算子：赛题 NWB 载入、赛道 baseline 解码推理、跨天漂移指标、LinshuFile 导出、评测侧车导出。
一条 pipeline YAML 即可对原始数据进行处理并输出 `.ls` 数据文件与同级评测侧车 JSON。

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
git clone https://gitee.com/MGAM/Linxi-LincePlugin
cd Linxi-LincePlugin
python -m pip install -e .
```

## 依赖说明

| 组 | 包 | 用途 |
|---|---|---|
| 必选 | scikit-learn | WF baseline 解码（`BaselineDecodeInfer` 的 `model="wf"` 路径，纯 CPU） |
| 可选 extras `[gru]` | torch | GRU baseline 解码（`model="gru"`） |

## 算子与注册

在配置文件中增加以下字段：

```yaml
linxi_plugin:
  - linxi_linceplugin.load_lince_session
  - linxi_linceplugin.decode_baseline
  - linxi_linceplugin.drift_analysis
  - linxi_linceplugin.export_wiring
  - linxi_linceplugin.eval_export
```

| stage | processor_name | 导入模块 | 描述 |
|---|---|---|---|
| load | `LoadLinceSession` | `linxi_linceplugin.load_lince_session` | 单个赛题 session 的 NWB 投影进临析内部表示 |
| postprocess | `BaselineDecodeInfer` | `linxi_linceplugin.decode_baseline` | 预置权重 WF/GRU baseline 解码推理与官方口径评分 |
| analyze | `LinceDriftAnalysis` | `linxi_linceplugin.drift_analysis` | 跨天漂移指标（cos_raw、cos_centered、norm_ratio、gap-days 相关） |
| export | `LinceWriteLinshuFile` | `linxi_linceplugin.export_wiring` | 表示层接线上游 `WriteLinshuFile` 写出 `.ls` |
| export | `ExportEvalMetrics` | `linxi_linceplugin.eval_export` | 将 `context.metrics` 暂存袋写出为 `.ls` 同级 `<数据文件名>.eval.json` |

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
解码结果在引擎默认的 OpenBLAS 单线程配置下计算，与官方发布的基线数值存在 10⁻⁶ 量级以内差异，不影响分数比较与排名。

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
| `session_scalar_cols` | `None` | trials 表中按 trial 广播的 session 级标量列名 |
| `on_lossy` | "warn" | 字段无法表达时的策略：`"warn"` / `"fail"` |
| `require_eval_cols` | `true` | 是否要求 trials 表含评测列 |
| `structural_cols` | `None` | 非评测结构列集合（默认 `start_time`/`stop_time`/`trial_id`） |
| `overwrite` / `skip_fields` / `skip_raw_signals` | `true` / `None` / `false` | 透传上游 `WriteLinshuFile` 的同名参数 |

## 端到端示例

`examples/e2e_wf_MA-CO-20231227-01.yaml`：easy 任务 `MA-CO-20231227-01` 的 WF 全链示例，链路为载入 → 解码推理 → 漂移指标 → 导出 `.ls` → 导出评测侧车 JSON。示例内的数据路径是占位符，用于自有流水线时替换为实际路径。

在仓库根目录执行驱动脚本：

```bash
python examples/run_e2e_wf.py \
  --data-root <challenge_data 根目录> \
  --code-root <challenge_code 目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/wf_challenge_results.json>
```

## 评测侧车文件

导出段末尾的 `ExportEvalMetrics` 把 `context.metrics` 暂存袋序列化为与 `.ls` 数据文件同目录、同命名根的 `<数据文件名>.eval.json`（如 `MA-CO-20231227-01.ls` → `MA-CO-20231227-01.eval.json`）。本算子须在写出 `.ls` 的导出算子之后同段执行。

顶层键固定为 `session_id`、`tier`、`span`、`metrics`、`drift_targets`，缺失键落 null；`metrics` 逐项为 `n_bins`、`r2_x`、`r2_y`、`r2_mean_raw`、`r2_mean`、`total_latency_ms`、`latency_per_bin_ms`、`latency_score`、`session_score`、`support_trials`、`query_trials`；`drift_targets` 为跨天漂移分析的目标会话清单。NaN 与 ±Inf 在序列化前统一转换为 null 并打印一条转换清单日志；非 JSON 原生对象即时抛 `TypeError`，不静默丢键。

驱动脚本从该侧车读取评测行。给出 `--golden` 时逐字段与官方黄金基准比较并打印绝对差，每字段判定为绝对差 ≤1e-4，session_score 另以黄金延迟重合成复核，末尾打印 `PARITY PASS/FAIL`；同一配置重复运行时，侧车与 store 的确定性字段完全一致。

## 行为序列与计数矩阵的落位

光标速度与位置序列（`cursor_vel_x`、`cursor_vel_y`、`cursor_pos_x`、`cursor_pos_y`）写在产物根字段 `behavior_recording`，键为信号标识，值为 TimeSeries。解码预测列（`cursor_vel_pred_x`、`cursor_vel_pred_y`）与 `eval_mask` 在 query 记录的 `auxiliary_channels`。

分箱放电计数矩阵写在根字段 `binned_spikes`，类型为 `linshu_format.core` 的 `BinnedSpikes`（`from linshu_format.core import BinnedSpikes`）：`counts` 为 (time, channel) 两维数组，行时间取自源数据真实时间戳（试次间隙可见），`bin_sec` 承载名义分箱宽度，溯源字段记录产出它的管线与分箱参数。

## 运行配置

- 导出段：上下文无可写数据时上游 `WriteLinshuFile` 只告警、不产出文件，`LinceWriteLinshuFile` 与 `ExportEvalMetrics` 在写入前断言可写对象在场、写入后断言产物存在，违例抛 `LinceExportError`。
- 长流水线：输出进 work_dir 并启用 `linxi process --incomplete-workdir-strategy delete`，强制中断留下的 `.ls.tmp-*` 残件由下一次启动的 work_dir 清扫覆盖；显式指定 `output_path` 时残件位于 work_dir 之外、清扫策略不可达，需手动清理。

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
├── linxi_linceplugin/        # 算子源码（含五个赛题算子与模板示例算子）
├── examples/                 # 端到端示例 YAML 与驱动脚本
├── NEXTSTEPS.md
├── pyproject.toml
└── README.md
```
