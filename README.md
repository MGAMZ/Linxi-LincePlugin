# Linxi-LincePlugin

临策算法竞赛通用算子插件仓库。面向"运动跨天解码"赛道提供五个 Linxi 算子：赛题 NWB 载入、赛道 baseline 解码推理、跨天漂移指标、LinshuFile 导出、评测侧车导出；面向"记忆状态跨个体跨天解码"赛道提供 DPA session 的 units-based NWB 载入算子与 baseline 分类解码推理算子。
一条 pipeline YAML 即可对原始数据进行处理并输出 `.ls` 数据文件与同级评测侧车 JSON。

## 安装前提

### 环境与网络要求

- Python >= 3.12
- 网络可访问 PyPI
- 数据集与赛道 baseline 代码目录 `challenge_code`

### Linxi 的获取

Linxi 已发布于 PyPI，通过 `pip install linxi` 安装；`linshu-format`、`linxi-trodes` 依赖随其自动装入。

### 安装 Linxi 与 linshu-format

1. `pip install linxi linshu-format`
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
  - linxi_linceplugin.load_dpa_session
  - linxi_linceplugin.decode_baseline
  - linxi_linceplugin.dpa_baseline_infer
  - linxi_linceplugin.drift_analysis
  - linxi_linceplugin.export_wiring
  - linxi_linceplugin.eval_export
```

| stage | processor_name | 导入模块 | 描述 |
|---|---|---|---|
| load | `LoadLinceSession` | `linxi_linceplugin.load_lince_session` | 单个赛题 session 的 NWB 投影进临析内部表示 |
| load | `LoadLinceDpaSession` | `linxi_linceplugin.load_dpa_session` | 单个 DPA session 的 units-based NWB 投影进临析内部表示 |
| postprocess | `BaselineDecodeInfer` | `linxi_linceplugin.decode_baseline` | 预置权重 WF/GRU baseline 解码推理与官方口径评分 |
| postprocess | `DpaBaselineInfer` | `linxi_linceplugin.dpa_baseline_infer` | DPA 赛道 baseline 分类解码（按子挑战路由，赛道运行时指针管线，评测载荷入暂存袋经侧车袋透传） |
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

### LoadLinceDpaSession

DPA 赛题 NWB 只有 `units`、`intervals/trials` 与 `processing/ecephys/Firing_rate_1000ms` 三个实体位置，无 acquisition/electrodes，上游 `LoadNWB` 与运动侧载入字段假设均不适用，本算子按该 schema 自定义只读载入并做全契约校验（nwb 2.9.0、文件名 subject/date/role 与文件内标识交叉、trial 角色↔列矩阵、FR 形状与索引列语义、spike 计数三方一致）。

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_path` | `None` | DPA session NWB 文件路径（必填传入） |
| `recording_key` | "query" | query = 链上主记录；support = 校准记录（只写 `context.ecephys[recording_key]`） |
| `name` | `None` | 算子实例名 |

query 侧落位：FR 速率矩阵（已剥除源中混入数据区的 trial/bin 索引列）写根容器 `binned_spikes`（`time` 置空，行→时间语义由 `time_reference="delay_concat_bins"` 声明；units 表整体挂 counts 的 `channel` 多级坐标）；units 表与逐单元 spike 序列写根容器 `neurons`（`spike_samples` 为延迟拼接轴秒值，1 Hz 下 sample 即秒）；试次表写顶层 `context.trials` 与 `context.recording`。两侧的 `auxiliary_channels` 均物化 `trial_index`/`bin_index` 两条逐行索引序列。session 标识取 NWB 文件名主干。

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

### DpaBaselineInfer

对 DPA 评测文件执行赛道 baseline 分类解码，`method="baseline"` 按评测文件所在子挑战目录路由：`challenge2` = 恒等 region 池化 linear SVC（6 c2 train 拼接训练、五折 CV 选 C、全训练集重训、赛道评分器打分）；`challenge1` = 教程 unit_space 同日五折 CV（CV 指标即 `metrics` 的 `mem_acc` / `corr_acc`，预测列为 final model 对该文件全部 trial 的样本内预测，CV 明细以 `lince_dpa_c1_cv` 顶层键入侧车）。特征构造、超参网格与评分公式全部经 `track_root` 运行时指针装载的赛道模块执行，插件零数值路径复刻；推理前后对赛道数值路径源码做 sha256 清单守卫。逐 trial 预测 `mem_pred_lbl` / `corr_pred_lbl` 挂入 query 侧试次表；评测载荷写 `context.metrics` 暂存袋标准键 `metrics`（`split`、`subject`、`session_date`、`mem_acc`、`corr_acc`、`session_score`、`n_trials`、`method`），由 `ExportEvalMetrics` 袋透传写侧车。c2 无标签评测角色（`eval` / `eval-2`）照常产出预测与提交契约件，`metrics` 三比值落 null。

| 参数 | 默认 | 说明 |
|---|---|---|
| `track_root` | 必填 | "记忆状态跨个体跨天解码"赛道仓库根目录（含 `lince_memory/`、`scripts/`、`paths.py`） |
| `eval_path` | 必填 | 评测 NWB 路径，命名主干须与上游 `LoadLinceDpaSession` 载入的 session 一致 |
| `method` | "baseline" | 方法配置名，当前可选 `"baseline"`；未知名 Fast Fail |
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
| `date_pattern` | `r"MA-[A-Z]{2}-(\d{8})-\d{2}"` | 会话目录名日期正则，首个捕获组为 8 位日期（gap_days 解析） |
| `train_level` | "train" | 质心与 train 末日取样的层名 |
| `holdout_levels` | `None` → ("easy", "normal", "hard") | gap×cos 相关的取样层集合 |
| `agg_level` | "hard" | 批次聚合均值针对的层名（聚合字段名保持 `*_mean_hard` 字面量） |
| `name` | `None` | 算子实例名 |

### LinceWriteLinshuFile

将流水线中的记录与结果导出为 LinshuFile `.ls` 产物。DPA 类 context 的 units 表以 pandas MultiIndex 挂在矩阵 channel 坐标上，xarray→zarr 编码层拒序列化多级索引：本算子在写出前把该坐标展开为逐 level 坐标列（层级序记入 `lince_units_levels` 属性、展开清单登记进根 `notes`），写出后恢复载入态；无该形态的 context（如运动链）零触碰。读回经 `linxi_linceplugin.load_dpa_session.units_frame` 还原本表。

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

### DPA 双链示例

`examples/e2e_dpa_sub-m091_ses-20210608.yaml`：challenge1 `sub-m091_ses-20210608` train 同天基线链（载入 → 推理 → 导出 `.ls` → 导出侧车），golden 对账 `baseline_c1_cv.json` 的 unit_space 五折 CV 数字。
`examples/e2e_dpa_sub-m090_ses-20210527.yaml`：challenge2 `sub-m090_ses-20210527` 评测日双输入（support=eval-2、query=eval-1）恒等跨天链，golden 对账 `methods_results.json` 中各标签实际执行法（`method_selection.json` 选定法，HeadRefit 选定时代跑规则见 `run_e2e_dpa.py` 文档串）的 m090 eval-1 行，判据含逐 trial 预测翻转数 = 0。

```bash
python examples/run_e2e_dpa.py --chain c1 \
  --data-root <DPA challenge_data 根目录> \
  --track-root <记忆状态跨个体跨天解码仓库根目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/baseline_c1_cv.json>

python examples/run_e2e_dpa.py --chain m090eval \
  --data-root <DPA challenge_data 根目录> \
  --track-root <记忆状态跨个体跨天解码仓库根目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/methods_results.json> \
  --selection <赛道仓库 output/csv/method_selection.json>
```

## 评测侧车文件

导出段末尾的 `ExportEvalMetrics` 把 `context.metrics` 暂存袋序列化为与 `.ls` 数据文件同目录、同命名根的 `<数据文件名>.eval.json`（如 `MA-CO-20231227-01.ls` → `MA-CO-20231227-01.eval.json`）。本算子须在写出 `.ls` 的导出算子之后同段执行。

顶层键固定为 `session_id`、`tier`、`span`、`metrics`、`drift_targets`，缺失键落 null；`metrics` 逐项由当链解码算子填充——运动链为 `n_bins`、`r2_x`、`r2_y`、`r2_mean_raw`、`r2_mean`、`total_latency_ms`、`latency_per_bin_ms`、`latency_score`、`session_score`、`support_trials`、`query_trials`，DPA 链为 `split`、`subject`、`session_date`、`mem_acc`、`corr_acc`、`session_score`、`n_trials`、`method`；`drift_targets` 为跨天漂移分析的目标会话清单。schema 之外的袋键一并写出。NaN 与 ±Inf 在序列化前统一转换为 null 并打印一条转换清单日志；非 JSON 原生对象即时抛 `TypeError`，不静默丢键。

驱动脚本从该侧车读取评测行。给出 `--golden` 时逐字段与官方黄金基准比较并打印绝对差（运动链容差 1e-4，DPA 链 1e-6 且 m090eval 链附逐 trial 翻转判据），session_score 另以黄金重合成复核，末尾打印 `PARITY PASS/FAIL`；同一配置重复运行时，侧车与 store 的确定性字段完全一致。

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
├── linxi_linceplugin/        # 算子源码（含两赛道七个赛题算子与模板示例算子）
├── examples/                 # 端到端示例 YAML 与驱动脚本
├── NEXTSTEPS.md
├── pyproject.toml
└── README.md
```
