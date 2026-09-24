# Linxi-LincePlugin

临策算法竞赛的临析引擎插件仓库，为手写运动跨天解码、精细手部运动解码、记忆状态跨个体跨天解码、硬膜外运动解码四条赛道提供数据载入、baseline 解码与漂移指标算子，并提供跨赛道共用的 LinshuFile 导出与评测结果文件导出算子。一条 pipeline YAML 即可处理原始数据，输出 `.ls` 数据文件与同级的 `<数据文件名>.eval.json` 评测结果文件。

## 安装

需要 Python 3.12 及以上，网络可访问 PyPI。请自行准备赛题数据 `challenge_data`，以及各赛道的 baseline 代码仓库。

```bash
pip install linxi linshu-format
git clone https://gitee.com/MGAM/Linxi-LincePlugin
cd Linxi-LincePlugin
pip install -e .
```

scikit-learn 与 pyedflib 随包自动安装。GRU baseline 解码额外需要 torch，用可选依赖安装：

```bash
pip install -e ".[gru]"
```

## 快速开始

仓库根目录执行驱动脚本。脚本把链配置里的路径占位符替换为命令行传入的根目录后运行，产物为 `.ls` 数据文件与同级的评测结果文件。给出 `--golden` 时与官方基准逐字段比较并打印 `PARITY PASS/FAIL`。

运动赛道。链路与数据全部来自官方赛包，`--data-root` 与 `--code-root` 直接指向赛包内的 `challenge_data` 与 `challenge_code`：

```bash
python examples/run_e2e_wf.py \
  --data-root <赛包 challenge_data 根目录> \
  --code-root <赛包 challenge_code 目录> \
  --output-root <产物输出目录> \
  [--golden <评测基准 JSON>]
```

记忆赛道。`c1` 链为 challenge1 `sub-m091_ses-20210608` 同天基线，`m090eval` 链为 challenge3 `sub-m090_ses-20210527` 评测日双输入，support 为 eval-2，query 为 eval-1，后者需 `--selection` 指定方法选择文件：

```bash
python examples/run_e2e_dpa.py --chain c1 \
  --data-root <challenge_data 根目录> \
  --track-root <记忆赛道仓库根目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/baseline_c1_cv.json>

python examples/run_e2e_dpa.py --chain m090eval \
  --data-root <challenge_data 根目录> \
  --track-root <记忆赛道仓库根目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/methods_results.json> \
  --selection <赛道仓库 output/csv/method_selection.json>
```

硬膜外赛道。使用 heldin 训练集，链路为载入 → 评测窗口 → baseline 解码 → 导出 `.ls` → 导出评测结果文件，`--chain both` 同时运行单动作与组合动作两条流水线：

```bash
python examples/run_e2e_epi.py --chain both \
  --data-root <NEO heldin 目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/heldin_predictions.csv>
```

精细手部赛道。链配置为 `examples/e2e_finehand_mini.yaml`，链路为载入 → GRU 推理 → 导出 `.ls`，评测结果文件由推理算子写出。驱动对同一输入连跑两次链并做自身决定论对账，打印 `E2E MINI PASS/FAIL`；`--golden` 指向的指纹文件不存在时留存本轮指纹，存在时逐成员比对：

```bash
python examples/run_e2e_finehand.py \
  --data-root <赛题数据包根目录> \
  --track-root <精细手部赛道仓库根目录> \
  --output-root <产物输出目录> \
  [--golden <指纹文件>]
```

示例配置内的数据路径都是占位符，用于自有流水线时替换为实际路径，从代码构建并运行流水线的完整方法以 `examples/run_e2e_wf.py` 的 `main` 函数为准。各示例的链配置与判据说明见 `examples/README.md`。

## 算子注册

算子模块按赛道分为五个子包，分别对应运动跨天解码 `hand_motion_decode`、精细手部运动解码 `fine_hand_decode`、记忆状态跨个体跨天解码 `memory_state_decode`、硬膜外运动解码 `epidural_motion_decode`、共用导出 `export`。在流水线配置中用 `linxi_plugin` 字段逐模块导入触发注册，一条链只需列出该链用到的模块：

```yaml
linxi_plugin:
  # hand_motion_decode 赛道
  - linxi_linceplugin.hand_motion_decode.load_session
  - linxi_linceplugin.hand_motion_decode.decode_baseline
  - linxi_linceplugin.hand_motion_decode.drift_analysis
  # fine_hand_decode 赛道
  - linxi_linceplugin.fine_hand_decode.load_session
  - linxi_linceplugin.fine_hand_decode.decode_baseline
  # memory_state_decode 赛道
  - linxi_linceplugin.memory_state_decode.load_session
  - linxi_linceplugin.memory_state_decode.decode_baseline
  # epidural_motion_decode 赛道
  - linxi_linceplugin.epidural_motion_decode.load_session
  - linxi_linceplugin.epidural_motion_decode.action_windows
  - linxi_linceplugin.epidural_motion_decode.decode_baseline
  # 共用导出
  - linxi_linceplugin.export.linshufile
  - linxi_linceplugin.export.ieeg_linshufile
  - linxi_linceplugin.export.eval_metrics
```

| stage | processor_name | 功能 |
|---|---|---|
| load | `LoadLinceSession` | 载入单个运动赛题 session 的 NWB 数据 |
| load | `LoadLinceFineHandSession` | 载入单个精细手部赛题 session 的 npz 形态数据 |
| load | `LoadLinceDpaSession` | 载入单个 DPA session 的 units-based NWB 数据 |
| load | `LoadLinceEpiSession` | 载入含 BDF 三件套的单个硬膜外 session 目录 |
| preprocess | `EpiExtractActionWindows` | 按评测契约截取逐试次动作窗口并扩展试次表 |
| postprocess | `BaselineDecodeInfer` | 预置权重 WF/GRU baseline 解码与评分 |
| postprocess | `FineHandGruInfer` | 精细手部赛道预置权重 GRU 推理与四分量评分 |
| postprocess | `DpaBaselineInfer` | DPA 赛道 baseline 分类解码，按子挑战选择方法 |
| postprocess | `EpiBaselineInfer` | 硬膜外赛道 baseline 解码，统一 8 分类 |
| analyze | `LinceDriftAnalysis` | 跨天漂移指标 |
| export | `LinceWriteLinshuFile` | 将 ecephys 记录导出为 `.ls` |
| export | `LinceWriteIeegLinshuFile` | 将 `context.ieeg` 连续电位记录导出为 `.ls` |
| export | `ExportEvalMetrics` | 将 `context.metrics` 写出为 `.ls` 同级 `<数据文件名>.eval.json` |

## 算子参数

### LoadLinceSession

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_path` | 必填 | session NWB 文件路径 |
| `recording_key` | "query" | 评测集用 `query`，校准集用 `support` |
| `data_root` | 必填 | 数据根目录路径 |
| `name` | `None` | 算子实例名 |

### LoadLinceDpaSession

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_path` | 必填 | DPA session NWB 文件路径 |
| `recording_key` | "query" | `query` 为链上主记录，`support` 为校准记录 |
| `name` | `None` | 算子实例名 |

### BaselineDecodeInfer

按官方提交接口执行解码：`reset` 与 `predict` 全档位执行，`adapt` 仅在 `level` 为 `normal` 且上下文含 `support` 记录时执行，`easy` 与 `hard` 档不接收校准数据。

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | "wf" | 解码模型：`wf` 或 `gru` |
| `baseline_code_path` | 必填 | 官方赛包 `challenge_code` 目录，内含 `Participant/` 子目录 |
| `data_root` | 必填 | `challenge_data` 根目录 |
| `weights_dir` | `""` | 预置权重目录 |
| `task` | `""` | 任务类型 `MA_CO` 或 `MA_RT`，必填 |
| `level` | `""` | 评测难度 `easy`、`normal` 或 `hard`，必填 |
| `horizon` | `""` | normal 与 hard 的时程片 |
| `session_key` | `""` | session 目录名 |
| `name` | `None` | 算子实例名 |

### DpaBaselineInfer

| 参数 | 默认 | 说明 |
|---|---|---|
| `track_root` | 必填 | 记忆赛道仓库根目录，内含 `lince_memory/`、`scripts/` 与 `paths.py`，其 `paths.py` 中的 `DATA_ROOT` 须与 `--data-root` 指向同一份 `challenge_data` |
| `eval_path` | 必填 | 评测 NWB 路径，文件主干须与上游 `LoadLinceDpaSession` 载入的 session 一致 |
| `method` | "baseline" | 方法配置名，当前仅 `baseline`，未知名报错 |
| `name` | `None` | 算子实例名 |

### LoadLinceEpiSession

信号以伏特为单位写入 `context.ieeg[recording_key]` 的 iEEGRecording 容器，试次表写入其 `events`，query 侧同步写入顶层 `context.trials`。试次表列为 `trial_id`、`start_time`、`stop_time`、`trigger`、`label`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_path` | 必填 | session 目录路径，目录名以 `-single-MA` 或 `-dual-MA` 结尾 |
| `recording_key` | "query" | `query` 为链上主记录并填顶层 `context.trials`，`support` 为辅助记录 |
| `name` | `None` | 算子实例名 |

### LoadLinceFineHandSession

读取赛题 session 目录内的逐 trial npz（`tx`、`sbp`、`keypoint`、`frame_timestamps_ns` 等字段）与 taskinfos.csv，TX 反算计数写入根容器 `binned_spikes`，关键点序列以单条多维 TimeSeries（键 `hand_keypoints`，形状 (time, keypoint, xyz)）写入 `behavior_recording`，试次表写入顶层 `trials`，`context.recording` 挂内存计数记录供下游算子使用；`support` 侧仅挂 `context.ecephys` 槽。SBP 全矩阵与 44 列任务信息表经 `sidecar_dir` 伴随导出到 `.ls` 之外。

| 参数 | 默认 | 说明 |
|---|---|---|
| `session_dir` | 必填 | 赛题 session 目录，内含 `heldin/` 或 `heldout/` 一个分区 |
| `recording_key` | "query" | `query` 侧写根容器 `binned_spikes`、`behavior_recording` 与顶层 `trials`，`support` 侧仅挂 `context.ecephys` |
| `sidecar_dir` | `None` | 设定时写出 `<session>_sbp.npy` 与 `<session>_taskinfos_full.csv`，未设时日志声明不导出 |
| `name` | `None` | 算子实例名 |

### EpiExtractActionWindows

每试次截取动作开始 trigger 后 0.2 秒起的 2.0 秒窗口，1000 Hz 采样下为 2000 点，以 (time, channel) float32 存入试次表 `window_signal` 列，并追加 `window_start_sample`、`window_stop_sample`、`sample_id` 列。提交样本命名为 `{session}_onset_{trigger_sample}`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `recording_key` | "query" | 读写的 `context.ieeg` 键 |
| `window_start` | 0.2 | 窗口起点相对动作开始的秒数 |
| `window_duration` | 2.0 | 窗口秒数 |
| `name` | `None` | 算子实例名 |

### EpiBaselineInfer

对 query 试次表的评测窗口执行统一 8 分类解码，预测写回 `pred_label` 列，评测指标写入 `context.metrics`，键含 `macro_f1`、`per_class_f1`、`n_eval_trials`、`n_train_trials`、`classes` 与 `method`。训练集从 `train_root` 下全部 session 目录重建。

| 参数 | 默认 | 说明 |
|---|---|---|
| `train_root` | 必填 | 训练 session 集合根目录，其下每个含 `data.bdf` 的子目录为一个 session |
| `recording_key` | "query" | 读写的 `context.ieeg` 键 |
| `name` | `None` | 算子实例名 |

### FineHandGruInfer

以 DATA_ROOT 官方预置 GRU 权重对精细手部 session 执行与官方 runtime 同构的推理，trial 起点重置隐状态、512 帧分块，输入按 checkpoint 统计量 z-score。写出 `<session_id>_pred.h5` 与 `<算子实例名>.eval.json`，四分量评分经 `track_root` 挂接赛道包 `lince_finehand.scoring` 计算，五分数 dict 连同 `session_id` 两键写入 `context.metrics`。写前守卫拒绝输出目录位于只读输入根之内。

| 参数 | 默认 | 说明 |
|---|---|---|
| `session_dir` | 必填 | 赛题 session 目录，主干名须与上游载入的 session 一致 |
| `model_dir` | 必填 | 预置权重目录，读取 `<model_dir>/<session_id>/checkpoint.pt` |
| `track_root` | 必填 | 精细手部赛道仓库根目录，评分 import `lince_finehand.scoring` |
| `output_dir` | 必填 | 产物目录，位于只读输入根之内时推理前拒绝写入 |
| `device` | "cpu" | `cpu`、`cuda` 或 `auto` |
| `num_threads` | 4 | torch 线程数，`None` 表示不设置 |
| `name` | `None` | 算子实例名 |

### LinceDriftAnalysis

计算跨天漂移指标，输出列含 `cos_raw`、`cos_centered`、`norm_ratio` 及与 gap_days 的相关值。

| 参数 | 默认 | 说明 |
|---|---|---|
| `source` | "context" | `context` 流水线已载入记录，`sessions` 显式会话清单，`sweep` 按 `data_root` 扫描发现 |
| `sessions` | `None` | 会话清单，`sessions` 模式必填，逐项含 `task`/`level`/`session_key` 与 (T,512) 计数矩阵或 NWB 路径 |
| `data_root` | `None` | `sweep` 模式必填，`challenge_data` 根目录 |
| `tasks` / `levels` | `None` | sweep 的任务与级别范围 |
| `target_keys` | `["query"]` | 被分析的记录键 |
| `centroid_key` / `centroid` | `None` | 质心来源为记录键或 512 维向量，至少提供其一 |
| `session_key` / `train_session_keys` | `None` | 目标 session 与训练 session 名，用于 gap_days |
| `result_prefix` | "drift" | 输出列名前缀 |
| `date_pattern` | `r"MA-[A-Z]{2}-(\d{8})-\d{2}"` | 会话目录名日期正则，首个捕获组为 8 位日期 |
| `train_level` | "train" | 质心与 train 末日取样的层名 |
| `holdout_levels` | `None` → ("easy", "normal", "hard") | gap×cos 相关的取样层集合 |
| `agg_level` | "hard" | 批次聚合均值针对的层名 |
| `name` | `None` | 算子实例名 |

### LinceWriteLinshuFile

| 参数 | 默认 | 说明 |
|---|---|---|
| `output_path` | `None` | `.ls` 产物路径，产物为 Zarr 目录而非单文件 |
| `ecephys_key` | "query" | 导出的记录键 |
| `session_scalar_cols` | `None` | trials 表中按 trial 广播的 session 级标量列名 |
| `on_lossy` | "warn" | 字段无法表达时的策略：`warn` 或 `fail` |
| `require_eval_cols` | `true` | 是否要求 trials 表含评测列 |
| `structural_cols` | `None` | 非评测结构列集合，默认为 `start_time`/`stop_time`/`trial_id` |
| `overwrite` / `skip_fields` / `skip_raw_signals` | `true` / `None` / `false` | 传给上游 `WriteLinshuFile` 的同名参数 |

写出前将 DPA 记录 units 表的 MultiIndex channel 坐标展开为逐层坐标列，读回用 `linxi_linceplugin.memory_state_decode.load_session.units_frame` 还原。

### LinceWriteIeegLinshuFile

| 参数 | 默认 | 说明 |
|---|---|---|
| `output_path` | 必填 | `.ls` store 路径 |
| `recording_key` | "query" | `context.ieeg` 键 |
| `overwrite` / `skip_fields` | `true` / `None` | 覆盖策略，`skip_fields` 传给 `exclude_fields` |
| `require_eval_cols` / `structural_cols` | `true` / `None` | 与 `LinceWriteLinshuFile` 同名参数同语义 |
| `on_lossy` | "warn" | 有损策略：`warn` 或 `fail` |

### ExportEvalMetrics

将 `context.metrics` 写出为与 `.ls` 同目录、同命名根的 `<数据文件名>.eval.json`，即数据文件名换 `.eval.json` 后缀。本算子须在写出 `.ls` 的导出算子之后、同一导出段执行。

## 数据产物格式

`.ls` 数据文件是 LinshuFile Zarr 目录。行为序列与计数矩阵的存放位置如下。

光标速度与位置序列 `cursor_vel_x`、`cursor_vel_y`、`cursor_pos_x`、`cursor_pos_y` 写入根字段 `behavior_recording`，键为信号标识，值为 TimeSeries。解码预测列 `cursor_vel_pred_x`、`cursor_vel_pred_y` 与 `eval_mask` 在 query 记录的 `auxiliary_channels`。

分箱放电计数矩阵写入根字段 `binned_spikes`，类型为 `linshu_format.core.BinnedSpikes`。`counts` 为 (time, channel) 两维数组，行时间取自源数据时间戳，`bin_sec` 为名义分箱宽度。

评测副本 `<数据文件名>.eval.json` 顶层键为 `session_id`、`tier`、`span`、`metrics`、`drift_targets`，缺失键写入 null。`tier` 在运动链为该 session 的评测难度层级，为 easy、normal 或 hard，在硬膜外链为动作类型，单动作记 single、组合动作记 dual。`span` 在运动链为 normal 与 hard 任务的时程片。记忆链不写 `tier` 与 `span`，硬膜外链不写 `span`，对应键输出为 null。`metrics` 由各链的解码算子填充：运动链含 `n_bins`、`r2_x`、`r2_y`、`r2_mean_raw`、`r2_mean`、`total_latency_ms`、`latency_per_bin_ms`、`latency_score`、`session_score`、`support_trials`、`query_trials`。DPA 链含 `split`、`subject`、`session_date`、`mem_acc`、`corr_acc`、`session_score`、`n_trials`、`method`。硬膜外链含 `macro_f1`、`per_class_f1`、`n_eval_trials`、`n_train_trials`、`classes`、`method`。`drift_targets` 为漂移分析的目标会话清单。schema 之外的键一并写出，NaN 与 ±Inf 序列化为 null。运动链的官方赛包未公布计分公式，其 `latency_score` 与 `session_score` 按插件内部约定权重计算，用于横向比较而非复算官方分数。
