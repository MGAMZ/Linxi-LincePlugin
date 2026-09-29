# Linxi-LincePlugin

临策算法竞赛的临析引擎插件仓库，为五条赛道提供数据载入、基线解码与结果导出算子。
五条赛道分别是手写运动跨天解码、精细手部运动解码、记忆状态跨个体跨天解码、硬膜外运动解码、高保真脑电数据压缩。
准备赛题数据后，运行一条流水线配置和随附的驱动脚本，即可处理原始数据，产出 `.ls` 数据文件与配套的 `<数据文件名>.eval.json` 评测结果文件。

## 安装

需要 Python 3.12 及以上，网络可访问 PyPI。赛题数据与各赛道的基线代码仓库需要自行准备。

```bash
pip install linxi linshu-format
git clone https://gitee.com/MGAM/Linxi-LincePlugin
cd Linxi-LincePlugin
pip install -e .
```

scikit-learn 与 pyedflib 随包自动安装。
GRU 基线解码额外需要 torch，用可选依赖安装：

```bash
pip install -e ".[gru]"
```

## 快速开始

`examples/` 下每条赛道有一个流水线配置 YAML 和对应的驱动脚本。
驱动脚本把配置里的路径占位符替换为命令行传入的实际目录后运行流水线。
运动赛道与记忆赛道的驱动接受 `--golden` 参数，给出后与官方基准文件逐字段比较并报告结果。
硬膜外赛道的驱动必须给出 `--golden`。

运动赛道，配置 `e2e_wf_MA-CO-20231227-01.yaml`，流水线依次完成载入、解码、漂移分析、导出：

```bash
python examples/run_e2e_wf.py \
  --data-root <challenge_data 根目录> \
  --code-root <challenge_code 目录> \
  --output-root <产物输出目录>
```

记忆赛道，`c1` 流水线运行 challenge1 的同天基线，配置 `e2e_dpa_sub-m091_ses-20210608.yaml`；`m090eval` 流水线运行 challenge3 的评测日双输入，support 校准集为 eval-2，query 评测集为 eval-1，需比较结果时另用 `--selection` 给出方法选择文件，配置 `e2e_dpa_sub-m090_ses-20210527.yaml`：

```bash
python examples/run_e2e_dpa.py --chain c1 \
  --data-root <challenge_data 根目录> \
  --track-root <记忆赛道仓库根目录> \
  --output-root <产物输出目录>

python examples/run_e2e_dpa.py --chain m090eval \
  --data-root <challenge_data 根目录> \
  --track-root <记忆赛道仓库根目录> \
  --output-root <产物输出目录>
```

硬膜外赛道，配置分为单动作与组合动作两条流水线，`--chain` 取 `single`、`dual` 或 `both`：

```bash
python examples/run_e2e_epi.py --chain both \
  --data-root <NEO heldin 目录> \
  --output-root <产物输出目录> \
  --golden <官方参考预测文件>
```

精细手部赛道，配置 `e2e_finehand_mini.yaml`，流水线依次完成载入、GRU 推理、导出：

```bash
python examples/run_e2e_finehand.py \
  --data-root <赛题数据包根目录> \
  --track-root <精细手部赛道仓库根目录> \
  --output-root <产物输出目录>
```

高保真脑电数据压缩赛道，配置 `e2e_eeg_compression.yaml`，对应 np 子赛道的最小流水线，依次完成载入、重建质量评估、导出、指标表写出：

```bash
python examples/run_e2e_eeg_compression.py \
  --track-root <高保真脑电数据压缩赛道仓库根目录> \
  --output-root <产物输出目录>
```

各配置里的数据路径都是占位符，搭自己的流水线时替换为实际路径即可。

## 算子注册

算子按赛道分为六个子包：
- 运动赛道 `hand_motion_decode`
- 精细手部赛道 `fine_hand_decode`
- 记忆赛道 `memory_state_decode`
- 硬膜外赛道 `epidural_motion_decode`
- 压缩赛道 `eeg_compression`
- 共用导出 `export`。

在配置文件头添加如下内容即可导入插件包
```yaml
linxi_plugin:
  - linxi_linceplugin
```

| stage | processor_name | 功能 |
|---|---|---|
| load | `LoadLinceSession` | 载入运动赛道的 session NWB 数据 |
| load | `LoadLinceFineHandSession` | 载入精细手部赛道的 session 数据，含逐 trial 的 npz 文件与任务信息表 |
| load | `LoadLinceDpaSession` | 载入记忆赛道的 session NWB 数据 |
| load | `LoadLinceEpiSession` | 载入硬膜外赛道的 session 目录，含 BDF 格式连续信号 |
| load | `LoadCompress2BciNpStream` | 载入压缩赛道 np 子赛道的 SpikeGLX 连续信号 |
| load | `LoadCompress2BciIeegEdf` | 载入压缩赛道 ieeg 子赛道的 BIDS EDF 信号 |
| load | `LoadCompress2BciTrodesRec` | 载入压缩赛道 flex 子赛道的 Trodes `.rec` 信号 |
| preprocess | `EpiExtractActionWindows` | 硬膜外赛道按试次截取动作开始后 2 秒的信号窗口 |
| postprocess | `BaselineDecodeInfer` | 运动赛道预置权重解码与评分，支持 wf 与 gru 两种模型 |
| postprocess | `FineHandGruInfer` | 精细手部赛道以官方预置 GRU 权重推理并评分 |
| postprocess | `DpaBaselineInfer` | 记忆赛道基线分类解码，按子挑战选择方法 |
| postprocess | `EpiBaselineInfer` | 硬膜外赛道解码，统一 8 分类 |
| analyze | `LinceDriftAnalysis` | 计算运动赛道的跨天漂移指标 |
| quality | `Compress2BciReconQuality` | 比较压缩赛道原始数据与重建数据，给出压缩比、PRD 与合规检查结果 |
| export | `LinceWriteLinshuFile` | 将电生理记录导出为 `.ls` 数据文件 |
| export | `LinceWriteIeegLinshuFile` | 将连续电位记录导出为 `.ls` 数据文件 |
| export | `ExportEvalMetrics` | 将评测指标写出为 `.ls` 同名的 `<数据文件名>.eval.json` |

各算子的参数以代码 docstring 为准，`examples/` 的配置与驱动脚本给出了完整调用方式。

## 数据产物

`.ls` 数据文件遵循 LinshuFile 格式，保存为 Zarr 目录。

主要数据字段的存放方式如下：
- 行为序列写入根字段 `behavior_recording`
- 分箱放电计数写入根字段 `binned_spikes`
- 硬膜外赛道与压缩赛道 ieeg 子赛道的连续信号写入 `context.ieeg` 容器

评测结果文件 `<数据文件名>.eval.json` 与 `.ls` 同目录、同命名根，顶层键包含：
- `session_id`
- `tier`
  - 运动赛道为评测难度
    - `easy`
    - `normal`
    - `hard`
  - 硬膜外解码为动作类型
    - 单动作记 `single`
    - 组合动作记 `dual`
- `span`: normal 与 hard 任务的时程片段
- `metrics`: 由各类流水线自行填充，详情可见附表
- `drift_targets`: 漂移分析的目标会话清单
- 缺失字段写入 null。

### 附表：metric字段

| 流水线 | metrics 内容 |
| --- | --- |
| 运动 | `n_bins`、`r2_x`、`r2_y`、`r2_mean_raw`、`r2_mean`、`total_latency_ms`、`latency_per_bin_ms`、`latency_score`、`session_score`、`support_trials`、`query_trials` |
| 记忆 | `split`、`subject`、`session_date`、`mem_acc`、`corr_acc`、`session_score`、`n_trials`、`method` |
| 硬膜外 | `macro_f1`、`per_class_f1`、`n_eval_trials`、`n_train_trials`、`classes`、`method` |
| 压缩 | 写入形如 `recon_quality_<子赛道>` 的键，字段含 `cr`、`prd`、`metrics`、`compliance` |
