# 示例

五条赛道的端到端示例。每条赛道含链配置 YAML 与驱动脚本，配置中的机器本地路径写成 `__DATA_ROOT__`、`__CODE_ROOT__`、`__TRACK_ROOT__`、`__OUTPUT_ROOT__` 四个占位符，由驱动在执行前替换为命令行传入的根目录。产物为 `.ls` 数据文件与同级的 `<数据文件名>.eval.json` 评测结果文件。

在仓库根目录、已安装 linxi 的 Python 环境中运行，安装方式见根目录 `README.md`。给出 `--golden` 时与官方基准逐字段比较并打印 `PARITY PASS/FAIL`。

## 运动赛道

`e2e_wf_MA-CO-20231227-01.yaml`：easy 任务 `MA-CO-20231227-01` 的 WF 基线全链，链路为载入 → 解码推理 → 漂移分析 → 导出 `.ls` → 导出评测结果文件。

```bash
python examples/run_e2e_wf.py \
  --data-root <challenge_data 根目录> \
  --code-root <challenge_code 目录> \
  --output-root <产物输出目录> \
  [--golden <赛道仓库 output/csv/wf_challenge_results.json>]
```

驱动从评测结果文件读取评测行，打印 session 读数与漂移目标。`--golden` 可不给，不给时只运行链并打印产物路径，不执行对账。给出 `--golden` 时每字段绝对差不超过 1e-4。

## 记忆赛道

`e2e_dpa_sub-m091_ses-20210608.yaml`：challenge1 `sub-m091_ses-20210608` 同天基线链，链路为载入 → 基线解码 → 导出 `.ls` → 导出评测结果文件。

`e2e_dpa_sub-m090_ses-20210527.yaml`：challenge3 `sub-m090_ses-20210527` 评测日双输入链，support 为 eval-2，query 为 eval-1。

```bash
python examples/run_e2e_dpa.py --chain c1 \
  --data-root <challenge_data 根目录> \
  --track-root <记忆赛道仓库根目录> \
  --output-root <产物输出目录> \
  [--golden <赛道仓库 output/csv/baseline_c1_cv.json>]

python examples/run_e2e_dpa.py --chain m090eval \
  --data-root <challenge_data 根目录> \
  --track-root <记忆赛道仓库根目录> \
  --output-root <产物输出目录> \
  [--golden <赛道仓库 output/csv/methods_results.json>] \
  [--selection <赛道仓库 output/csv/method_selection.json>]
```

`--golden` 可不给，不给时只运行链并打印产物路径，不执行对账。m090eval 链执行对账时还需 `--selection` 给出 `method_selection.json`，逐字段对账容差不超过 1e-6，并附逐 trial 预测翻转数为 0 的判据。赛道仓库 `paths.py` 中的 `DATA_ROOT` 必须与 `--data-root` 指向同一份 `challenge_data`，m090eval 链的训练集由 `DATA_ROOT` 下的 `challenge3` 子目录发现。

## 硬膜外赛道

`e2e_epi_P01_20240905-single-MA.yaml` 与 `e2e_epi_P01_20241112-dual-MA.yaml`：heldin 单动作与组合动作 session 端到端链，链路为载入 → 评测窗口 → baseline 解码 → 导出 `.ls` → 导出评测结果文件。

```bash
python examples/run_e2e_epi.py --chain single|dual|both \
  --data-root <NEO heldin 目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/heldin_predictions.csv>
```

`--golden` 必填，判据为 `.ls` 读回的逐样本预测与官方权重推理的参考标签完全一致，评测结果文件的 `macro_f1` 与按参考文件重算的数值一致，信号与窗口形状符合官方评测规定。

## 精细手部赛道

`e2e_finehand_mini.yaml`：单 session `2026060501` 的最小链，链路为载入 → GRU 推理 → 导出 `.ls`，评测结果文件由推理算子写出。

```bash
python examples/run_e2e_finehand.py \
  --data-root <赛题数据包根目录> \
  --track-root <精细手部赛道仓库根目录> \
  --output-root <产物输出目录> \
  [--golden <指纹文件>]
```

驱动对同一输入连跑两次链并做自身决定论对账，一致时打印 `E2E MINI PASS`。`--golden` 指向的指纹文件不存在时留存本轮指纹，存在时逐成员比对。

## 高保真脑电数据压缩赛道

`e2e_eeg_compression.yaml`：np 子赛道冒烟夹具的最小链，链路为载入 → 重建质量 → 导出 `.ls`（置 `skip_raw_signals`，约 2 GB 原始信号不写入磁盘）→ 导出指标表。夹具是赛道仓库磁盘留存的冒烟产物，本链只读不写，不引入全量数据。

```bash
python examples/run_e2e_eeg_compression.py \
  --track-root <高保真脑电数据压缩赛道仓库根目录> \
  --output-root <产物输出目录> \
  [--golden <指纹文件>]
```

驱动对同一输入连跑两次链并做自身决定论对账，判据为指标表两轮逐字节一致、`.ls` 排除集外成员两轮逐字节一致、指标 `cr` 与 `prd` 对赛道冒烟基准 `output/csv/baseline_np_smoke_meta_fixed.json` 逐位一致、合规五门全过，全部通过时打印 `E2E EEG COMPRESSION PASS` 并以 0 退出。`--golden` 指向的指纹文件不存在时留存本轮指纹，存在时把占位符化后的指标表逐字段比对，路径字段以 `__TRACK_ROOT__` 记法留存，指纹跨机器可迁移。
