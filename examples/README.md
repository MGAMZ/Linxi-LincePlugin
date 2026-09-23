# 示例

三条赛道的端到端示例。每条赛道含链配置 YAML 与驱动脚本，配置中的机器本地路径写成 `__DATA_ROOT__`、`__CODE_ROOT__`、`__TRACK_ROOT__`、`__OUTPUT_ROOT__` 四个占位符，由驱动在执行前替换为命令行传入的根目录。产物为 `.ls` 数据文件与同级 `<数据文件名>.eval.json` 评测侧车。

在仓库根目录、已安装 linxi 的 Python 环境中运行，安装方式见根目录 `README.md`。给出 `--golden` 时与官方基准逐字段比较并打印 `PARITY PASS/FAIL`。

## 运动赛道

`e2e_wf_MA-CO-20231227-01.yaml`：easy 任务 `MA-CO-20231227-01` 的 WF 基线全链，链路为载入 → 解码推理 → 漂移分析 → 导出 `.ls` → 导出评测侧车。

```bash
python examples/run_e2e_wf.py \
  --data-root <challenge_data 根目录> \
  --code-root <challenge_code 目录> \
  --output-root <产物输出目录> \
  [--golden <赛道仓库 output/csv/wf_challenge_results.json>]
```

驱动从侧车读取评测行，打印 session 读数与漂移目标。`--golden` 可不给，不给时只运行链并打印产物路径，不执行对账。给出 `--golden` 时每字段绝对差不超过 1e-4。

## 记忆赛道

`e2e_dpa_sub-m091_ses-20210608.yaml`：challenge1 `sub-m091_ses-20210608` 同天基线链，链路为载入 → 基线解码 → 导出 `.ls` → 导出评测侧车。

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

`e2e_epi_P01_20240905-single-MA.yaml` 与 `e2e_epi_P01_20241112-dual-MA.yaml`：heldin 单动作与组合动作 session 端到端链，链路为载入 → 评测窗口 → baseline 解码 → 导出 `.ls` → 导出评测侧车。

```bash
python examples/run_e2e_epi.py --chain single|dual|both \
  --data-root <NEO heldin 目录> \
  --output-root <产物输出目录> \
  --golden <赛道仓库 output/csv/heldin_predictions.csv>
```

`--golden` 必填，判据为 `.ls` 读回的逐样本预测与官方权重推理的参考标签完全一致，评测侧车的 `macro_f1` 与按参考文件重算的数值一致，信号与窗口形状符合官方评测规定。
