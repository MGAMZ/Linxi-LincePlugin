# examples

- `e2e_wf_MA-CO-20231227-01.yaml`：easy session `MA_CO/easy/MA-CO-20231227-01` 的 WF 基线端到端流水线（载入 → 解码推理 → 漂移分析 → LinshuFile 导出 `.ls` → 评测侧车导出）；机器本地路径一律写成 `__DATA_ROOT__` / `__CODE_ROOT__` / `__OUTPUT_ROOT__` 占位符，由驱动在执行前注入。
- 运行（仓库根目录、已安装 linxi 的 Python 环境）：`python examples/run_e2e_wf.py --data-root <challenge_data 根目录> --code-root <challenge_code 目录> --output-root <产物输出目录> [--golden <wf_challenge_results.json>]`；产物为 `.ls` store 与同级 `<数据文件名>.eval.json` 侧车，驱动从侧车读取评测行并打印 session 读数与漂移目标；给出 `--golden` 时逐字段与官方黄金基准比较（每字段绝对差 ≤1e-4）并打印 `PARITY PASS/FAIL`。
- `run_e2e_wf.py`：驱动入口，以 `PipelineDefinition.from_file` + TaskRunner 执行替换占位符后的配置。
- `e2e_dpa_sub-m091_ses-20210608.yaml`：DPA challenge1 `sub-m091_ses-20210608` train 同天基线链（载入 → 基线推理 → 导出 `.ls` → 评测侧车导出）。
- `e2e_dpa_sub-m090_ses-20210527.yaml`：DPA challenge3 `sub-m090_ses-20210527` 评测日双输入链（support=eval-2、query=eval-1）。两条 DPA 链的路径占位符为 `__DATA_ROOT__` / `__TRACK_ROOT__`（"记忆状态跨个体跨天解码"赛道仓库根）/ `__OUTPUT_ROOT__`。
- 运行（仓库根目录、已安装 linxi 的 Python 环境）：`python examples/run_e2e_dpa.py --chain c1|m090eval --data-root <DPA challenge_data 根目录> --track-root <赛道仓库根目录> --output-root <产物输出目录> --golden <c1 链 baseline_c1_cv.json；m090eval 链 methods_results.json，须同时给 --selection <method_selection.json>>`；逐字段对账容差 ≤1e-6，m090eval 链附逐 trial 预测翻转数 =0 判据，末尾打印 `PARITY PASS/FAIL`。
