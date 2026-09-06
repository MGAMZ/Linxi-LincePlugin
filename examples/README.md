# examples

- `e2e_wf_MA-CO-20231227-01.yaml`：钉死 easy session `MA_CO/easy/MA-CO-20231227-01` 的 WF 基线端到端流水线（载入 → 解码推理 → 漂移分析 → LinshuFile 导出 `.ls`）；机器本地路径一律写成 `__DATA_ROOT__` / `__CODE_ROOT__` / `__OUTPUT_ROOT__` 占位符，由驱动在执行前注入。
- 运行（仓库根目录、已安装 linxi 的 Python 环境）：`python examples/run_e2e_wf.py --data-root <challenge_data 根目录> --code-root <challenge_code 目录> --output-root <产物输出目录> [--golden <wf_challenge_results.json>]`；给出 `--golden` 时执行逐字段 ≤1e-6 对账并打印 `PARITY PASS/FAIL`。
- `run_e2e_wf.py`：驱动入口，以 `PipelineDefinition.from_file` + TaskRunner 执行替换后的配置；numpy 先于 linxi 导入的黄金复现契约见该文件导入处注释。
