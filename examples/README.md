# examples

- `e2e_wf_MA-CO-20231227-01.yaml`：钉死 easy session `MA_CO/easy/MA-CO-20231227-01` 的 WF 基线端到端流水线（载入 → 解码推理 → 漂移分析 → LinshuFile 导出 `.ls`）。
- 运行（dbci 环境，仓库根目录）：`PYTHONPATH=. conda run -n dbci --no-capture-output python examples/run_e2e_wf.py --golden "/mnt/f/mgam_repos/Lince/运动跨天解码/output/csv/wf_challenge_results.json"`
- `run_e2e_wf.py`：驱动入口，以 `PipelineDefinition.from_file` + TaskRunner 执行 YAML 并对黄金逐字段对账；numpy 先于 linxi 导入以锁定与官方 harness 一致的 BLAS 线程配置（`linxi process` CLI 入口下 `linxi/__init__` 强制 `OPENBLAS_NUM_THREADS=1`，数值确定性偏离黄金 ~1e-5，头部 docstring 有完整说明）。
