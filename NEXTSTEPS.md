# 开发指引

1. 环境与安装步骤见 `README.md` 的"安装前提"一节，本仓库以可编辑模式安装。
2. 新增算子：在对应赛道子包（`hand_motion_decode/`、`memory_state_decode/`）或共用 `export/` 下新建模块，继承 `DefaultProcessor`，用 `register_as_linxi_processor(stage=...)` 注册，`_process()` 返回更新后的 context。赛道内部支撑模块以下划线前缀命名。
3. 注册名写进类的 `PROCESSOR_NAME`，与 pipeline YAML 中的 `processor_name` 一致。
4. 在 pipeline YAML 的 `linxi_plugin` 字段列出用到的模块点路径即完成装载，加载入口（`linxi process -c <yaml>` 或 `PipelineDefinition.from_file`）见 `README.md` 的"开发自定义算子"一节。
5. 端到端验证按 `README.md` 的"端到端示例"两条命令运行对应链的驱动脚本，golden 对账打印 `PARITY PASS` 为通过。
