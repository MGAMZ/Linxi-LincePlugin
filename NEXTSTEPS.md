# 下一步

1. 确保 Linxi 安装在与你的插件相同的 Python 环境中。Linxi 竞赛正式上线时提供 PyPI 版本，执行 `python -m pip install linxi`（连同 `linshu-format` 依赖一并装入当前环境）。
   - 校验：`python -c "import linxi"` 无报错。
2. 执行 `python -m pip install -e .`，把当前插件仓库安装为可编辑模式。
3. 根据你的业务需要修改 `linxi_linceplugin` 包中的示例代码。
4. 如果你不需要某个示例文件，可以直接删除它，并同步更新 `__init__.py`。
5. 在 Linxi 的 pipeline YAML 中配置 `linxi_plugin`，验证你的 processor 或 probe 能被发现。YAML 样例与加载入口（`linxi process -c <yaml>` 或 `PipelineDefinition.from_file`）见仓库 `README.md` 的"算子与注册"与"开发自定义算子"两节。

## 代码修改建议

### processor

如果你要提供自定义 processor：

- 继承 `DefaultProcessor`
- 用 `register_as_linxi_processor(stage=...)` 注册
- 在 `_process()` 中返回更新后的 context

### probe

如果你要提供自定义 probe：

- 编写返回 `probeinterface.Probe` 或 `probeinterface.ProbeGroup` 的函数
- 用 `register_probe_definition(name)` 注册
- 保持注册名与实际用途一致，避免命名冲突

## 发布前检查

- 更新 `pyproject.toml` 中的元数据
- 删除模板中的示例类名和示例 probe 名称
- 把 README 改成描述真实插件功能的文档