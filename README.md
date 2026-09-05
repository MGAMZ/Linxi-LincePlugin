# Linxi-LincePlugin

临策算法竞赛通用算子插件仓库

这个仓库是一个 Linxi 插件项目骨架。它使用根目录包布局和 hatchling 打包，适合直接作为独立插件仓库继续开发。当前仓库内的 `ExamplePluginProcessor` 与 `example_linear_probe` 为模板示例算子（保留为可运行示例），赛题通用算子见后续提交与本 README 后文增补。

## 目录结构

```text
Linxi-LincePlugin/
├── linxi_linceplugin/
├── tests/
├── .gitignore
├── NEXTSTEPS.md
├── pyproject.toml
└── README.md
```

## 插件加载方式

Linxi 会根据 pipeline 配置中的 `linxi_plugin` 导入插件模块。对这个模板来说，通常直接填写包名即可。把下面的示例保存为 `demo_pipeline.yaml`：

```yaml
name: demo_pipeline
linxi_plugin:
  - linxi_linceplugin
steps:
  - stage: preprocess
    processor_name: ExamplePluginProcessor
    params: {}
```

加载这个 YAML 有两种方式，任选其一：

- 命令行：

```bash
linxi process -c demo_pipeline.yaml -i <输入数据路径> -o <输出目录>
```

- 脚本：`PipelineDefinition.from_file` 解析 YAML 时会自动导入 `linxi_plugin` 列出的模块并完成注册，随后 `TaskRunner` 即可发现这些 processor 与 probe：

```python
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner

pipeline = PipelineDefinition.from_file("demo_pipeline.yaml")
runner = TaskRunner(pipeline)
```

## 开发原则

- 导入插件包时会立即执行注册逻辑
- 不要在模块顶层写重型副作用代码
- 同名 processor 或 probe 会在注册时直接报错

更具体的开发步骤见 `NEXTSTEPS.md`。