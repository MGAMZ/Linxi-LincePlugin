"""临策算法竞赛通用算子插件仓库。

按赛道划分子包：hand_motion_decode/（运动跨天解码）、memory_state_decode/（记忆状态跨个体跨天解码）、export/（两赛道共用的 LinshuFile 导出与评测结果附属文件导出）。
算子注册由 pipeline YAML 的 linxi_plugin 字段逐模块导入触发，见 README。
"""
