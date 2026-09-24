"""高保真脑电数据压缩赛道评分核心与判题格式合规检查"""

from .compliance import check_flex, check_ieeg, check_np

__all__ = ["check_np", "check_ieeg", "check_flex"]
