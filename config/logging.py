"""
日志配置模块
使用 loguru 提供统一的日志管理
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from .settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
) -> None:
    """
    配置应用日志系统
    
    使用 loguru 提供结构化日志，支持：
    - 控制台彩色输出
    - 文件日志轮转
    - 自动清理过期日志
    - 异常追踪
    
    Args:
        log_level: 日志级别，默认从配置读取
        log_file: 日志文件路径，默认从配置读取
        rotation: 日志轮转规则，默认从配置读取
        retention: 日志保留时间，默认从配置读取
    """
    # 使用配置中的默认值
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    rotation = rotation or settings.log_rotation
    retention = retention or settings.log_retention
    
    # 移除默认的 handler
    logger.remove()
    
    # ==================== 控制台日志 ====================
    # 添加彩色控制台输出，格式化更易读
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
        backtrace=True,  # 显示完整的异常追踪
        diagnose=True,   # 显示变量值
    )
    
    # ==================== 文件日志 ====================
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加文件日志，支持轮转和自动清理
    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=log_level,
        rotation=rotation,      # 文件大小达到限制时轮转
        retention=retention,    # 保留指定时间的日志
        compression="zip",      # 压缩旧日志
        backtrace=True,
        diagnose=True,
        enqueue=True,          # 异步写入，提高性能
    )
    
    logger.info(f"📝 日志系统初始化完成 - 级别: {log_level}, 文件: {log_file}")


def get_logger(name: str):
    """
    获取指定名称的 logger
    
    Args:
        name: logger 名称，通常使用模块的 __name__
        
    Returns:
        配置好的 logger 实例
        
    Example:
        >>> from config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条日志")
    """
    return logger.bind(name=name)


# 在模块导入时自动初始化日志系统
if "pytest" not in sys.modules:
    setup_logging()

