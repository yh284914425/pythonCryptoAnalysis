#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志管理系统
支持控制台输出、文件日志、结构化日志等多种模式
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Optional, Union, Dict, Any
import json
from pathlib import Path

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OutputMode(Enum):
    """输出模式枚举"""
    CONSOLE_ONLY = "console_only"      # 仅控制台输出
    LOGGER_ONLY = "logger_only"        # 仅logger输出
    BOTH = "both"                      # 同时输出到控制台和logger
    STRUCTURED = "structured"          # 结构化日志输出
    SILENT = "silent"                  # 静默模式

class UnifiedLogger:
    """
    统一日志管理器
    支持多种输出模式，适应不同的使用场景
    """
    
    def __init__(
        self, 
        name: str = "TechnicalAnalysis",
        default_mode: OutputMode = OutputMode.CONSOLE_ONLY,
        log_file: Optional[str] = None,
        enable_emoji: bool = True
    ):
        self.name = name
        self.default_mode = default_mode
        self.enable_emoji = enable_emoji
        
        # 设置标准logger
        self.logger = logging.getLogger(name)
        
        # 如果指定了日志文件，配置文件处理器
        if log_file:
            self._setup_file_handler(log_file)
        
        # Emoji映射
        self.emoji_map = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🚨"
        } if enable_emoji else {}
        
        # 统计信息
        self.stats = {
            "debug": 0,
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0
        }
    
    def _setup_file_handler(self, log_file: str):
        """设置文件处理器"""
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def _format_message(self, message: str, level: LogLevel, context: Optional[Dict[str, Any]] = None) -> str:
        """格式化消息"""
        emoji = self.emoji_map.get(level, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 基础消息
        if emoji:
            formatted = f"{emoji} {message}"
        else:
            formatted = f"[{level.value}] {message}"
        
        # 添加上下文信息
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            formatted += f" ({context_str})"
        
        return formatted
    
    def _output_to_console(self, message: str, level: LogLevel):
        """输出到控制台"""
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            print(message, file=sys.stderr)
        else:
            print(message)
    
    def _output_to_logger(self, message: str, level: LogLevel):
        """输出到logger"""
        level_map = {
            LogLevel.DEBUG: self.logger.debug,
            LogLevel.INFO: self.logger.info,
            LogLevel.WARNING: self.logger.warning,
            LogLevel.ERROR: self.logger.error,
            LogLevel.CRITICAL: self.logger.critical
        }
        level_map[level](message)
    
    def _output_structured(self, message: str, level: LogLevel, context: Optional[Dict[str, Any]] = None):
        """输出结构化日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message,
            "logger": self.name
        }
        
        if context:
            log_entry["context"] = context
        
        print(json.dumps(log_entry, ensure_ascii=False))
    
    def log(
        self, 
        message: str, 
        level: LogLevel = LogLevel.INFO,
        mode: Optional[OutputMode] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        统一日志输出方法
        
        Args:
            message: 日志消息
            level: 日志级别
            mode: 输出模式（如果不指定则使用默认模式）
            context: 上下文信息
        """
        output_mode = mode or self.default_mode
        
        # 更新统计
        self.stats[level.value.lower()] += 1
        
        # 静默模式直接返回
        if output_mode == OutputMode.SILENT:
            return
        
        # 格式化消息
        formatted_message = self._format_message(message, level, context)
        
        # 根据模式输出
        if output_mode == OutputMode.CONSOLE_ONLY:
            self._output_to_console(formatted_message, level)
        
        elif output_mode == OutputMode.LOGGER_ONLY:
            self._output_to_logger(message, level)  # logger不需要emoji
        
        elif output_mode == OutputMode.BOTH:
            self._output_to_console(formatted_message, level)
            self._output_to_logger(message, level)
        
        elif output_mode == OutputMode.STRUCTURED:
            self._output_structured(message, level, context)
    
    # 便捷方法
    def debug(self, message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
        """调试日志"""
        self.log(message, LogLevel.DEBUG, mode, context)
    
    def info(self, message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
        """信息日志"""
        self.log(message, LogLevel.INFO, mode, context)
    
    def warning(self, message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
        """警告日志"""
        self.log(message, LogLevel.WARNING, mode, context)
    
    def error(self, message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
        """错误日志"""
        self.log(message, LogLevel.ERROR, mode, context)
    
    def critical(self, message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
        """严重错误日志"""
        self.log(message, LogLevel.CRITICAL, mode, context)
    
    # 特殊场景方法
    def progress(self, message: str, current: int, total: int, context: Optional[Dict[str, Any]] = None):
        """进度日志"""
        progress_context = {"current": current, "total": total, "percent": f"{current/total*100:.1f}%"}
        if context:
            progress_context.update(context)
        self.info(message, context=progress_context)
    
    def performance(self, operation: str, duration: float, context: Optional[Dict[str, Any]] = None):
        """性能日志"""
        perf_context = {"operation": operation, "duration_ms": f"{duration*1000:.3f}"}
        if context:
            perf_context.update(context)
        self.info(f"Performance: {operation}", context=perf_context)
    
    def report_section(self, title: str, separator: str = "=", width: int = 60):
        """报告章节标题"""
        self.info("")
        self.info(separator * width)
        self.info(title)
        self.info(separator * width)
    
    def table_header(self, columns: list, width: int = 15):
        """表格头部"""
        header = " | ".join([f"{col:>{width}}" for col in columns])
        self.info(header)
        self.info("-" * len(header))
    
    def table_row(self, values: list, width: int = 15):
        """表格行"""
        row = " | ".join([f"{str(val):>{width}}" for val in values])
        self.info(row)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        return {
            "stats": self.stats.copy(),
            "total": sum(self.stats.values()),
            "logger_name": self.name,
            "default_mode": self.default_mode.value
        }
    
    def set_mode(self, mode: OutputMode):
        """设置默认输出模式"""
        self.default_mode = mode
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0

# 全局实例
_global_logger = None

def get_logger(
    name: str = "TechnicalAnalysis",
    mode: OutputMode = OutputMode.CONSOLE_ONLY,
    log_file: Optional[str] = None
) -> UnifiedLogger:
    """
    获取全局logger实例
    
    Args:
        name: logger名称
        mode: 默认输出模式
        log_file: 日志文件路径（可选）
    
    Returns:
        UnifiedLogger实例
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = UnifiedLogger(name, mode, log_file)
    
    return _global_logger

def set_global_mode(mode: OutputMode):
    """设置全局日志模式"""
    logger = get_logger()
    logger.set_mode(mode)

# 便捷函数
def log_info(message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
    """便捷的信息日志函数"""
    logger = get_logger()
    logger.info(message, mode, context)

def log_warning(message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
    """便捷的警告日志函数"""
    logger = get_logger()
    logger.warning(message, mode, context)

def log_error(message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
    """便捷的错误日志函数"""
    logger = get_logger()
    logger.error(message, mode, context)

def log_debug(message: str, mode: Optional[OutputMode] = None, context: Optional[Dict[str, Any]] = None):
    """便捷的调试日志函数"""
    logger = get_logger()
    logger.debug(message, mode, context)

def log_performance(operation: str, duration: float, context: Optional[Dict[str, Any]] = None):
    """便捷的性能日志函数"""
    logger = get_logger()
    logger.performance(operation, duration, context)

def log_progress(message: str, current: int, total: int, context: Optional[Dict[str, Any]] = None):
    """便捷的进度日志函数"""
    logger = get_logger()
    logger.progress(message, current, total, context)

if __name__ == "__main__":
    # 测试示例
    print("🧪 测试统一日志系统...")
    
    # 测试不同模式
    logger = UnifiedLogger("TestLogger", OutputMode.CONSOLE_ONLY)
    
    print("\n1. 控制台模式测试:")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    print("\n2. 结构化日志测试:")
    logger.set_mode(OutputMode.STRUCTURED)
    logger.info("结构化日志测试", context={"user": "test", "action": "login"})
    
    print("\n3. 报告格式测试:")
    logger.set_mode(OutputMode.CONSOLE_ONLY)
    logger.report_section("📊 性能报告")
    logger.table_header(["指标", "耗时", "状态"])
    logger.table_row(["RSI", "0.05s", "成功"])
    logger.table_row(["MACD", "0.03s", "成功"])
    
    print("\n4. 统计信息:")
    stats = logger.get_stats()
    print(f"总日志数: {stats['total']}")
    print(f"各级别: {stats['stats']}") 