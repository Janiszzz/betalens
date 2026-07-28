#%%By Janis 251122
"""
配置管理模块
功能：
- 加载和管理datafeed模块的配置参数
- 支持从config.json文件读取配置
- 支持运行时动态修改配置
- 提供配置验证和默认值
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from copy import deepcopy


# 默认配置（作为fallback）
DEFAULT_CONFIG = {
    "database": {
        "dbname": "datafeed",
        "user": "postgres",
        "password": "",
        "host": "localhost",
        "port": "5432"
    },
    "logging": {
        "log_dir": "./logs",
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为当前模块目录下的config.json
        """
        # 确定配置文件路径
        if config_file is None:
            configured = os.environ.get("BETALENS_CONFIG")
            module_file = Path(__file__).parent / "config.json"
            local_file = Path(__file__).parent / "config.local.json"
            user_root = Path(os.environ.get("APPDATA", Path.home() / ".config"))
            user_file = user_root / "betalens" / "config.json"
            if configured:
                config_file = Path(configured).expanduser()
            elif user_file.exists():
                config_file = user_file
            elif local_file.exists():
                config_file = local_file
            elif module_file.exists():
                # Legacy local config remains readable but is no longer tracked.
                config_file = module_file
            else:
                config_file = user_file
        
        self.config_file = Path(config_file)
        self._config = None
        self._loaded = False
        
        # 加载配置
        self.load()
    
    def load(self) -> None:
        """
        从文件加载配置
        
        如果文件不存在或加载失败，使用默认配置
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # 合并文件配置和默认配置（文件配置优先）
                self._config = self._merge_config(DEFAULT_CONFIG, file_config)
                self._loaded = True
                
                logger = logging.getLogger('ConfigManager')
                logger.info(f"成功从 {self.config_file} 加载配置")
            else:
                # 文件不存在，使用默认配置
                self._config = deepcopy(DEFAULT_CONFIG)
                self._loaded = False
                
                logger = logging.getLogger('ConfigManager')
                logger.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
        
        except Exception as e:
            # 加载失败，使用默认配置
            self._config = deepcopy(DEFAULT_CONFIG)
            self._loaded = False
            
            logger = logging.getLogger('ConfigManager')
            logger.error(f"加载配置文件失败: {str(e)}，使用默认配置")
    
    def save(self, config_file: Optional[str] = None) -> None:
        """
        保存当前配置到文件
        
        Args:
            config_file: 配置文件路径，默认使用初始化时的路径
        """
        if config_file is not None:
            self.config_file = Path(config_file)
        
        try:
            # 确保目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            
            logger = logging.getLogger('ConfigManager')
            logger.info(f"配置已保存到 {self.config_file}")
        
        except Exception as e:
            logger = logging.getLogger('ConfigManager')
            logger.error(f"保存配置文件失败: {str(e)}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔，如 'database.dbname'
            default: 默认值，如果键不存在则返回此值
            
        Returns:
            配置值
            
        Example:
            >>> config = ConfigManager()
            >>> config.get('database.dbname')
            'datafeed'
            >>> config.get('database.port')
            '5432'
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔，如 'database.dbname'
            value: 配置值
            
        Example:
            >>> config = ConfigManager()
            >>> config.set('database.dbname', 'my_database')
            >>> config.get('database.dbname')
            'my_database'
        """
        keys = key_path.split('.')
        target = self._config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # 设置值
        target[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 配置节名称，如 'database', 'logging'
            
        Returns:
            配置节字典
        """
        return self.get(section, {})
    
    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """
        合并两个配置字典（递归）
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并字典
                result[key] = self._merge_config(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return deepcopy(self._config)
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)


# 全局配置实例
_global_config = None


def get_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置实例
    
    Args:
        config_file: 配置文件路径，仅在首次调用时有效
        
    Returns:
        ConfigManager实例
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigManager(config_file)
    
    return _global_config


def reset_config() -> None:
    """重置全局配置实例"""
    global _global_config
    _global_config = None


# 便捷访问函数
def get_database_config() -> Dict[str, str]:
    """获取数据库配置"""
    config = dict(get_config().get_section('database'))
    env_names = {
        "dbname": "BETALENS_DB_NAME",
        "user": "BETALENS_DB_USER",
        "password": "BETALENS_DB_PASSWORD",
        "host": "BETALENS_DB_HOST",
        "port": "BETALENS_DB_PORT",
    }
    for key, env_name in env_names.items():
        if env_name in os.environ:
            config[key] = os.environ[env_name]
    return config


def get_logging_config() -> Dict[str, str]:
    """获取日志配置"""
    return get_config().get_section('logging')

