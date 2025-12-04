"""
模型工厂 - 动态注册模式（简化版）
"""

import importlib
from typing import Dict, Any

from models.base_model import BaseModel
from configs.config import TrainingConfig


class ModelFactory:
    """模型工厂类 - 使用动态注册模式"""
    
    # 模型注册表
    _registry = {}
    
    @classmethod
    def register(cls, model_type: str, config_section: str = None, 
                 description: str = None, dependencies: list = None):
        """注册模型装饰器"""
        def decorator(model_class):
            config_key = config_section or model_type
            
            cls._registry[model_type] = {
                'class': model_class,
                'config_key': config_key,
                'description': description or getattr(model_class, '__doc__', '').strip().split('\n')[0],
                'dependencies': dependencies or [],
                'module': model_class.__module__
            }
            
            # 设置属性，便于自动发现
            model_class._model_type = model_type
            model_class._config_section = config_key
            model_class._description = description
            model_class._dependencies = dependencies or []
            
            return model_class
        return decorator
    
    @classmethod
    def create_model(cls, config: TrainingConfig) -> BaseModel:
        """创建模型实例"""
        model_type = config.model.model_type
        
        if model_type not in cls._registry:
            # 尝试动态导入
            cls._try_import_model(model_type)
        
        if model_type not in cls._registry:
            raise ValueError(f"未注册的模型类型: {model_type}")
        
        registry_info = cls._registry[model_type]
        
        # 检查依赖
        missing_deps = cls._check_dependencies(registry_info['dependencies'])
        if missing_deps:
            print(f"警告: {model_type} 缺少依赖: {missing_deps}")
            return cls._create_fallback_model(config)
        
        # 提取配置
        model_config = cls._extract_config(config, registry_info['config_key'])
        model_config['model_type'] = model_type
        model_config['model_name'] = config.model.model_name
        
        # 创建模型实例
        return registry_info['class'](model_config)
    
    @classmethod
    def _try_import_model(cls, model_type: str):
        """尝试动态导入模型"""
        module_name = f"models.{model_type}_model"
        try:
            module = importlib.import_module(module_name)
            # 查找注册了该类型的类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '_model_type') and attr._model_type == model_type:
                    # 这个类已经通过装饰器注册了
                    return True
        except ImportError:
            pass
        return False
    
    @classmethod
    def _check_dependencies(cls, dependencies: list) -> list:
        """检查依赖包是否安装"""
        missing = []
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        return missing
    
    @classmethod
    def _extract_config(cls, config: TrainingConfig, config_key: str) -> Dict[str, Any]:
        """从配置中提取模型特定配置"""
        if hasattr(config, config_key):
            config_obj = getattr(config, config_key)
            if hasattr(config_obj, '__dict__'):
                return config_obj.__dict__.copy()
            elif hasattr(config_obj, '__dataclass_fields__'):
                return {k: getattr(config_obj, k) for k in config_obj.__dataclass_fields__}
        return {}
    
    @classmethod
    def _create_fallback_model(cls, config: TrainingConfig) -> BaseModel:
        """创建备用模型（随机森林）"""
        print(f"使用随机森林替代 {config.model.model_type}")
        
        # 修改配置为随机森林
        from dataclasses import replace
        config = replace(config, model=replace(config.model, model_type='random_forest'))
        
        # 递归调用创建随机森林模型
        return cls.create_model(config)
    
    @classmethod
    def get_available_models(cls) -> list:
        """获取所有已注册的模型类型"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_model_description(cls, model_type: str) -> str:
        """获取模型描述"""
        if model_type in cls._registry:
            return cls._registry[model_type]['description']
        return f"未知模型类型: {model_type}"
    
    @classmethod
    def manual_register(cls, model_type: str, model_class, 
                       config_section: str = None, description: str = None, 
                       dependencies: list = None):
        """手动注册模型"""
        config_key = config_section or model_type
        
        cls._registry[model_type] = {
            'class': model_class,
            'config_key': config_key,
            'description': description or getattr(model_class, '__doc__', '').strip().split('\n')[0],
            'dependencies': dependencies or [],
            'module': model_class.__module__
        }
        print(f"手动注册模型: {model_type}")


# 自动注册现有模型
def _register_existing_models():
    """自动注册所有使用装饰器的模型"""
    import glob
    import importlib
    
    # 查找所有模型文件
    model_files = glob.glob("models/*_model.py")
    
    for model_file in model_files:
        # 提取模块名（去掉 .py 扩展名）
        module_name = model_file.replace("/", ".").replace("\\", ".").replace(".py", "")
        
        try:
            module = importlib.import_module(module_name)
            
            # 查找所有类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # 检查是否是类且继承了BaseModel
                if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                    # 检查是否有 _model_type 属性（通过装饰器设置）
                    if hasattr(attr, '_model_type'):
                        model_type = attr._model_type
                        
                        # 检查是否已注册
                        if model_type not in ModelFactory._registry:
                            # 获取装饰器参数
                            config_section = getattr(attr, '_config_section', model_type)
                            description = getattr(attr, '_description', getattr(attr, '__doc__', '').strip().split('\n')[0])
                            dependencies = getattr(attr, '_dependencies', [])
                            
                            # 注册模型
                            ModelFactory._registry[model_type] = {
                                'class': attr,
                                'config_key': config_section,
                                'description': description,
                                'dependencies': dependencies,
                                'module': module_name
                            }
                            
                            print(f"自动注册模型: {model_type} -> {attr.__name__}")
                            
        except ImportError as e:
            print(f"无法导入模块 {module_name}: {e}")
        except Exception as e:
            print(f"处理模块 {module_name} 时出错: {e}")

# 初始化时注册
_register_existing_models()