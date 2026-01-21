import os
import importlib

# 自动导入当前目录下的所有API适配器，确保所有API适配器触发向APIBase注册
def auto_import_apis():
    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"apis.{filename[:-3]}"
            importlib.import_module(module_name)

# 在模块导入时自动触发
auto_import_apis()


