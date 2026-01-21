'''
这里实现一个Router类:
Router类根据api_name。访问APIBase的注册表_registry，获取对应API调用类。
并返回实例化后的调用器类。
'''

from base.api_base import APIBase

class Router:

    @staticmethod
    def get_api_adapter(api_name: str) -> APIBase:
        """
        根据 api_name 返回对应的API Adapter实例
        """
        print("[DEBUG][Router] 注册表：", APIBase._registry)
        api_class = APIBase._registry.get(api_name)

        if not api_class:
            raise ValueError(f"未找到对应的API调用器 api_name: {api_name}")
        return api_class()