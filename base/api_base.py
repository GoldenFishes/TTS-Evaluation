'''
定义API Base基础类，所有的实际tts api adapter都继承自改基类
Router类通过api的str返回一个具体api适配器，这个适配器具备api基础类的通用实现方法
'''
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union

import wave
import pathlib

class APIBase(ABC):
    '''
    一个抽象的基类API Base，所有具体的 API调用器 继承此类，并实现voice_clone方法
    用于维护API调用器的注册表，并实现一些基础通用方法

    在 APIBase 基类中维护注册表，并通过API名称注册子类，可以实现动态路由和解耦设计，
    好处是无需每次新增一个API调用器，就去修改Router类的代码，添加新的条件分支，而只需添加新的API适配器类并注册，不需要修改现有路由逻辑

    通过register方法注册API适配器类，register方法接受一个参数，api_name，表示API适配器名称
    使用方法：@APIBase.register("minimax")

    路由器Routor会通过这个注册表来查找并返回对应的调用类
    '''
    # 注册表：键为 api_name，值为对应的API调用类
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, api_name: str):
        """显式注册执行器类（替代装饰器）"""

        def wrapper(subclass: type):
            cls._registry[api_name] = subclass
            return subclass

        return wrapper

    @abstractmethod
    def voice_clone(self, target_text: List[str], reference_audio: str, reference_text: str = None, other_args: dict = None):
        """
        由子类必须实现的具体voice_clone方法

        其中：
        - target_text: 切分好的要生成的目标文本列表
        - reference_audio: Voice Clone 的参考音色的音频路径
        - reference_text: 参考音频对应的文本转录（可传空，部分TTS模型不需要这个）
        - other_args: 字典形式包裹的其他特定参数，用于控制不同TTS模型的

        流式返回 yield (
            chunk,              # PCM格式的音频
            sample_rate,        # 采样率
            channels,           # 声道数
            bit_depth,          # 位深
            call_start_ts,      # 开始调用时的时间戳 排除首次创建音色的时间 时间戳必须以time.perf_counter()
        )
        """
        raise NotImplementedError("须在子类具体APIAdapter中实现该方法！")

    @abstractmethod
    def setup_model(self, model_name: str):
        """
        在API调用器中设置/指定具体模型名称
        不返回
        """
        raise NotImplementedError("须在子类具体APIAdapter中实现该方法！")

    @abstractmethod
    def get_support_model(self):
        """
        获取当前API调用器中支持的模型名称
        返回：Dict[key为模型名称,value为模型描述]
        """
        raise NotImplementedError("须在子类具体APIAdapter中实现该方法！")

    @abstractmethod
    def get_support_language(self) -> List[str]:
        """
        获取当前API调用器，当前设置的模型支持的语言
        返回：Dict[Key为支持的语言选项（可能缩写）,value为该语言缩写的全称]
        """
        raise NotImplementedError("须在子类具体APIAdapter中实现该方法！")


    # 上：基础方法
    # --------------------------------------------------------------------------------------------
    # 下：一些通用工具方法
    def get_model(self):
        """
        获取当前API调用器中的模型名称
        返回：Str
        """
        return self.model_name

    def get_config(self, config_name: str) -> Dict[str, Any]:
        '''
        读取根目录apis文件夹下config_name的yaml文件
        '''
        # 确保文件名有.yaml后缀
        if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
            config_name = f"{config_name}.yaml"

        config_path = f"apis/{config_name}"
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        return config_data

    # 将pcm文件生成wav文件
    def pcm_to_wav(
        self,
        pcm_path: pathlib.Path,
        wav_path: pathlib.Path,
        sample_rate=24000,
        channels=1,
        sample_width=2,  # 16-bit = 2 bytes
    ):
        with pcm_path.open("rb") as pcm_f:
            pcm_data = pcm_f.read()

        with wave.open(str(wav_path), "wb") as wav_f:
            wav_f.setnchannels(channels)
            wav_f.setsampwidth(sample_width)
            wav_f.setframerate(sample_rate)
            wav_f.writeframes(pcm_data)

    # 将pcm数据生成wav文件
    def pcm_data_to_wav(
        self,
        pcm_data,
        wav_path: pathlib.Path,
        sample_rate=24000,
        channels=1,
        sample_width=2,  # 16-bit = 2 bytes
    ):
        with wave.open(str(wav_path), "wb") as wav_f:
            wav_f.setnchannels(channels)
            wav_f.setsampwidth(sample_width)
            wav_f.setframerate(sample_rate)
            wav_f.writeframes(pcm_data)
