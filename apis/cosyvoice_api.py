'''
!! 此cosyvoice_api没有以最新标准实现，暂时无法使用

'''

import os
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback

from base.api_base import APIBase


# 注册 CosyVoice 到 APIBase 注册表
@APIBase.register("cosyvoice")
class CosyVoiceAPI(APIBase):
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            dashscope.api_key = api_key
        elif os.getenv("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        else:
            raise ValueError("必须提供 api_key 或设置 DASHSCOPE_API_KEY 环境变量")

        self.model_name = "cosyvoice-v3-flash"  # 默认模型

    # ✅ 实现抽象方法：voice_clone
    def voice_clone(self,
                    target_text: str,
                    reference_audio: str,
                    reference_text: str = None,
                    other_args: dict = None) -> Dict[str, Any]:
        """
        使用 CosyVoice 的语音克隆功能（复刻音色）生成语音。
        注意：CosyVoice 的复刻音色需要预先通过声音复刻接口创建，并使用对应的 voice 名称。

        Args:
            target_text: 要合成的目标文本
            reference_audio: 参考音频文件路径（用于复刻音色，需预先上传并注册）
            reference_text: 参考音频对应的文本（可选）
            other_args: 其他控制参数，如 voice, format, sample_rate 等

        Returns:
            包含音频文件路径、请求ID、耗时等信息的字典
        """
        other_args = other_args or {}

        # 默认使用复刻音色（前提是你已经在阿里云平台上创建了复刻音色）
        voice = other_args.get("voice") or "longxiaoyu"  # 或你上传的复刻音色名称
        output_file = other_args.get("output_file", "output.mp3")
        format = other_args.get("format", "mp3")
        sample_rate = other_args.get("sample_rate", 22050)

        callback = CosyVoiceCallback(output_file)

        try:
            synthesizer = SpeechSynthesizer(
                model=self.model_name,
                voice=voice,
                callback=callback,
                additional_params=other_args.get("additional_params")
            )

            start = time.time()
            synthesizer.call(target_text)
            synthesizer.wait_completed()
            callback.save_audio()

            return {
                "success": True,
                "audio_file": output_file,
                "model": self.model_name,
                "voice": voice,
                "text": target_text,
                "duration_ms": (time.time() - start) * 1000,
                "request_id": synthesizer.get_last_request_id()
            }

        except Exception as e:
            logger.exception("CosyVoice voice_clone 失败")
            return {"success": False, "error": str(e)}

    # ✅ 实现抽象方法：setup_model
    def setup_model(self, model_name: str):
        """
        设置当前使用的模型版本。
        可选值: cosyvoice-v1, cosyvoice-v2, cosyvoice-v3-flash, cosyvoice-v3-plus
        """
        supported_models = [
            "cosyvoice-v1",
            "cosyvoice-v2",
            "cosyvoice-v3-flash",
            "cosyvoice-v3-plus"
        ]
        if model_name not in supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        self.model_name = model_name
        logger.info(f"模型切换为: {model_name}")

    # ✅ 实现抽象方法：get_model
    def get_model(self) -> str:
        return self.model_name

    # ✅ 实现抽象方法：get_support_language
    def get_support_language(self) -> List[str]:
        """
        返回当前模型支持的语言列表。
        实际上 CosyVoice 支持语言提示，但这里我们返回常见语言。
        """
        return ["zh", "en", "fr", "de", "ja", "ko", "ru"]


# ✅ 辅助类：处理回调和音频保存
class CosyVoiceCallback(ResultCallback):
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.audio_data = bytearray()

    def on_open(self):
        logger.info("CosyVoice 合成连接已建立")

    def on_data(self, data: bytes) -> None:
        self.audio_data.extend(data)

    def on_complete(self):
        logger.info("CosyVoice 合成完成")

    def on_error(self, message: str):
        logger.error(f"CosyVoice 合成出错: {message}")

    def on_close(self):
        with open(self.output_file, "wb") as f:
            f.write(self.audio_data)
        logger.info(f"音频已保存至: {self.output_file}")




