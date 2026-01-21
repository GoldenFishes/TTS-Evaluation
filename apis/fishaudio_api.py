'''
这里实现 fishaudio tts 的API

'''
# pip install fish-audio-sdk
from fishaudio import FishAudio
from fishaudio.types import ReferenceAudio
from fishaudio.utils import save

import time
import pathlib
from typing import Iterable, List

# TTS Evaluation的API基类
from base.api_base import APIBase


# 注册到 APIBase 注册表
@APIBase.register("fishaudio")
class FishAudioAPI(APIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 所有API Adapter都需要实现的初始化
        self.config = self.get_config("fishaudio_api_config.yaml")  # 包含 API_KEY 为key的字典
        self.model_name = None  # 当前设置的模型名称

        # 不同API调用方法的客制化变量
        self.client = None

    # 获取当前API调用器中支持的模型名称
    def get_support_model(self):
        return {
            "s1": "Fish Audio S1 旗舰型号，拥有行业领先的品质，40亿参数，0.008 WER（0.8%字错误率），最佳表演与自然，完全的情绪控制能力"
        }

    # 根据当前模型获取其支持的语种
    def get_support_language(self):
        if self.model_name in ["s1"]:
            return {'zh': 'Chinese',
                    'en': 'English',
                    'es': 'Spanish',
                    'ru': 'Russian',
                    'it': 'Italian',
                    'fr': 'French',
                    'ko': 'Korean',
                    'ja': 'Japanese',
                    'de': 'German',
                    'pt': 'Portuguese',
                    'ar': 'Arabic',
                    'nl': 'Dutch',
                    'pl': 'Polish'}
        else:
            return {}

    def get_model(self):
        return self.model_name

    def setup_model(self, model_name: str):
        if model_name not in self.get_support_model():
            raise ValueError(f"不支持的模型: {model_name}")

        # 设置模型名称
        self.model_name = model_name

        # 设置模型
        self.client = FishAudio(api_key=self.config["API_KEY"])

    def voice_clone(self, target_text: list[str], reference_audio: str, reference_text: str = None, other_args: dict = None):
        '''
        参考文档：https://docs.fish.audio/developer-guide/sdk-guide/python/websocket
        实现voice_clone

        流式返回 yield (
            chunk,              # PCM格式的音频
            sample_rate,        # 采样率
            channels,           # 声道数
            bit_depth,          # 位深
            call_start_ts,      # 开始调用时的时间戳 排除首次创建音色的时间
        )
        '''
        sample_rate = 44100
        channels = 1
        bit_depth = 16

        # 1. 创建持久语音模型
        samples = [
            (reference_audio, reference_text)
        ]
        voices = []
        texts = []
        for audio_file, transcript in samples:
            with open(audio_file, "rb") as f:
                voices.append(f.read())
            texts.append(transcript)

        voice = self.client.voices.create(
            title="Custom Voice",
            voices=voices,
            texts=texts,
            description="Voice with accurate transcripts",
            # enhance_audio_quality=True  # 启用自动音频增强以清理噪杂参考音频
        )  # 后续使用voice.id调取

        # # 1. 即时语音克隆
        # with open(reference_audio, "rb") as f:
        #     audio_file = f.read()

        try:
            call_start_ts = time.perf_counter()  # 记录正式调用voice clone生成的起始时刻
            # 2. 通过 WebSocket 流式返回 audio
            audio_stream = self.client.tts.stream_websocket(
                self._text_chunks(target_text),
                # references=[ReferenceAudio(
                #     audio=audio_file,
                #     text=reference_text,
                # )],  # 使用即时语音克隆
                reference_id=voice.id,  # 使用已创建的持久音色模型
                latency="balanced",  # Use "balanced" for real-time, "normal" for quality
                format="pcm",
                model=self.model_name,
            )

            # 流式返回音频数据
            for chunk in audio_stream:
                yield chunk, sample_rate, channels, bit_depth, call_start_ts  # 使用生成器流式返回每个音频片段

        finally:
            # 确保总是删除创建的持久语音模型
            self.client.voices.delete(voice.id)
            # pass


    def _text_chunks(self, target_text:list[str]) -> Iterable[str]:
        for chunk in target_text:
            yield chunk


if __name__ == "__main__":
    '''
    单元测试 python -m apis.fishaudio_api
    '''
    api = FishAudioAPI()
    api.setup_model("s1")

    out_pcm = pathlib.Path("result/test_out.pcm")

    with out_pcm.open("wb") as f:
        for pcm_chunk, sample_rate, channels, bit_depth, _ in api.voice_clone(
                target_text=[
                    "你好，这是一个流式语音合成测试。",
                    "我们正在验证是否可以正确返回音频数据。",
                ],
                reference_audio="data/voice_prompt/base_voice_prompt/voice_ZH_zhongli.wav",
                reference_text="茶叶各有特征，喝得多了自然有些心得。茶品好坏，闻香味便可略知一二了。"
        ):
            print(f"[TEST] got chunk: {len(pcm_chunk)} bytes")
            f.write(pcm_chunk)

    # 将写入的pcm转成wav进行验证
    wav_path = out_pcm.with_suffix(".wav")
    api.pcm_to_wav(
        out_pcm, wav_path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=bit_depth // 8,
    )

    print("[TEST] finished, pcm saved:", out_pcm)
    print("[TEST] wav generated:", wav_path)



