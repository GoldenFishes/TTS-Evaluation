'''
这里实现 qwen3 tts 的API

'''
import pyaudio
'''
Installation instructions for pyaudio:
APPLE Mac OS X
  brew install portaudio
  pip install pyaudio
Debian/Ubuntu
  sudo apt-get install python-pyaudio python3-pyaudio
  or
  pip install pyaudio
CentOS
  sudo yum install -y portaudio portaudio-devel && pip install pyaudio
Microsoft Windows
  python -m pip install pyaudio
'''
import queue
import base64
import threading
import time
import pathlib
import requests
import dashscope  # DashScope Python SDK 版本需要不低于1.23.9
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
    AudioFormat
)

# TTS Evaluation的API基类
from base.api_base import APIBase


# 注册到 APIBase 注册表
@APIBase.register("qwen")
class QWenAPI(APIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 所有API Adapter都需要实现的初始化
        self.config = self.get_config("qwen_api_config.yaml")  # 包含 CN_API_KEY 和 INTL_API_KEY 为key的字典
        self.model_name = None  # 当前设置的模型名称

        # 不同API调用方法的客制化变量
        self.region = "cn"  # cn / intl 这里控制该API是走中国北京url还是走新加坡国际url
        self.ws_url = None
        self.customize_url = None

    # 获取当前API调用器中支持的模型名称
    def get_support_model(self):
        return {
            "qwen3-tts-vc-realtime-2025-11-27": "支持声音复刻，基于真实音频样本快速复刻音色，打造拟人化品牌声纹，确保音色高度还原与一致性"
        }

    # 根据当前模型获取其支持的语种
    def get_support_language(self):
        if self.model_name in ["qwen3-tts-vc-realtime-2025-11-27"]:
            return {'zh': 'Chinese',
                    'en': 'English',
                    'es': 'Spanish',
                    'ru': 'Russian',
                    'it': 'Italian',
                    'fr': 'French',
                    'ko': 'Korean',
                    'ja': 'Japanese',
                    'de': 'German',
                    'pt': 'Portuguese'}
        else:
            return {}

    def get_model(self):
        return self.model_name

    def setup_model(self, model_name: str):
        if model_name not in self.get_support_model():
            raise ValueError(f"不支持的模型: {model_name}")

        # 设置模型名称
        self.model_name = model_name

        # 向dashscope设置API
        if self.region == "cn":
            if not self.config["CN_API_KEY"]:
                raise RuntimeError("未配置 CN_API_KEY")
            dashscope.api_key = self.config["CN_API_KEY"]
            self.ws_url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
            self.customize_url = (
                "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
            )
        else:
            if not self.config["INTL_API_KEY"]:
                raise RuntimeError("未配置 INTL_API_KEY")
            dashscope.api_key = self.config["INTL_API_KEY"]
            self.ws_url = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
            self.customize_url = (
                "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization"
            )

        # # 初始化 qwen_tts_realtime 并连接
        # self.qwen_tts_realtime = QwenTtsRealtime(
        #     model=self.model_name,
        #     callback=self.callback,
        #     url=self.ws_url,
        # )
        # self.qwen_tts_realtime.connect()


    def voice_clone(self, target_text: list[str], reference_audio: str, reference_text: str = None, other_args: dict = None):
        '''
        参考文档：https://www.alibabacloud.com/help/zh/model-studio/qwen-tts-realtime?spm=a2c63.p38356.0.i1#6011832a3b7lc
        实现voice_clone

        流式返回 yield (
            chunk,              # PCM格式的音频
            sample_rate,        # 采样率
            channels,           # 声道数
            bit_depth,          # 位深
            call_start_ts,      # 开始调用时的时间戳 排除首次创建音色的时间
        )
        '''
        sample_rate = 24000
        channels = 1
        bit_depth = 16

        # 每个样本一个 callback
        callback = StreamCallback()

        # 初始化 qwen_tts_realtime 并连接
        qwen_tts_realtime = QwenTtsRealtime(
            model=self.model_name,
            callback=callback,
            url=self.ws_url,
        )
        qwen_tts_realtime.connect()

        # 1. 创建新的音色 根据 reference_audio
        voice = self._create_voice(pathlib.Path(reference_audio))

        # 2. 提交设置
        qwen_tts_realtime.update_session(
            voice=voice, # 将voice参数替换为复刻生成的专属音色
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode='server_commit'
        )

        call_start_ts = time.perf_counter()  # 记录正式调用voice clone生成的起始时刻
        # 3. 发送文本
        for text_chunk in target_text:
            qwen_tts_realtime.append_text(text_chunk)
            time.sleep(0.05)

        # 4. 流式返回结果
        qwen_tts_realtime.finish()

        while True:
            try:
                audio_chunk = callback.audio_queue.get(timeout=0.1)
                yield audio_chunk, sample_rate, channels, bit_depth, call_start_ts
            except queue.Empty:
                if callback.complete_event.is_set():
                    break

        if callback.error:
            raise callback.error


    def _create_voice(self, reference_audio):
        '''
        创建音色，并返回 voice 参数
        '''
        # 解码音频
        base64_str = base64.b64encode(reference_audio.read_bytes()).decode()
        data_uri = f"data:audio/wav;base64,{base64_str}"

        # 当前API Key
        if self.region == "cn":
            api_key = self.config["CN_API_KEY"]
        else:
            api_key = self.config["INTL_API_KEY"]

        # 创建请求
        payload = {
            "model": "qwen-voice-enrollment",  # 不要修改该值
            "input": {
                "action": "create",
                "target_model": self.model_name,
                "preferred_name": "new_voice",  # TODO 这里应该是暂时随便给当前音色起个名字
                "audio": {"data": data_uri}
            }
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(self.customize_url, json=payload, headers=headers)

        # 请求结果
        if resp.status_code != 200:
            raise RuntimeError(f"创建 voice 失败: {resp.status_code}, {resp.text}")

        try:
            return resp.json()["output"]["voice"]
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"解析 voice 响应失败: {e}")



# ======= 回调类 =======
# 完全复制于文档https://www.alibabacloud.com/help/zh/model-studio/qwen-tts-realtime?spm=a2c63.p38356.0.i1#6011832a3b7lc
# 中 实时语音合成-通义千问 的 使用声音复刻音色进行语音合成
class MyCallback(QwenTtsRealtimeCallback):
    """
    自定义 TTS 流式回调
    """
    def __init__(self):
        self.complete_event = threading.Event()
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )

    def on_open(self) -> None:
        print('[TTS] 连接已建立')

    def on_close(self, close_status_code, close_msg) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()
        print(f'[TTS] 连接关闭 code={close_status_code}, msg={close_msg}')

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get('type', '')
            if event_type == 'session.created':
                print(f'[TTS] 会话开始: {response["session"]["id"]}')
            elif event_type == 'response.audio.delta':
                audio_data = base64.b64decode(response['delta'])
                self._stream.write(audio_data)
            elif event_type == 'response.done':
                print(f'[TTS] 响应完成')
            elif event_type == 'session.finished':
                print('[TTS] 会话结束')
                self.complete_event.set()
        except Exception as e:
            print(f'[Error] 处理回调事件异常: {e}')

    def wait_for_finished(self):
        self.complete_event.wait()


class StreamCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.complete_event = threading.Event()
        self.error = None

    def on_open(self):
        # print('[TTS] 连接已建立')
        pass

    def on_close(self, close_status_code, close_msg):
        # print(f'[TTS] 连接关闭 code={close_status_code}, msg={close_msg}')
        self.complete_event.set()

    def on_event(self, response: dict):
        try:
            event_type = response.get("type", "")

            if event_type == "response.audio.delta":
                audio_bytes = base64.b64decode(response["delta"])
                self.audio_queue.put(audio_bytes)

            elif event_type == "session.finished":
                self.complete_event.set()

        except Exception as e:
            self.error = e
            self.complete_event.set()

if __name__ == "__main__":
    '''
    单元测试 python -m apis.qwen_api
    '''
    api = QWenAPI()
    api.setup_model("qwen3-tts-vc-realtime-2025-11-27")

    out_pcm = pathlib.Path("result/test_out.pcm")

    with out_pcm.open("wb") as f:
        for pcm_chunk, sample_rate, channels, bit_depth, _ in api.voice_clone(
                target_text=[
                    "你好，这是一个流式语音合成测试。",
                    "我们正在验证是否可以正确返回音频数据。",
                ],
                reference_audio="data/voice_prompt/base_voice_prompt/voice_ZH_zhongli.wav",
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



