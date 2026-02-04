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
        preferred_name = pathlib.Path(reference_audio).stem
        voice = self._get_or_create_voice(
            reference_audio_path=pathlib.Path(reference_audio),
            preferred_name=preferred_name
        )
        # print("preferred_name:", preferred_name)
        # print("voice:", voice)

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

    def _get_or_create_voice(
            self,
            preferred_name: str,
            reference_audio_path: pathlib.Path,
    ):
        """
        查找或创建音色：
        - 在 voice 列表里查找 voice 字段中是否包含 preferred_name
        - 命中则返回完整 voice
        - 未命中则创建新的音色
        """
        if not preferred_name:
            raise ValueError("preferred_name 不能为空")

        # 1️⃣ 查询已有音色
        voices = self._get_voice_list(page_size=50, page_index=0)

        for item in voices:
            voice_full = item.get("voice", "")
            if preferred_name in voice_full:
                # print(
                #     f"[voice] 命中已有音色: preferred_name={preferred_name}, "
                #     f"voice={voice_full}"
                # )
                return voice_full

        # 2️⃣ 未命中 → 创建新音色
        # print(
        #     f"[voice] 未找到音色 preferred_name={preferred_name}，开始创建"
        # )

        voice_full = self._create_voice(
            reference_audio=reference_audio_path,
            preferred_name=preferred_name,
        )

        # print(
        #     f"[voice] ✅ 创建成功: preferred_name={preferred_name}, "
        #     f"voice={voice_full}"
        # )

        return voice_full

    def _delete_all_voices(self, page_size: int = 20, max_rounds: int = 100):
        """
        删除当前账号下的所有音色（带打印日志）

        - 每轮打印查询到的全部音色
        - 每个成功删除的音色都会打印提示
        """
        round_idx = 0
        deleted_count = 0

        while True:
            if round_idx >= max_rounds:
                raise RuntimeError(
                    f"[voice-clean] 超过最大轮数 {max_rounds}，可能存在异常"
                )

            print(f"\n[voice-clean] ===== 第 {round_idx + 1} 轮查询 =====")

            # 每一轮都从 page_index = 0 拉
            voice_list = self._get_voice_list(
                page_size=page_size,
                page_index=0
            )

            if not voice_list:
                print("[voice-clean] 当前已无任何音色，清理完成 ✅")
                break

            print(f"[voice-clean] 查询到 {len(voice_list)} 个音色：")
            for item in voice_list:
                print(
                    f"  - voice={item.get('voice')} | "
                    f"name={item.get('preferred_name')} | "
                    f"model={item.get('target_model')} | "
                    f"create={item.get('gmt_create')}"
                )

            # 删除本轮音色
            for item in voice_list:
                voice = item.get("voice")
                if not voice:
                    continue

                try:
                    self._delete_voice(voice)
                    deleted_count += 1
                    print(f"[voice-clean] ✅ 已删除音色: {voice}")
                except Exception as e:
                    print(f"[voice-clean] ❌ 删除音色失败: {voice}, error={e}")
                    raise

            round_idx += 1

        print(
            f"\n[voice-clean] 🎉 清理完成，总共删除音色数量: {deleted_count}"
        )

        return {
            "status": "completed",
            "deleted": deleted_count
        }

    def _create_voice(self, reference_audio, preferred_name):
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
                "preferred_name": preferred_name,
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

    def _get_voice_list(self, page_size: int = 10, page_index: int = 0):
        """
        查询已创建的音色列表

        :param page_size: 每页数量
        :param page_index: 页码，从 0 开始
        :return: voice_list (list[dict])
        """
        # 选择 API Key
        if self.region == "cn":
            api_key = self.config["CN_API_KEY"]
        else:
            api_key = self.config["INTL_API_KEY"]

        if not api_key:
            raise RuntimeError("未配置 API KEY")

        payload = {
            "model": "qwen-voice-enrollment",  # 固定值
            "input": {
                "action": "list",
                "page_size": page_size,
                "page_index": page_index
            }
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        resp = requests.post(self.customize_url, json=payload, headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(
                f"查询 voice list 失败: {resp.status_code}, {resp.text}"
            )

        try:
            data = resp.json()
            return data["output"]["voice_list"]
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"解析 voice list 响应失败: {e}")

    def _delete_voice(self, voice: str):
        """
        删除指定音色

        :param voice: 要删除的音色 ID（如 voice_xxx）
        """
        if not voice:
            raise ValueError("voice 不能为空")

        # 选择 API Key
        if self.region == "cn":
            api_key = self.config["CN_API_KEY"]
        else:
            api_key = self.config["INTL_API_KEY"]

        if not api_key:
            raise RuntimeError("未配置 API KEY")

        payload = {
            "model": "qwen-voice-enrollment",  # 固定值
            "input": {
                "action": "delete",
                "voice": voice
            }
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        resp = requests.post(self.customize_url, json=payload, headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(
                f"删除 voice 失败: {resp.status_code}, {resp.text}"
            )

        try:
            data = resp.json()
            return {
                "request_id": data.get("request_id"),
                "voice": voice,
                "status": "deleted"
            }
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"解析删除 voice 响应失败: {e}")


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
    # api._delete_all_voices()  # 删除所有音色

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



