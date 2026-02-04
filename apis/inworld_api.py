'''
这里实现 inworld tts 的API

'''
import os
import queue
import base64
import threading
import time
import json
import wave
import io
import pathlib
import requests

# TTS Evaluation的API基类
from base.api_base import APIBase
from requests import delete


# 注册到 APIBase 注册表
@APIBase.register("inworld")
class InworldAPI(APIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 所有API Adapter都需要实现的初始化
        self.config = self.get_config("inworld_api_config.yaml")
        self.model_name = None  # 当前设置的模型名称

        # 不同API调用方法的客制化变量

    # 获取当前API调用器中支持的模型名称
    def get_support_model(self):
        return {
            "inworld-tts-1-max": "我们最强大且富有表现力的模型，更具表现力、更符合语境感知的语言，更强的多语言能力"
        }

    # 根据当前模型获取其支持的语种
    def get_support_language(self):
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
                'nl': 'Dutch',
                'pl': 'Polish'}

    def get_model(self):
        return self.model_name

    def setup_model(self, model_name: str):
        if model_name not in self.get_support_model():
            raise ValueError(f"不支持的模型: {model_name}")

        # 设置模型名称
        self.model_name = model_name



    def voice_clone(self, target_text: list[str], reference_audio: str, reference_text: str = None, other_args: dict = None):
        '''
        参考文档：https://docs.inworld.ai/docs/quickstart-tts
        https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech-websocket
        实现voice_clone

        1) 先调用 Clone Voice API 创建自定义 voiceId
        2) 再调用 Streaming TTS API 输出语音数据（PCM, 实时 yield）

        流式返回 yield (
            chunk,              # PCM格式的音频
            sample_rate,        # 采样率
            channels,           # 声道数
            bit_depth,          # 位深
            call_start_ts,      # 开始调用时的时间戳 排除首次创建音色的时间
        )
        '''
        # --- 1) 调用 Clone Voice API 创建 voice ---
        '''
        该API存在bug，无法正常删除声音，故我们首先查找声音，如果存在已创建的对应音色则不再继续创建
        '''

        audio_b64 = base64.b64encode(pathlib.Path(reference_audio).read_bytes()).decode()  # 传入引用音频base64

        filename_with_ext = os.path.basename(reference_audio)
        voice_name= os.path.splitext(filename_with_ext)[0]
        # print(f"[InWorld] 音色名称: {voice_name}")

        if "EN" in voice_name.upper():
            langCode = "EN_US"
        elif "ZH" in voice_name.upper():
            langCode = "ZH_CN"
        else:
            langCode = "AUTO"

        # 获取当前工作区中所有已创建的声音
        filtered = self.list_voices_by_language(
            api_key=self.config["API_KEY"],
            languages=[langCode]
        )
        # print(f"[DEBUG] 获取所有已创建的声音：{filtered}")

        exist_voice = next(
            (v for v in filtered if voice_name.lower() in v["voiceId"].lower()),
            None  # 如果找不到就返回 None
        )
        # print(f"[InWorld] 过滤出已存在的匹配的声音 {exist_voice}")

        if exist_voice:
            voice_id = exist_voice["voiceId"]
            # print(f"[InWorld] ✓ 找到已存在音色: {voice_id}")
        else:
            # print(f"[InWorld] 未找到音色，开始克隆...")
            clone_body = {
                "displayName": voice_name,  # 提示音频的文件名不带后缀
                "langCode": langCode,
                "voiceSamples": [
                    {
                        "audioData": audio_b64,
                        "transcription": reference_text.strip(),
                    }
                ],
            }
            r_clone = requests.post(
                "https://api.inworld.ai/voices/v1/voices:clone",
                headers=self._auth_headers(),
                json=clone_body
            )
            if r_clone.status_code != 200:
                print("Clone failed:", r_clone.text)
            r_clone.raise_for_status()
            result_clone = r_clone.json()
            # print(f"result_clone: {result_clone}")
            # 得到克隆后的 voiceId
            voice_id = result_clone["voice"]["voiceId"]
            # print(f"[InWorld] ✓ Clone成功, voice_id: {voice_id}")

        # print(f"[Debug]:{voice_id}")

        # --- 2) 调用 Streaming TTS API 开始流式合成 ---
        start = time.perf_counter()

        '''
        这里样本中每个chunk都作为一次单独请求，如果希望只使用一个请求包含所有chunk的话应当在并发测试中
        TTSEvaluator.concurrency_test()传入single_chunk_mode=True
        来控制只有一个chunk，而不是在此处直接使用 full_text = " ".join(target_text) 修改
        '''
        for text_piece in target_text:
            payload = {
                "text": text_piece,  # 单条文本
                "voiceId": voice_id,
                "modelId": self.model_name,
                "audio_config": {
                    "audio_encoding": "LINEAR16",  # PCM WAV
                    "sample_rate_hertz": 48000,
                },
            }

            response = requests.post(
                "https://api.inworld.ai/tts/v1/voice:stream",
                json=payload,
                headers=self._auth_headers(),
                stream=True
            )
            # print(f"[InWorld] 响应状态码: {response.status_code}")
            # print(f"[InWorld] 响应URL: {response.url}")
            if response.status_code != 200:
                print(f"[InWorld] ✗ 错误响应内容: {response.text[:500]}")

            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line_obj = json.loads(line)
                # 取出base64编码的音频字段
                b64_audio = line_obj.get("result", {}).get("audioContent")
                if not b64_audio:
                    continue

                raw_chunk = base64.b64decode(b64_audio)
                if len(raw_chunk) <= 44:
                    continue
                pcm_data = raw_chunk[44:]  # 跳 WAV header
                yield (
                    pcm_data,  # 原始 PCM
                    48000,  # sample_rate
                    1,  # channels
                    16,  # bit_depth
                    start
                )

        # --- 3) 删除已创建的声音
        if voice_id:
            pass
            # # FIXME:修复删除音色时总是失败的问题
            # try:
            #     self.delete_voice(self.config["API_KEY"], self.config["WORKSPACE_ID"], voice_id)
            #     # print(f"[inworld] deleted voiceId={voice_id}")
            # except Exception as e:
            #     # 不要影响上游流程
            #     print(f"[inworld][WARN] failed to delete voice {voice_id}: {e}")

    def _auth_headers(self):
        return {
            "Authorization": f"Basic {self.config["API_KEY"]}",
            "Content-Type": "application/json",
        }

    def delete_voice(self, api_key: str, voice_id: str):
        url = f"https://api.inworld.ai/voices/v1/voices/{voice_id}"
        headers = {
            "Authorization": f"Basic {api_key}"
        }
        resp = requests.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()  # 返回 {} 或成功 JSON

    def list_voices_by_language(self, api_key: str, languages: list[str]):
        """
        列出 workspace 下指定语言的声音
        """
        url = f"https://api.inworld.ai/voices/v1/voices"
        headers = {"Authorization": f"Basic {api_key}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        voices = resp.json().get("voices", [])

        # 过滤指定语言
        filtered = [v for v in voices if v.get("langCode") in languages]
        return filtered


if __name__ == "__main__":
    '''
    单元测试 python -m apis.inworld_api
    '''
    api = InworldAPI()
    api.setup_model("inworld-tts-1-max")

    out_pcm = pathlib.Path("result/test_out.pcm")

    with out_pcm.open("wb") as f:
        chunk_count = 0
        vc_start_time = time.perf_counter()
        for pcm_chunk, sample_rate, channels, bit_depth, ts in api.voice_clone(
                target_text=[
                    "晨间阅读半小时，完成健身计划。处理邮件与日程安排，推进项目A的测试阶段。晚间参加线上技术分享会，收获颇丰。",
                ],
                reference_audio="data/voice_prompt/base_voice_prompt/voice_ZH_zhongli.wav",
                reference_text="茶叶各有特征，喝得多了自然有些心得。茶品好坏，闻香味便可略知一二了。"
        ):
            current_time = time.perf_counter()
            chunk_count += 1
            elapsed = current_time - ts
            if chunk_count == 1:
                preprocess_time = ts - vc_start_time
                print(f"[TEST] 预处理时间（音色查找/创建） {preprocess_time:.4f}s")

            print(f"[TEST] Chunk #{chunk_count:02d} | "
                  f"Size: {len(pcm_chunk):5d} bytes | "
                  f"Elapsed: {elapsed:.4f}s")
            f.write(pcm_chunk)
        '''
        [TEST] 预处理时间（音色查找/创建） 0.8343s
        [TEST] Chunk #01 | Size: 216960 bytes | Elapsed: 2.4416s
        [TEST] Chunk #02 | Size: 130080 bytes | Elapsed: 2.5917s
        [TEST] Chunk #03 | Size: 212640 bytes | Elapsed: 2.7500s
        [TEST] Chunk #04 | Size: 190560 bytes | Elapsed: 2.9987s
        [TEST] Chunk #05 | Size: 274080 bytes | Elapsed: 3.6538s
        [TEST] Chunk #06 | Size: 106560 bytes | Elapsed: 3.8078s
        '''

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