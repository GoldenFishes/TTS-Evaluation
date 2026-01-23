'''
这里实现 minimax tts 的API
注：minimax海外和国内分开，国内API用国内的URL并且按照国内文档进行
'''
import os
import uuid
import time
import soundfile as sf
import asyncio
import websockets
import websocket
from websockets.exceptions import ConnectionClosedOK, ConnectionClosed
import json
import ssl
import subprocess
import pathlib
from typing import Iterable, List
import requests

import torch, torchaudio, math

# TTS Evaluation的API基类
from base.api_base import APIBase


# 注册到 APIBase 注册表
@APIBase.register("minimax")
class MiniMaxAPI(APIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 所有API Adapter都需要实现的初始化
        self.config = self.get_config("minimax_api_config.yaml")  # 包含 API_KEY 和 GROUP_ID 为key的字典
        self.model_name = None  # 当前设置的模型名称

        # 不同API调用方法的客制化变量
        self.base_url = None


    # 获取当前API调用器中支持的模型名称
    def get_support_model(self):
        return {
            "speech-2.6-hd": "超低延迟、智能解析和增强的自然度。",
            "speech-2.6-turbo": "速度更快、价格更实惠，是您经纪人的理想之选。",
            "speech-02-hd": "节奏和稳定性极佳，在复制相似性和音质方面表现出色。",
            "speech-02-turbo": "节奏和稳定性极佳，多语言功能增强，性能卓越。"
        }

    # 根据当前模型获取其支持的语种
    def get_support_language(self):
        # 所有模型支持语言相同
        return {'zh': 'Chinese',
                'yue': 'Cantonese', # 粤语常用 yue
                'en': 'English',
                'es': 'Spanish',
                'fr': 'French',
                'ru': 'Russian',
                'de': 'German',
                'pt': 'Portuguese',
                'ar': 'Arabic',
                'it': 'Italian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'id': 'Indonesian',
                'vi': 'Vietnamese',
                'tr': 'Turkish',
                'nl': 'Dutch',
                'uk': 'Ukrainian',
                'th': 'Thai',
                'pl': 'Polish',
                'ro': 'Romanian',
                'el': 'Greek',
                'cs': 'Czech',
                'fi': 'Finnish',
                'hi': 'Hindi',
                'bg': 'Bulgarian',
                'da': 'Danish',
                'he': 'Hebrew', # 希伯来语
                'ms': 'Malay', # 马来语
                'fa': 'Persian', # 波斯语
                'sk': 'Slovak', # 斯洛伐克语
                'sv': 'Swedish', # 瑞典语
                'hr': 'Croatian', # 克罗地亚语
                'tl': 'Filipino', # 菲律宾语 (Tagalog)
                'hu': 'Hungarian', # 匈牙利语
                'no': 'Norwegian', # 挪威语
                'sl': 'Slovenian', # 斯洛文尼亚语
                'ca': 'Catalan', # 加泰罗尼亚语
                'nn': 'Nynorsk', # 新挪威语
                'ta': 'Tamil', # 泰米尔语
                'af': 'Afrikaans'} # 南非荷兰语

    def get_model(self):
        return self.model_name

    def setup_model(self, model_name: str):
        if model_name not in self.get_support_model():
            raise ValueError(f"不支持的模型: {model_name}")

        # 设置模型名称
        self.model_name = model_name

    # FIXME:调用结束的时候有一点小bug，之前批量跑实验正常，后面为了改其他的问题好像把调用结束的部分改坏了。反正暂时没有继续测试该API所以暂时没管。
    def voice_clone(self, target_text: list[str], reference_audio: str, reference_text: str = None, other_args: dict = None):
        '''
        参考文档 https://platform.minimaxi.com/docs/guides/speech-voice-clone 创建音色
        参考文档 https://platform.minimax.com/docs/guides/speech-t2a-websocket 调用音色进行流式生成
        实现voice_clone

        注：MiniMax音色克隆要求参考音频大于10秒，我们的所有测试数据音频均不满足，故我们需要延长该音频至大于10s

        流式返回 yield (
            chunk,              # PCM格式的音频
            sample_rate,        # 采样率
            channels,           # 声道数
            bit_depth,          # 位深
            call_start_ts,      # 开始调用时的时间戳 排除首次创建音色的时间
        )
        '''
        upload_url = "https://api.minimaxi.com/v1/files/upload"
        clone_url = "https://api.minimaxi.com/v1/voice_clone"
        get_voice_url = "https://api.minimaxi.com/v1/get_voice"
        delete_voice_url = "https://api.minimaxi.com/v1/delete_voice"
        ws_url = "wss://api.minimaxi.com/ws/v1/t2a_v2"
        headers = {"Authorization": f"Bearer {self.config["API_KEY"]}"}

        '''
        我们首先查找声音，如果存在已创建的对应音色则不再继续创建
        '''
        filename_with_ext = os.path.basename(reference_audio)
        voice_name = os.path.splitext(filename_with_ext)[0]
        voice_id = voice_name
        # print("voice_id =", voice_id)

        # self.delete_all_flies()  # 清理所有文件

        response = requests.post(
            get_voice_url,
            headers={
                "Authorization": f"Bearer {self.config["API_KEY"]}",
                "Content-Type": "application/json"
            },
            json={
                "voice_type": "voice_cloning"  # 只查询克隆音色
            }
        )
        response.raise_for_status()
        voices = response.json().get("voice_cloning", [])
        # print("voices",voices)

        # self._delete_all_voices(voices, delete_voice_url)  # 删除所有音色

        exist_voice = any(v.get("voice_id") == voice_id for v in voices)
        # print("exist_voice",exist_voice)

        # 如果未创建则创建，否则直接使用已创建的
        if exist_voice:
            voice_id = voice_id
        else:
            print(f"[创建音色]：{voice_id}")
            # 1. 读取 reference_audio wav，延长至 ≥10秒。 同步延长 reference_text 至相应倍数。
            # MiniMax音色克隆要求参考音频大于10秒，我们的所有测试数据音频均不满足，故我们需要延长该音频至大于10s
            wav, sr = self.load_wav(reference_audio)
            extended_wav, reps = self.repeat_with_crossfade(wav, sr, target_sec=10.0)
            # 同步复制文本
            reference_text = reference_text * reps
            temp_wav = f"temp/{voice_id}.wav"
            self.save_wav(extended_wav, temp_wav, sr)

            # 2. 创建音色
            # （可选）上传该音色的更多示例 见 https://platform.minimaxi.com/docs/guides/speech-voice-clone

            # 上传文件
            with open(temp_wav, "rb") as f:  # 传入延长后的音色参考文件
                response = requests.post(
                    upload_url,
                    headers=headers,
                    data={"purpose": "voice_clone"},
                    files={"file": (os.path.basename(reference_audio), f)},
                )
            response.raise_for_status()
            file_id = response.json().get("file", {}).get("file_id")
            # print("file_id", file_id)

            # 克隆音色
            clone_payload = {
                "file_id": file_id,
                "voice_id": voice_id,
                # "clone_prompt": {
                #     "prompt_audio": prompt_file_id,
                #     "prompt_text": "后来认为啊，是有人抓这鸡，可是抓鸡的地方呢没人听过鸡叫。"
                # },
                "text": reference_text,
                "model": self.model_name
            }
            clone_headers = {
                "Authorization": f"Bearer {self.config["API_KEY"]}",
                "Content-Type": "application/json"
            }
            response = requests.post(clone_url, headers=clone_headers, json=clone_payload)
            # print("create clone voice:",response.text)

            # 清楚保存的临时音频
            os.remove(temp_wav)

        # 3. 声音克隆
        # 同步调用声音克隆
        for chunk_info in self._stream_tts_sync(ws_url, headers, voice_id, target_text):
            yield chunk_info

        # # 同步调用异步生成器
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # agen = self._stream_tts(ws_url, headers, voice_id, target_text)
        # try:
        #     while True:
        #         chunk_info = loop.run_until_complete(agen.__anext__())
        #         yield chunk_info
        # except StopAsyncIteration:
        #     loop.close()

    # ---------------- 内部函数 ----------------
    def _stream_tts_sync(self, ws_url, headers, voice_id, target_text):
        # 固定参数
        sample_rate = 32000
        channels = 1
        bit_depth = 16

        ws = websocket.create_connection(
            ws_url,
            header=[f"{k}: {v}" for k, v in headers.items()],
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )

        # print("[DEBUG] WebSocket connected")

        try:
            # ----------------- 建立连接 -----------------
            while True:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                if msg.get("event") == "connected_success":
                    # print("[DEBUG] connected_success received")
                    break

            # ----------------- 发送 task_start -----------------
            ws.send(json.dumps({
                "event": "task_start",
                "model": self.model_name,
                "voice_setting": {"voice_id": voice_id, "speed": 1, "vol": 1, "pitch": 0},
                "audio_setting": {
                    "sample_rate": sample_rate,  # 可选 [8000，16000，22050，24000，32000，44100]
                    # "bitrate": 128000  # 可选 [32000，64000，128000，256000]，仅对mp3格式音频生效
                    "format": "pcm",
                    "channel": channels
                }
            }))
            # print("[DEBUG] task_start sent")

            # 等待 task_started
            while True:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except:
                    continue
                if msg.get("event") == "task_started":
                    call_start_ts = time.perf_counter()
                    print("[DEBUG] task_started received")
                    break
                elif msg.get("event") == "task_failed":
                    raise RuntimeError("task_failed during start")

            # ----------------- 发送文本 & 接收 chunk -----------------
            for txt in target_text:
                print(f"[DEBUG] sending text: {txt}")
                ws.send(json.dumps({"event": "task_continue", "text": txt}))

                while True:
                    raw = ws.recv()
                    if not raw:
                        continue
                    try:
                        msg = json.loads(raw)
                        # handle task_failed
                        if msg.get("event") == "task_failed":
                            error_msg = msg.get("base_resp", {}).get("status_msg", "unknown error")
                            print(f"[ERROR] task_failed: {error_msg}")
                            return
                    except Exception as e:
                        print(f"[DEBUG] task failed: {e}")
                        continue

                    # chunk 音频
                    if "data" in msg and "audio" in msg["data"]:
                        pcm_chunk = bytes.fromhex(msg["data"]["audio"])
                        yield pcm_chunk, sample_rate, channels, bit_depth, call_start_ts

                    # 文本生成完成
                    if msg.get("is_final"):
                        print("[DEBUG] text chunk finished")
                        break

            # ----------------- 完成任务 -----------------
            ws.send(json.dumps({"event": "task_finish"}))
            print("[DEBUG] task_finish sent")

            while True:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except:
                    continue
                if msg.get("event") == "task_finished":
                    print("[DEBUG] task_finished received")
                    return

        except websocket.WebSocketConnectionClosedException:
            # 服务端主动关闭连接，正常结束
            return

        finally:
            ws.close()
            print("[DEBUG] WebSocket closed")

    async def _stream_tts(
        self,
        ws_url,
        headers,
        voice_id,
        target_text,
    ):
        '''
        websockets连接文档：https://platform.minimaxi.com/docs/api-reference/speech-t2a-websocket
        '''
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        ws = await websockets.connect(ws_url, additional_headers=headers, ssl=ssl_context)

        try:
            # 1. 连接成功
            connected = json.loads(await ws.recv())
            if connected.get("event") != "connected_success":
                await ws.close()
                raise RuntimeError("WebSocket connect failed")

            # 2. 任务开始
            start_msg = {
                "event": "task_start",
                "model": self.model_name,
                "voice_setting": {"voice_id": voice_id, "speed": 1, "vol": 1, "pitch": 0},
                "audio_setting": {
                    "sample_rate": 32000,  # 可选 [8000，16000，22050，24000，32000，44100]
                    # "bitrate": 128000  # 可选 [32000，64000，128000，256000]，仅对mp3格式音频生效
                    "format": "pcm",
                    "channel": 1
                }
            }
            await ws.send(json.dumps(start_msg))

            resp = json.loads(await ws.recv())
            if resp.get("event") != "task_started":
                await ws.close()
                raise RuntimeError("TTS task start failed")

            call_start_ts = time.perf_counter()

            # 3. 顺序发送多个task_continue事件，发送要合成的文本
            for txt in target_text:
                # await asyncio.sleep(0.05)  # ✅ 不阻塞 event loop
                await ws.send(json.dumps({"event": "task_continue", "text": txt}))

                while True:
                    try:
                        raw = await ws.recv()
                    except ConnectionClosedOK:
                        # 服务端正常关闭，直接结束
                        return
                    except ConnectionClosed as e:
                        # 非正常关闭，抛出
                        raise

                    msg = json.loads(raw)

                    if "data" in msg and "audio" in msg["data"]:
                        pcm_chunk = bytes.fromhex(msg["data"]["audio"])
                        yield pcm_chunk, 32000, 1, 16, call_start_ts

                    if msg.get("is_final"):
                        break

            # 4. 正常结束
            try:
                await ws.send(json.dumps({"event": "task_finish"}))
            except ConnectionClosed:
                pass

        finally:
            await ws.close()

    # 延长音频
    def repeat_with_crossfade(self, wav, sr, target_sec=10.0, fade_ms=5):
        """
        wav: [channels, samples]

        返回 (extended_wav, reps)
            extended_wav: 整句重复 reps 次、接缝交叉淡入淡出后的音频
            reps: 实际复制倍数（≥1 的整数）
        """
        ch, n = wav.shape
        dur = n / sr
        reps = math.ceil(target_sec / dur)  # 最少需要几次才能≥target_sec
        fade_samples = int(fade_ms / 1000 * sr)
        fade = torch.linspace(0, 1, fade_samples)

        out = torch.zeros(ch, 0)
        for i in range(reps):
            chunk = wav.clone()
            if i > 0:  # 交叉淡入淡出
                chunk[:, :fade_samples] *= fade.flip(0)
                out[:, -fade_samples:] *= fade
            out = torch.cat([out, chunk], dim=1)

        return out, reps

    def load_wav(self, path: str):
        """
        读取 WAV 文件
        返回：
            wav: Tensor [channels, samples], float32 [-1, 1]
            sr: 采样率
        """
        data, sr = sf.read(path, dtype='float32')  # shape: [samples, channels] 或 [samples] 单声道
        if data.ndim == 1:
            data = data[:, None]  # [samples, 1]
        data = torch.from_numpy(data.T)  # 转置为 [channels, samples]
        return data, sr

    def save_wav(self, wav: torch.Tensor, path: str, sr: int):
        """
        保存 WAV 文件
        wav: [channels, samples], float32 [-1,1]
        """
        wav_np = wav.detach().cpu().numpy().T  # 转为 [samples, channels]
        sf.write(path, wav_np, sr, subtype='PCM_16')
        # print(f"Saved WAV to {path}")

    def _delete_all_voices(self, voices, delete_voice_url):
        print("删除所有创建的音色中...")
        for voice in voices:
            voice_id = voice.get("voice_id")
            response = requests.post(
                delete_voice_url,
                headers={
                    "Authorization": f"Bearer {self.config["API_KEY"]}",
                    "Content-Type": "application/json"
                },
                json={
                    "voice_type": "voice_cloning",
                    "voice_id": voice.get("voice_id")
                }
            )  # 删除音色
            response.raise_for_status()
            resp_json = response.json()
            if resp_json.get("base_resp", {}).get("status_code") == 0:
                print(f"Voice {voice_id} deleted successfully")
                pass
            else:
                print(f"Failed to delete voice {voice_id}: {resp_json}")
            time.sleep(0.1)

    def delete_all_flies(self):
        '''
        FIXME: 无法删除文件，返回报错 {'file_id': 0, 'base_resp': {'status_code': 1000, 'status_msg': 'unknown error'}}
        '''
        response = requests.get(
            "https://api.minimaxi.com/v1/files/list",
            headers={"Authorization": f"Bearer {self.config["API_KEY"]}"}
        )
        response.raise_for_status()
        for file_info in response.json().get("files", []):
            file_id = file_info.get("file_id")
            purpose = file_info.get("purpose")
            # 删除该文件
            print(f"[Delete FLIES] file_id: {int(file_id)} purpose: {purpose}]")
            del_resp = requests.post(
                "https://api.minimaxi.com/v1/files/delete",
                headers={
                    "Authorization": f"Bearer {self.config['API_KEY']}",
                    "Content-Type": "multipart/form-data"
                },
                data=json.dumps({
                  "file_id": int(file_id),
                  "purpose": str(purpose)
                })
            )
            del_resp.raise_for_status()
            resp_json = del_resp.json()
            if resp_json.get("base_resp", {}).get("status_code") == 0:
                print(f"File {file_id} deleted successfully")
            else:
                print(f"Failed to delete file {file_id}: {resp_json}")
            time.sleep(0.1)


if __name__ == "__main__":
    '''
    单元测试 python -m apis.minimax_api
    '''
    api = MiniMaxAPI()
    api.setup_model("speech-2.6-hd")

    out_pcm = pathlib.Path("result/test_out.pcm")

    with out_pcm.open("wb") as f:
        for pcm_chunk, sample_rate, channels, bit_depth, ts in api.voice_clone(
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






