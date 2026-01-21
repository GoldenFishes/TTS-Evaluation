'''
这里为评估TTS的入口

一般效果测试候选API入口函数： TTSEvaluator.evaluate_all()
    我们在 TTSEvaluator.single_api_evaluate() 实现单个TTS API的调用与测试

并发测试候选API入口函数： TTSEvaluator.concurrency_test()
    进行并发测试时务必设置 TTSEvaluator初始化任务为 text_task_support=["long_stream_prompt"]
    以保证获取到用于并发的单个样本本身耗时足够长。

'''
import apis  # 触发 apis.__init__ 的 auto_import_apis() ; 如果不这么做，API适配器将无法正常向router注册
import yaml
import time
import pathlib
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional, Tuple

import queue
import threading
from dataclasses import dataclass, asdict, field

from base.data_collector import TTSEvalDataCollector  # 用于获取数据集路径
from base.router import Router  # 用于返回不同API调用类的路由器（每个API调用类都继承自APIBase基类）

class TTSStreamMetrics:
    '''
    用于单次收集TTS流式生成指标类
    '''
    # 首chunk延迟与总生成时长
    ttfb: Optional[float] = None
    total_time: Optional[float] = None

    audio_duration: Optional[float] = None  # 生成音频时长
    rtf: Optional[float] = None  # Real-Time Factor 生成总耗时 / 音频总时长

@dataclass
class ConcurrencyRequestMetrics:
    '''
    用于统计并发测试中的单个请求指标
    '''
    success: bool
    # 首chunk延迟与总生成时长
    ttfb: float | None
    total_time: float | None

    audio_duration: float | None
    rtf: float | None

    error: str | None = None

@dataclass
class ConcurrencyState:
    '''
    单API并发测试的全局状态
    '''
    # ===== 生命周期控制 =====
    start_time: float = 0.0
    test_finished: bool = False  # 用于防止无报错运行完后也能正常结束
    error_occurred: bool = False  # 产生报错后停止增加并发数

    # ===== 并发统计 =====
    active_requests: int = 0
    total_started: int = 0
    total_finished: int = 0

    # ===== 音频窗口统计 =====
    window_audio_bytes: int = 0
    audio_format: Optional[Tuple[int, int, int]] = None  # (sr, ch, bd)

    # ===== 结果存储 =====
    metrics: List["ConcurrencyRequestMetrics"] = field(default_factory=list)
    chunk_stats: list[dict] = field(default_factory=list)  # 用于存储每个 chunk 的 ts, concurrency, audio_duration, rtf

    # ===== 线程同步 =====
    lock: threading.Lock = field(default_factory=threading.Lock)

class TTSEvaluator:
    def __init__(
        self,
        api_and_model_names: Dict,  # 待测试的tts api与模型名称，api名称需要与向APIBase中注册的名称一致
        output_dir: str,  # 用于保存输出结果的文件夹
        text_task_support: List[str],  # 评估涉及文本任务类型, 与data中文本提示的文件夹名一致
        voice_task_support: List[str],  # 评估涉及的音色提示类型，与data中音色提示的文件夹名一致
    ):
        '''
        api_and_model_names:
            {"api_name1" : [model_name1, model_name2, ...],
             "api_name2" : [model_name1, model_name2, ...]}
        '''
        self.api_and_model_names = api_and_model_names
        self.output_dir = output_dir

        self.router = Router()
        data_collector = TTSEvalDataCollector(
            data_path="data",
            text_task_support=text_task_support,
            voice_task_support=voice_task_support,
        )
        self.text_tasks = data_collector.get_text_prompts()  # -> Dict[task, List[TextPrompt]]
        self.voice_prompts = data_collector.get_voice_prompts()  # -> Dict[voice_type, List[VoicePrompt]]

    # -----------------------------------------------------------------------------------------------------------
    # 遍历数据,效果评估
    def evaluate_all(self):
        '''
        为每个模型生成统计报告保存于: {self.output_dir}/{api_name}_{model_name}/results_summary.yaml
            计算相关指标的平均值
        '''
        for api_name, model_name_list in self.api_and_model_names.items():
            for model_name in model_name_list:
                print(f"Evaluating API={api_name} Model={model_name}")

                # 单模型相关任务评测
                results_summary = self.single_api_evaluate(api_name, model_name)

                # 计算每个指标的平均值
                # 支持: ttfb, total_time, audio_duration, rtf
                agg_metrics = defaultdict(list)
                for res in results_summary:
                    m = res["metrics"]
                    if m.ttfb is not None:
                        agg_metrics["ttfb"].append(m.ttfb)
                    if m.total_time is not None:
                        agg_metrics["total_time"].append(m.total_time)
                    if m.audio_duration is not None:
                        agg_metrics["audio_duration"].append(m.audio_duration)
                    if m.rtf is not None:
                        agg_metrics["rtf"].append(m.rtf)

                # 计算平均值
                avg_metrics = {}
                for k, v_list in agg_metrics.items():
                    avg_metrics[k] = sum(v_list) / len(v_list) if v_list else None

                # 保存 YAML
                summary_dir = pathlib.Path(f"{self.output_dir}/{api_name}_{model_name}")
                summary_dir.mkdir(parents=True, exist_ok=True)

                summary_path = summary_dir / "results_summary.yaml"
                with open(summary_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump({
                        "api_name": api_name,
                        "model_name": model_name,
                        "avg_metrics": avg_metrics,
                        "total_samples": len(results_summary),
                        "samples": [
                            {
                                "task": r["task"],
                                "sample": r["sample"],
                                "voice": r["voice"],
                                "wav_path": r["wav_path"],
                                "metrics_path": r["metrics_path"],
                                "metrics": {
                                    "ttfb": r["metrics"].ttfb,
                                    "total_time": r["metrics"].total_time,
                                    "audio_duration": r["metrics"].audio_duration,
                                    "rtf": r["metrics"].rtf,
                                }
                            }
                            for r in results_summary
                        ]
                    }, f, allow_unicode=True)

    def single_api_evaluate(self, api_name: str, model_name: str) -> Dict:
        '''
        对单个模型的API进行评估调用，遍历适配的所有任务数据并执行调用生成.
        一个API类可以用于该API支持的所有系列模型，故我们需要同时指定我们实现的API适配器名称 api_name 和具体的模型名称 model_name

        将生成结果保存为：
            wav：{self.output_dir}/{api_name}_{model_name}/{task_name}/{sample_name}_{voice_name}.wav
            Metrics: {self.output_dir}/{api_name}_{model_name}/{task_name}/{sample_name}_{voice_name}.yaml
        '''
        # 根据名称获取实际的API调用类
        api_adapter = self.router.get_api_adapter(api_name)
        # 设置该API调用的模型名称
        api_adapter.setup_model(model_name)

        base_result_dir = pathlib.Path(f"{self.output_dir}/{api_name}_{model_name}")
        results_summary = []

        # 遍历需要评估的样本
        for task_name, txt_prompts in self.text_tasks.items():
            task_dir = base_result_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            for txt in txt_prompts:
                for voice_type, voice_list in self.voice_prompts.items():
                    for vp in voice_list:

                        # 特定任务中测试样本过滤，只测试部分样本，防止消耗太多token =====================================
                        # 长文本流式任务中仅对zhongli和furina音色进行voice clone
                        if vp.voice_name not in ["voice_ZH_zhongli", "voice_ZH_furina"] and task_name in ["long_stream"]:
                            continue

                        # emotion任务中仅对zhongli和furina音色进行voice clone
                        if vp.voice_name not in ["voice_ZH_zhongli", "voice_ZH_furina"] and task_name in ["emotion"]:
                            continue

                        # mixed-lingual_in-context任务中仅对zhongli和furina音色进行voice clone
                        if vp.voice_name not in ["voice_ZH_zhongli", "voice_ZH_furina"] and task_name in ["mixed-lingual_in-context"]:
                            continue

                        if vp.voice_name not in ["voice_EN_man1", "voice_EN_woman1", "voice_ZH_zhongli", "voice_ZH_furina"] and task_name in ["hardcase"]:
                            continue
                        # =====================================================================================

                        # 测试每个样本时：
                        print(f"Generating | Task={task_name} | Sample={txt.sample_name} | Voice={vp.voice_name}")

                        metrics = None
                        last_chunk_ts = None
                        audio_bytes = bytearray()  # 用于缓存 PCM

                        # 调用API的声音克隆 流式生成
                        for chunk, sample_rate, channels, bit_depth, call_start_ts in api_adapter.voice_clone(
                            target_text=txt.text,           # 目标文本
                            reference_audio=vp.wav_path,    # 参考音频
                            reference_text=vp.text,         # 参考音频的文本转录
                        ):
                            now = time.perf_counter()

                            # 第一个chunk
                            if metrics is None:
                                metrics = TTSStreamMetrics()
                                metrics.ttfb = now - call_start_ts
                            else:
                                # chunk间隔可以在这里扩展收集
                                pass

                            audio_bytes.extend(chunk)
                            last_chunk_ts = now

                        # 生成指标
                        if metrics and last_chunk_ts:
                            metrics.total_time = last_chunk_ts - call_start_ts
                            metrics.audio_duration = len(audio_bytes) / (sample_rate * channels * (bit_depth // 8))
                            metrics.rtf = metrics.total_time / metrics.audio_duration

                        # 保存 WAV 文件
                        wav_path = pathlib.Path(task_dir / f"{txt.sample_name}_{vp.voice_name}.wav")
                        api_adapter.pcm_data_to_wav(
                            pcm_data=audio_bytes,
                            wav_path=wav_path,
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_width=bit_depth // 8,
                        )

                        # 保存 YAML 指标
                        metrics_path = task_dir / f"{txt.sample_name}_{vp.voice_name}.yaml"
                        with open(metrics_path, "w", encoding="utf-8") as f:
                            yaml.safe_dump({
                                "ttfb": metrics.ttfb,
                                "total_time": metrics.total_time,
                                "audio_duration": metrics.audio_duration,
                                "rtf": metrics.rtf
                            }, f, allow_unicode=True)

                        # 记录到 summary
                        results_summary.append({
                            "task": task_name,
                            "sample": txt.sample_name,
                            "voice": vp.voice_name,
                            "wav_path": str(wav_path),
                            "metrics_path": str(metrics_path),
                            "metrics": metrics
                        })

                        # 避免QPM上限 非流式调用间隔2s，连续流式调用间隔60s
                        if task_name in ["long_stream"]:
                            print("..")
                            time.sleep(60)
                        else:
                            time.sleep(2)

        return results_summary

    # -----------------------------------------------------------------------------------------------------------
    # 用于测试并发数量的方法
    def concurrency_test(self):
        '''
        为每个模型生成统计报告保存于: {self.output_dir}/{api_name}_{model_name}/results_concurrency_summary.yaml
            计算并发测试的相关指标
        '''
        for api_name, model_name_list in self.api_and_model_names.items():
            for model_name in model_name_list:
                print(f"Evaluating API={api_name} Model={model_name} concurrency test")
                result = self.single_api_concurrency_test(api_name, model_name)

                out_dir = pathlib.Path(self.output_dir) / f"{api_name}_{model_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # 保存 yaml
                yaml_path = out_dir / "results_concurrency_summary.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(result, f, allow_unicode=True)

                # 画图
                self._plot_concurrency_rtf(
                    result["chunk_stats"],
                    out_dir / "concurrency_rtf.png"
                )

    def single_api_concurrency_test(self, api_name: str, model_name: str) -> Dict:
        """
        Chunk 级并发测试实现
        1. 使用一个重复样本进行 ramp-up 并发，每 0.5s 新增请求
        2. 一旦出现错误 → 停止增加并发，继续接收剩余请求
        3. 记录每个 chunk 的到达时间、RTF 和并发数
        4. 返回 summary，包括总请求数、QPM、平均 TTFB、平均 RTF、chunk_stats
        """

        # 获取 API Adapter 并设置模型
        api_adapter = self.router.get_api_adapter(api_name)
        api_adapter.setup_model(model_name)

        # 使用固定样本
        task_name, txt_prompts = next(iter(self.text_tasks.items()))
        text_sample = txt_prompts[0]
        voice_type, voice_list = next(iter(self.voice_prompts.items()))
        voice_prompt = voice_list[0]

        state = ConcurrencyState()
        state.start_time = time.perf_counter()

        ramp_interval = 0.5
        max_duration = 60.0  # 1 分钟 ramp
        controller_sleep = 0.05

        threads: list[threading.Thread] = []

        start_ts = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start_ts
            if elapsed > max_duration:
                break

            with state.lock:
                if getattr(state, "error_occurred", False):
                    break

                target_concurrency = int(elapsed / ramp_interval) + 1
                need_to_start = target_concurrency - state.active_requests

                # ===== 实时打印当前活跃线程数 =====
                print(
                    f"[{elapsed:.2f}s] Active requests: {state.active_requests}, total started: {state.total_started}"
                )

            for _ in range(need_to_start):
                t = threading.Thread(
                    target=self._run_single_request_chunk_stats,
                    args=(api_adapter, text_sample, voice_prompt, state),
                    daemon=True
                )
                t.start()
                threads.append(t)

            time.sleep(controller_sleep)

        # 等待所有请求结束
        for t in threads:
            t.join()

        with state.lock:
            state.test_finished = True

        # 汇总统计
        success_metrics = [m for m in state.metrics if m.success]
        chunk_stats = [s for s in state.chunk_stats]

        actual_duration = max((s["ts"] for s in chunk_stats), default=max_duration)

        summary = {
            "api_name": api_name,
            "model_name": model_name,
            "total_started": state.total_started,
            "total_finished": state.total_finished,
            "max_concurrency": max((s["concurrency"] for s in chunk_stats), default=0),
            "qpm": state.total_started / (actual_duration / 60),
            "avg_ttfb": sum(m.ttfb for m in success_metrics) / len(success_metrics) if success_metrics else None,
            "avg_rtf": sum(m.rtf for m in success_metrics) / len(success_metrics) if success_metrics else None,
            "error_occurred": getattr(state, "error_occurred", False),
            "chunk_stats": chunk_stats,
        }

        return summary

    def _run_single_request_chunk_stats(self, api_adapter, text_sample, voice_prompt, state: ConcurrencyState):
        """
        单个请求线程体，每个 chunk 记录：
        ts, concurrency, audio_duration, rtf
        """
        with state.lock:
            state.active_requests += 1
            state.total_started += 1

        try:
            last_chunk_ts = None
            audio_bytes = 0

            for chunk, sr, ch, bd, call_start_ts in api_adapter.voice_clone(
                    target_text=text_sample.text,
                    reference_audio=voice_prompt.wav_path,
                    reference_text=voice_prompt.text,
            ):
                now = time.perf_counter()
                audio_bytes_chunk = len(chunk)
                audio_duration_chunk = audio_bytes_chunk / (sr * ch * (bd // 8))

                # chunk 耗时 = 上一个 chunk 到当前 chunk 的时间
                chunk_elapsed = now - last_chunk_ts if last_chunk_ts else now - call_start_ts
                rtf = chunk_elapsed / audio_duration_chunk if audio_duration_chunk > 0 else None

                with state.lock:
                    state.chunk_stats.append({
                        "ts": now - state.start_time,
                        "concurrency": state.active_requests,
                        "audio_duration": audio_duration_chunk,
                        "rtf": rtf,
                    })
                    if not getattr(state, "audio_format", None):
                        state.audio_format = (sr, ch, bd)

                last_chunk_ts = now
                audio_bytes += audio_bytes_chunk

            # 请求结束生成指标
            total_time = last_chunk_ts - call_start_ts if last_chunk_ts else None
            total_audio_duration = audio_bytes / (sr * ch * (bd // 8))
            total_rtf = total_time / total_audio_duration if total_audio_duration > 0 else None

            req_metrics = ConcurrencyRequestMetrics(
                success=True,
                ttfb=state.chunk_stats[0]["ts"] if state.chunk_stats else None,
                total_time=total_time,
                audio_duration=total_audio_duration,
                rtf=total_rtf,
            )

        except Exception as e:
            print("[Exception]:\n", e)
            req_metrics = ConcurrencyRequestMetrics(
                success=False,
                ttfb=None,
                total_time=None,
                audio_duration=None,
                rtf=None,
                error=str(e),
            )
            with state.lock:
                state.error_occurred = True

        finally:
            with state.lock:
                state.metrics.append(req_metrics)
                state.active_requests -= 1
                state.total_finished += 1

    def _plot_concurrency_rtf(self, chunk_stats: list[dict], save_path: pathlib.Path):
        '''
        蓝线（左侧纵坐标与底部横坐标），随测试时间存在的当前并发数。
        红线（右侧纵坐标与底部横坐标），某一时刻下系统接收的返回chunk的平均RTF。反映在并发变化的情况下，RTF 是如何随负载劣化的。其中RTF=1时用虚线标明。

            请求 A: |----compute----| chunk |----compute----| chunk |
            请求 B:       |----compute----| chunk |----compute----|
            请求 C:                  |----compute----| chunk |

        红线与蓝线代表：在某一并发水平下，所有在该并发水平期间完成的请求的平均 RTF

        音频chunk的返回并不均匀，也不一定连续。但是我们记录chunk音频时长、上一个chunk的到达时间与这个chunk的到达时间，能够得到该chunk在这个时间段中的RTF。
        每个请求中返回的每个chunk都能从中得到一个代表某一个时间段该chunk产生的RTF指标
        通过某一时刻下可能的多个重叠的chunk，得到该时刻下平均RTF。此时RTF与应当并发数量强相关。
        '''
        import matplotlib.pyplot as plt

        ts = [s["ts"] for s in chunk_stats]
        concurrency = [s["concurrency"] for s in chunk_stats]
        rtf = [s["rtf"] for s in chunk_stats]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Concurrency")
        ax1.plot(ts, concurrency, label="Concurrency", color="blue")
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.set_ylabel("RTF")
        ax2.plot(ts, rtf, label="RTF", color="red")
        ax2.axhline(1.0, linestyle='--', color='gray')
        ax2.tick_params(axis='y')

        fig.legend(loc="upper right")
        plt.title("Concurrency vs RTF (chunk-level)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    '''
    python evaluator.py
    
    TTSEvaluator.evaluate_all() 一般效果评测，指定voice clone的任务与音色，自动遍历进行
    TTSEvaluator.concurrency_test() 并发测试，指定voice clone的任务text_task_support为["long_stream_prompt"]，使用其中用于流式生成的样本进行并发测试
    '''
    # API名称与具体模型名称的配置, 测试工具会自动遍历所有配置
    api_and_model_names = {
        "qwen": ["qwen3-tts-vc-realtime-2025-11-27"],
        "fishaudio": ["s1"],
        "minimax": ["speech-2.6-turbo", "speech-02-hd"],
        "inworld": ["inworld-tts-1-max"]
    }

    evaluator = TTSEvaluator(
        api_and_model_names=api_and_model_names,
        output_dir="result",
        text_task_support=["hardcase_prompt"],  # "emotion_prompt", "hardcase_prompt", "mixed-lingual_in-context_prompt"
        # text_task_support=["long_stream_prompt"],  # 测试流式生成时的长样本
        voice_task_support=["base_voice_prompt"],
    )

    evaluator.evaluate_all()  # 效果评测

    # evaluator.concurrency_test()  # 使用并发测试时，任务必须选择 text_task_support=["long_stream_prompt"]，保证样本生成的正常长度大于1分钟



