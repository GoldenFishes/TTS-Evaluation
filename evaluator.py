'''
这里为评估TTS的入口

一般效果测试候选API入口函数： TTSEvaluator.evaluate_all()
    我们在 TTSEvaluator.single_api_evaluate() 实现单个TTS API的调用与测试

并发测试候选API入口函数： TTSEvaluator.concurrency_test()
    进行并发测试时务必设置 TTSEvaluator初始化任务为 text_task_support=["long_stream_prompt"]
    以保证获取到用于并发的单个样本本身耗时足够长。

QPM测试候选API入口函数： TTSEvaluator.qpm_test()
    QPM测试时务必设置 TTSEvaluator初始化任务为 text_task_support=["short_prompt"]
    以保证获取到用于QPM测试的单个样本本身耗时足够短。

稳定性测试候选API入口函数： TTSEvaluator.stability_test()
    稳定性测试是设置 TTSEvaluator初始化任务为 text_task_support=["neutral_prompt"]
    使用自然样本进行测试
    - 确保传入的 concurrency 和 qpm 是小于前面实验测出的最大负载上限的
    - 设置allow_overlap模式:
        True为允许上一轮请求未结束的时候进行下一轮请求，优先满足设置的QPM，此时长样本可能会出现同时活跃数>concurrency
        False为不允许请求重叠，优先满足同时活跃数≤concurrency，此时长样本可能会出现实际运行时每分钟没有发送足够QPM设置值的请求
    - 设置测试时长，建议先80s预测试，正式测试180s

'''
import apis  # 触发 apis.__init__ 的 auto_import_apis() ; 如果不这么做，API适配器将无法正常向router注册
import yaml
import time
import copy
import pathlib
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional, Tuple

import queue
import threading
from dataclasses import dataclass, asdict, field

from base.data_collector import TTSEvalDataCollector  # 用于获取数据集路径
from base.router import Router  # 用于返回不同API调用类的路由器（每个API调用类都继承自APIBase基类）

import matplotlib.pyplot as plt

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
    # 记录完整生命周期
    start_ts: float | None  # call_start_ts - state.start_time,
    end_ts: float | None  # last_chunk_ts - state.start_time,

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
    stop_ramp: bool = False  # 产生报错后 冻结 ramp-up

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
    def concurrency_test(self, single_chunk_mode):
        '''
        并发能力测试主入口（按 API × Model 维度执行）

        功能：
        1. 遍历所有 API / Model 组合
        2. 对每个组合执行一次 chunk 级并发 ramp-up 测试
        3. 将统计结果保存为 YAML
        4. 生成样本并发数 vs RTF 的可视化图表

        输出目录结构：
        {output_dir}/{api_name}_{model_name}/
            ├── results_concurrency_summary.yaml
            └── concurrency_rtf.png
        '''
        for api_name, model_name_list in self.api_and_model_names.items():
            for model_name in model_name_list:
                print(f"Evaluating API={api_name} Model={model_name} concurrency test")

                # 执行单 API + 单模型的并发测试
                summary, success_metrics = self.single_api_concurrency_test(api_name, model_name, single_chunk_mode)

                out_dir = pathlib.Path(self.output_dir) / f"{api_name}_{model_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # 保存 yaml
                yaml_path = out_dir / "results_concurrency_summary.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(summary, f, allow_unicode=True)

                # 画图
                self._plot_concurrency_rtf_v2(
                    summary["chunk_stats"],
                    success_metrics,
                    out_dir / "concurrency_rtf.png"
                )

    def single_api_concurrency_test(self, api_name: str, model_name: str, single_chunk_mode: bool = False) -> Dict:
        """
        单 API + 单 Model 的并发 ramp-up 测试（chunk 级）

        single_chunk_mode=False:
            测试流式输入长样本（单样本22个chunk，样本平均总耗时30s+）时，测试的时最大样本并发数（一个样本可能多个请求）
        single_chunk_mode=True:
            测试非流式输入长样本（单样本仅1个chunk，样本平均总耗时30s+）时，测试的是最大请求并发数

        最大样本并发数：
            1. 使用固定文本 + 固定参考音频，避免输入变量影响结果
            2. 采用 ramp-up 策略：
               - 每 ramp_interval 秒增加 1 个新的样本并发请求
            3. 一旦任意请求发生异常：
               - 停止继续加并发
               - 等待已启动请求自然结束
            4. 记录每个 chunk 的：
               - 到达时间（ts）
               - 当前系统活跃并发数（concurrency）
               - chunk 对应音频时长
               - chunk 级 RTF
            5. 汇总生成 summary 指标

        返回 summary 字段说明：
        - total_started / total_finished
        - max_concurrency
        - qpm（基于 chunk 实际时间跨度）
        - avg_ttfb / avg_rtf（请求级）
        - error_list 记录所有报错信息
        - chunk_stats（用于画图）
        """

        # 获取 API Adapter 并设置模型
        api_adapter = self.router.get_api_adapter(api_name)
        api_adapter.setup_model(model_name)

        # 使用固定样本
        task_name, txt_prompts = next(iter(self.text_tasks.items()))
        text_sample = txt_prompts[0]
        if single_chunk_mode:
            # 将样本中所有chunk合并为一个
            text_sample.text = ["".join(text_sample.text)]

        voice_type, voice_list = next(iter(self.voice_prompts.items()))
        voice_prompt = voice_list[0]

        # 初始化并发测试状态对象（包含锁、计数器、统计数据等）
        state = ConcurrencyState()
        state.start_time = time.perf_counter()

        ramp_interval = 0.5  # 0.5 并发 ramp-up 间隔（每 0.5 秒增加 1 个并发）
        max_duration = 60.0  # 最大 ramp-up 持续时间（60 秒）
        controller_sleep = 0.1  # 控制器循环的 sleep 时间，避免 busy loop

        # 保存所有启动的线程，便于后续 join
        threads: list[threading.Thread] = []
        start_ts = time.perf_counter()  # 记录 ramp-up 起始时间

        while True:
            elapsed = time.perf_counter() - start_ts  # 当前已运行时间
            if elapsed > max_duration:
                break  # 若超过最大测试时间则退出 ramp-up

            # 进入临界区，读取/修改共享状态
            with state.lock:
                if state.stop_ramp:  # 如果停止ramp_up (遇到首个报错时)
                    break

                # 根据时间计算目标样本并发数
                # 每 ramp_interval 秒增加 1
                target_concurrency = int(elapsed / ramp_interval) + 1
                need_to_start = target_concurrency - state.active_requests  # 需要新启动的请求数 = 目标并发 - 当前活跃请求

                # ===== 实时打印当前活跃线程数 =====
                print(
                    f"[{elapsed:.2f}s] Active requests: {state.active_requests}, total started: {state.total_started}"
                )

            # 启动需要补齐的请求线程
            for _ in range(need_to_start):
                with state.lock:
                    if state.stop_ramp:  # 二次防线 , 如果停止ramp_up (遇到首个报错时) 则不再补齐请求线程
                        break

                t = threading.Thread(
                    target=self._run_single_request_chunk_stats,
                    args=(api_adapter, text_sample, voice_prompt, state),
                    daemon=True
                )
                t.start()
                threads.append(t)
            # 控制器休眠，避免过高 CPU 占用
            time.sleep(controller_sleep)

        # ramp-up 结束后，等待所有已启动请求完成
        for t in threads:
            t.join(timeout=300)  # 最多等待5min

        # 标记测试已结束
        with state.lock:
            state.test_finished = True

        # 汇总统计
        success_metrics = [m for m in state.metrics if m.success]       # 成功请求指标
        failed_metrics = [m for m in state.metrics if not m.success]    # 失败请求指标
        chunk_stats = [s for s in state.chunk_stats]  # 拷贝所有 chunk 级统计数据

        actual_duration = max((s["ts"] for s in chunk_stats), default=max_duration)

        summary = {
            "api_name": api_name,
            "model_name": model_name,

            "total_started": state.total_started,
            "total_finished": state.total_finished,
            "total_success": len(success_metrics),
            "total_failed": len(failed_metrics),

            "max_concurrency": self.compute_max_success_concurrency(success_metrics),  # 在测试过程中，任意时刻同时处于成功请求生命周期内的最大请求数量
            "qpm": state.total_started / (actual_duration / 60),

            "avg_ttfb": sum(m.ttfb for m in success_metrics) / len(success_metrics) if success_metrics else None,
            "avg_rtf": sum(m.rtf for m in success_metrics) / len(success_metrics) if success_metrics else None,

            "error_list": [m.error for m in failed_metrics],
            "chunk_stats": chunk_stats,
        }

        return summary, success_metrics

    def _run_single_request_chunk_stats(self, api_adapter, text_sample, voice_prompt, state: ConcurrencyState):
        """
        单个请求线程体，每个 chunk 记录：
        ts, concurrency, audio_duration, rtf
        """
        # 请求开始，更新当前活跃请求数和总启动数
        with state.lock:
            state.active_requests += 1
            state.total_started += 1

        req_metrics = ConcurrencyRequestMetrics(
            success=False,
            start_ts=0.0,
            end_ts=0.0,
            ttfb=None,
            total_time=None,
            audio_duration=None,
            rtf=None,
            error=str("连接未开始"),
        )   # 防止 try / except 没接住异常，例如在建立连接截断就抛异常，此时尚未返回任何东西。

        try:
            last_chunk_ts = None  # 上一个 chunk 的到达时间
            audio_bytes = 0  # 累计返回的音频字节数
            ttfb = None

            # 遍历 TTS 接口返回的音频 chunk
            for chunk, sr, ch, bd, call_start_ts in api_adapter.voice_clone(
                    target_text=text_sample.text,
                    reference_audio=voice_prompt.wav_path,
                    reference_text=voice_prompt.text,
            ):
                # print(f"[返回Chunk] {time.perf_counter()}")
                now = time.perf_counter()  # 当前 chunk 到达时间
                if ttfb is None:
                    ttfb = now - call_start_ts
                    # print(f"[request] ttfb: {ttfb:.4f}s")
                audio_bytes_chunk = len(chunk)  # 当前 chunk 的字节数
                audio_duration_chunk = audio_bytes_chunk / (sr * ch * (bd // 8))   # 根据采样率 / 通道数 / 位深计算 chunk 对应音频时长

                # 计算 chunk 生成耗时 = 上一个 chunk 到当前 chunk 的时间
                chunk_elapsed = now - last_chunk_ts if last_chunk_ts else now - call_start_ts
                rtf = chunk_elapsed / audio_duration_chunk if audio_duration_chunk > 0 else None

                # 记录 chunk 级统计信息
                with state.lock:
                    state.chunk_stats.append({
                        "ts": now - state.start_time,  # 距离测试开始的时间
                        "concurrency": state.active_requests,  # 当前系统活跃并发数
                        "audio_duration": audio_duration_chunk,  # 当前 chunk 对应音频时长
                        "rtf": rtf,  # 当前 chunk 的 RTF
                    })
                    if not getattr(state, "audio_format", None):
                        state.audio_format = (sr, ch, bd)

                last_chunk_ts = now
                audio_bytes += audio_bytes_chunk

            # 请求结束生成指标
            total_time = last_chunk_ts - call_start_ts if last_chunk_ts else None
            total_audio_duration = audio_bytes / (sr * ch * (bd // 8))
            total_rtf = total_time / total_audio_duration if total_audio_duration > 0 else None

            # 构造请求级成功指标
            req_metrics = ConcurrencyRequestMetrics(
                success=True,
                start_ts=call_start_ts - state.start_time,
                end_ts=last_chunk_ts - state.start_time,
                ttfb=ttfb,
                total_time=total_time,
                audio_duration=total_audio_duration,
                rtf=total_rtf,
            )

        except Exception as e:
            now = time.perf_counter()
            print("[Exception]:\n", e)
            # 构造失败的请求指标，补充时间
            req_metrics = ConcurrencyRequestMetrics(
                success=False,
                start_ts=now - state.start_time,  # 补 start_ts
                end_ts=now - state.start_time,  # 补 end_ts
                ttfb=None,
                total_time=None,
                audio_duration=None,
                rtf=None,
                error=str(e),
            )
            with state.lock:
                state.stop_ramp = True

        finally:
            now = time.perf_counter()
            # 如果失败请求没有时间戳，保底赋值
            if req_metrics.start_ts is None:
                req_metrics.start_ts = now - state.start_time
            if req_metrics.end_ts is None:
                req_metrics.end_ts = now - state.start_time

            with state.lock:
                state.metrics.append(req_metrics)
                state.active_requests -= 1
                state.total_finished += 1

    def compute_max_success_concurrency(self, success_metrics):
        '''
        每个成功请求 = 一个区间 [start_ts, end_ts]
        我们只计算在测试过程中，任意时刻，同时处于「最终成功的请求」生命周期内的最大数量
        '''

        events = []

        for m in success_metrics:
            events.append((m.start_ts, +1))  # 成功请求开始
            events.append((m.end_ts, -1))  # 成功请求结束

        # 时间相同：先结束(-1)，再开始(+1)，避免虚高
        events.sort(key=lambda x: (x[0], x[1]))

        cur = 0
        max_cur = 0
        for _, delta in events:
            cur += delta
            max_cur = max(max_cur, cur)

        return max_cur

    def _plot_concurrency_rtf_v2(
            self,
            chunk_stats: list[dict],
            success_metrics: list,
            save_path: pathlib.Path,
    ):
        """
        并发退化曲线（chunk 级，语义增强版）

        左 Y 轴：
            - 浅蓝色实线：所有活跃请求并发（系统压力）
            - 深蓝色虚线：实时成功请求并发（有效产出压力）

        右 Y 轴：
            - 红色实线：chunk 级 RTF
            - 灰色虚线：RTF = 1 临界线
        """
        if not chunk_stats:
            return

        # ---------- 时间轴（以 chunk 为采样点） ----------
        ts = [s["ts"] for s in chunk_stats]
        active_concurrency = [s["concurrency"] for s in chunk_stats]
        rtf = [s["rtf"] for s in chunk_stats]

        # ---------- 成功请求的生命周期区间 ----------
        # 只使用最终成功的请求
        success_intervals = [
            (m.start_ts, m.end_ts)
            for m in success_metrics
            if m.start_ts is not None and m.end_ts is not None
        ]

        # ---------- 计算每个 ts 对应的“成功并发数” ----------
        def success_concurrency_at(t):
            return sum(1 for start, end in success_intervals if start <= t <= end)

        success_concurrency = [success_concurrency_at(t) for t in ts]

        # ---------- 开始画图 ----------
        fig, ax1 = plt.subplots(figsize=(11, 5))

        # 左 Y 轴：并发
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Concurrency")

        ax1.plot(
            ts,
            active_concurrency,
            label="Active Concurrency (All)",
            linewidth=2,
            alpha=0.9,
            color='#87CEEB', #浅蓝色
        )

        ax1.plot(
            ts,
            success_concurrency,
            label="Active Concurrency (Success-only)",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            color='#00008B'  # 深蓝色
        )

        ax1.tick_params(axis="y")

        # 右 Y 轴：RTF
        ax2 = ax1.twinx()
        ax2.set_ylabel("RTF")

        ax2.plot(
            ts,
            rtf,
            label="RTF (chunk-level)",
            linewidth=1.5,
            alpha=0.8,
            color='#FFA500',  # 橙色
        )

        ax2.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="RTF = 1.0",
            color='#7f7f7f',  # 灰色
        )

        ax2.tick_params(axis="y")

        # ---------- 图例 & 标题 ----------
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        fig.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
        )

        plt.title("Concurrency vs RTF (Chunk-level, Effective Load Aware)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # -----------------------------------------------------------------------------------------------------------
    # 用于测试QPM的方法
    def qpm_test(self, max_concurrency: int):
        '''
        QPM测试主入口（按 API × Model 维度执行）

        测试时间是 60s，在这 60s 内「发起」的请求里，所有最终成功完成的请求，都计入 QPM

        功能：
        1. 遍历所有 API / Model 组合
        2. 对每个组合执行一次 chunk 级QPM测试
        3. 将统计结果保存为 YAML

        输出目录结构：
        {output_dir}/{api_name}_{model_name}/
            ├── results_qpm_summary.yaml
            └── qpm_rtf.png
        '''
        for api_name, model_name_list in self.api_and_model_names.items():
            for model_name in model_name_list:
                print(f"Evaluating API={api_name} Model={model_name} QPM test")

                # 执行单 API + 单模型的QPM测试
                summary, success_metrics = self.single_api_qpm_test(api_name, model_name, max_concurrency)

                out_dir = pathlib.Path(self.output_dir) / f"{api_name}_{model_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # 保存 yaml
                yaml_path = out_dir / "results_qpm_summary.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(summary, f, allow_unicode=True)

                # 画图
                self._plot_qpm_rtf(
                    summary["chunk_stats"],
                    success_metrics,
                    out_dir / "qpm_rtf.png"
                )

    def single_api_qpm_test(self, api_name: str, model_name: str, max_concurrency: int = 10):
        """
        Steady-state QPM 测试：
        - 始终维持 max_concurrency 活跃请求
        - 忽略单请求异常
        - 唯一停止条件：达到 test_duration
        """

        api_adapter = self.router.get_api_adapter(api_name)
        api_adapter.setup_model(model_name)

        # 固定样本
        task_name, txt_prompts = next(iter(self.text_tasks.items()))
        text_sample = txt_prompts[0]
        text_sample.text = ["".join(text_sample.text)]  # 永远保证样本里只有1个chunk

        voice_type, voice_list = next(iter(self.voice_prompts.items()))
        voice_prompt = voice_list[0]

        state = ConcurrencyState()
        state.start_time = time.perf_counter()

        # ---------- QPM 参数 ----------
        threads: list[threading.Thread] = []
        start_ts = time.perf_counter()

        controller_sleep = 0.1
        test_duration = 60  # 持续发送请求的时间， 1分钟

        while True:
            now = time.perf_counter()
            elapsed = now - start_ts

            if elapsed > test_duration:
                break

            with state.lock:
                need_to_start = max_concurrency - state.active_requests

                print(
                    f"[{elapsed:6.2f}s] "
                    f"Active={state.active_requests}/{max_concurrency} "
                    f"Started={state.total_started} "
                    f"Finished={state.total_finished}"
                )

            for _ in range(need_to_start):
                t = threading.Thread(
                    target=self._run_single_request_chunk_stats,
                    args=(api_adapter, text_sample, voice_prompt, state),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            time.sleep(controller_sleep)

        # ---------------- drain 阶段 ----------------
        for t in threads:
            t.join(timeout=300)

        with state.lock:
            state.test_finished = True

        # ---------- 汇总 ----------
        success_metrics = [m for m in state.metrics if m.success]
        failed_metrics = [m for m in state.metrics if not m.success]
        chunk_stats = list(state.chunk_stats)

        actual_duration = max(
            (s["ts"] for s in chunk_stats),
            default=test_duration,
        )

        summary = {
            "api_name": api_name,
            "model_name": model_name,

            "test_mode": "steady_state_qpm",
            "max_concurrency": max_concurrency,
            "test_total_duration": actual_duration,  # 整个测试完成所花的时间

            "total_started": state.total_started,
            "total_finished": state.total_finished,
            "total_success": len(success_metrics),
            "total_failed": len(failed_metrics),

            # steady-state QPM
            "qpm": len(success_metrics) / (test_duration / 60),

            "max_success_concurrency": self.compute_max_success_concurrency(success_metrics),

            "avg_ttfb": (
                sum(m.ttfb for m in success_metrics) / len(success_metrics)
                if success_metrics else None
            ),
            "avg_rtf": (
                sum(m.rtf for m in success_metrics) / len(success_metrics)
                if success_metrics else None
            ),

            "error_list": [m.error for m in failed_metrics],
            "chunk_stats": chunk_stats,
        }

        return summary, success_metrics

    def _plot_qpm_rtf(
            self,
            chunk_stats: list[dict],
            success_metrics: list,
            save_path: pathlib.Path,
    ):
        """
        Steady-state QPM 测试可视化（chunk 级）

        左 Y 轴：
            - 浅蓝实线：所有活跃请求并发（系统施加负载）
            - 深蓝虚线：成功请求并发（有效吞吐负载）

        右 Y 轴：
            - 橙色实线：chunk 级 RTF
            - 灰色虚线：RTF = 1 实时生成临界线
        """
        if not chunk_stats:
            return

        # ---------- chunk 级时间轴 ----------
        ts = [s["ts"] for s in chunk_stats]
        active_concurrency = [s["concurrency"] for s in chunk_stats]
        rtf = [s["rtf"] for s in chunk_stats]

        # ---------- 成功请求生命周期 ----------
        success_intervals = [
            (m.start_ts, m.end_ts)
            for m in success_metrics
            if m.start_ts is not None and m.end_ts is not None
        ]

        # 计算每个 ts 时刻的 success 并发
        def success_concurrency_at(t):
            return sum(1 for start, end in success_intervals if start <= t <= end)

        success_concurrency = [success_concurrency_at(t) for t in ts]

        # ---------- 画图 ----------
        fig, ax1 = plt.subplots(figsize=(11, 5))

        # 左轴：并发
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Concurrency")

        ax1.plot(
            ts,
            active_concurrency,
            label="Active Concurrency (All)",
            linewidth=2,
            alpha=0.9,
            color="#87CEEB",  # 浅蓝
        )

        ax1.plot(
            ts,
            success_concurrency,
            label="Active Concurrency (Success-only)",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            color="#00008B",  # 深蓝
        )

        ax1.tick_params(axis="y")

        # 右轴：RTF
        ax2 = ax1.twinx()
        ax2.set_ylabel("RTF")

        ax2.plot(
            ts,
            rtf,
            label="RTF (chunk-level)",
            linewidth=1.5,
            alpha=0.8,
            color="#FFA500",  # 橙色
        )

        ax2.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="RTF = 1.0",
            color="#7f7f7f",  # 灰色
        )

        ax2.tick_params(axis="y")

        # ---------- Legend ----------
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        fig.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
        )

        plt.title("Steady-state QPM Test: Concurrency vs RTF")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # -----------------------------------------------------------------------------------------------------------
    # 用于测试稳定性的方法
    def stability_test(self, concurrency:int, qpm:int, allow_overlap:bool, test_duration:int):
        '''
        稳定性测试（SLA Test），
        固定并发上限 concurrency 固定发送速率 qpm 。 测试时长 3 分钟 并 统计每个请求是否成功（success rate 是核心指标）

        如果允许请求重叠，则优先根据每次发送 concurrency 数量的请求来计算一分钟内均匀发送满 qpm 的时间节点：
            每两次发送之间允许上一次发送的请求未结束的情况下发送下一轮请求。

        如果不允许请求重叠，则优先保证同时活跃的请求数量不超过 concurrency。
            在每次发送窗口时判断过去1分钟的发送的请求是否达到QPM，未达到则补充，已达到等待。
        '''

        for api_name, model_name_list in self.api_and_model_names.items():
            for model_name in model_name_list:
                print(f"Evaluating API={api_name} Model={model_name} Stability Test")

                summary, success_metrics = self.single_api_stability_test(
                    api_name=api_name,
                    model_name=model_name,
                    concurrency=concurrency,
                    qpm=qpm,
                    allow_overlap=allow_overlap,
                    test_duration=test_duration
                )

                out_dir = pathlib.Path(self.output_dir) / f"{api_name}_{model_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # 保存 YAML
                yaml_path = out_dir / "results_stability_summary.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(summary, f, allow_unicode=True)

                # 画图
                self._plot_stability_rtf_and_failures(
                    summary["chunk_stats"],
                    summary["failed_metrics"],
                    summary["success_metrics"],
                    out_dir / "stability_rtf.png",
                )

                self._plot_stability_ttfb_and_failures(
                    summary["chunk_stats"],
                    summary["failed_metrics"],
                    summary["success_metrics"],
                    out_dir / "stability_ttfb.png",
                )

    def single_api_stability_test(
        self,
        api_name: str,
        model_name: str,
        concurrency: int,
        qpm: int,
        allow_overlap: bool,
        test_duration = 180  # 默认180s
    ):
        """
        单 API / 单 Model 稳定性测试（SLA 模型）
        - 滑动 60 秒窗口控制 QPM
        - 每条请求间隔 >= 60 / QPM 秒
        - 固定并发上限
        - 每 5 秒打印日志
        """
        import time
        import threading

        api_adapter = self.router.get_api_adapter(api_name)
        api_adapter.setup_model(model_name)

        # ---------- 固定样本 ----------
        task_name, txt_prompts = next(iter(self.text_tasks.items()))
        text_sample = txt_prompts[0]
        text_sample.text = ["".join(text_sample.text)]

        voice_type, voice_list = next(iter(self.voice_prompts.items()))
        voice_prompt = voice_list[0]

        # ---------- 状态 ----------
        state = ConcurrencyState()
        state.start_time = time.perf_counter()
        threads: list[threading.Thread] = []

        start_ts = time.perf_counter()
        end_ts = start_ts + test_duration

        # ---------- 滑动窗口 QPM ----------
        request_timestamps = []

        # 打印日志
        log_interval = 5.0
        last_log_ts = start_ts

        request_interval_min = 60.0 / qpm  # 每条请求最小间隔

        while True:
            now = time.perf_counter()
            elapsed = now - start_ts
            if elapsed >= test_duration:
                break

            # ---------- 每 5 秒打印一次运行状态 ----------
            if now - last_log_ts >= log_interval:
                with state.lock:
                    active = state.active_requests
                    started = state.total_started
                    finished = state.total_finished
                    success = sum(1 for m in state.metrics if m.success)
                    failed = finished - success

                approx_qpm = success / (elapsed / 60.0) if elapsed > 0 else 0.0

                mode = "allow quest overlap" if allow_overlap else "strict"
                print(
                    f"[{elapsed:6.1f}s][{mode}] act={active}/{concurrency} "
                    f"started={started} fin={finished} succ={success} fail={failed} "
                    f"qpm≈{approx_qpm:.1f} window_req={len(request_timestamps)}"
                )
                last_log_ts = now

            # ---------- 清理滑动窗口中过期请求 ----------
            request_timestamps = [t for t in request_timestamps if t > now - 60.0]

            # ---------- 滑动窗口 QPM 已满 ----------
            if len(request_timestamps) >= qpm:
                time.sleep(0.05)
                continue

            # ---------- 请求间隔控制 ----------
            if request_timestamps:
                last_req_ts = request_timestamps[-1]
                sleep_needed = request_interval_min - (now - last_req_ts)
                if sleep_needed > 0:
                    time.sleep(min(sleep_needed, 0.05))
                    continue

            # ---------- 并发控制 ----------
            with state.lock:
                if not allow_overlap and state.active_requests >= concurrency:
                    time.sleep(0.05)
                    continue

                # 发起请求
                t = threading.Thread(
                    target=self._run_single_request_chunk_stats,
                    args=(api_adapter, text_sample, voice_prompt, state),
                    daemon=True,
                )
                t.start()
                threads.append(t)
                state.total_started += 1
                request_timestamps.append(now)

            # 循环小 sleep 保证调度，不空转
            time.sleep(0.001)

        # ---------- drain ----------
        for t in threads:
            t.join(timeout=300)

        with state.lock:
            state.test_finished = True

        # ---------- 汇总 ----------
        success_metrics = [m for m in state.metrics if m.success]
        failed_metrics = [m for m in state.metrics if not m.success]
        total_success = len(success_metrics)
        total_finished = state.total_finished

        # 把 ConcurrencyRequestMetrics 对象转成 dict，便于 YAML 序列化
        def metrics_to_dict(metrics_list):
            return [
                {
                    "success": m.success,
                    "start_ts": m.start_ts,
                    "end_ts": m.end_ts,
                    "ttfb": m.ttfb,
                    "total_time": m.total_time,
                    "audio_duration": m.audio_duration,
                    "rtf": m.rtf,
                    "error": m.error,
                }
                for m in metrics_list
            ]

        summary = {
            "api_name": api_name,
            "model_name": model_name,
            "test_mode": "stability_test",
            "concurrency": concurrency,
            "target_qpm": qpm,
            "test_duration_sec": test_duration,
            "total_started": state.total_started,
            "total_finished": total_finished,
            "total_success": total_success,
            "total_failed": len(failed_metrics),
            "success_rate": total_success / total_finished if total_finished > 0 else None,
            "actual_qpm": total_success / (test_duration / 60.0),
            "avg_ttfb": sum(m.ttfb for m in success_metrics) / total_success if total_success else None,
            "avg_rtf": sum(m.rtf for m in success_metrics) / total_success if total_success else None,
            "error_list": [m.error for m in failed_metrics],
            "chunk_stats": list(state.chunk_stats),
            "metrics": metrics_to_dict(state.metrics),
            "failed_metrics": metrics_to_dict(failed_metrics),
            "success_metrics": metrics_to_dict(success_metrics),
        }

        return summary, success_metrics

    def _plot_stability_rtf_and_failures(
            self,
            chunk_stats: list[dict],
            failed_metrics: list[dict],
            success_metrics: list[dict],
            save_path: pathlib.Path,
    ):
        """
        可视化：
        - 左轴：每个 chunk 的 RTF
        - 右轴：累计失败请求数 + 滑动窗口成功 QPM（过去1分钟）
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not chunk_stats:
            print("No chunk stats to plot.")
            return

        ts = [s["ts"] for s in chunk_stats]
        rtf = [s["rtf"] for s in chunk_stats]

        # ---------- 失败请求时间 ----------
        fail_ts = sorted([m["end_ts"] for m in failed_metrics if m["end_ts"] is not None])
        cumulative_fail = np.zeros(len(ts), dtype=int)
        fail_idx = 0
        for i, t in enumerate(ts):
            while fail_idx < len(fail_ts) and fail_ts[fail_idx] <= t:
                fail_idx += 1
            cumulative_fail[i] = fail_idx

        # ---------- 成功 QPM (滑动1分钟窗口) ----------
        # 将成功请求按 start_ts 排序
        success_start_ts = sorted([m["start_ts"] for m in success_metrics if m["start_ts"] is not None])
        success_qpm = np.zeros(len(ts), dtype=float)
        window_sec = 60.0

        for i, t in enumerate(ts):
            # 统计过去60秒内发起的成功请求数
            count = sum(1 for s_ts in success_start_ts if t - window_sec <= s_ts <= t)
            success_qpm[i] = count * (60.0 / window_sec)  # 换算为 QPM

        # ---------- 绘图 ----------
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # 左轴：Chunk RTF
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Chunk RTF")
        ax1.plot(ts, rtf, color="#FFA500", linewidth=1.6, alpha=0.9, label="Chunk RTF")
        ax1.axhline(1.0, color="#808080", linestyle="--", linewidth=1.2, alpha=0.7, label="RTF = 1.0")

        # 右轴：累计失败请求 + 成功 QPM
        ax2 = ax1.twinx()
        ax2.set_ylabel("Failed Requests / Success QPM")
        ax2.plot(ts, cumulative_fail, color="#FF0000", linewidth=2.0, alpha=0.9, label="Failed Requests")
        ax2.plot(ts, success_qpm, color="#008000", linewidth=2.0, alpha=0.9, label="Success QPM (1min)")

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Stability Test: Chunk RTF vs Failures & Success QPM (Sliding 1min)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_stability_ttfb_and_failures(
            self,
            chunk_stats: list[dict],
            failed_metrics: list[dict],
            success_metrics: list[dict],
            save_path: pathlib.Path,
    ):
        """
        稳定性测试可视化（TTFB 视角）

        左轴：
            - 每个成功请求的 TTFB（scatter）
            - SLA 参考线（可选）

        右轴：
            - 累计失败请求数
            - 成功 QPM（滑动 1 分钟）
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not success_metrics:
            print("No success metrics to plot TTFB.")
            return

        # ---------- 成功请求 TTFB ----------
        ttfb_ts = [
            m["start_ts"]
            for m in success_metrics
            if m["start_ts"] is not None and m["ttfb"] is not None
        ]
        ttfb_values = [
            m["ttfb"]
            for m in success_metrics
            if m["start_ts"] is not None and m["ttfb"] is not None
        ]

        if not ttfb_ts:
            print("No valid TTFB data.")
            return

        # ---------- 失败请求 ----------
        fail_ts = sorted(
            m["end_ts"]
            for m in failed_metrics
            if m["end_ts"] is not None
        )

        # 时间轴（统一用 chunk_stats 的 ts，保证对齐）
        ts = [s["ts"] for s in chunk_stats]
        if not ts:
            return

        # 累计失败数
        cumulative_fail = np.zeros(len(ts), dtype=int)
        fail_idx = 0
        for i, t in enumerate(ts):
            while fail_idx < len(fail_ts) and fail_ts[fail_idx] <= t:
                fail_idx += 1
            cumulative_fail[i] = fail_idx

        # ---------- 成功 QPM（滑动 1 分钟） ----------
        window_sec = 60.0
        success_start_ts = sorted(ttfb_ts)

        success_qpm = np.zeros(len(ts), dtype=float)
        for i, t in enumerate(ts):
            count = sum(1 for s_ts in success_start_ts if t - window_sec <= s_ts <= t)
            success_qpm[i] = count * (60.0 / window_sec)

        # ---------- 绘图 ----------
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # 左轴：TTFB
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("TTFB (s)")

        ax1.scatter(
            ttfb_ts,
            ttfb_values,
            color="#1f77b4",
            alpha=0.7,
            s=20,
            label="TTFB (per request)",
        )

        # SLA 参考线（可选，按需调整）
        ax1.axhline(
            1.0,
            color="#808080",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="TTFB = 1.0s",
        )

        # 右轴：失败 & QPM
        ax2 = ax1.twinx()
        ax2.set_ylabel("Failed Requests / Success QPM")

        ax2.plot(
            ts,
            cumulative_fail,
            color="#FF0000",
            linewidth=2.0,
            alpha=0.9,
            label="Failed Requests",
        )

        ax2.plot(
            ts,
            success_qpm,
            color="#008000",
            linewidth=2.0,
            alpha=0.9,
            label="Success QPM (1min)",
        )

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Stability Test: TTFB vs Failures & Success QPM (Sliding 1min)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()



if __name__ == "__main__":
    '''
    python evaluator.py
    
    TTSEvaluator.evaluate_all() 一般效果评测，指定voice clone的任务与音色，自动遍历进行
    TTSEvaluator.concurrency_test() 并发测试，指定voice clone的任务text_task_support为["long_stream_prompt"]，使用其中用于流式生成的样本进行并发测试
    TTSEvaluator.qpm_test() QPM测试，指定voice clone的任务text_task_support为["short_prompt"]，使用其中的短样本进行QPM上限的测试
    TTSEvaluator.stability_test() 稳定性测试，指定voice clone的任务text_task_support为["neutral_prompt"]，使用自然样本进行稳定性测试
    '''
    # API名称与具体模型名称的配置, 测试工具会自动遍历所有配置
    api_and_model_names = {
        # "qwen": ["qwen3-tts-vc-realtime-2025-11-27"],
        # "fishaudio": ["s1"],
        # "minimax": ["speech-02-hd"],  # "speech-2.6-turbo", "speech-02-hd"
        "inworld": ["inworld-tts-1-max"]
    }

    evaluator = TTSEvaluator(
        api_and_model_names=api_and_model_names,
        output_dir="result",
        # text_task_support=["hardcase_prompt"],  # "emotion_prompt", "hardcase_prompt", "mixed-lingual_in-context_prompt"
        # text_task_support=["long_stream_prompt"],  # 测试流式生成时 / 并发测试时的长样本
        # text_task_support=["short_prompt"],  # 测试QPM时的短样本
        text_task_support=["neutral_prompt"],  # 测试稳定性时使用的样本
        voice_task_support=["base_voice_prompt"],
    )

    # 一般效果评测
    # evaluator.evaluate_all()

    # 并发测试
    # evaluator.concurrency_test(single_chunk_mode=False)  # 使用并发测试时，任务必须选择 text_task_support=["long_stream_prompt"]，保证样本生成的正常长度大于1分钟
    '''
    1. 确保初始化 TTSEvaluator() 中任务 text_task_support 为 ["long_stream_prompt"]
    2. 确保执行 TTSEvaluator.concurrency_test() 的 single_chunk_mode 是正确的：
        single_chunk_mode=False:
            测试流式输入长样本（单样本22个chunk，样本平均总耗时30s+）时，测试的时最大样本并发数（一个样本可能多个请求）
        single_chunk_mode=True:
            测试非流式输入长样本（单样本仅1个chunk，样本平均总耗时30s+）时，测试的是最大请求并发数
    '''

    # QPM 测试
    # evaluator.qpm_test(max_concurrency=32)  # 测试QPM时，任务必须选择 text_task_support=["short_prompt"]，且max_concurrency设置为并发测试时得到的最大并发数。
    '''
    1. 确保初始化 TTSEvaluator() 中任务 text_task_support 为 ["short_prompt"]
    2. 确保执行 TTSEvaluator.qpm_test() 的 max_concurrency 是前面实验测试出的该模型最大并发数
    '''

    # 稳定性测试
    evaluator.stability_test(concurrency=1, qpm=10, allow_overlap=False, test_duration=80)
    '''
    1. 确保初始化 TTSEvaluator() 中任务 text_task_support 为 ["neutral_prompt"]
    2. 确保传入的 concurrency 和 qpm 是小于前面实验测出的最大负载上限的
    3. 设置allow_overlap模式:
        True为允许上一轮请求未结束的时候进行下一轮请求，优先满足设置的QPM，此时长样本可能会出现同时活跃数>concurrency
        False为不允许请求重叠，优先满足同时活跃数≤concurrency，此时长样本可能会出现实际运行时每分钟没有发送足够QPM设置值的请求
    4. 设置测试时长，建议开始80s测试，正式测试180s
    '''