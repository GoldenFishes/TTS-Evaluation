## 兼容不同API的评估工具

`evaluator.py` 为评估TTS的入口

一般效果测试候选API入口函数： `TTSEvaluator.evaluate_all()`
    我们在 `TTSEvaluator.single_api_evaluate()` 实现单个TTS API的调用与测试

并发测试候选API入口函数： `TTSEvaluator.concurrency_test()`
    进行并发测试时务必设置 TTSEvaluator 初始化任务为 text_task_support=["long_stream_prompt"]
    以保证获取到用于并发的单个样本本身耗时足够长。



结果输出于 `result/`



### API 适配器

在 `apis/` 下实现各个不同TTS API的适配