'''
这里实现数据收集器，用于遍历data/下数据，组装TTS的输入。

data/
    text_prompt/                         # 要生成的目标文本
        emotion_prompt/                  # 包含特点情感语境下的文本，可用于测试从文本中推断情感的能力
        hardcase_prompt/                 # 绕口令，测试生成的语言清晰度
        mixed-lingual_in-context_prompt/ # 上下文混合语言，用于测试TTS多语种混合生成能力
        neutral_prompt/                  # 中性文本，用于测试对其施加情感控制后的效果

    voice_prompt/                        # voice clone的音色示例
        base_voice_prompt/               # 基础音色提示，包含多个的wav和txt对
        emotion_voice_prompt/            # 带感情的音频提示，包含以angry，happy，neutral，sad，surprise命名的wav和txt对


'''
from pathlib import Path
from typing import List, Dict, NamedTuple

class VoicePrompt(NamedTuple):
    """一条音色提示：音频路径 + 对应音频转录文本 + 音色名（去掉扩展名）"""
    wav_path: Path  # 音频路径
    text: str  # 音频对应的转录文本
    voice_name: str  # 音色名称

class TextPrompt(NamedTuple):
    """一条文本样本：文本内容 + 样本名（去掉扩展名）"""
    text: List[str]  # 文本内容
    sample_name: str  # 样本名称


class TTSEvalDataCollector:
    def __init__(
        self,
        data_path: str | Path,
        text_task_support: list[str] | None = None,
        voice_task_support: list[str] | None = None,
    ):
        self.data_path = Path(data_path)
        self.text_task_support = set(text_task_support or [])   # 用于限制遍历哪个子文件夹下的text, 任务类型
        self.voice_task_support = set(voice_task_support or [])  # 用于限制遍历哪个子文件夹下的voice, 任务类型

        # 预计算好路径，后续方法直接复用
        self._text_root = self.data_path / "text_prompt"
        self._voice_root = self.data_path / "voice_prompt"

        # 合法性检查
        assert self._text_root.exists(), f"{self._text_root} not found"
        assert self._voice_root.exists(), f"{self._voice_root} not found"


    # 对外接口 -----------------------------------------------------------------------------------
    def get_text_prompts(self) -> Dict[str, List[TextPrompt]]:
        """
        返回：
        {
            'emotion': [TextPrompt(...), ...],
            'hardcase': [...],
            ...
        }
        只包含用户在 text_task_support 中指定的任务。
        """
        tasks: Dict[str, List[TextPrompt]] = {}
        for task_dir in sorted(self._text_root.iterdir()):
            if not task_dir.is_dir():
                continue
            # 根据 text_task_support 限制获取用于不同任务的目标文本
            if task_dir.name in self.text_task_support:
                task_name = task_dir.name.replace("_prompt", "")
                tasks[task_name] = self._load_text_prompts(task_dir)
        return tasks

    def get_voice_prompts(self) -> Dict[str, List[VoicePrompt]]:
        """
        返回：
        {
            'base_voice': [VoicePrompt(...), ...],
            'emotion_voice': [...],
        }
        只包含用户在 voice_task_support 中指定的任务。
        """
        voices: Dict[str, List[VoicePrompt]] = {}
        for voice_subdir in sorted(self._voice_root.iterdir()):
            if not voice_subdir.is_dir():
                continue
            # 根据 voice_task_support 限制获取用于不同任务的音频路径
            if voice_subdir.name in self.voice_task_support:
                task_name = voice_subdir.name.replace("_voice_prompt", "").replace("_prompt", "")
                voices[task_name] = self._load_voice_prompts(voice_subdir)
        return voices



    # 内部工具 -----------------------------------------------------------------------------------
    def _load_text_prompts(self, task_dir: Path) -> List[TextPrompt]:
        '''
        获取task_dir下的所有txt文件，读取其中内容
        根据换行符将txt内容收集为list[str]
        '''
        prompts = []
        for txt_file in sorted(task_dir.glob("*.txt")):
            raw_text = self._read_text_file(txt_file)
            if not raw_text:
                continue

            # 核心：按行拆分，并过滤空行
            text_chunks = [
                line.strip()
                for line in raw_text.splitlines()
                if line.strip()
            ]

            if not text_chunks:
                continue

            prompts.append(TextPrompt(text=text_chunks, sample_name=txt_file.stem))
        return prompts

    def _load_voice_prompts(self, voice_subdir: Path) -> List[VoicePrompt]:
        """
        遍历 voice_subdir 下所有 wav 文件，找到同名 txt 作为转录文本路径。
        如果缺失对应 txt，则跳过该条音色提示。
        """
        prompts = []
        for wav_file in sorted(voice_subdir.glob("*.wav")):
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                print(f"[Skip] 缺失转录文本: {txt_file}")
                continue
            text = self._read_text_file(txt_file)
            prompts.append(
                VoicePrompt(
                    wav_path=wav_file,
                    text=text,
                    voice_name=wav_file.stem,
                )
            )
        return prompts

    def _read_text_file(self, txt_file: Path) -> str:
        for encoding in ("utf-8", "gbk", "gb2312"):
            try:
                return txt_file.read_text(encoding=encoding).strip()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            f"Cannot decode file {txt_file} with utf-8 / gbk / gb2312"
        )



if __name__ == "__main__":
    '''
    python base/data_collector.py
    '''
    collector = TTSEvalDataCollector(
        data_path="data",
        text_task_support=["long_stream_prompt"],  # 文本任务
        voice_task_support=["base_voice_prompt"],  # 只用基础音色
    )

    text_tasks = collector.get_text_prompts()  # -> Dict[task, List[TextPrompt]]
    voice_prompts = collector.get_voice_prompts()  # -> Dict[voice_task, List[VoicePrompt]]

    print("text_tasks \n", text_tasks)
    print("voice_prompts \n", voice_prompts)

    for task_name, txt_prompts in text_tasks.items():
        for txt in txt_prompts:
            for voice_type, voice_list in voice_prompts.items():
                for vp in voice_list:
                    print(
                        f"Generating | Task={task_name} | Sample={txt.sample_name} | Voice={vp.voice_name}"
                    )
