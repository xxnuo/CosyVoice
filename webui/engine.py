import os
import random
import sys
from typing import Literal

import librosa
import numpy as np
import torch

from webui.config import Config

# vLLM
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 根目录
sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # noqa: E402
from vllm import ModelRegistry  # type: ignore  # noqa: E402

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# CosyVoice
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402
from cosyvoice.utils.file_utils import load_wav, logging  # noqa: E402

logger = logging.getLogger()


class CosyVoiceEngine:
    def __init__(self):
        self.ok = False
        self.cosyvoice = None
        self.default_data = np.zeros(Config.sample_rate)

    def postprocess(
        self,
        speech,
        max_val=Config.max_val,
        sample_rate=Config.sample_rate,
        top_db=60,
        hop_length=220,
        win_length=440,
    ):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(sample_rate * 0.2))], dim=1)
        return speech

    def load_model(self, model_name: str = Config.model_name):
        """加载指定的模型"""
        if self.cosyvoice is not None and self.ok:
            return

        logger.info(f"Loading model: {model_name}")
        try:
            self.cosyvoice = CosyVoice2(
                model_dir=Config.model_dir + "/" + model_name, load_vllm=True
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

        logger.info(f"Model {model_name} loaded")
        self.ok = True

    def unload_model(self):
        """卸载当前加载的模型并释放资源"""
        if self.cosyvoice is not None and self.ok:
            del self.cosyvoice
            self.cosyvoice = None
            torch.cuda.empty_cache()
            self.ok = False

    def get_available_spks(self):
        """获取可用的预训练音色列表"""
        if self.cosyvoice is not None and self.ok:
            return self.cosyvoice.list_available_spks()
        return []

    def generate(
        self,
        mode: Literal[
            "sft", "clone", "crosslingual"
        ] = "sft",  # 可选值：sft（预训练音色）、clone（3s极速复刻）、crosslingual（跨语种复刻）
        sft_spk: str = "",  # 要使用的预训练音色，如果为空，则使用默认音色
        tts_text: str = "",  # 要生成的文本
        prompt_wav_text: str = "",  # 要使用的提示文本
        prompt_wav_path: str = "",  # 要使用的提示音频
        prompt_wav_sr: int = Config.prompt_sr,  # 提示音频的采样率
        tts_sr: int = Config.sample_rate,  # 要生成的音频的采样率，默认与模型一致
        randomize_seed: bool = True,  # 是否随机化种子
        seed: int = 0,  # 随机种子
        stream: bool = False,  # 是否流式生成
        speed: float = 1.0,  # 说话速度
    ):
        """生成图像并返回生成的图像列表和使用的种子"""
        # 自动加载模型
        if self.cosyvoice is None or not self.ok:
            self.load_model(Config.model_name)

        # 随机化种子
        if randomize_seed:
            seed = random.randint(1, 100000000)

        # 检查必须参数
        if tts_text == "":
            raise ValueError("请输入要生成的文本！")

        # 检查可选参数
        if mode == "sft":
            # 预训练生成需要指定预训练音色
            if sft_spk == "":
                sft_spks = self.get_available_spks()
                if len(sft_spks) == 0:
                    raise ValueError("没有可用的预训练音色")
                sft_spk = sft_spks[0]
            if sft_spk not in sft_spks:
                raise ValueError(f"预训练音色 {sft_spk} 不存在")

        elif mode == "clone" or mode == "crosslingual":
            # 克隆和跨语种生成需要指定提示音频和提示文本
            if not os.path.exists(prompt_wav_path):
                raise ValueError("需要提示音频")
            if prompt_wav_text == "":
                raise ValueError("需要提示文本")

        else:
            raise ValueError(f"不支持的模式: {mode}")

        # 生成
        if mode == "sft":
            logger.info("Get SFT speech")
            set_all_random_seed(seed)
            for i in self.cosyvoice.inference_sft(
                tts_text=tts_text,
                sft_spk=sft_spk,
                stream=stream,
                speed=speed,
            ):
                yield (tts_sr, i["tts_speech"].numpy().flatten())
        elif mode == "clone":
            logger.info("Get clone speech")
            prompt_speech_16k = self.postprocess(
                load_wav(prompt_wav_path, prompt_wav_sr)
            )
            set_all_random_seed(seed)
            for i in self.cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                yield (tts_sr, i["tts_speech"].numpy().flatten())
        elif mode == "crosslingual":
            logger.info("Get crosslingual speech")
            prompt_speech_16k = self.postprocess(
                load_wav(prompt_wav_path, prompt_wav_sr)
            )
            set_all_random_seed(seed)
            for i in self.cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                yield (tts_sr, i["tts_speech"].numpy().flatten())


if __name__ == "__main__":
    pass
