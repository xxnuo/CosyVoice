import os
import random
import sys
import time
import threading
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
        self.last_used_time = 0  # 最后一次使用时间
        self.idle_timeout = Config.idle_timeout  # 闲置超时时间（秒），默认10分钟
        self.timer_thread = None
        self.timer_stop_event = threading.Event()
        self._start_idle_timer()  # 启动闲置检查定时器

    def _start_idle_timer(self):
        """启动闲置检查定时器线程"""
        if self.timer_thread is not None and self.timer_thread.is_alive():
            self.timer_stop_event.set()  # 停止旧的定时器线程
            self.timer_thread.join()  # 等待线程结束

        self.timer_stop_event.clear()  # 重置停止事件
        self.timer_thread = threading.Thread(target=self._idle_timer_task)
        self.timer_thread.daemon = True  # 设为守护线程，主程序退出时自动结束
        self.timer_thread.start()

    def _idle_timer_task(self):
        """定时器线程任务，定期检查模型闲置状态"""
        check_interval = 60  # 每60秒检查一次
        while not self.timer_stop_event.is_set():
            if self.ok and self.last_used_time > 0:
                current_time = time.time()
                if current_time - self.last_used_time > self.idle_timeout:
                    logger.info(
                        f"Model idle for {self.idle_timeout} seconds, unloading..."
                    )
                    self.unload_model()
            time.sleep(check_interval)

    def reset_timer(self):
        """重置闲置计时器"""
        self.last_used_time = time.time()

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
            self.reset_timer()  # 重置计时器
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
        self.reset_timer()  # 重置计时器

    def unload_model(self):
        """卸载当前加载的模型并释放资源"""
        if self.cosyvoice is not None and self.ok:
            del self.cosyvoice
            self.cosyvoice = None
            torch.cuda.empty_cache()
            self.ok = False
            self.last_used_time = 0  # 重置最后使用时间

    def get_available_spks(self):
        """获取可用的模型自带音色列表"""
        # if self.cosyvoice is not None and self.ok:
        #     self.reset_timer()  # 重置计时器
        #     return self.cosyvoice.list_available_spks()
        # return []

        self.reset_timer()  # 重置计时器
        return ["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"]

    def generate(
        self,
        mode: Literal[
            "sft", "clone", "crosslingual"
        ] = "sft",  # 可选值：sft（模型自带音色）、clone（克隆音色）、crosslingual（跨语种克隆音色）
        spk_id: str = Config.default_spk,  # 要使用的模型自带音色，如果为空，则使用默认音色
        tts_text: str = "",  # 要生成的文本
        prompt_wav_text: str = "",  # 要使用的提示文本
        prompt_wav_path: str = "",  # 要使用的提示音频
        prompt_wav_sr: int = Config.prompt_sr,  # 提示音频的采样率
        max_val: float = Config.max_val,  # 音量最大值
        tts_sr: int = Config.sample_rate,  # 要生成的音频的采样率，默认与模型一致
        randomize_seed: bool = True,  # 是否随机化种子
        seed: int = 0,  # 随机种子
        stream: bool = False,  # 是否流式生成
        speed: float = 1.0,  # 说话速度
    ):
        """生成音频并返回生成的音频流式数据：yield (tts_sr, i["tts_speech"].numpy().flatten())"""
        # 自动加载模型
        if self.cosyvoice is None or not self.ok:
            self.load_model(Config.model_name)
        else:
            self.reset_timer()  # 重置计时器

        # 随机化种子
        if randomize_seed:
            seed = random.randint(1, 100000000)

        # 检查必须参数
        if tts_text == "":
            raise ValueError("请输入要生成的文本！")

        # 检查可选参数
        if mode == "sft":
            # 模型自带生成需要指定模型自带音色
            sft_spks = self.get_available_spks()
            if spk_id == "":
                if len(sft_spks) == 0:
                    raise ValueError("没有可用的模型自带音色")
                spk_id = sft_spks[0]
            if spk_id not in sft_spks:
                raise ValueError(f"模型自带音色 {spk_id} 不存在")

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
                spk_id=spk_id,
                stream=stream,
                speed=speed,
            ):
                yield (tts_sr, i["tts_speech"].numpy().flatten())
        elif mode == "clone":
            logger.info("Get clone speech")
            prompt_speech_16k = self.postprocess(
                load_wav(prompt_wav_path, prompt_wav_sr), max_val=max_val
            )
            set_all_random_seed(seed)
            for i in self.cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                yield (tts_sr, i["tts_speech"].numpy().flatten())
        elif mode == "crosslingual":
            logger.info("Get crosslingual speech")
            prompt_speech_16k = self.postprocess(
                load_wav(prompt_wav_path, prompt_wav_sr), max_val=max_val
            )
            set_all_random_seed(seed)
            for i in self.cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                yield (tts_sr, i["tts_speech"].numpy().flatten())

        # 重置计时器
        self.reset_timer()

    def __del__(self):
        """析构函数，确保定时器线程被正确停止"""
        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_stop_event.set()
            self.timer_thread.join(timeout=1.0)  # 等待线程结束，最多等待1秒


if __name__ == "__main__":
    pass
