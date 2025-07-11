import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # 模型
    model_dir = os.getenv("MODEL_DIR", "/data/models/huggingface")
    model_name = os.getenv("MODEL", "iic/CosyVoice2-0.5B")
    sample_rate = os.getenv("SAMPLE_RATE", 24000)
    max_val = os.getenv("MAX_VAL", 1.0)
    prompt_sr = os.getenv("PROMPT_SR", 16000)
    default_spk = os.getenv("DEFAULT_SPK", "中文女")

    # 预装音色
    voice_dir = os.getenv("VOICE_DIR", "/app/voices")
