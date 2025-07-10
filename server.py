import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import random
import librosa
import io
import base64
from typing import List, Optional, Literal, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI(
    title="CosyVoice TTS API",
    description="OpenAI TTS API compatible server using CosyVoice",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义推理模式，与webui.py保持一致
INFERENCE_MODES = ["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"]


# Models for API requests
class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = ""
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav"]] = "mp3"
    speed: Optional[float] = 1.0
    prompt_text: Optional[str] = ""
    prompt_audio: Optional[str] = None  # Base64 encoded audio
    instruct_text: Optional[str] = ""
    seed: Optional[int] = 0
    stream: Optional[bool] = False
    mode: Optional[str] = "预训练音色"  # 默认使用预训练音色模式


class VoiceResponse(BaseModel):
    id: str
    name: str
    preview_url: Optional[str] = None
    category: str = "cosyvoice"


class VoicesResponse(BaseModel):
    voices: List[VoiceResponse]


# Global variables
max_val = 0.8
prompt_sr = 16000


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat(
        [speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1
    )
    return speech


def audio_to_bytes(audio_array, sample_rate, format="mp3"):
    """Convert numpy audio array to bytes in the specified format"""
    buffer = io.BytesIO()
    if format == "wav":
        torchaudio.save(
            buffer, torch.tensor(audio_array).unsqueeze(0), sample_rate, format="wav"
        )
    else:
        # Default to mp3 for other formats (OpenAI mainly uses mp3)
        torchaudio.save(
            buffer, torch.tensor(audio_array).unsqueeze(0), sample_rate, format="mp3"
        )
    buffer.seek(0)
    return buffer.read()


def decode_audio(base64_audio):
    """Decode base64 audio to a temporary file and return the path"""
    if not base64_audio:
        return None

    try:
        # Decode base64 string
        audio_bytes = base64.b64decode(base64_audio)

        # Create a temporary file
        temp_file = os.path.join(
            os.getcwd(), f"temp_audio_{random.randint(1000, 9999)}.wav"
        )
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        return temp_file
    except Exception as e:
        logging.error(f"Failed to decode audio: {str(e)}")
        return None


def generate_seed():
    """Generate a random seed"""
    return random.randint(1, 100000000)


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """OpenAI compatible TTS endpoint"""
    try:
        tts_text = request.input
        voice = request.voice
        speed = request.speed
        prompt_text = request.prompt_text
        instruct_text = request.instruct_text
        seed = request.seed if request.seed != 0 else generate_seed()
        stream = request.stream
        response_format = request.response_format
        mode = request.mode

        # 处理prompt音频
        prompt_wav = None
        if request.prompt_audio:
            prompt_wav = decode_audio(request.prompt_audio)

        # 设置随机种子
        set_all_random_seed(seed)

        # 根据模式选择不同的推理方法
        audio_data = None

        if mode == "预训练音色":
            if not voice and len(sft_spk) > 0:
                voice = sft_spk[0]

            if voice not in sft_spk:
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice {voice} not found in available voices",
                )

            logging.info(f"Using SFT inference with voice {voice}")
            for i in cosyvoice.inference_sft(
                tts_text, voice, stream=stream, speed=speed
            ):
                audio_data = i["tts_speech"].numpy().flatten()

        elif mode == "3s极速复刻":
            if not prompt_wav:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt audio is required for 3s极速复刻 mode",
                )
            if not prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt text is required for 3s极速复刻 mode",
                )

            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            logging.info("Using zero-shot inference")
            for i in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                audio_data = i["tts_speech"].numpy().flatten()

        elif mode == "跨语种复刻":
            if not prompt_wav:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt audio is required for 跨语种复刻 mode",
                )

            if cosyvoice.instruct:
                raise HTTPException(
                    status_code=400,
                    detail="Cross-lingual mode is not supported with instruct model",
                )

            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            logging.info("Using cross-lingual inference")
            for i in cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=stream, speed=speed
            ):
                audio_data = i["tts_speech"].numpy().flatten()

        elif mode == "自然语言控制":
            if not instruct_text:
                raise HTTPException(
                    status_code=400,
                    detail="Instruct text is required for 自然语言控制 mode",
                )

            if not cosyvoice.instruct:
                raise HTTPException(
                    status_code=400,
                    detail="Instruct mode requires an instruct-capable model",
                )

            if not voice and len(sft_spk) > 0:
                voice = sft_spk[0]

            logging.info("Using instruct inference")
            prompt_speech_16k = None
            if prompt_wav:
                prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))

            for i in cosyvoice.inference_instruct2(
                tts_text,
                instruct_text,
                prompt_speech_16k,
                voice,
                stream=stream,
                speed=speed,
            ):
                audio_data = i["tts_speech"].numpy().flatten()

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mode: {mode}. Available modes: {INFERENCE_MODES}",
            )

        # 清理临时文件
        if prompt_wav and os.path.exists(prompt_wav):
            os.remove(prompt_wav)

        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # 转换为请求的格式并返回
        audio_bytes = audio_to_bytes(
            audio_data, cosyvoice.sample_rate, format=response_format
        )

        content_type = f"audio/{response_format}"
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=content_type)

    except Exception as e:
        # 确保清理临时文件
        if "prompt_wav" in locals() and prompt_wav and os.path.exists(prompt_wav):
            os.remove(prompt_wav)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices (OpenAI compatible)"""
    voices = []
    for voice_id in sft_spk:
        if voice_id:  # Skip empty voice IDs
            voices.append(
                {
                    "id": voice_id,
                    "name": voice_id,
                    "preview_url": None,
                    "category": "cosyvoice",
                }
            )

    return {"voices": voices}


@app.get("/modes")
async def list_modes():
    """List available inference modes"""
    return {"modes": INFERENCE_MODES}


@app.post("/upload_prompt")
async def upload_prompt(file: UploadFile = File(...)):
    """Upload prompt audio file and return base64 encoded string"""
    try:
        contents = await file.read()
        # Save to temporary file to verify it's valid audio
        temp_file = f"temp_{random.randint(1000, 9999)}.wav"
        with open(temp_file, "wb") as f:
            f.write(contents)

        # Check if it's a valid audio file
        try:
            info = torchaudio.info(temp_file)
            if info.sample_rate < prompt_sr:
                os.remove(temp_file)
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Audio sample rate {info.sample_rate} is lower than required {prompt_sr}Hz"
                    },
                )
        except Exception as e:
            os.remove(temp_file)
            return JSONResponse(
                status_code=400, content={"error": f"Invalid audio file: {str(e)}"}
            )

        # Convert to base64
        base64_audio = base64.b64encode(contents).decode("utf-8")
        os.remove(temp_file)

        return {"base64_audio": base64_audio}
    except Exception as e:
        if "temp_file" in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to process audio: {str(e)}"}
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/app/models/iic/CosyVoice2-0.5B",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()

    global cosyvoice, sft_spk, default_data

    try:
        cosyvoice = CosyVoice2(args.model_dir, load_vllm=True)
    except Exception:
        raise TypeError("No valid model_type!")

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = [""]
    default_data = np.zeros(cosyvoice.sample_rate)

    logging.info(f"Server started on {args.host}:{args.port}")
    logging.info(f"Available voices: {sft_spk}")
    logging.info(f"Available modes: {INFERENCE_MODES}")

    # Start the FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
