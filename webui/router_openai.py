import io
import logging
import os
import wave

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from webui.config import Config
from webui.router_model import engine, global_engine_lock
from webui.schemas_openai import OpenAISpeechRequest

logger = logging.getLogger()
router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/audio/voices_sft",
    description="[专属] 获取模型自带音色列表",
)
def voice_sft_list():
    """[专属] 获取模型自带音色列表"""
    return {"status": "success", "voices": engine.get_available_spks()}


@router.get("/audio/voices", description="获取预装音色列表")
def voice_list():
    """获取预装音色列表"""
    voices = [file for file in os.listdir(Config.voice_dir) if file.endswith(".wav")]
    return {"status": "success", "voices": voices}


@router.get("/audio/voices/listen", description="[专属] 试听预装音色")
def voice_listen(voice: str):
    """[专属] 试听预装音色"""
    voice_wav_path = os.path.join(Config.voice_dir, voice)
    if not os.path.exists(voice_wav_path):
        raise HTTPException(status_code=404, detail="音色不存在")
    return FileResponse(voice_wav_path)


@router.post("/audio/speech", description="生成音频，自动加载模型")
def create_speech(
    request: OpenAISpeechRequest,
    # client_request: Request,  # 兼容性参数
    # x_raw_response: str,  # = Header(None, alias="x-raw-response"),  # 兼容性参数
):
    """生成音频，自动加载模型"""
    # 解析请求
    tts_text = request.input
    if tts_text == "":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request_error",
                "message": "input 不能为空",
                "type": "invalid_request_error",
            },
        )

    mode = ""  # 模式
    voice = ""  # 音色
    try:
        parts = request.voice.split("_", 1)
        if len(parts) < 2:
            raise ValueError("音色格式不合法，应为'模式_对应音色名称'格式")
        mode = parts[0]
        voice = parts[1]
    except Exception as e:
        logger.error(f"voice 参数不合法: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request_error",
                "message": f"voice 参数不合法: {str(e)}",
                "type": "invalid_request_error",
            },
        )

    # 检查音色是否存在
    prompt_wav_path = ""  # 提示音色音频路径
    prompt_wav_text = ""  # 提示音色音频文本
    if mode == "sft":
        if voice not in engine.get_available_spks():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "invalid_request_error",
                    "message": f"音色不存在: {voice}",
                    "type": "invalid_request_error",
                },
            )
    else:
        prompt_wav_path = os.path.join(Config.voice_dir, voice)
        prompt_wav_text_path = os.path.join(Config.voice_dir, voice + ".txt")
        if not os.path.exists(prompt_wav_path):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "invalid_request_error",
                    "message": f"音色不存在: {voice}",
                    "type": "invalid_request_error",
                },
            )
        if os.path.exists(prompt_wav_text_path):
            with open(prompt_wav_text_path, "r") as f:
                prompt_wav_text = f.read()
                if prompt_wav_text == "":
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "invalid_request_error",
                            "message": f"程序损坏，音色文本为空: {voice}",
                            "type": "invalid_request_error",
                        },
                    )
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "invalid_request_error",
                    "message": f"程序损坏，音色文本丢失: {voice}",
                    "type": "invalid_request_error",
                },
            )

    # 生成音频
    try:
        with global_engine_lock:
            generator = engine.generate(
                mode=mode,
                spk_id=voice,
                tts_text=tts_text,
                prompt_wav_path=prompt_wav_path,
                prompt_wav_text=prompt_wav_text,
                speed=request.speed,
                stream=request.stream,
                max_val=request.volume_multiplier,
            )
            if request.stream:
                # 流式返回音频数据
                async def stream_audio():
                    try:
                        audio_chunks = []  # 收集所有音频块用于计算总大小
                        sample_rate = None
                        
                        # 首先收集所有音频数据
                        for sr, audio_data in generator:
                            # 检查是否为空白音频块
                            if len(audio_data) == 0 or np.all(np.abs(audio_data) < 1e-6):
                                logger.debug("跳过空白音频块")
                                continue
                                
                            # 将 numpy 数组转换为 16 位整数
                            audio_int16 = (audio_data * (2**15)).astype("int16")
                            audio_chunks.append(audio_int16.tobytes())
                            sample_rate = sr  # 保存采样率
                        
                        if not audio_chunks:
                            logger.error("没有生成任何有效音频数据")
                            raise HTTPException(status_code=500, detail="没有生成任何有效音频数据")
                            
                        # 计算总音频数据大小
                        total_audio_size = sum(len(chunk) for chunk in audio_chunks)
                        
                        # 创建完整的 WAV 文件
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, "wb") as wav_file:
                            wav_file.setnchannels(1)  # 单声道
                            wav_file.setsampwidth(2)  # 16位 = 2字节
                            wav_file.setframerate(sample_rate)
                            # 写入所有音频数据
                            for chunk in audio_chunks:
                                wav_file.writeframes(chunk)
                        
                        # 获取完整的 WAV 文件数据
                        wav_buffer.seek(0)
                        wav_data = wav_buffer.read()
                        
                        # 一次性返回完整的 WAV 文件
                        yield wav_data

                    except Exception as e:
                        logger.error(f"流式音频生成错误: {str(e)}")
                        raise HTTPException(
                            status_code=500, detail=f"流式音频生成错误: {str(e)}"
                        )

                return StreamingResponse(
                    stream_audio(),
                    media_type=f"audio/{request.response_format}",
                    headers={
                        "X-Accel-Buffering": "no",
                        "Cache-Control": "no-cache",
                        "Content-Type": f"audio/{request.response_format}",
                        "Transfer-Encoding": "chunked",
                        "Content-Encoding": "identity",  # 禁用压缩
                    },
                )
            else:
                # 非流式模式：收集所有音频片段并合并
                audio_chunks = []
                sample_rate = Config.sample_rate  # 默认采样率

                try:
                    for sr, audio_data in generator:
                        # 检查是否为空白音频块
                        if len(audio_data) == 0 or np.all(np.abs(audio_data) < 1e-6):
                            logger.debug("跳过空白音频块")
                            continue

                        audio_chunks.append(audio_data)
                        sample_rate = sr  # 保存采样率
                except Exception as e:
                    logger.error(f"非流式音频生成错误: {str(e)}")
                    raise HTTPException(
                        status_code=500, detail=f"非流式音频生成错误: {str(e)}"
                    )

                # 合并所有音频片段
                if audio_chunks:
                    combined_audio = np.concatenate(audio_chunks)

                    # 转换为16位整数PCM
                    audio_int16 = (combined_audio * (2**15)).astype("int16")

                    # 创建内存中的 WAV 文件
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)  # 单声道
                        wav_file.setsampwidth(2)  # 16位 = 2字节
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())

                    wav_buffer.seek(0)
                    wav_data = wav_buffer.read()

                    return Response(
                        content=wav_data,
                        media_type=f"audio/{request.response_format}",
                        headers={
                            "Cache-Control": "no-cache",
                            "Content-Type": f"audio/{request.response_format}",
                            "Content-Encoding": "identity",  # 禁用压缩
                        },
                    )
                else:
                    raise HTTPException(
                        status_code=500, detail="音频生成失败：没有生成任何音频数据"
                    )
    except Exception as e:
        logger.error(f"生成音频失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成音频失败: {str(e)}")


# 兼容性接口


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


@router.get("/models")
async def list_models():
    """List all available models"""
    try:
        # Create standard model list
        models = [
            {
                "id": Config.model_name,
                "object": "model",
                "created": 1686935002,
                "owned_by": "CosyVoice",
            },
        ]

        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model list",
                "type": "server_error",
            },
        )


@router.get("/models/{model}")
async def retrieve_model(model: str):
    """Retrieve a specific model"""
    try:
        # Define available models
        models = {
            Config.model_name: {
                "id": Config.model_name,
                "object": "model",
                "created": 1686935002,
                "owned_by": "CosyVoice",
            },
        }

        # Check if requested model exists
        if model not in models:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "message": f"Model '{model}' not found",
                    "type": "invalid_request_error",
                },
            )

        # Return the specific model
        return models[model]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve model information",
                "type": "server_error",
            },
        )
