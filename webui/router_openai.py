import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import FileResponse

from webui.config import Config
from webui.router_model import engine, global_engine_lock
from CosyVoice.webui.schemas_openai import OpenAISpeechRequest

logger = logging.getLogger()
router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


@router.post("/audio/voices_sft", description="[专属] 获取模型自带音色列表")
def voice_sft_list():
    """[专属] 获取模型自带音色列表"""
    return {"status": "success", "voices": engine.get_available_spks()}


@router.post("/audio/voices", description="获取预装音色列表")
def voice_list():
    """获取预装音色列表"""
    return {"status": "success", "voices": os.listdir(Config.voice_dir)}


@router.post("/audio/voices/listen", description="[专属] 试听预装音色")
def voice_listen(voice: str):
    """[专属] 试听预装音色"""
    voice_wav_path = os.path.join(Config.voice_dir, voice)
    if not os.path.exists(voice_wav_path):
        raise HTTPException(status_code=404, detail="音色不存在")
    return FileResponse(voice_wav_path)


@router.post("/audio/speech", description="生成音频")
def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """生成音频"""
    try:
        # if request.stream: # 默认是流式输出，暂时不支持非流式
        with global_engine_lock:
            pass

    except Exception as e:
        logger.error(f"生成音频失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成音频失败: {str(e)}")


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
