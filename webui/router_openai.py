import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from webui.config import Config
from webui.router_model import engine, global_engine_lock
from webui.schemas import OpenAISpeechRequest

logger = logging.getLogger()
router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


@router.post("/audio/voices_sft", description="获取预训练音色列表")
def voice_list_sft():
    """获取预训练音色列表"""
    return {"status": "success", "voices": engine.get_available_spks()}


@router.post("/audio/voices", description="获取预装音色列表")
def voice_list():
    """获取预装音色列表"""
    return {"status": "success", "voices": os.listdir(Config.voice_dir)}


@router.post("/audio/voices/listen", description="试听预装音色")
def voice_listen(voice: str):
    """试听预装音色"""
    voice_wav_path = os.path.join(Config.voice_dir, voice)
    if not os.path.exists(voice_wav_path):
        raise HTTPException(status_code=404, detail="音色不存在")
    return FileResponse(voice_wav_path)


@router.post("/generate", description="生成音频")
def generate(request: OpenAISpeechRequest):
    try:
        with global_engine_lock:
            pass
    except Exception as e:
        logger.error(f"生成图片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成图片失败: {str(e)}")


@router.get("/output/{filename}", description="获取生成的图片")
def get_image(filename: str):
    file_path = Path(engine.output_folder) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(file_path)
