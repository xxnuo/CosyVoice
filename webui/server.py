import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from webui.config import Config
from webui.engine import CosyVoiceEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


global_engine_lock = threading.RLock()
engine = CosyVoiceEngine()

# 初始化 FastAPI 应用
app = FastAPI(description="CosyVoice Server")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头部的中间件"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["x-process-time"] = str(process_time)
    return response


# 添加Gzip压缩中间件
app.add_middleware(GZipMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/model/status", description="获取模型状态信息")
def model_status():
    """获取模型加载状态和最后使用时间"""
    return {"status": "ok", "model_loaded": engine.ok}


@app.post("/voice/list_sft", description="获取预训练音色列表")
def voice_list_sft():
    """获取预训练音色列表"""
    return {"status": "success", "voices": engine.get_available_spks()}


@app.post("/voice/list", description="获取预装音色列表")
def voice_list():
    """获取预装音色列表"""
    return {"status": "success", "voices": os.listdir(Config.voice_dir)}


@app.post("/voice/listen", description="试听预装音色")
def voice_listen(voice: str):
    """试听预装音色"""
    voice_wav_path = os.path.join(Config.voice_dir, voice)
    if not os.path.exists(voice_wav_path):
        raise HTTPException(status_code=404, detail="音色不存在")
    return FileResponse(voice_wav_path)


@app.post("/model/unload", description="手动卸载模型")
def manual_unload_model():
    """手动卸载模型接口"""
    with global_engine_lock:
        engine.unload_model()
    return {"status": "success", "message": "模型已卸载"}


@app.post("/model/load", description="手动加载模型")
def manual_load_model(model: str = Config.model_name):
    """手动加载模型接口"""
    try:
        with global_engine_lock:
            engine.load_model(model_name=model)
        return {"status": "success", "message": f"模型 {model} 已加载"}
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


class GenerateRequest(BaseModel):
    prompt: str = "A girl in a cyberpunk city at night"
    model: Optional[str] = default_model
    steps: Optional[int] = 4
    height: Optional[int] = 512
    width: Optional[int] = 512
    images: Optional[int] = 1
    guidance_scale: Optional[float] = 3.5
    seed: Optional[int] = 0
    randomize_seed: Optional[bool] = True


class GenerateResponse(BaseModel):
    status: str
    seed: int
    image_paths: List[str]


@app.post("/generate", description="生成图片", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        with global_engine_lock:
            _, seed, saved_paths = engine.generate(
                prompt=request.prompt,
                checkpoint=request.model,
                num_images_per_prompt=request.images,
                randomize_seed=request.randomize_seed,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                auto_unload=False,
            )

            return {
                "status": "success",
                "seed": seed,
                "image_paths": [p.lstrip(default_output_folder) for p in saved_paths],
            }
    except Exception as e:
        logger.error(f"生成图片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成图片失败: {str(e)}")


@app.get("/output/{filename}", description="获取生成的图片")
def get_image(filename: str):
    file_path = Path(engine.output_folder) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(file_path)
