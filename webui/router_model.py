import logging
import threading

from fastapi import APIRouter, HTTPException

from webui.config import Config
from webui.engine import CosyVoiceEngine

logger = logging.getLogger()

global_engine_lock = threading.RLock()
engine = CosyVoiceEngine()

router = APIRouter(
    tags=["Model"],
    responses={404: {"description": "Not found"}},
)


@router.get("/model/status", description="获取模型状态信息")
def model_status():
    """获取模型加载状态和最后使用时间"""
    return {"status": "ok", "model_loaded": engine.ok}


@router.post("/model/unload", description="手动卸载模型")
def manual_unload_model():
    """手动卸载模型接口"""
    with global_engine_lock:
        engine.unload_model()
    return {"status": "success", "message": "模型已卸载"}


@router.post("/model/load", description="手动加载模型")
def manual_load_model(model: str = Config.model_name):
    """手动加载模型接口"""
    try:
        with global_engine_lock:
            engine.load_model(model_name=model)
        return {"status": "success", "message": f"模型 {model} 已加载"}
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")
