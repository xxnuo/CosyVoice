import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from webui import router_model, router_openai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


# 初始化 FastAPI 应用
app = FastAPI(
    description="""CosyVoice OpenAI Compatible TTS Server

音色：有三种类型的音色，一种是模型自带音色，一种是预装音色，一种是用户提供音频及音频文本。
模型自带音色：在 /audio/voices_sft 接口中获取，是模型自带的音色，可以用于模型自带音色（mode=sft）生成任务。
预装音色：在 /audio/voices 接口中获取，是系统预装的音色，可以用于克隆音色（mode=clone）、跨语种克隆（mode=crosslingual）任务。
用户音频：与预装音色等价，可以用于克隆音色（mode=clone）、跨语种克隆（mode=crosslingual）任务。（注意：用户提供音频时，需要提供对应音频文本，否则会报错。）
"""
)

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

# 注册路由
app.include_router(router_openai.router, prefix="/v1")
app.include_router(router_model.router)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
