import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from webui import router_model, router_openai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


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

# 注册路由
app.include_router(router_openai.router, prefix="/v1")
app.include_router(router_model.router)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
