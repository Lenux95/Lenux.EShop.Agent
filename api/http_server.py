"""
FastAPI 服务器入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings, setup_logging, get_logger

# 初始化日志
setup_logging()
logger = get_logger("http_server")

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG 知识库问答系统 API"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入路由
from api.routers import test

# 注册路由
app.include_router(test.router, prefix="/api/test", tags=["agent测试接口"])

@app.get("/")
async def root():
    """健康检查接口"""
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 启动 {settings.app_name} 服务")
    logger.info(f"📍 监听地址: http://{settings.server_host}:{settings.server_port}")
    
    uvicorn.run(
        "api.http_server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload
    )