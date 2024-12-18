import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from controllers.memory_lane_controller import MemoryLaneController
from scripts.check_env import check_environment
import uvicorn
from Utils.logger import setup_logger

# 设置日志
logger = setup_logger("main")

# 检查环境
if not check_environment():
    logger.error("环境检查失败")
    raise SystemExit(1)

app = FastAPI(title="MemoryLane API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建控制器实例
controller = MemoryLaneController()

class ChatInput(BaseModel):
    user_input: str

@app.post("/start")
async def start_conversation():
    """开始新的对话"""
    try:
        first_question = await controller.start_conversation()
        return {"system_query": first_question}
    except Exception as e:
        logger.error(f"启动对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(chat_input: ChatInput):
    """处理用户输入的对话接口"""
    try:
        result = await controller.process_user_input(chat_input.user_input)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"处理对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/contents")
async def get_contents(topic: str = None):
    """获取生成的内容"""
    try:
        contents = await controller.get_generated_contents(topic)
        return {"contents": contents}
    except Exception as e:
        logger.error(f"获取内容失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """关闭应用时的清理工作"""
    try:
        await controller.close()
    except Exception as e:
        logger.error(f"关闭应用失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 