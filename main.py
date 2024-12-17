# 加载环境变量
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from Agent.MemoryLaneAgent import MemoryLaneAgent
from langchain_chroma import Chroma
from langchain.schema import Document
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_vector_store(embeddings):
    """初始化向量数据库，添加错误处理"""
    try:
        # 使用标准的生产环境向量存储路径
        persist_directory = os.getenv('PRODUCTION_VECTOR_STORE_PATH', 'db/production/vector_store')
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化Chroma
        db = Chroma(
            collection_name="chat_history",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # 如果集合为空，添加一个初始文档
        if db._collection.count() == 0:
            db.add_documents([Document(page_content="对话初始化")])
            
        return db
    except Exception as e:
        logger.error(f"初始化向量数据库失败: {str(e)}")
        raise

def launch_chat(agent: MemoryLaneAgent):
    """运行聊天循环"""
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"

    print(f"{ai_icon}：你好！我是你的AI助手，让我们开始聊天吧！")
    
    # 获取第一个问题
    first_question = agent.start_chat()
    print(f"{ai_icon}：{first_question}")
    
    while True:
        try:
            user_input = input(f"{human_icon}：")
            if user_input.strip().lower() in ["quit", "exit", "bye"]:
                print(f"{ai_icon}：再见！")
                break
                
            reply = agent.chat(user_input, verbose=False)
            print(f"{ai_icon}：{reply}\n")
        except Exception as e:
            logger.error(f"聊天过程出错: {str(e)}")
            print(f"{ai_icon}：抱歉，我遇到了一些问题，请重试。")

def main():
    # 检查环境变量是否存在
    required_env_vars = [
        "ZHIPUAI_API_KEY",
        "ZHIPUAI_MODEL_NAME",
        "ZHIPUAI_EMBEDDING_KEY",
        "ZHIPUAI_EMBEDDING_MODEL"
    ]
    
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"环境变量 {var} 未设置！请在.env文件中设置该变量。")

    try:
        # 初始化语言模型
        llm = ChatZhipuAI(
            model=os.getenv("ZHIPUAI_MODEL_NAME"),
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            temperature=0.7,
        )

        # 初始化embeddings
        embeddings = ZhipuAIEmbeddings(
            model=os.getenv("ZHIPUAI_EMBEDDING_MODEL"),
            api_key=os.getenv("ZHIPUAI_EMBEDDING_KEY"),
        )

        # 初始化向量数据库
        db = init_vector_store(embeddings)
        retriever = db.as_retriever(
            search_kwargs={"k": 3}  # 检索最相关的3条历史记录
        )

        # 初始化聊天智能体
        agent = MemoryLaneAgent(
            llm=llm,
            prompts_path="./prompts/main",
            main_prompt_file="main.json",
            memory_retriever=retriever,
            max_token_limit=4000
        )

        # 启动聊天
        launch_chat(agent)
        
    except Exception as e:
        logger.error(f"程序初始化失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
