import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from core.dialogue_manager import DialogueManager
from core.vector_store import VectorStoreManager
from core.content_processor import ContentProcessor
from core.theme_manager import ThemeManager
from models.schemas import DialogueContext, DialogueTurn
import uuid
from datetime import datetime

async def test_content_processing():
    """测试内容处理功能"""
    print("\n=== 测试内容处理 ===")
    
    # 初始化组件
    llm = ChatZhipuAI(
        model=os.getenv('Extract_Model'),
        api_key=os.getenv('Extract_API_key')
    )
    content_processor = ContentProcessor(
        extract_llm=llm,
        identify_llm=llm,
        vector_store=None
    )
    
    # 测试用例
    test_text = "去年春节，我和父母、姐姐一起在北京的家里团聚，大家一起包饺子，很开心。"
    
    # 1. 测试实体提取
    entities_and_keywords = await content_processor._extract_entities_and_keywords(test_text)
    print("\n1. 实体提取结果:")
    print(f"实体: {entities_and_keywords['entities']}")
    print(f"关键词: {entities_and_keywords['keywords']}")
    
    # 2. 测试主题识别
    themes = await content_processor._identify_themes(test_text, entities_and_keywords)
    print("\n2. 主题识别结果:")
    print(f"识别的主题: {themes}")

async def test_storage_mechanism():
    """测试存储机制"""
    print("\n=== 测试存储机制 ===")
    
    # 初始化组件
    embeddings = ZhipuAIEmbeddings(
        model=os.getenv('Embedding_model'),
        api_key=os.getenv('Embedding_API_key')
    )
    vector_store = VectorStoreManager(embeddings)
    
    # 测试数据
    test_content = {
        "text": "去年春节，我和父母、姐姐一起在北京的家里团聚。",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "entities": {
                "人物": ["我", "父母", "姐姐"],
                "时间": ["去年春节"],
                "地点": ["北京", "家里"],
                "事件": ["团聚"]
            },
            "themes": ["家庭", "节日"],
            "dialogue_id": str(uuid.uuid4())
        }
    }
    
    # 1. 测试向量存储
    success = await vector_store.add_memory(
        text=test_content["text"],
        metadata=test_content["metadata"]
    )
    print("\n1. 向量存储测试:")
    print(f"存储状态: {'成功' if success else '失败'}")
    
    # 2. 测试相似度检索
    similar_contents = await vector_store.search_similar(test_content["text"], k=1)
    if similar_contents:
        print("\n2. 相似度检索测试:")
        for content in similar_contents:
            print(f"相似内容: {content['content']}")
            print(f"相似度分数: {content['score']:.2f}")
            print(f"元数据: {content['metadata']}")

async def test_context_management():
    """测试上下文管理"""
    print("\n=== 测试上下文管理 ===")
    
    # 创建测试对话上下文
    dialogue_context = [
        DialogueTurn(
            id=str(uuid.uuid4()),
            question="请谈谈你的家庭",
            answer="我有一个温暖的家庭，父母和姐姐都很关心我。",
            topic="家庭",
            emotion_score=0.8,
            interest_score=0.7,
            depth_level=1
        ),
        DialogueTurn(
            id=str(uuid.uuid4()),
            question="你们平时会一起做什么？",
            answer="春节的时候我们会一起包饺子，聊天，看春晚。",
            topic="家庭",
            emotion_score=0.9,
            interest_score=0.8,
            depth_level=2
        )
    ]
    
    # 打印上下文信息
    print("\n当前对话上下文:")
    for turn in dialogue_context:
        print(f"\n问题: {turn.question}")
        print(f"回答: {turn.answer}")
        print(f"主题: {turn.topic}")
        print(f"情感分数: {turn.emotion_score}")
        print(f"兴趣分数: {turn.interest_score}")
        print(f"深度等级: {turn.depth_level}")

async def main():
    """运行所有测试"""
    _ = load_dotenv(find_dotenv())
    
    await test_content_processing()
    await test_storage_mechanism()
    await test_context_management()

if __name__ == "__main__":
    asyncio.run(main()) 