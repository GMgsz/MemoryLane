import os
import sys
import pytest
import asyncio
import sqlite3
import tempfile
import pytest_asyncio
from pathlib import Path

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.append(project_root)

# 在设置路径后再导入模块
from Utils.DatabaseManager import DatabaseManager
from unittest.mock import MagicMock
from langchain.schema import Document

@pytest.fixture(scope="function")
def event_loop():
    """创建一个函数级别的事件循环"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def db_manager(event_loop):
    """提供测试数据库管理器实例"""
    # 使用环境变量中的测试数据库路径
    test_db_path = os.getenv('TEST_DB_PATH', 'tests/test_data/test.db')
    test_dir = os.path.dirname(test_db_path)
    os.makedirs(test_dir, exist_ok=True)
    
    # 如果存在旧的测试文件，先删除
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # 初始化数据库管理器
    db_manager = DatabaseManager(test_db_path)
    
    # Mock向量存储
    mock_results = [
        (Document(page_content="测试对话1", metadata={"id": "test_1"}), 0.8),
        (Document(page_content="测试对话2", metadata={"id": "test_2"}), 0.6)
    ]
    db_manager.vector_store = MagicMock()
    db_manager.vector_store.similarity_search_with_score = MagicMock(return_value=mock_results)
    
    # 创建一个测试话题
    db_manager.save_topic("topic_1", "兴趣", "兴趣")
    
    try:
        yield db_manager
    finally:
        db_manager.close()
        # 不删除vector_store目录，避免文件锁定问题

@pytest_asyncio.fixture
async def setup_test_entities(db_manager):
    """准备测试用的实体数据"""
    entities = [
        {
            "id": f"test_entity_{i}",
            "name": f"测试实体_{i}",
            "type": "activity",
            "attributes": {"time": "测试时间"},
            "primary_topic": "兴趣"
        }
        for i in range(5)
    ]
    # 先保存实体
    for entity in entities:
        db_manager.save_entity(
            entity["id"],
            entity["name"],
            entity["type"],
            entity["attributes"]
        )
        # 关联到话题
        db_manager.save_entity_topic_relation(entity["id"], "topic_1")
    return entities

@pytest_asyncio.fixture
async def setup_test_dialogues(db_manager, setup_test_entities):
    """准备测试用的对话数据"""
    dialogues = [
        {
            "id": f"test_dialogue_{i}",
            "system_query": "测试问题",
            "user_response": "测试回答",
            "entity_ids": [f"test_entity_{i}"]
        }
        for i in range(5)
    ]
    
    for dialogue in dialogues:
        await db_manager.save_dialogue(
            dialogue["id"],
            dialogue["system_query"],
            dialogue["user_response"],
            dialogue["entity_ids"]
        )
    
    return dialogues

@pytest_asyncio.fixture
async def llm_client():
    """模拟LLM客户端"""
    mock_llm = MagicMock()
    
    async def mock_analyze_clustering(*args, **kwargs):
        existing_clusters = kwargs.get("existing_clusters", [])
        
        if not existing_clusters:
            # 第一个实体：创建新聚类
            return {
                "decision": "CREATE_NEW",
                "target_cluster": "测试聚类",
                "reasoning": "第一个实体，创建新聚类"
            }
        else:
            # 后续实体：加入第一个聚类
            return {
                "decision": "JOIN_EXISTING",
                "target_cluster": existing_clusters[0]["id"],
                "reasoning": "加入已有聚类"
            }
    
    async def mock_evaluate_cluster(*args, **kwargs):
        return {
            "can_generate": True,
            "completeness_score": 0.8,
            "content_outline": "1. 背景介绍\n2. 主要事件\n3. 感受总结"
        }
    
    async def mock_invoke(*args, **kwargs):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(
            "这是一段生成的测试内容。\n\n"
            "在这个故事中，主人公展现了对篮球的热爱。从小就开始打篮球，"
            "经常和朋友一起切磋球技。到了高中时期，更是勇于挑战高年级的学长，"
            "虽然输多赢少，但在这个过程中学到了很多宝贵的经验和技术。\n\n"
            "这段经历不仅锻炼了球技，更培养了永不言弃的精神。"
        )
    
    mock_llm.analyze_clustering = mock_analyze_clustering
    mock_llm.evaluate_cluster = mock_evaluate_cluster
    mock_llm.invoke = mock_invoke
    
    return mock_llm