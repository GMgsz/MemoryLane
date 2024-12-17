import os
import pytest
from Utils.DatabaseManager import DatabaseManager
from datetime import datetime
from langchain_community.embeddings import ZhipuAIEmbeddings
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

@pytest.fixture
def db_manager():
    """创建测试用的数据库管理器"""
    # 使用临时数据库文件
    test_db_path = "test_dialogue_analysis.db"
    
    # 如果存在旧的测试数据库，先删除
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db_manager = DatabaseManager(db_path=test_db_path)
    yield db_manager
    
    # 测试完成后清理
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_save_and_get_dialogue(db_manager):
    """测试对话的保存和获取"""
    # 准备测试数据
    dialogue_id = f"test_dialogue_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    system_query = "测试系统问题"
    user_response = "测试用户回答"
    entity_ids = ["entity_1", "entity_2"]
    
    # 保存对话
    db_manager.save_dialogue(dialogue_id, system_query, user_response, entity_ids)
    
    # 获取对话相关的实体
    entities = db_manager.get_dialogue_entities(dialogue_id)
    assert len(entities) == 0  # 因为还没有保存实体

def test_save_and_get_entity(db_manager):
    """测试实体的保存和获取"""
    # 准备测试数据
    entity_id = "entity_1"
    name = "篮球"
    entity_type = "activity"
    attributes = {
        "time": "从小",
        "nature": "运动"
    }
    
    # 保存实体
    db_manager.save_entity(entity_id, name, entity_type, attributes)
    
    # 获取实体
    entity = db_manager.get_entity_by_id(entity_id)
    assert entity is not None
    assert entity["name"] == name
    assert entity["type"] == entity_type
    assert entity["attributes"] == attributes

def test_save_and_get_topic(db_manager):
    """测试话题的保存和获取"""
    # 准备测试数据
    topic_id = "topic_1"
    name = "兴趣"
    primary_category = "兴趣"
    
    # 保存话题
    db_manager.save_topic(topic_id, name, primary_category)
    
    # 获取话题
    topic = db_manager.get_topic_by_id(topic_id)
    assert topic is not None
    assert topic["name"] == name
    assert topic["primary_category"] == primary_category

def test_entity_topic_relation(db_manager):
    """测试实体-话题���联"""
    # 准备测试数据
    entity_id = "entity_1"
    topic_id = "topic_1"
    score = 0.9
    
    # 保存实体和话题
    db_manager.save_entity(entity_id, "篮球", "activity", {})
    db_manager.save_topic(topic_id, "兴趣", "兴趣")
    
    # 保存关联关系
    db_manager.save_entity_topic_relation(entity_id, topic_id, score)
    
    # 获取实体的话题
    topics = db_manager.get_entity_topics(entity_id)
    assert len(topics) == 1
    assert topics[0]["id"] == topic_id
    assert topics[0]["score"] == score

@pytest.mark.skipif(not os.getenv("ZHIPUAI_Embedding_key"), reason="需要智谱API key")
def test_vector_store(db_manager):
    """测试向量数据库功能"""
    # 初始化embeddings
    embeddings = ZhipuAIEmbeddings(
        model=os.getenv("ZHIPUAI_Embedding_model"),
        api_key=os.getenv("ZHIPUAI_Embedding_key"),
    )
    
    # 初始化向量数据库
    db_manager.init_vector_store(embeddings)
    
    # 保存文本
    dialogue_id = "test_dialogue_1"
    text = "从小打篮球"
    metadata = {"timestamp": datetime.now().isoformat()}
    
    db_manager.save_to_vector_store(dialogue_id, text, metadata)
    
    # 搜索相似内容
    results = db_manager.search_similar_dialogues("篮球运动")
    assert len(results) > 0

def test_batch_operations(db_manager):
    """测试批量操作"""
    # 准备测试数据
    entities = [
        {
            "id": "entity_1",
            "name": "篮球",
            "type": "activity",
            "attributes": {"time": "从小"}
        },
        {
            "id": "entity_2",
            "name": "足球",
            "type": "activity",
            "attributes": {"time": "最近"}
        }
    ]
    
    # 批量保存实体
    db_manager.save_entities_batch(entities)
    
    # 验证
    for entity_data in entities:
        entity = db_manager.get_entity_by_id(entity_data["id"])
        assert entity is not None
        assert entity["name"] == entity_data["name"] 