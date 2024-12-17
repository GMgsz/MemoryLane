import pytest
from Agent.ContentGenerationAgent import ContentGenerationAgent
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_entities():
    return [
        {
            "id": "entity_1",
            "name": "篮球",
            "type": "activity",
            "attributes": {"time": "小时候"},
            "primary_topic": "兴趣"
        },
        {
            "id": "entity_2",
            "name": "组队",
            "type": "activity",
            "attributes": {"time": "高中"},
            "primary_topic": "兴趣"
        }
    ]

@pytest.fixture
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

@pytest.fixture
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

@pytest.mark.asyncio
async def test_entity_clustering(db_manager, llm_client, mock_entities, setup_test_dialogues):
    """测试实体聚类功能"""
    agent = ContentGenerationAgent(db_manager, llm_client)
    
    # 先保存实体和话题关联
    for entity in mock_entities:
        db_manager.save_entity(
            entity["id"],
            entity["name"],
            entity["type"],
            entity["attributes"]
        )
        db_manager.save_entity_topic_relation(entity["id"], "topic_1")
    
    # 测试实体组织
    await agent._organize_entities(mock_entities)
    
    # 验证聚类结果
    clusters = db_manager.get_cluster_by_topic("兴趣")
    assert len(clusters) > 0
    logger.info(f"找到的聚类: {clusters}")
    
    # 验证每个聚类的实体
    total_entities = []
    for cluster in clusters:
        entities = db_manager.get_cluster_entities(cluster["id"])
        logger.info(f"聚类 {cluster['id']} 的实体: {entities}")
        total_entities.extend(entities)
    
    # 验证所有实体都被分配到聚类中
    assert len(total_entities) == len(mock_entities), f"期望 {len(mock_entities)} 个实体，实际有 {len(total_entities)} 个"
    
    # 验证所有实体都被正确分配
    entity_ids = {entity["id"] for entity in total_entities}
    mock_entity_ids = {entity["id"] for entity in mock_entities}
    assert entity_ids == mock_entity_ids, f"实体ID不匹配: \n期望: {mock_entity_ids}\n实际: {entity_ids}"

@pytest.mark.asyncio
async def test_cluster_evaluation(db_manager, llm_client, setup_test_entities):
    """测试聚类评估功能"""
    agent = ContentGenerationAgent(db_manager, llm_client)
    
    # 创建测试聚类
    cluster_id = db_manager.create_cluster("测试聚类", "兴趣")
    
    # 添加足够数量的实体到聚类
    for i in range(agent.ENTITY_THRESHOLD):
        entity_id = f"test_entity_{i}"
        db_manager.save_entity(
            entity_id,
            f"测试实体_{i}",
            "activity",
            {"time": "测试时间"}
        )
        db_manager.add_entity_to_cluster(entity_id, cluster_id)
    
    # 触发评估
    await agent._evaluate_single_cluster({"id": cluster_id})
    
    # 验证评估结果
    cluster = db_manager.get_cluster(cluster_id)
    assert cluster["state"] == "GENERATED"  # 应该已生成内容
    
    # 验证内容生成
    contents = db_manager.get_cluster_contents(cluster_id)
    assert len(contents) > 0
    assert contents[0]["content"] == "这是测试生成的内容"

@pytest.mark.asyncio
async def test_content_generation_trigger(db_manager, llm_client):
    """测试内容生成触发机制"""
    agent = ContentGenerationAgent(db_manager, llm_client)
    
    # 测试对话计数触发
    for i in range(agent.TRIGGER_INTERVAL - 1):
        agent.on_dialogue_complete(f"dialogue_{i}")
        assert agent.dialogue_count == i + 1
    
    # 触发生成
    agent.on_dialogue_complete(f"dialogue_{agent.TRIGGER_INTERVAL}")
    assert agent.dialogue_count == 0  # 计数应该重置

@pytest.mark.asyncio
async def test_content_generation_process(db_manager, llm_client, mock_entities):
    """测试完整的内容生成流程"""
    agent = ContentGenerationAgent(db_manager, llm_client)
    
    # 准备测试数据
    for entity in mock_entities:
        db_manager.save_entity(
            entity["id"],
            entity["name"],
            entity["type"],
            entity["attributes"]
        )
        db_manager.save_entity_topic_relation(entity["id"], "topic_1")
    
    # 触发容生成
    await agent.trigger_content_generation()
    
    # 验证结果
    clusters = db_manager.get_cluster_by_topic("兴趣")
    assert len(clusters) > 0
    
    # 验证聚类状态和内容
    for cluster in clusters:
        assert cluster["state"] in ["GENERATED", "ACCUMULATING"]
        if cluster["state"] == "GENERATED":
            contents = db_manager.get_cluster_contents(cluster["id"])
            assert len(contents) > 0