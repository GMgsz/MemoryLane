import pytest
import logging
from Agent.ContentGenerationAgent import ContentGenerationAgent
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_dialogues():
    """准备测试对话数据"""
    return [
        {
            "id": "dialogue_001",
            "system_query": "您小时候有什么特别喜欢的运动或游戏吗？",
            "user_response": "从小打篮球，经常和朋友一起打比赛",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "dialogue_002",
            "system_query": "能具体说说打篮球的经历吗？",
            "user_response": "记得高中时经常和高年级的学长比赛",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "dialogue_003",
            "system_query": "这些比赛给您留下了什么印象？",
            "user_response": "虽然输多赢少，但是学到了很多技术",
            "timestamp": datetime.now().isoformat()
        }
    ]

@pytest.fixture
def mock_entities():
    """准备测试实体数据"""
    return [
        {
            "id": "entity_001",
            "name": "篮球",
            "type": "activity",
            "attributes": {
                "time": "从小",
                "frequency": "经常",
                "nature": "运动"
            },
            "primary_topic": "兴趣"
        },
        {
            "id": "entity_002",
            "name": "比赛",
            "type": "event",
            "attributes": {
                "time": "高中",
                "participants": "高年级学长",
                "nature": "竞技"
            },
            "primary_topic": "挑战与困难"
        }
    ]

@pytest.mark.asyncio
async def test_content_generation_detailed(db_manager, llm_client, mock_dialogues, mock_entities):
    """详细测试内容生成流程"""
    logger.info("=== 开始测试内容生成流程 ===")
    
    # 1. 初始化智能体
    agent = ContentGenerationAgent(db_manager, llm_client)
    logger.info("智能体初始化完成")
    
    # 2. 保存测试数据
    logger.info("\n=== 保存测试数据 ===")
    for dialogue in mock_dialogues:
        await db_manager.save_dialogue(
            dialogue["id"],
            dialogue["system_query"],
            dialogue["user_response"],
            []  # 初始时没有关联实体
        )
        logger.info(f"保存对话: {dialogue['id']}")
    
    for entity in mock_entities:
        db_manager.save_entity(
            entity["id"],
            entity["name"],
            entity["type"],
            entity["attributes"]
        )
        logger.info(f"保存实体: {entity['name']}")
        
        # 关联实体和话题
        db_manager.save_entity_topic_relation(
            entity["id"],
            f"topic_{entity['primary_topic']}"
        )
        logger.info(f"实体 {entity['name']} 关联到话题 {entity['primary_topic']}")
    
    # 3. 测试实体组织
    logger.info("\n=== 测试实体组织 ===")
    await agent._organize_entities(mock_entities)
    
    # 检查聚类结果
    clusters = db_manager.get_cluster_by_topic("兴趣")
    logger.info(f"兴趣话题下的聚类: {json.dumps(clusters, indent=2, ensure_ascii=False)}")
    
    for cluster in clusters:
        entities = db_manager.get_cluster_entities(cluster["id"])
        logger.info(f"\n聚类 {cluster['id']} 的实体:")
        logger.info(json.dumps(entities, indent=2, ensure_ascii=False))
    
    # 4. 测试聚类评估
    logger.info("\n=== 测试聚类评估 ===")
    for cluster in clusters:
        logger.info(f"\n评估聚类: {cluster['id']}")
        await agent._evaluate_single_cluster(cluster)
        
        # 检查评估后的状态
        updated_cluster = db_manager.get_cluster(cluster["id"])
        logger.info(f"评估后的聚类状态: {updated_cluster['state']}")
        
        # 如果生成了内容，显示内容
        if updated_cluster["state"] == "GENERATED":
            contents = db_manager.get_cluster_contents(cluster["id"])
            logger.info("\n生成的内容:")
            for content in contents:
                logger.info(json.dumps(content, indent=2, ensure_ascii=False))
    
    # 5. 测试完整触发流程
    logger.info("\n=== 测试完整触发流程 ===")
    for i in range(agent.TRIGGER_INTERVAL):
        agent.on_dialogue_complete(f"dialogue_{i}")
        logger.info(f"对话计数: {agent.dialogue_count}")
    
    # 6. 验证最终状态
    logger.info("\n=== 最终状态 ===")
    all_clusters = []
    for topic in ["兴趣", "挑战与困难"]:
        clusters = db_manager.get_cluster_by_topic(topic)
        all_clusters.extend(clusters)
    
    logger.info(f"\n所有聚类数量: {len(all_clusters)}")
    for cluster in all_clusters:
        logger.info(f"\n聚类 {cluster['id']}:")
        logger.info(f"- 状态: {cluster['state']}")
        logger.info(f"- 主题: {cluster.get('primary_topic', '未知')}")
        
        entities = db_manager.get_cluster_entities(cluster["id"])
        logger.info("- 包含实体:")
        for entity in entities:
            logger.info(f"  * {entity['name']}: {entity['attributes']}")
        
        if cluster["state"] == "GENERATED":
            contents = db_manager.get_cluster_contents(cluster["id"])
            logger.info("- 生成内容:")
            for content in contents:
                logger.info(f"  * {content['content'][:100]}...")

if __name__ == "__main__":
    pytest.main(["-v", "test_content_generation_detailed.py"]) 