from Utils.DatabaseManager import DatabaseManager
from langchain_community.embeddings import ZhipuAIEmbeddings
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def prepare_basketball_story(db_manager):
    """准备篮球故事相关的数据"""
    # 1. 保存对话
    dialogues = [
        {
            "id": "dialogue_001",
            "system_query": "您生活中最难忘的教训是什么?",
            "user_response": "篮球",
            "timestamp": "2024-03-14T10:00:00",
            "entities": [
                {
                    "id": "entity_001",
                    "name": "篮球",
                    "type": "activity",
                    "attributes": {
                        "nature": "运动"
                    },
                    "topics": [
                        {"id": "topic_兴趣", "score": 0.9},
                        {"id": "topic_早年生活", "score": 0.8}
                    ]
                }
            ]
        },
        {
            "id": "dialogue_002",
            "system_query": "您是如何面对这个挑战的呢?",
            "user_response": "组队",
            "timestamp": "2024-03-14T10:01:00",
            "entities": [
                {
                    "id": "entity_002",
                    "name": "组队",
                    "type": "activity",
                    "attributes": {
                        "nature": "应对策略",
                        "context": "篮球比赛"
                    },
                    "topics": [
                        {"id": "topic_挑战与困难", "score": 0.9},
                        {"id": "topic_友谊", "score": 0.8}
                    ]
                }
            ]
        }
    ]
    
    for dialogue in dialogues:
        # 保存实体
        for entity in dialogue["entities"]:
            db_manager.save_entity(
                entity_id=entity["id"],
                name=entity["name"],
                entity_type=entity["type"],
                attributes=entity["attributes"]
            )
            
            # 保存话题关系
            for topic in entity["topics"]:
                db_manager.save_topic(
                    topic_id=topic["id"],
                    name=topic["id"].replace("topic_", ""),
                    primary_category=topic["id"].replace("topic_", "")
                )
                db_manager.save_entity_topic_relation(
                    entity_id=entity["id"],
                    topic_id=topic["id"],
                    score=topic["score"]
                )
        
        # 保存对话
        entity_ids = [entity["id"] for entity in dialogue["entities"]]
        db_manager.save_dialogue(
            dialogue_id=dialogue["id"],
            system_query=dialogue["system_query"],
            user_response=dialogue["user_response"],
            entity_ids=entity_ids
        )
        
        # 保存到向量数据库
        combined_text = f"{dialogue['system_query']}\n{dialogue['user_response']}"
        db_manager.save_to_vector_store(
            dialogue_id=dialogue["id"],
            text=combined_text,
            metadata={
                "timestamp": dialogue["timestamp"],
                "entity_ids": entity_ids
            }
        )

def prepare_childhood_story(db_manager):
    """准备童年烤番薯故事相关的数据"""
    # 1. 保存对话
    dialogues = [
        {
            "id": "dialogue_003",
            "system_query": "童年是一个充满回忆和故事的时期。我很想听听您童年时的趣事或者特别���经历。有没有哪个瞬间或者故事让您特别怀念?",
            "user_response": "小时候在沙里烤番薯",
            "timestamp": "2024-03-14T10:02:00",
            "entities": [
                {
                    "id": "entity_003",
                    "name": "烤番薯",
                    "type": "activity",
                    "attributes": {
                        "time": "小时候",
                        "location": "沙里"
                    },
                    "topics": [
                        {"id": "topic_早年生活", "score": 0.9},
                        {"id": "topic_友谊", "score": 0.7}
                    ]
                }
            ]
        },
        {
            "id": "dialogue_004",
            "system_query": "烤番薯听起来是个有趣的童年回忆!您是在什么样的场合或者和谁一起烤的呢?",
            "user_response": "和我的朋友少欣",
            "timestamp": "2024-03-14T10:03:00",
            "entities": [
                {
                    "id": "entity_004",
                    "name": "少欣",
                    "type": "person",
                    "attributes": {
                        "relationship": "朋友"
                    },
                    "topics": [
                        {"id": "topic_友谊", "score": 0.9},
                        {"id": "topic_早年生活", "score": 0.7}
                    ]
                }
            ]
        }
    ]
    
    # 保存数据的逻辑与篮球故事类似
    for dialogue in dialogues:
        # ... 保存实体、话题、对话等
        pass  # 实现类似上面的保存逻辑

def prepare_japan_camp_story(db_manager):
    """准备日本夏令营故事相关的数据"""
    dialogues = [
        {
            "id": "dialogue_005",
            "system_query": "您是在日本的哪个地方欣赏的夕阳呢?",
            "user_response": "夏令营",
            "timestamp": "2024-03-14T10:04:00",
            "entities": [
                {
                    "id": "entity_005",
                    "name": "日本夏令营",
                    "type": "event",
                    "attributes": {
                        "location": "日本",
                        "activity": "夏令营",
                        "time": "过去"
                    },
                    "topics": [
                        {"id": "topic_旅行", "score": 0.9},
                        {"id": "topic_友谊", "score": 0.8}
                    ]
                }
            ]
        },
        {
            "id": "dialogue_006",
            "system_query": "在夏令营中，您和这些朋友一起做了哪些有趣的活动呢?",
            "user_response": "我们一起做了一个项目",
            "timestamp": "2024-03-14T10:05:00",
            "entities": [
                {
                    "id": "entity_006",
                    "name": "智能胸针项目",
                    "type": "project",
                    "attributes": {
                        "nature": "创新项目",
                        "function": "识别图片和录音，充当第二大脑",
                        "context": "夏令营"
                    },
                    "topics": [
                        {"id": "topic_成就", "score": 0.9},
                        {"id": "topic_友谊", "score": 0.8}
                    ]
                }
            ]
        }
    ]
    
    # 实现保存逻辑，与篮球故事类似
    for dialogue in dialogues:
        # ... 保存实体、话题、对话等
        pass

def prepare_travel_story(db_manager):
    """准备旅行相关的故事数据"""
    dialogues = [
        {
            "id": "dialogue_007",
            "system_query": "您有没有特别想去的地方?",
            "user_response": "我想去西藏",
            "timestamp": "2024-03-14T10:06:00",
            "entities": [
                {
                    "id": "entity_007",
                    "name": "西藏旅行",
                    "type": "plan",
                    "attributes": {
                        "destination": "西藏",
                        "activity": "看日照西山",
                        "companion": "女朋友"
                    },
                    "topics": [
                        {"id": "topic_旅行", "score": 0.9},
                        {"id": "topic_兴趣", "score": 0.7}
                    ]
                }
            ]
        },
        {
            "id": "dialogue_008",
            "system_query": "你们之前去过哪些地方旅行呢?",
            "user_response": "北京",
            "timestamp": "2024-03-14T10:07:00",
            "entities": [
                {
                    "id": "entity_008",
                    "name": "北京之行",
                    "type": "experience",
                    "attributes": {
                        "location": "北京",
                        "activity": "球场散步",
                        "memory": "聊到了未来"
                    },
                    "topics": [
                        {"id": "topic_旅行", "score": 0.9},
                        {"id": "topic_关键事件", "score": 0.8}
                    ]
                }
            ]
        }
    ]
    
    # 实现保存逻辑
    for dialogue in dialogues:
        pass

def prepare_tradition_story(db_manager):
    """准备传统文化相关的故事数据"""
    dialogues = [
        {
            "id": "dialogue_009",
            "system_query": "这些祭拜活动对你个人有什么特别的意义或感受吗?",
            "user_response": "熟悉传统文化吧",
            "timestamp": "2024-03-14T10:08:00",
            "entities": [
                {
                    "id": "entity_009",
                    "name": "家庭祭拜",
                    "type": "tradition",
                    "attributes": {
                        "activity": "祭拜",
                        "item": "纸钱",
                        "significance": "传统文化传承"
                    },
                    "topics": [
                        {"id": "topic_家庭", "score": 0.9},
                        {"id": "topic_信仰", "score": 0.8}
                    ]
                }
            ]
        },
        {
            "id": "dialogue_010",
            "system_query": "在婚闹中有没有发生什么有趣或难忘的事情?",
            "user_response": "我们一群人做了很多游戏才让新娘新郎相见",
            "timestamp": "2024-03-14T10:09:00",
            "entities": [
                {
                    "id": "entity_010",
                    "name": "表姐婚礼",
                    "type": "event",
                    "attributes": {
                        "activity": "婚闹",
                        "relationship": "表姐",
                        "detail": "玩猜谜游戏"
                    },
                    "topics": [
                        {"id": "topic_家庭", "score": 0.9},
                        {"id": "topic_关键事件", "score": 0.8}
                    ]
                }
            ]
        }
    ]
    
    # 实现保存逻辑
    for dialogue in dialogues:
        pass

def main():
    # 初始化数据库管理器
    db_manager = DatabaseManager(db_path="test_dialogue_analysis.db")
    
    # 初始化向量数据库
    embeddings = ZhipuAIEmbeddings(
        model=os.getenv("ZHIPUAI_Embedding_model"),
        api_key=os.getenv("ZHIPUAI_Embedding_key"),
    )
    db_manager.init_vector_store(embeddings)
    
    # 准备测试数据
    prepare_basketball_story(db_manager)
    prepare_childhood_story(db_manager)
    prepare_japan_camp_story(db_manager)
    prepare_travel_story(db_manager)
    prepare_tradition_story(db_manager)

if __name__ == "__main__":
    main() 