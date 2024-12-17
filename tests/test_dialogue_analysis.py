import os

# 确保测试目录存在
test_dir = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(test_dir, exist_ok=True)

from Agent.DialogueAnalysisAgent import DialogueAnalysisAgent
import json
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_community.chat_models import ChatZhipuAI


def print_state(dialogue, agent):
    """打印当前状态"""
    print("\n处理对话:")
    print(f"系统: {dialogue['system']}")
    print(f"用户: {dialogue['user']}")
    
    print("\n当前实体:")
    for entity in agent.entities.values():
        print(f"\n- {entity.name} ({entity.type}):")
        print(f"  属性: {json.dumps(entity.attributes, ensure_ascii=False, indent=2)}")
        print(f"  话题: {entity.topic_belongings}")
    
    print("\n当前话题:")
    for topic in agent.topics.values():
        print(f"\n- {topic.name}:")
        print(f"  相关实体: {topic.related_entities}")
    
    print("\n" + "="*50)

def test_dialogue_analysis():
    """测试对话分析智能体"""
    
    # 1. 初始化智能体
    llm = ChatZhipuAI(
        model=os.getenv("ZHIPUAI_MODEL_NAME"),
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        temperature=0,
    )
    # 使用测试数据库路径
    test_db_path = os.getenv('TEST_DB_PATH', 'tests/test_data/test.db')
    agent = DialogueAnalysisAgent(llm=llm, db_path=test_db_path)
    
    # 2. 测试对话片段
    dialogues = [
        {
            "system": "童年是一个充满回忆和故事的时期。我很想听听童年时的趣事或者特别的经历。有没有哪个瞬间或者故事让您特别怀念?",
            "user": "小时候在沙里烤番薯"
        },
        {
            "system": "烤番薯听起来是个有趣的童年回忆!您是在什么样的场合或者和谁一起烤的呢?",
            "user": "和我的朋友少欣"
        },
        {
            "system": "和朋友一起烤番薯一定很有趣!您们是在哪里找到沙子来烤的呢?这个活动有什么特别的回忆吗?",
            "user": "在家隔壁的农田里"
        }
    ]
    
    # 3. 处理对话
    for dialogue in dialogues:
        # 处理单轮对话
        agent.process_dialogue(
            system_query=dialogue["system"],
            user_response=dialogue["user"]
        )
        
        # 打印状态
        print_state(dialogue, agent)

def test_dialogue_analysis_basketball():
    """测试篮球相关的对话分析"""
    
    # 1. 初始化智能体
    llm = ChatZhipuAI(
        model=os.getenv("ZHIPUAI_MODEL_NAME"),
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        temperature=0,
    )
    # 使用测试数据库路径
    test_db_path = os.getenv('TEST_DB_PATH', 'tests/test_data/test.db')
    agent = DialogueAnalysisAgent(llm=llm, db_path=test_db_path)
    
    # 2. 测试对话片段
    dialogues = [
        {
            "system": "您小时候有什么特别喜欢的运动或游戏吗？",
            "user": "从小打篮球"
        },
        {
            "system": "篮球是一很好的运动！您在打篮球的过程中有什么特别难忘的经历吗？",
            "user": "挑战高年级"
        },
        {
            "system": "面对高年级同学的挑战，您是怎么应对的呢？",
            "user": "组队"
        }
    ]
    
    # 3. 处理对话
    for dialogue in dialogues:
        agent.process_dialogue(
            system_query=dialogue["system"],
            user_response=dialogue["user"]
        )
        
        print_state(dialogue, agent)

if __name__ == "__main__":
    print("测试1：烤番薯故事")
    test_dialogue_analysis()
    
    print("\n\n测试2：篮球故事")
    test_dialogue_analysis_basketball() 