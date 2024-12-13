import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import SystemMessage, HumanMessage

async def make_api_call(llm: ChatZhipuAI, message: str, call_id: int):
    """进行单次API调用"""
    try:
        system_message = SystemMessage(content="你是一个助手。")
        human_message = HumanMessage(content=message)
        
        print(f"开始调用 {call_id}")
        response = await llm.ainvoke([system_message, human_message])
        print(f"调用 {call_id} 成功: {response.content[:50]}...")
        return response.content
    except Exception as e:
        print(f"调用 {call_id} 失败: {str(e)}")
        return None

async def test_concurrent_calls():
    """测试并发API调用"""
    # 加载环境变量
    _ = load_dotenv(find_dotenv())
    
    # 创建不同的LLM实例
    llm1 = ChatZhipuAI(
        model=os.getenv('Extract_Model'),
        api_key=os.getenv('Extract_API_key')
    )
    
    llm2 = ChatZhipuAI(
        model=os.getenv('Identify_Model'),
        api_key=os.getenv('Identify_API_key')
    )
    
    llm3 = ChatZhipuAI(
        model=os.getenv('Generate_Model'),
        api_key=os.getenv('Generate_API_key')
    )
    
    # 测试场景1：使用相同的API key进行并发调用
    print("\n测试场景1：使用相同的API key进行并发调用")
    tasks = [
        make_api_call(llm1, f"这是测试消息 {i}", i)
        for i in range(3)
    ]
    await asyncio.gather(*tasks)
    
    # 等待一段时间
    print("\n等待10秒...")
    await asyncio.sleep(10)
    
    # 测试场景2：使用不同的API key进行并发调用
    print("\n测试场景2：使用不同的API key进行并发调用")
    tasks = [
        make_api_call(llm1, "提取实体的测试", 1),
        make_api_call(llm2, "识别主题的测试", 2),
        make_api_call(llm3, "生成内容的测试", 3)
    ]
    await asyncio.gather(*tasks)
    
    # 测试场景3：间隔调用
    print("\n测试场景3：间隔调用")
    for i in range(3):
        await make_api_call(llm1, f"间隔测试 {i}", i)
        await asyncio.sleep(2)  # 等待2秒

if __name__ == "__main__":
    asyncio.run(test_concurrent_calls()) 