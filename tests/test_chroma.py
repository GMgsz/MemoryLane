import os
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vector_store():
    """加载向量数据库"""
    try:
        # 加载环境变量
        _ = load_dotenv(find_dotenv())
        
        # 初始化embeddings
        embeddings = ZhipuAIEmbeddings(
            model=os.getenv("ZHIPUAI_Embedding_model"),
            api_key=os.getenv("ZHIPUAI_Embedding_key"),
        )
        
        # 加载现有的向量数据库
        db = Chroma(
            collection_name="chat_history",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        return db
    except Exception as e:
        logger.error(f"加载向量数据库失败: {str(e)}")
        raise

def print_collection_info(db):
    """打印集合信息"""
    try:
        # 获取集合
        collection = db._collection
        
        # 打印基本信息
        print(f"\n=== 向量数据库信息 ===")
        print(f"集合名称: {collection.name}")
        print(f"文档数量: {collection.count()}")
        
        # 获取所有文档
        results = db.get()
        
        print("\n=== 存储的文档内容 ===")
        if results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"\n文档 {i+1}:")
                print(f"内容: {doc}")
                print(f"元数据: {metadata}")
        else:
            print("数据库中没有文档")
            
    except Exception as e:
        logger.error(f"获取数据库信息失败: {str(e)}")
        raise

def main():
    try:
        # 加载向量数据库
        db = load_vector_store()
        
        # 打印信息
        print_collection_info(db)
        
        # 测试搜索
        print("\n=== 测试相似性搜索 ===")
        query = "你好"
        results = db.similarity_search(query, k=2)
        
        print(f"\n与 '{query}' 最相似的文档:")
        for i, doc in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
            
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 