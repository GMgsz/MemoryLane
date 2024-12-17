from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
_ = load_dotenv(find_dotenv())

def inspect_chroma_db():
    """检查向量数据库中的数据"""
    try:
        # 初始化embeddings
        embeddings = ZhipuAIEmbeddings(
            model=os.getenv("ZHIPUAI_Embedding_model"),
            api_key=os.getenv("ZHIPUAI_Embedding_key"),
        )
        
        # 连接到现有的Chroma数据库
        db = Chroma(
            collection_name="dialogue_analysis",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # 1. 获取集合信息
        print("\n=== Collection Info ===")
        print(f"Collection name: {db._collection.name}")
        print(f"Number of documents: {db._collection.count()}")
        
        # 2. 获取所有文档
        print("\n=== All Documents ===")
        results = db.get()
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\nDocument {i+1}:")
            print("Text:", doc)
            print("Metadata:", metadata)
            
        # 3. 执行一些示例查询
        print("\n=== Sample Queries ===")
        queries = [
            "童年的回忆",
            "运动经历",
            "旅行计划",
            "传统文化"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = db.similarity_search_with_score(query, k=2)
            for doc, score in results:
                print(f"\nScore: {score}")
                print("Content:", doc.page_content)
                print("Metadata:", doc.metadata)

    except Exception as e:
        print(f"Error inspecting vector store: {str(e)}")

if __name__ == "__main__":
    inspect_chroma_db() 