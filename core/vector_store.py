from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from typing import List, Dict, Any
import json

class VectorStoreManager:
    def __init__(self, embeddings: ZhipuAIEmbeddings):
        self.embeddings = embeddings
        self.vector_store = Chroma(
            collection_name="memory_lane",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
    
    async def add_memory(self, text: str, metadata: Dict):
        """添加记忆到向量存储"""
        try:
            formatted_metadata = {
                "id": metadata.get("id"),
                "timestamp": metadata.get("timestamp"),
                "themes": ",".join(metadata.get("themes", [])),
                "entities": json.dumps(metadata.get("entities", {})),
                "keywords": ",".join(metadata.get("keywords", [])),
                "dialogue_id": metadata.get("dialogue_id")
            }
            
            self.vector_store.add_texts(
                texts=[text],
                metadatas=[formatted_metadata]
            )
            return True
        except Exception as e:
            print(f"存储失败: {e}")
            return False
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """改进相似度计算"""
        # 添加预处理
        # 考虑语义特征
        # 优化阈值设置
    
    async def search_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """搜索相似的记忆，返回内容和相似度分数"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                'content': doc.page_content,
                'score': score,
                'metadata': doc.metadata
            } for doc, score in results
        ] 