import sqlite3
from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path
import logging
import os
from langchain_chroma import Chroma
import uuid
from langchain_community.embeddings import ZhipuAIEmbeddings
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, db_path: str = None):
        """初始化数据库管理器"""
        if db_path is None:
            # 使用默认的生产环境数据库路径
            db_path = os.getenv('PRODUCTION_DB_PATH', 'db/production/memory_lane.db')
            
        # 确保数据库目录存在
        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
        self._init_vector_store()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建dialogues表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dialogues (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        system_query TEXT NOT NULL,
                        user_response TEXT NOT NULL
                    )
                """)
                
                # 创建entities表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        attributes TEXT NOT NULL
                    )
                """)
                
                # 创建dialogue_entity_relations表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dialogue_entity_relations (
                        dialogue_id TEXT,
                        entity_id TEXT,
                        PRIMARY KEY (dialogue_id, entity_id),
                        FOREIGN KEY (dialogue_id) REFERENCES dialogues (id),
                        FOREIGN KEY (entity_id) REFERENCES entities (id)
                    )
                """)
                
                # 创建topics表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        primary_category TEXT NOT NULL
                    )
                """)
                
                # 创建entity_topic_relations表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entity_topic_relations (
                        entity_id TEXT,
                        topic_id TEXT,
                        score REAL DEFAULT 1.0,
                        PRIMARY KEY (entity_id, topic_id),
                        FOREIGN KEY (entity_id) REFERENCES entities (id),
                        FOREIGN KEY (topic_id) REFERENCES topics (id)
                    )
                """)
                
                # 创建clusters表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS clusters (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        primary_topic TEXT NOT NULL,
                        state TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建entity_cluster_relations表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entity_cluster_relations (
                        entity_id TEXT,
                        cluster_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (entity_id, cluster_id),
                        FOREIGN KEY (entity_id) REFERENCES entities (id),
                        FOREIGN KEY (cluster_id) REFERENCES clusters (id)
                    )
                """)
                
                # 创建generated_contents表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS generated_contents (
                        id TEXT PRIMARY KEY,
                        cluster_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        version INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (cluster_id) REFERENCES clusters (id)
                    )
                """)
                
                conn.commit()
        except Exception as e:
            logger.error(f"初始化数据库失败: {str(e)}")
            raise
    
    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            # 根据数据库路径确定向量存储路径
            if 'test' in self.db_path:
                vector_store_path = os.getenv('TEST_VECTOR_STORE_PATH', 'tests/test_data/vector_store')
            else:
                vector_store_path = os.getenv('PRODUCTION_VECTOR_STORE_PATH', 'db/production/vector_store')
            
            os.makedirs(vector_store_path, exist_ok=True)
            
            embedding_function = ZhipuAIEmbeddings(
                model=os.getenv("ZHIPUAI_EMBEDDING_MODEL"),
                api_key=os.getenv("ZHIPUAI_EMBEDDING_KEY")
            )
            self.vector_store = Chroma(
                collection_name="dialogues",
                persist_directory=vector_store_path,
                embedding_function=embedding_function
            )
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            self.vector_store = None  # 测试时可以没有向量存储
    
    async def save_dialogue(self, dialogue_id: str, system_query: str, user_response: str, entity_ids: List[str]):
        """保存对话及其关联的实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 先创建表（如果不存在）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dialogues (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        system_query TEXT NOT NULL,
                        user_response TEXT NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dialogue_entity_relations (
                        dialogue_id TEXT,
                        entity_id TEXT,
                        PRIMARY KEY (dialogue_id, entity_id),
                        FOREIGN KEY (dialogue_id) REFERENCES dialogues (id),
                        FOREIGN KEY (entity_id) REFERENCES entities (id)
                    )
                """)
                
                # 使用 INSERT OR REPLACE 而不是 INSERT
                cursor.execute(
                    "INSERT OR REPLACE INTO dialogues (id, timestamp, system_query, user_response) VALUES (?, ?, ?, ?)",
                    (dialogue_id, datetime.now().isoformat(), system_query, user_response)
                )
                
                # 先删除旧的关联关系
                cursor.execute(
                    "DELETE FROM dialogue_entity_relations WHERE dialogue_id = ?",
                    (dialogue_id,)
                )
                
                # 保存新的对话-实体关系
                for entity_id in entity_ids:
                    cursor.execute(
                        "INSERT INTO dialogue_entity_relations (dialogue_id, entity_id) VALUES (?, ?)",
                        (dialogue_id, entity_id)
                    )
                
                conn.commit()
        except Exception as e:
            logger.error(f"保存对话失败: {str(e)}")
            raise

    def save_entity(self, entity_id: str, name: str, entity_type: str, attributes: Dict):
        """保存实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 先创建表（如果不存在）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        attributes TEXT NOT NULL
                    )
                """)
                
                # 创建entity_topic_relations表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entity_topic_relations (
                        entity_id TEXT,
                        topic_id TEXT,
                        score REAL DEFAULT 1.0,
                        PRIMARY KEY (entity_id, topic_id),
                        FOREIGN KEY (entity_id) REFERENCES entities (id),
                        FOREIGN KEY (topic_id) REFERENCES topics (id)
                    )
                """)
                
                # 保存实体
                cursor.execute(
                    "INSERT OR REPLACE INTO entities (id, name, type, attributes) VALUES (?, ?, ?, ?)",
                    (entity_id, name, entity_type, json.dumps(attributes, ensure_ascii=False))
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"保存实体失败: {str(e)}")
            raise

    def save_topic(self, topic_id: str, name: str, primary_category: str):
        """保存话题"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 先创建表（如果不存在）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        primary_category TEXT NOT NULL
                    )
                """)
                
                # 创建entity_topic_relations表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entity_topic_relations (
                        entity_id TEXT,
                        topic_id TEXT,
                        score REAL DEFAULT 1.0,
                        PRIMARY KEY (entity_id, topic_id),
                        FOREIGN KEY (entity_id) REFERENCES entities (id),
                        FOREIGN KEY (topic_id) REFERENCES topics (id)
                    )
                """)
                
                # 保存话题
                cursor.execute(
                    "INSERT OR REPLACE INTO topics (id, name, primary_category) VALUES (?, ?, ?)",
                    (topic_id, name, primary_category)
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"保存话题失败: {str(e)}")
            raise

    def save_entity_topic_relation(self, entity_id: str, topic_id: str, score: float = 1.0):
        """保存实体-话题关联"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 保存关联关系
                cursor.execute(
                    "INSERT OR REPLACE INTO entity_topic_relations (entity_id, topic_id, score) VALUES (?, ?, ?)",
                    (entity_id, topic_id, score)
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"保存实体-话题关联失败: {str(e)}")
            raise

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """根据ID获取实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT id, name, type, attributes FROM entities WHERE id = ?",
                    (entity_id,)
                )
                
                result = cursor.fetchone()
                if result:
                    return {
                        "id": result[0],
                        "name": result[1],
                        "type": result[2],
                        "attributes": json.loads(result[3])
                    }
                return None
        except Exception as e:
            logger.error(f"获取实体失败: {str(e)}")
            raise

    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """根据ID获取话题"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT id, name, primary_category FROM topics WHERE id = ?",
                    (topic_id,)
                )
                
                result = cursor.fetchone()
                if result:
                    return {
                        "id": result[0],
                        "name": result[1],
                        "primary_category": result[2]
                    }
                return None
        except Exception as e:
            logger.error(f"获取话题失败: {str(e)}")
            raise

    def get_entity_topics(self, entity_id: str) -> List[Dict]:
        """获取实体相关的所有话题"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT t.id, t.name, t.primary_category, etr.score
                    FROM topics t
                    JOIN entity_topic_relations etr ON t.id = etr.topic_id
                    WHERE etr.entity_id = ?
                """, (entity_id,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "primary_category": row[2],
                        "score": row[3]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取实体话题失败: {str(e)}")
            raise

    async def save_entities_batch(self, entities: List[Dict]):
        """批量保存实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 先创建表（如果不存在）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        attributes TEXT NOT NULL
                    )
                """)
                
                cursor.executemany(
                    "INSERT OR REPLACE INTO entities (id, name, type, attributes) VALUES (?, ?, ?, ?)",
                    [
                        (
                            entity["id"],
                            entity["name"],
                            entity["type"],
                            json.dumps(entity["attributes"], ensure_ascii=False)
                        )
                        for entity in entities
                    ]
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"批量保存实体失败: {str(e)}")
            raise

    def save_topics_batch(self, topics: List[Dict]):
        """批量保存话题"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.executemany(
                    "INSERT OR REPLACE INTO topics (id, name, primary_category) VALUES (?, ?, ?)",
                    [
                        (
                            topic["id"],
                            topic["name"],
                            topic["primary_category"]
                        )
                        for topic in topics
                    ]
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"批量保存话题失败: {str(e)}")
            raise

    def save_entity_topic_relations_batch(self, relations: List[Dict]):
        """批量保存实体-话题关联"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.executemany(
                    "INSERT OR REPLACE INTO entity_topic_relations (entity_id, topic_id, score) VALUES (?, ?, ?)",
                    [
                        (
                            relation["entity_id"],
                            relation["topic_id"],
                            relation.get("score", 1.0)
                        )
                        for relation in relations
                    ]
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"批量保存实体-话题关联失败: {str(e)}")
            raise

    def get_dialogue_entities(self, dialogue_id: str) -> List[Dict]:
        """获取对话相关的所有实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT e.id, e.name, e.type, e.attributes
                    FROM entities e
                    JOIN dialogue_entity_relations der ON e.id = der.entity_id
                    WHERE der.dialogue_id = ?
                """, (dialogue_id,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3])
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取对话实体失败: {str(e)}")
            raise

    def get_topic_entities(self, topic_id: str) -> List[Dict]:
        """获取话题相关的所有实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT e.id, e.name, e.type, e.attributes, etr.score
                    FROM entities e
                    JOIN entity_topic_relations etr ON e.id = etr.entity_id
                    WHERE etr.topic_id = ?
                """, (topic_id,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3]),
                        "score": row[4]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取话题实体失败: {str(e)}")
            raise

    def init_vector_store(self, embeddings):
        """初始化向量数据库"""
        try:
            # 创建持久化目录
            persist_directory = "./chroma_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            # 初始化Chroma
            self.vector_store = Chroma(
                collection_name="dialogue_analysis",
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
        except Exception as e:
            logger.error(f"初始化向量数据库失败: {str(e)}")
            raise

    def save_to_vector_store(self, dialogue_id: str, text: str, metadata: Dict = None):
        """保存文本到向量数据库"""
        try:
            if metadata is None:
                metadata = {}
            
            # 将列表转换为字符串
            if "entity_ids" in metadata:
                metadata["entity_ids"] = ",".join(metadata["entity_ids"])
            
            metadata["dialogue_id"] = dialogue_id
            
            self.vector_store.add_texts(
                texts=[text],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"保存到向量数据库失败: {str(e)}")
            raise

    def search_similar_dialogues(self, query: str, k: int = 3) -> List[Dict]:
        """搜索相似对话"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            return [
                {
                    "text": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        # 将字符串转回列表
                        "entity_ids": doc.metadata.get("entity_ids", "").split(",") if doc.metadata.get("entity_ids") else []
                    },
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"搜索相似对话失败: {str(e)}")
            raise

    def get_dialogues_by_time_range(self, start_time: str, end_time: str) -> List[Dict]:
        """按时间范围获取对话"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, timestamp, system_query, user_response
                    FROM dialogues
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """, (start_time, end_time))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "timestamp": row[1],
                        "system_query": row[2],
                        "user_response": row[3]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"按时间范围获取对话失败: {str(e)}")
            raise

    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """按类型获取实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, type, attributes
                    FROM entities
                    WHERE type = ?
                """, (entity_type,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3])
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"按类型获取实体失败: {str(e)}")
            raise

    def get_topics_with_entity_count(self) -> List[Dict]:
        """获取话题及其关联的实体数量"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT t.id, t.name, t.primary_category, COUNT(etr.entity_id) as entity_count
                    FROM topics t
                    LEFT JOIN entity_topic_relations etr ON t.id = etr.topic_id
                    GROUP BY t.id
                    ORDER BY entity_count DESC
                """)
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "primary_category": row[2],
                        "entity_count": row[3]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取话题统计失败: {str(e)}")
            raise

    def create_cluster(self, name: str, primary_topic: str, state: str = "ACCUMULATING") -> str:
        """创建新聚类"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cluster_id = f"cluster_{str(uuid.uuid4())[:8]}"
                cursor.execute("""
                    INSERT INTO clusters (id, name, primary_topic, state)
                    VALUES (?, ?, ?, ?)
                """, (cluster_id, name, primary_topic, state))
                conn.commit()
                logger.info(f"创建新聚类: {cluster_id}, {name}, {primary_topic}, {state}")
                return cluster_id
        except Exception as e:
            logger.error(f"创建聚类失败: {str(e)}")
            raise

    def add_entity_to_cluster(self, entity_id: str, cluster_id: str):
        """添加实体到聚类"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO entity_cluster_relations (entity_id, cluster_id)
                    VALUES (?, ?)
                """, (entity_id, cluster_id))
                conn.commit()
        except sqlite3.IntegrityError as e:
            logger.error(f"添加实体到聚类失败(外键约束): {str(e)}")
            raise
        except Exception as e:
            logger.error(f"添加实体到聚类失败: {str(e)}")
            raise

    def get_unprocessed_entities(self) -> List[Dict]:
        """获取未处理的实体（未分配到聚类的实体）"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT e.id, e.name, e.type, e.attributes, t.primary_category
                    FROM entities e
                    LEFT JOIN entity_topic_relations etr ON e.id = etr.entity_id
                    LEFT JOIN topics t ON etr.topic_id = t.id
                    LEFT JOIN entity_cluster_relations ecr ON e.id = ecr.entity_id
                    WHERE ecr.cluster_id IS NULL
                """)
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3]),
                        "primary_topic": row[4]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取未处理实体失败: {str(e)}")
            raise

    def get_cluster_by_topic(self, primary_topic: str) -> List[Dict]:
        """获取主题下的所有聚类"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, primary_topic, state, created_at, updated_at
                    FROM clusters
                    WHERE primary_topic = ?
                """, (primary_topic,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "primary_topic": row[2],
                        "state": row[3],
                        "created_at": row[4],
                        "updated_at": row[5]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取主题聚类失败: {str(e)}")
            raise

    def get_clusters_by_state(self, state: str) -> List[Dict]:
        """获取特定状态的聚类"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, primary_topic, state, created_at, updated_at
                    FROM clusters
                    WHERE state = ?
                """, (state,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "primary_topic": row[2],
                        "state": row[3],
                        "created_at": row[4],
                        "updated_at": row[5]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取聚类失败: {str(e)}")
            raise

    def get_cluster_entity_count(self, cluster_id: str) -> int:
        """获取聚类中的实体数量"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM entity_cluster_relations
                    WHERE cluster_id = ?
                """, (cluster_id,))
                
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"获取聚类实体数量失败: {str(e)}")
            raise

    def get_cluster(self, cluster_id: str) -> Dict:
        """获取聚类信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, primary_topic, state, created_at, updated_at
                    FROM clusters
                    WHERE id = ?
                """, (cluster_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "primary_topic": row[2],
                        "state": row[3],
                        "created_at": row[4],
                        "updated_at": row[5]
                    }
                return None
        except Exception as e:
            logger.error(f"获取聚类信息失败: {str(e)}")
            raise

    def get_cluster_entities(self, cluster_id: str) -> List[Dict]:
        """获取聚类中的所有实体"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.id, e.name, e.type, e.attributes
                    FROM entities e
                    JOIN entity_cluster_relations ecr ON e.id = ecr.entity_id
                    WHERE ecr.cluster_id = ?
                """, (cluster_id,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "attributes": json.loads(row[3])
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取聚类实体失败: {str(e)}")
            raise

    def update_cluster_state(self, cluster_id: str, new_state: str):
        """更新聚类状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE clusters
                    SET state = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_state, cluster_id))
                conn.commit()
        except Exception as e:
            logger.error(f"更新聚类状态失败: {str(e)}")
            raise

    def save_generated_content(self, cluster_id: str, content: str) -> str:
        """保存生成的内容"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                content_id = f"content_{str(uuid.uuid4())[:8]}"
                cursor.execute("""
                    INSERT INTO generated_contents (id, cluster_id, content)
                    VALUES (?, ?, ?)
                """, (content_id, cluster_id, content))
                conn.commit()
                return content_id
        except Exception as e:
            logger.error(f"保存生成内容失败: {str(e)}")
            raise

    def get_cluster_contents(self, cluster_id: str) -> List[Dict]:
        """获取聚类的生成内容"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, content, version, created_at
                    FROM generated_contents
                    WHERE cluster_id = ?
                    ORDER BY version DESC
                """, (cluster_id,))
                
                results = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "content": row[1],
                        "version": row[2],
                        "created_at": row[3]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"获取聚类内容失败: {str(e)}")
            raise

    def close(self):
        """关闭数据库连接"""
        try:
            if hasattr(self, 'vector_store'):
                if hasattr(self.vector_store, '_client') and not isinstance(self.vector_store, MagicMock):
                    self.vector_store._client.close()
                del self.vector_store
        except Exception as e:
            logger.warning(f"关闭向量存储失败: {str(e)}")