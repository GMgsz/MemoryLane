from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from langchain.schema.output_parser import StrOutputParser
from Utils.DatabaseManager import DatabaseManager


@dataclass
class DialogueTurn:
    """对话单元"""
    id: str
    timestamp: str
    system_query: str
    user_response: str
    entity_ids: List[str] = field(default_factory=list)


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    type: str  # person, place, time, event, activity
    attributes: Dict[str, str] = field(default_factory=dict)
    topic_belongings: List[str] = field(default_factory=list)


@dataclass
class Topic:
    """话题"""
    id: str
    name: str
    primary_category: str
    related_entities: Set[str] = field(default_factory=set)
    dialogue_turns: List[str] = field(default_factory=list)


class DialogueAnalysisAgent:
    """对话分析智能体"""

    # 获取项目根目录
    ROOT_DIR = Path(__file__).parent.parent

    # 预设的一级主题
    PRIMARY_TOPICS = [
        "挑战与困难", "家庭", "早年生活", "友谊", "影响",
        "成就", "职业生涯", "兴趣", "信仰", "关键事件",
        "旅行", "其他"
    ]

    def __init__(
            self,
            llm: BaseChatModel,
            embeddings=None,
            window_size: int = 3,
            db_path: str = None,
            db_manager=None
    ):
        self.llm = llm
        self.window_size = window_size

        # 初始化数据库管理器
        if db_manager is None:
            if db_path is None:
                # 使用默认的生产环境数据库路径
                db_path = os.getenv('PRODUCTION_DB_PATH', 'db/production/memory_lane.db')
            db_manager = DatabaseManager(db_path=db_path)
        self.db_manager = db_manager
        if embeddings:
            self.db_manager.init_vector_store(embeddings)

        # 对话窗口
        self.dialogue_window: List[DialogueTurn] = []

        # 内存中的实体和话题缓存
        self.entities: Dict[str, Entity] = {}
        self.topics: Dict[str, Topic] = {}

        # 初始化提示模板
        self._init_prompts()

    def _init_prompts(self):
        """初始化提示模板"""
        try:
            # 实体提取模板
            entity_prompt_builder = PromptTemplateBuilder(
                prompt_path=os.path.join(self.ROOT_DIR, "prompts/dialogue_analysis"),
                prompt_file="entity_extraction.json"
            )
            # print("\n=== Entity Extraction Template ===")
            self.entity_prompt = entity_prompt_builder.build()
            # print(self.entity_prompt.template)
            # print("================================\n")
            self.entity_chain = (self.entity_prompt | self.llm | StrOutputParser())

            # 共指消解模板
            coreference_prompt_builder = PromptTemplateBuilder(
                prompt_path=os.path.join(self.ROOT_DIR, "prompts/dialogue_analysis"),
                prompt_file="coreference_resolution.json"
            )
            self.coreference_prompt = coreference_prompt_builder.build().partial()
            self.coreference_chain = (self.coreference_prompt | self.llm | StrOutputParser())

            # 话题归属模板
            topic_prompt_builder = PromptTemplateBuilder(
                prompt_path=os.path.join(self.ROOT_DIR, "prompts/dialogue_analysis"),
                prompt_file="topic_belonging.json"
            )
            self.topic_prompt = topic_prompt_builder.build().partial()
            self.topic_chain = (self.topic_prompt | self.llm | StrOutputParser())
        except Exception as e:
            print(f"初始化提示模板失败: {str(e)}")
            raise

    def process_dialogue(
            self,
            dialogue_turn: DialogueTurn
    ) -> None:
        """处理单轮对话"""


        # 2. 更新对话窗口
        self._update_dialogue_window(dialogue_turn)

        # 3. 提取实体
        entities = self._extract_entities(dialogue_turn)

        # 4. 共指消解
        references = self._resolve_coreference(dialogue_turn, entities)

        # 5. 更新实体
        self._update_entities(entities, references)

        # 6. 分析话题归属
        self._analyze_topic_belonging(entities)

        # 7. 更新对话单元的实体ID列表
        dialogue_turn.entity_ids = list(self.entities.keys())

        # 8. 持久化数据
        self._persist_data(dialogue_turn)

    def _update_dialogue_window(self, dialogue_turn: DialogueTurn) -> None:
        """更新对话窗口"""
        self.dialogue_window.append(dialogue_turn)
        if len(self.dialogue_window) > self.window_size:
            self.dialogue_window.pop(0)

    def _get_dialogue_history(self) -> str:
        """获取对话历史"""
        history = []
        for turn in self.dialogue_window[:-1]:  # 除去最新的对话
            history.append(
                f"系统：{turn.system_query}\n"
                f"用户：{turn.user_response}"
            )
        return "\n\n".join(history)

    def _clean_llm_response(self, response: str) -> str:
        """清理LLM的响应，移除Markdown代码块标记"""
        # 移除开头的```json���```
        if response.startswith('```'):
            response = response.split('\n', 1)[1]
        # 移除结尾的```
        if response.endswith('```'):
            response = response.rsplit('\n', 1)[0]
        # 移除可能的空行
        response = response.strip()
        return response

    def _extract_entities(self, dialogue_turn: DialogueTurn) -> List[Dict]:
        """提取实体"""
        try:
            response = self.entity_chain.invoke({
                "system_query": dialogue_turn.system_query,
                "user_response": dialogue_turn.user_response,
                "dialogue_history": self._get_dialogue_history()
            })
            print("\n=== LLM Response (Entity Extraction) ===")
            print(response)
            print("=======================================\n")

            if hasattr(response, 'content'):
                response = response.content
            # 清理响应
            response = self._clean_llm_response(response)
            result = json.loads(response)
            return result["entities"]
        except Exception as e:
            print(f"实体提取失败: {str(e)}")
            print(f"原始响应: {response}")
            return []

    def _resolve_coreference(
            self,
            dialogue_turn: DialogueTurn,
            new_entities: List[Dict]
    ) -> List[Dict]:
        """共指消解"""
        # 1. 获取窗口内的实体信息
        window_entities = []
        for turn in self.dialogue_window[:-1]:  # 不包括当前对话
            turn_entities = []
            for entity_id in turn.entity_ids:
                if entity_id in self.entities:
                    entity = self.entities[entity_id]
                    turn_entities.append({
                        "dialogue_id": turn.id,
                        "name": entity.name,
                        "type": entity.type,
                        "attributes": entity.attributes
                    })
            if turn_entities:
                window_entities.append({
                    "dialogue_id": turn.id,
                    "entities": turn_entities
                })

        response = self.coreference_chain.invoke({
            "system_query": dialogue_turn.system_query,
            "user_response": dialogue_turn.user_response,
            "dialogue_history": self._get_dialogue_history(),
            "window_entities": json.dumps(window_entities, ensure_ascii=False),
            "new_entities": json.dumps(new_entities, ensure_ascii=False)
        })

        try:
            if hasattr(response, 'content'):
                response = response.content
            # 清理响应
            response = self._clean_llm_response(response)
            result = json.loads(response)
            return result["references"]
        except Exception as e:
            print(f"共指消解失败: {str(e)}")
            return []

    def _update_entities(self, new_entities: List[Dict], references: List[Dict]) -> None:
        """更新实体"""
        # 1. 处理新实体，添加去重逻辑
        for entity_info in new_entities:
            # 检查是否已存在相同名称和类型的实体
            existing_entity = None
            for entity in self.entities.values():
                if entity.name == entity_info["name"] and entity.type == entity_info["type"]:
                    existing_entity = entity
                    break

            if existing_entity:
                # 更新现有实体的属性
                existing_entity.attributes.update(entity_info.get("attributes", {}))
            else:
                # 创建新实体
                entity_id = f"entity_{len(self.entities)}"
                entity = Entity(
                    id=entity_id,
                    name=entity_info["name"],
                    type=entity_info["type"],
                    attributes=entity_info.get("attributes", {})
                )
                self.entities[entity_id] = entity

    def _analyze_topic_belonging(self, entities: List[Dict]) -> None:
        """分析话题归属"""
        try:
            # 准备输入
            input_data = {
                "system_query": self.dialogue_window[-1].system_query,
                "user_response": self.dialogue_window[-1].user_response,
                "dialogue_history": self._get_dialogue_history(),
                "entity_info": json.dumps(entities, ensure_ascii=False),
                "primary_topics": ", ".join(self.PRIMARY_TOPICS)
            }
            print("\n=== Topic Analysis Input ===")
            print(json.dumps(input_data, ensure_ascii=False, indent=2))
            print("===========================\n")

            response = self.topic_chain.invoke(input_data)

            # print("\n=== LLM Response (Topic Analysis) ===")
            # print(response)
            # print("===================================\n")

            if hasattr(response, 'content'):
                response = response.content
            # 清理响应
            response = self._clean_llm_response(response)

            print("\n=== Cleaned Response ===")
            print(response)
            print("=======================\n")

            result = json.loads(response)
            self._update_topic_belongings(result["topic_belongings"])
        except Exception as e:
            print(f"话题归属分析失败: {str(e)}")
            print(f"原始响应: {response}")

    def _update_topic_belongings(self, topic_belongings: List[Dict]) -> None:
        """更新话题归属"""
        for belonging in topic_belongings:
            # 只处理预定义的主题
            if belonging['primary_topic'] not in self.PRIMARY_TOPICS:
                continue

            # 1. 更新主话题
            topic_id = f"topic_{belonging['primary_topic']}"
            if topic_id not in self.topics:
                self.topics[topic_id] = Topic(
                    id=topic_id,
                    name=belonging["primary_topic"],
                    primary_category=belonging["primary_topic"]
                )

            # 2. 关联实体和话题
            for entity in self.entities.values():
                if entity.name == belonging["entity"]:
                    entity.topic_belongings.append(topic_id)
                    self.topics[topic_id].related_entities.add(entity.id)

            # 3. 处理相关话题
            for related in belonging.get("related_topics", []):
                related_id = f"topic_{related['topic']}"
                if related_id not in self.topics:
                    self.topics[related_id] = Topic(
                        id=related_id,
                        name=related["topic"],
                        primary_category=related["topic"]
                    )

    def _persist_data(self, dialogue_turn: DialogueTurn):
        """持久化数据"""
        try:
            # 1. 保存对话
            self.db_manager.save_dialogue(
                dialogue_id=dialogue_turn.id,
                system_query=dialogue_turn.system_query,
                user_response=dialogue_turn.user_response,
                entity_ids=dialogue_turn.entity_ids
            )

            # 2. 保存实体
            for entity in self.entities.values():
                self.db_manager.save_entity(
                    entity_id=entity.id,
                    name=entity.name,
                    entity_type=entity.type,
                    attributes=entity.attributes
                )

            # 3. 保存话题
            for topic in self.topics.values():
                self.db_manager.save_topic(
                    topic_id=topic.id,
                    name=topic.name,
                    primary_category=topic.primary_category
                )

            # 4. 保存实体-话题关联
            for entity in self.entities.values():
                for topic_id in entity.topic_belongings:
                    self.db_manager.save_entity_topic_relation(
                        entity_id=entity.id,
                        topic_id=topic_id
                    )

            # 5. 保存到向量数据库
            if hasattr(self.db_manager, 'vector_store'):
                combined_text = f"{dialogue_turn.system_query}\n{dialogue_turn.user_response}"
                self.db_manager.save_to_vector_store(
                    dialogue_id=dialogue_turn.id,
                    text=combined_text,
                    metadata={
                        "timestamp": dialogue_turn.timestamp,
                        "entity_ids": dialogue_turn.entity_ids
                    }
                )

        except Exception as e:
            print(f"持久化数据失败: {str(e)}")
            raise