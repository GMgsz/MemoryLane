from typing import Dict, List, Optional
import logging
from Utils.DatabaseManager import DatabaseManager
import asyncio
import json
import os
from Utils.PromptTemplateBuilder import PromptTemplateBuilder

logger = logging.getLogger(__name__)

class ContentGenerationAgent:
    def __init__(self, db_manager: DatabaseManager, llm_client):
        self.db_manager = db_manager
        self.llm = llm_client
        self.dialogue_count = 0
        self.TRIGGER_INTERVAL = 10
        self.ENTITY_THRESHOLD = 5
        
        # 设置提示模板路径
        self.PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "../prompts/content_generation")
        
    def on_dialogue_complete(self, dialogue_id: str):
        """每轮对话完成后的处理"""
        self.dialogue_count += 1
        if self.dialogue_count >= self.TRIGGER_INTERVAL:
            asyncio.create_task(self.trigger_content_generation())
            self.dialogue_count = 0
            
    async def trigger_content_generation(self):
        """触发内容生成流程"""
        try:
            # 1. 获取新增实体
            new_entities = self.db_manager.get_unprocessed_entities()
            if not new_entities:
                logger.info("没有新的未处理实体")
                return
            
            # 2. 组织实体到聚类
            await self._organize_entities(new_entities)
            
            # 3. 评估聚类
            await self._evaluate_clusters()
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise

    async def _organize_entities(self, new_entities: List[Dict]):
        """组织实体到聚类"""
        try:
            for entity in new_entities:
                # 打印当前处理的实体
                logger.debug(f"处理实体: {entity}")
                
                existing_clusters = self.db_manager.get_cluster_by_topic(
                    entity["primary_topic"]
                )
                logger.debug(f"获取到的现有聚类: {existing_clusters}")
                
                analysis_result = await self._analyze_entity_clustering(
                    entity, 
                    existing_clusters
                )
                
                if analysis_result["decision"] == "JOIN_EXISTING" and existing_clusters:
                    target_cluster_id = analysis_result["target_cluster"]
                    logger.debug(f"加入现有聚类: {target_cluster_id}")
                    self.db_manager.add_entity_to_cluster(
                        entity["id"],
                        target_cluster_id
                    )
                else:
                    # 创建新聚类
                    cluster_id = self.db_manager.create_cluster(
                        name=analysis_result["target_cluster"],
                        primary_topic=entity["primary_topic"],
                        state="ACCUMULATING"  # 添加初始状态
                    )
                    logger.debug(f"创建新聚类: {cluster_id}")
                    self.db_manager.add_entity_to_cluster(
                        entity["id"],
                        cluster_id
                    )
                    
        except Exception as e:
            logger.error(f"组织实体失败: {str(e)}")
            raise

    async def _analyze_entity_clustering(self, entity: Dict, existing_clusters: List[Dict]) -> Dict:
        """分析实体应该加入哪个聚类"""
        try:
            # 打印调试信息
            logger.debug(f"分析实体: {entity}")
            logger.debug(f"现有聚类: {existing_clusters}")
            
            context = await self._prepare_cluster_context(entity, existing_clusters)
            analysis_result = await self.llm.analyze_clustering(context)
            
            # 打印分析结果
            logger.debug(f"分析结果: {analysis_result}")
            return analysis_result
        except Exception as e:
            logger.error(f"分析实体聚类失败: {str(e)}")
            raise

    async def _prepare_cluster_context(self, entity: Dict, existing_clusters: List[Dict]) -> Dict:
        """准备聚类分析的上下文"""
        try:
            # 1. 获取实体相关的对话
            entity_dialogues = self.db_manager.get_dialogue_entities(entity["id"])
            
            # 2. 获取相似的对话
            similar_dialogues = self.db_manager.search_similar_dialogues(
                entity["name"],
                k=3
            )
            
            # 3. 获取相关的实体
            related_entities = []
            for dialogue in entity_dialogues:
                dialogue_entities = self.db_manager.get_dialogue_entities(dialogue["id"])
                related_entities.extend(dialogue_entities)
            
            # 4. 整理上下文
            context = {
                "entity_dialogues": entity_dialogues,
                "existing_clusters": existing_clusters,
                "similar_dialogues": similar_dialogues,
                "related_entities": related_entities
            }
            
            return context
            
        except Exception as e:
            logger.error(f"准备聚类上下文失败: {str(e)}")
            raise

    async def _evaluate_clusters(self):
        """评估聚类"""
        try:
            # 1. 获取所有需要评估的聚类
            clusters = self._get_clusters_to_evaluate()
            
            # 2. 评估每个聚类
            for cluster in clusters:
                await self._evaluate_single_cluster(cluster)
                
        except Exception as e:
            logger.error(f"评估聚类失败: {str(e)}")
            raise

    def _get_clusters_to_evaluate(self) -> List[Dict]:
        """获取需要评估的聚类"""
        try:
            # 获取状态为ACCUMULATING的聚类
            clusters = self.db_manager.get_clusters_by_state("ACCUMULATING")
            
            # 筛选出实体数量达到阈值的聚类
            clusters_to_evaluate = []
            for cluster in clusters:
                entity_count = self.db_manager.get_cluster_entity_count(cluster["id"])
                if entity_count >= self.ENTITY_THRESHOLD:
                    clusters_to_evaluate.append(cluster)
                    
            return clusters_to_evaluate
            
        except Exception as e:
            logger.error(f"获取待评估聚类失败: {str(e)}")
            raise

    async def _evaluate_single_cluster(self, cluster: Dict):
        """评估单个聚类"""
        try:
            # 1. 获取聚类相关数据
            cluster_data = await self._prepare_cluster_data(cluster["id"])
            
            # 2. 调用模型评估
            evaluation = await self.llm.evaluate_cluster(cluster_data)
            
            # 3. 处理评估结果
            if evaluation["can_generate"]:
                # 可以生成内容
                await self._generate_content(cluster["id"], evaluation)
                # 更新聚类状态
                self.db_manager.update_cluster_state(
                    cluster["id"], 
                    "GENERATED"
                )
            else:
                # 继续积累
                self.db_manager.update_cluster_state(
                    cluster["id"],
                    "ACCUMULATING"
                )
                
        except Exception as e:
            logger.error(f"评估聚类失败: {str(e)}")
            raise

    async def _prepare_cluster_data(self, cluster_id: str) -> Dict:
        """准备聚类数据"""
        try:
            # 1. 获取聚类信息
            cluster = self.db_manager.get_cluster(cluster_id)
            
            # 2. 获取聚类中的实体
            entities = self.db_manager.get_cluster_entities(cluster_id)
            
            # 3. 获取相关的对话
            dialogues = []
            for entity in entities:
                entity_dialogues = self.db_manager.get_dialogue_entities(entity["id"])
                dialogues.extend(entity_dialogues)
            
            return {
                "cluster": cluster,
                "entities": entities,
                "dialogues": dialogues
            }
            
        except Exception as e:
            logger.error(f"准备聚类数据失败: {str(e)}")
            raise

    async def _generate_content(self, cluster_id: str, evaluation: Dict):
        """生成内容"""
        try:
            # 1. 准备生成所需的数据
            generation_data = await self._prepare_generation_data(cluster_id, evaluation)
            
            # 2. 构建提示模板
            prompt_builder = PromptTemplateBuilder(
                prompt_path=self.PROMPTS_PATH,
                prompt_file="content_generation.json"
            )
            prompt = prompt_builder.build()
            
            # 3. 调用大模型生成内容
            response = await self.llm.invoke(prompt.format(
                cluster_info=json.dumps(generation_data["cluster_info"], ensure_ascii=False),
                entities=json.dumps(generation_data["entities"], ensure_ascii=False),
                dialogues=json.dumps(generation_data["dialogues"], ensure_ascii=False),
                outline=generation_data["outline"] or "自由发挥"
            ))
            
            content = {
                "content": response.content,
                "word_count": len(response.content)
            }
            
            # 4. 保存生成的内容
            content_id = self.db_manager.save_generated_content(
                cluster_id,
                content["content"]
            )
            
            return content_id
            
        except Exception as e:
            logger.error(f"生成内容失败: {str(e)}")
            raise

    async def _prepare_generation_data(self, cluster_id: str, evaluation: Dict) -> Dict:
        """准备内容生成所需的数据"""
        try:
            # 1. 获取聚类数据
            cluster_data = await self._prepare_cluster_data(cluster_id)
            
            # 2. 整理生成数据
            generation_data = {
                "cluster_info": cluster_data["cluster"],
                "entities": cluster_data["entities"],
                "dialogues": cluster_data["dialogues"],
                "evaluation": evaluation,
                "outline": evaluation.get("content_outline", None)
            }
            
            return generation_data
            
        except Exception as e:
            logger.error(f"准备生成数据失败: {str(e)}")
            raise