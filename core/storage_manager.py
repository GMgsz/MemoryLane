import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from models.content_manager import ThematicContent, ContentSegment
from models.schemas import DialogueTurn
from core.version_manager import VersionManager

class StorageManager:
    def __init__(self, storage_dir: str = "./data"):
        self.storage_dir = storage_dir
        self.version_manager = VersionManager(storage_dir)
        self.ensure_storage_structure()
        
    def ensure_storage_structure(self):
        """确保存储目录结构存在"""
        directories = [
            self.storage_dir,
            f"{self.storage_dir}/generated_content",
            f"{self.storage_dir}/dialogue_history",
            f"{self.storage_dir}/themes"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    async def save_generated_content(self, 
                                   theme: str, 
                                   content: str, 
                                   version: int = None):
        """保存生成的内容"""
        theme_dir = f"{self.storage_dir}/generated_content/{theme}"
        os.makedirs(theme_dir, exist_ok=True)
        
        # 获取最新版本号
        if version is None:
            existing_versions = [
                int(f.split('_')[1].split('.')[0])
                for f in os.listdir(theme_dir)
                if f.startswith('version_')
            ]
            version = max(existing_versions, default=0) + 1
            
        # 保存内容
        filename = f"{theme_dir}/version_{version}.json"
        data = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "version": version
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    async def load_generated_content(self, 
                                   theme: str, 
                                   version: int = None) -> Optional[str]:
        """加载生成的内容"""
        theme_dir = f"{self.storage_dir}/generated_content/{theme}"
        
        if not os.path.exists(theme_dir):
            return None
            
        if version is None:
            # 获取最新版本
            versions = [
                int(f.split('_')[1].split('.')[0])
                for f in os.listdir(theme_dir)
                if f.startswith('version_')
            ]
            if not versions:
                return None
            version = max(versions)
            
        filename = f"{theme_dir}/version_{version}.json"
        if not os.path.exists(filename):
            return None
            
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["content"]
            
    async def save_dialogue_history(self, dialogue_history: List[DialogueTurn]):
        """保存对话历史"""
        filename = f"{self.storage_dir}/dialogue_history/history.json"
        
        # 转换为可序列化的格式
        serializable_history = [
            {
                "id": turn.id,
                "question": turn.question,
                "answer": turn.answer,
                "topic": turn.topic,
                "emotion_score": turn.emotion_score,
                "interest_score": turn.interest_score,
                "depth_level": turn.depth_level,
                "timestamp": datetime.now().isoformat()
            }
            for turn in dialogue_history
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)
            
    async def load_dialogue_history(self) -> List[DialogueTurn]:
        """加载对话历史"""
        filename = f"{self.storage_dir}/dialogue_history/history.json"
        
        if not os.path.exists(filename):
            return []
            
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [DialogueTurn(**turn) for turn in data]
            
    async def save_theme_data(self, themes: Dict[str, ThematicContent]):
        """保存主题数据"""
        theme_dir = f"{self.storage_dir}/themes"
        
        for theme_name, theme_content in themes.items():
            filename = f"{theme_dir}/{theme_name}.json"
            
            # 转换为可序列化的格式
            serializable_content = {
                "main_theme": theme_content.main_theme,
                "sub_themes": {
                    name: {
                        "name": sub_theme.name,
                        "first_mentioned": sub_theme.first_mentioned.isoformat(),
                        "last_updated": sub_theme.last_updated.isoformat(),
                        "related_entities": sub_theme.related_entities,
                        "content_segments": [
                            self._serialize_content_segment(seg)
                            for seg in sub_theme.content_segments
                        ]
                    }
                    for name, sub_theme in theme_content.sub_themes.items()
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_content, f, ensure_ascii=False, indent=2)
                
    def _serialize_content_segment(self, segment: ContentSegment) -> Dict:
        """将ContentSegment转换为可序列化的格式"""
        return {
            "id": segment.id,
            "content": segment.content,
            "timestamp": segment.timestamp.isoformat(),
            "entities": segment.entities,
            "themes": segment.themes,
            "keywords": segment.keywords,
            "dialogue_context": [
                {
                    "id": turn.id,
                    "question": turn.question,
                    "answer": turn.answer,
                    "topic": turn.topic,
                    "emotion_score": turn.emotion_score,
                    "interest_score": turn.interest_score,
                    "depth_level": turn.depth_level
                }
                for turn in segment.dialogue_context
            ]
        } 
    async def create_backup(self, description: str = "") -> str:
        """创建当前状态的备份"""
        # 获取当前状态的序列化数据
        dialogue_history = await self._get_serialized_dialogue_history()
        generated_contents = await self._get_serialized_generated_contents()
        theme_data = await self._get_serialized_theme_data()
        
        # 创建快照
        version_id = await self.version_manager.create_snapshot(
            dialogue_history,
            generated_contents,
            theme_data,
            description
        )
        
        return version_id
        
    async def restore_backup(self, version_id: str):
        """恢复到指定版本"""
        data = await self.version_manager.restore_version(version_id)
        
        # 恢复数据
        await self.save_dialogue_history(data["dialogue_history"])
        
        # 恢复生成的内容
        for theme, content in data["generated_contents"].items():
            await self.save_generated_content(theme, content)
            
        # 恢复主题数据
        await self.save_theme_data(data["theme_data"]) 