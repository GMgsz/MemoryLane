import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import shutil

class VersionManager:
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        self.versions_dir = f"{base_dir}/versions"
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保必要的目录结构存在"""
        os.makedirs(self.versions_dir, exist_ok=True)
        
    async def create_snapshot(self, 
                            dialogue_history: List[Dict],
                            generated_contents: Dict[str, str],
                            theme_data: Dict,
                            description: str = "") -> str:
        """创建数据快照"""
        # 生成版本ID
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = f"{self.versions_dir}/{version_id}"
        
        try:
            # 创建版本目录
            os.makedirs(version_dir)
            
            # 保存版本信息
            version_info = {
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "status": "complete"
            }
            
            # 保存数据
            with open(f"{version_dir}/dialogue_history.json", 'w', encoding='utf-8') as f:
                json.dump(dialogue_history, f, ensure_ascii=False, indent=2)
                
            with open(f"{version_dir}/generated_contents.json", 'w', encoding='utf-8') as f:
                json.dump(generated_contents, f, ensure_ascii=False, indent=2)
                
            with open(f"{version_dir}/theme_data.json", 'w', encoding='utf-8') as f:
                json.dump(theme_data, f, ensure_ascii=False, indent=2)
                
            with open(f"{version_dir}/version_info.json", 'w', encoding='utf-8') as f:
                json.dump(version_info, f, ensure_ascii=False, indent=2)
                
            return version_id
            
        except Exception as e:
            # 如果出错，删除版本目录
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            raise e
            
    async def list_versions(self) -> List[Dict]:
        """列出所有可用的版本"""
        versions = []
        for version_id in os.listdir(self.versions_dir):
            info_file = f"{self.versions_dir}/{version_id}/version_info.json"
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    versions.append(info)
                    
        # 按时间戳排序
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        return versions
        
    async def restore_version(self, version_id: str) -> Dict:
        """恢复到指定版本"""
        version_dir = f"{self.versions_dir}/{version_id}"
        
        if not os.path.exists(version_dir):
            raise ValueError(f"Version {version_id} not found")
            
        try:
            # 读取版本数据
            with open(f"{version_dir}/dialogue_history.json", 'r', encoding='utf-8') as f:
                dialogue_history = json.load(f)
                
            with open(f"{version_dir}/generated_contents.json", 'r', encoding='utf-8') as f:
                generated_contents = json.load(f)
                
            with open(f"{version_dir}/theme_data.json", 'r', encoding='utf-8') as f:
                theme_data = json.load(f)
                
            return {
                "dialogue_history": dialogue_history,
                "generated_contents": generated_contents,
                "theme_data": theme_data
            }
            
        except Exception as e:
            raise ValueError(f"Error restoring version {version_id}: {str(e)}")
            
    async def compare_versions(self, 
                             version_id1: str, 
                             version_id2: str) -> Dict:
        """比较两个版本的差异"""
        v1_data = await self.restore_version(version_id1)
        v2_data = await self.restore_version(version_id2)
        
        differences = {
            "dialogue_history": {
                "added": len(v2_data["dialogue_history"]) - len(v1_data["dialogue_history"]),
                "changed_topics": self._compare_topics(
                    v1_data["dialogue_history"],
                    v2_data["dialogue_history"]
                )
            },
            "generated_contents": {
                "changed_themes": [
                    theme for theme in v2_data["generated_contents"]
                    if theme not in v1_data["generated_contents"] or
                    v1_data["generated_contents"][theme] != v2_data["generated_contents"][theme]
                ]
            }
        }
        
        return differences
        
    def _compare_topics(self, 
                       history1: List[Dict], 
                       history2: List[Dict]) -> Dict:
        """比较两个对话历史中的主题变化"""
        topics1 = set(turn["topic"] for turn in history1)
        topics2 = set(turn["topic"] for turn in history2)
        
        return {
            "added": list(topics2 - topics1),
            "removed": list(topics1 - topics2)
        } 