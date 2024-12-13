import json
from typing import Dict, List, Union

class ResponseParser:
    @staticmethod
    def parse_llm_response(response_text: str) -> Dict:
        """解析LLM的JSON响应"""
        try:
            # 尝试直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            try:
                # 查找第一个 { 和最后一个 }
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                # 如果还是失败，返回默认值
                return {
                    "entities": {},
                    "keywords": []
                } 