from typing import Dict, List, Optional
from datetime import datetime
from models.content_manager import ContentSegment, ThematicContent, SubTheme
from config.config import Config

class ThemeManager:
    def __init__(self):
        self.themes: Dict[str, ThematicContent] = {}
        self.config = Config.CONTENT_GENERATION
        self.theme_aspects = Config.THEME_STRUCTURE
        
    async def process_content(self, segment: ContentSegment) -> List[str]:
        """处理新的内容片段，返回需要生成内容的主题列表"""
        themes_to_generate = []
        
        # 处理每个相关主题
        for theme in segment.themes:
            # 更新或创建主题内容
            await self.update_theme_content(theme, segment)
            
            # 检查是否需要生成内容
            if await self._check_generation_trigger(theme):
                themes_to_generate.append(theme)
                
        return themes_to_generate
    
    async def update_theme_content(self, theme: str, segment: ContentSegment):
        """更新主题内容"""
        print(f"\n更新主题内容:")
        print(f"主题: {theme}")
        print(f"新内容: {segment.content}")
        
        # 确保主题存在
        if theme not in self.themes:
            self.themes[theme] = ThematicContent(
                main_theme=theme,
                sub_themes={},
                last_updated=datetime.now()
            )
        
        # 识别或创建子主题
        sub_theme_name = await self._identify_sub_theme(theme, segment)
        
        # 更新子主题
        await self._update_sub_theme(theme, sub_theme_name, segment)
        
    async def _identify_sub_theme(self, theme: str, segment: ContentSegment) -> str:
        """识别内容应该属于哪个子主题"""
        # 使用预定义的主题结构
        if theme in self.theme_aspects:
            aspects = self.theme_aspects[theme]["required_aspects"]
            
            # 基于内容分析确定属于哪个方面
            content = segment.content.lower()
            if "成员" in content or "父母" in content or "姐姐" in content:
                return "家庭成员"
            elif "传统" in content or "节日" in content or "习俗" in content:
                return "家庭传统"
            elif "氛围" in content or "关系" in content or "和睦" in content:
                return "家庭氛围"
            else:
                return "其他"
        return "general"
    
    def _count_chinese_words(self, text: str) -> int:
        """统计中文文本的字数"""
        # 移除空格和标点
        import re
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace(' ', '')
        return len(text)
    
    async def _update_sub_theme(self, theme: str, sub_theme_name: str, segment: ContentSegment):
        """更新子主题内容"""
        theme_content = self.themes[theme]
        
        # 第一阶段：简单地将所有内容存储在 "general" 子主题下
        if "general" not in theme_content.sub_themes:
            theme_content.sub_themes["general"] = SubTheme(
                name="general",
                content_segments=[],
                first_mentioned=datetime.now(),
                last_updated=datetime.now(),
                related_entities={}  # 使用空字典，第二阶段再完善实体关联
            )
        
        # 添加内容片段
        sub_theme = theme_content.sub_themes["general"]
        sub_theme.content_segments.append(segment)
        sub_theme.last_updated = datetime.now()
    
    async def _check_generation_trigger(self, theme: str) -> bool:
        """检查是否需要为主题生成内容"""
        theme_content = self.themes[theme]
        
        # 1. 基础条件检查
        all_segments = []
        print("\n当前主题内容状态:")
        for sub_name, sub_theme in theme_content.sub_themes.items():
            print(f"子主题 '{sub_name}':")
            for seg in sub_theme.content_segments:
                print(f"- 内容: {seg.content}")
                print(f"- 字数: {self._count_chinese_words(seg.content)}")
            all_segments.extend(sub_theme.content_segments)
        
        # 2. 内容完整度检查
        covered_aspects = set(theme_content.sub_themes.keys())
        if theme in self.theme_aspects:
            required_aspects = self.theme_aspects[theme]["required_aspects"]
            completion_ratio = len(covered_aspects) / len(required_aspects)
        else:
            # 对于未定义结构的主题，使用简单的完整度计算
            completion_ratio = 1.0 if len(theme_content.sub_themes) > 0 else 0.0
            required_aspects = ["基本信息"]
        
        print(f"\n检查生成触发条件:")
        print(f"1. 基础指标:")
        print(f"- 子主题数量: {len(theme_content.sub_themes)}")
        print(f"- 当前片段数: {len(all_segments)} (需要: {self.config['MIN_SEGMENTS']})")
        
        total_words = sum(self._count_chinese_words(seg.content) for seg in all_segments)
        print(f"- 总字数: {total_words} (需要: {self.config['MIN_WORDS']})")
        
        print(f"\n2. 内容完整度:")
        print(f"- 已覆盖方面: {covered_aspects}")
        print(f"- 需要方面: {required_aspects}")
        print(f"- 完整度比率: {completion_ratio:.2f}")
        
        # 3. 内容相关度检查
        content_relevance = await self._calculate_content_relevance(all_segments)
        print(f"\n3. 内容相关度: {content_relevance:.2f}")
        
        # 4. 用户兴趣度检查
        interest_level = self._calculate_interest_level(all_segments)
        print(f"4. 用户兴趣度: {interest_level:.2f}")
        
        # 5. 时间跨度检查
        if len(theme_content.sub_themes) > 0:
            earliest = min(st.first_mentioned for st in theme_content.sub_themes.values())
            latest = max(st.last_updated for st in theme_content.sub_themes.values())
            time_span = (latest - earliest).days
            print(f"5. 时间跨度: {time_span}天 (需要: {self.config['MIN_TIME_SPAN']}天)")
        
        # 综合评估
        should_generate = (
            len(all_segments) >= self.config['MIN_SEGMENTS'] and
            total_words >= self.config['MIN_WORDS'] and
            completion_ratio >= 0.5 and
            content_relevance >= self.config['SIMILARITY_THRESHOLD'] and
            interest_level >= self.config['INTEREST_THRESHOLD']
        )
        
        if should_generate:
            print("\n满足所有条件，将生成内容")
            return True
        else:
            print("\n条件不满足，原因:")
            if len(all_segments) < self.config['MIN_SEGMENTS']:
                print("- 片段数不足")
            if total_words < self.config['MIN_WORDS']:
                print("- 总字数不足")
            if completion_ratio < 0.5:
                print("- 内容完整度不足")
            if content_relevance < self.config['SIMILARITY_THRESHOLD']:
                print("- 内容相关度不足")
            if interest_level < self.config['INTEREST_THRESHOLD']:
                print("- 用户兴趣度不足")
            return False
            
    async def _calculate_content_relevance(self, segments: List[ContentSegment]) -> float:
        """计算内容相关度"""
        if not segments:
            return 0.0
            
        # 基于内容的相似性分析
        # TODO: 使用向量相似度计算
        return 0.7  # 临时返回固定值
        
    def _calculate_interest_level(self, segments: List[ContentSegment]) -> float:
        """计算用户兴趣度"""
        if not segments:
            return 0.0
            
        # 分析最近的对话内容中的兴趣指标
        # 1. 回答详细程度
        # 2. 情感投入度
        # 3. 主动分享程度
        return 0.8  # 临时返回固定值