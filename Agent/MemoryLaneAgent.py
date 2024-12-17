from typing import List, Optional, Tuple
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.base import VectorStoreRetriever
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PrintUtils import *
import os


def _format_long_term_memory(task_description: str, memory: BaseChatMemory) -> str:
    return memory.load_memory_variables(
        {"prompt": task_description}
    )["history"]

def _format_short_term_memory(memory: BaseChatMemory) -> str:
    messages = memory.chat_memory.messages
    string_messages = [messages[i].content for i in range(1,len(messages))]
    return "\n".join(string_messages)

class MemoryLaneAgent:
    """MemoryLaneAgent: 基于记忆的对话聊天智能体"""
    def __init__(
            self,
            llm: BaseChatModel,
            prompts_path: str,
            main_prompt_file: str = "main.json",
            memory_retriever: Optional[VectorStoreRetriever] = None,
            max_token_limit: int = 4000
    ):
        self.llm = llm
        self.prompts_path = prompts_path
        self.main_prompt_file = main_prompt_file
        self.memory_retriever = memory_retriever
        
        # 初始化短期记忆
        self.short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
        )
        
        # 初始化长期记忆(如果提供了retriever)
        if self.memory_retriever is not None:
            self.long_term_memory = VectorStoreRetrieverMemory(
                retriever=self.memory_retriever,
            )
        else:
            self.long_term_memory = None
            
        # 初始化提示模板
        self.prompt_template = PromptTemplateBuilder(
            self.prompts_path,
            self.main_prompt_file,
        ).build().partial()
        
        # 初始化LLM链
        self.chain = (self.prompt_template | self.llm | StrOutputParser())
        
        # 存储当前的系统问题
        self.current_question = None


    def start_chat(self) -> str:
        """开始对话，生成第一个问题
        这里是需要设计的——》初始化或者是新会话
        """
        # 生成初始问题
        context = {
            "short_term_memory": "",
            "long_term_memory": "",
            "user_message": "START_CHAT"
        }
        
        self.current_question = self.chain.invoke(context)
        return self.current_question
        
    def chat(self, user_message: str, verbose: bool = False) -> str:
        """处理用户输入并返回回复"""
        if self.current_question is None:
            # 如果没有当前问题，先生成一个
            return self.start_chat()
            
        # 保存当前的问答对到记忆中
        self.short_term_memory.save_context(
            {"input": "系统: " + self.current_question},
            {"output": "用户: " + user_message}
        )
        
        # 保存到长期记忆
        if self.long_term_memory is not None:
            self.long_term_memory.save_context(
                {"input": "系统: " + self.current_question},
                {"output": "用户: " + user_message}
            )
        
        # 准备上下文生成下一个问题
        context = {
            "short_term_memory": _format_short_term_memory(self.short_term_memory),
            "long_term_memory": _format_long_term_memory(user_message, self.long_term_memory) 
                if self.long_term_memory is not None else "",
            "user_message": user_message
        }
        
        # 生成下一个问题
        response = ""
        for s in self.chain.stream(context):
            if verbose:
                color_print(s, THOUGHT_COLOR, end="")
            response += s
            
        # 更新当前问题
        self.current_question = response
        return response

    def _init_prompts(self):
        """初始化提示模板"""
        prompt_builder = PromptTemplateBuilder(
            prompt_path=os.path.join(self.ROOT_DIR, "prompts"),  # 注意这里的路径
            prompt_file="main/main"  # 注意这里的文件路径格式
        )
        self.prompt = prompt_builder.build()


