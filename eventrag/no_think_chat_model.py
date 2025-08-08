import re
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class QwenNoThinkModel(BaseChatModel):
    """自定义的 ChatModel，自动去除响应中的 <think> 标签"""

    def __init__(self, **kwargs):
        super().__init__()
        # 使用私有属性存储底层 LLM
        self._llm = ChatOpenAI(**kwargs)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """生成聊天响应并去除 <think> 标签"""
        # 修改消息，在最后一条消息末尾添加 /no_think
        modified_messages = self._add_no_think_to_messages(messages)

        # 调用底层 LLM
        result = self._llm._generate(modified_messages, stop, run_manager, **kwargs)

        # 处理每个生成的响应
        processed_generations = []
        for generation in result.generations:
            content = generation.message.content
            # 去除 <think> 标签及其内容
            cleaned_content = self._remove_think_tags(content)

            # 创建新的 AIMessage
            new_message = AIMessage(content=cleaned_content)
            new_generation = ChatGeneration(message=new_message)
            processed_generations.append(new_generation)

        return ChatResult(generations=processed_generations, llm_output=result.llm_output)

    def _remove_think_tags(self, content: str) -> str:
        """去除内容中的 <think> 标签及其内容"""
        if not content:
            return content

        # 使用正则表达式去除 <think>...</think> 标签及其内容
        pattern = r'<think>.*?</think>'
        cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL)

        # 可能出现开头有 <think> 但无 </think> 的情况，去除 <think>
        pattern = r'^<think>'
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL)

        # 清理多余的空白字符
        cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content).strip()

        return cleaned_content

    def _add_no_think_to_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """在消息列表的最后一条消息末尾添加 /no_think"""
        if not messages:
            return messages

        modified_messages = messages[:-1]  # 除最后一条消息外的所有消息
        last_message = messages[-1]

        # 获取最后一条消息的内容并添加 /no_think
        last_content = last_message.content
        if isinstance(last_content, str):
            new_content = last_content + " /no_think"
        else:
            new_content = str(last_content) + " /no_think"

        # 根据消息类型创建新的消息对象
        if isinstance(last_message, HumanMessage):
            new_last_message = HumanMessage(content=new_content)
        elif isinstance(last_message, SystemMessage):
            new_last_message = SystemMessage(content=new_content)
        elif isinstance(last_message, AIMessage):
            new_last_message = AIMessage(content=new_content)
        else:
            # 对于其他类型的消息，使用原始类型
            new_last_message = type(last_message)(content=new_content)

        modified_messages.append(new_last_message)
        return modified_messages

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "custom_chat_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别参数"""
        return self._llm._identifying_params
