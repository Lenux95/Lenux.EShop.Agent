from typing import List,Optional,Any,Dict
from langchain_core.retrievers import BaseRetriever
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

from config import get_logger,settings

logger=get_logger(__name__)

# RAG 系统提示词
DEFAULT_RAG_SYSTEM_PROMPT = """你是一个智能问答助手，专门回答基于知识库的问题。

你的任务：
1. 使用 knowledge_base 工具搜索相关信息
2. 基于检索到的文档内容回答用户问题
3. 如果文档中没有相关信息，诚实地告诉用户
4. 在回答中引用来源文档（如果有 source 信息）

回答要求：
- 准确：严格基于文档内容，不要编造信息
- 完整：尽可能提供详细的回答
- 清晰：使用简洁明了的语言
- 引用：在回答末尾列出参考的文档来源

示例回答格式：
[回答内容]

参考来源：
- 文档1: [来源信息]
- 文档2: [来源信息]
"""

def create_rag_agent(
        retriever:BaseRetriever,
        model:Optional[str]=None,
        system_prompt: Optional[str] = None,
        tool_name:str="knowledge_base",
        tool_description:Optional[str]=None,
        **kwargs,
):
    """
    创建agent，默认使用ollama本地模型

    Args:
        retriever (BaseRetriever): _description_
        model (Optional[str], optional): _description_. Defaults to None.
        system_prompt (Optional[str], optional): _description_. Defaults to None.
        tool_name (str, optional): _description_. Defaults to "knowledge_base".
        tool_description (Optional[str], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    # todo 
    # 设置一个大模型，准备用ollama本地模型
    # 获取model后续需要设置配置项通过core模块引入
    # create_agent 需要的格式是 "openai:gpt-4o"，需要对模型进行格式化
    model_name=model or settings.ollama_model
    model_str=f"ollama:{model_name}"

    # 使用默认系统提示词
    if system_prompt is None:
        system_prompt = DEFAULT_RAG_SYSTEM_PROMPT

    if tool_description is None:
        tool_description = (
            "搜索知识库中的相关信息。"
            "当需要回答关于文档内容的问题时使用此工具。"
            "输入应该是一个搜索查询。"
        )

    logger.info("创建 RAG Agent")

    # 创建检索器工具
    retriever_tool=create_retriever_tool(
        retriever=retriever,
        name=tool_name,
        description=tool_description
    )

    tools=[retriever_tool]

    agent=create_agent(
        model=model_str,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )

    logger.info(f"RAG Agent 创建成功")
    logger.info(f"使用模型: {model}")

    return agent

# todo 格式化返回体
# def format_rag_response(
