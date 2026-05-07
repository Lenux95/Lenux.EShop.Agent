import os
from typing import List,Optional,Any,Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import get_logger,settings

logger=get_logger(__name__)

DEFAULT_RAG_SYSTEM_PROMPT = """你是一个智能问答助手，专门回答基于知识库的问题。

你的任务：
1. 使用 knowledge_base 工具搜索相关信息
2. 基于检索到的文档内容回答用户问题
3. 如果文档中没有相关信息，诚实地告诉用户
4. 在回答中引用来源文档（如果有 source 信息）

回答要求：
- 准确：只使用检索到的文档内容，严格基于文档内容，不要编造信息
- 完整：尽可能提供详细的回答
- 清晰：使用简洁明了的语言
- 引用：在回答末尾列出参考的文档来源
- 如果没有相关文档，直接说"不知道"
- 不要编造信息

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
        retriever (BaseRetriever): 检索器实例
        model (Optional[str], optional): 模型名称. Defaults to None.
        system_prompt (Optional[str], optional): 系统提示词. Defaults to None.
        tool_name (str, optional): 工具名称. Defaults to "knowledge_base".
        tool_description (Optional[str], optional): 工具描述. Defaults to None.

    Returns:
        Agent: RAG Agent 实例
    """
    from langchain.agents import create_agent
    
    # todo 统一标准化的模型
    # model_name=model or settings.ollama_model
    # model_str=f"openai:{model_name}"
    llm = ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        base_url=settings.openai_base_url,
        model="qwen-plus",
    )

    if system_prompt is None:
        system_prompt = DEFAULT_RAG_SYSTEM_PROMPT

    if tool_description is None:
        tool_description = (
            "搜索知识库中的相关信息。"
            "当需要回答关于文档内容的问题时使用此工具。"
            "输入应该是一个搜索查询。"
        )

    logger.info("创建 RAG Agent")

    retriever_tool=create_retriever_tool(
        retriever=retriever,
        name=tool_name,
        description=tool_description
    )

    tools=[retriever_tool]

    agent=create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )

    logger.info(f"RAG Agent 创建成功")
    logger.info(f"使用模型: {model}")

    return agent

def query_rag_agent(
    agent,
    query: str,
    return_sources: bool = True,
) -> Dict[str, Any]:
    """
    查询 RAG Agent 的便捷函数
    
    Args:
        agent: RAG Agent 实例
        query: 查询问题
        return_sources: 是否返回来源文档
        
    Returns:
        包含回答的字典
        
    Example:
        >>> agent = create_rag_agent(retriever)
        >>> result = query_rag_agent(agent, "什么是机器学习？")
        >>> print(result["answer"])
    """
    logger.info(f"🔍 查询 RAG Agent: {query[:50]}...")
    
    try:
        # 执行查询 - LangChain 1.0.3 的 agent 需要字典输入
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        # 提取回答
        if isinstance(result, dict) and "messages" in result:
            # 获取最后一条消息
            messages = result["messages"]
            if messages:
                answer = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
            else:
                answer = str(result)
        else:
            answer = str(result)
        
        # 格式化响应
        formatted = {"answer": answer}
        
        logger.info("✅ 查询完成")
        return formatted
        
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}")
        raise