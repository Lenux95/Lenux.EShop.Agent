"""
测试接口路由
"""

from fastapi import APIRouter
from config import get_logger

logger = get_logger("test_router")

router = APIRouter()

@router.post("/agent/query")
async def query_agent(data: dict):
    """
    RAG Agent 测试接口
    
    Args:
        data: 包含 query 的字典，例如 {"query": "你的问题"}
    
    Returns:
        包含回答的字典
    """
    query = data.get("query", "")
    if not query:
        return {"error": "缺少 query 参数"}
    
    logger.info(f"🔍 RAG 查询: {query}")
    
    try:
        from langchain_core.documents import Document
        from rag.splitters import split_document
        from rag.embeddings import get_embddings
        from rag.vector_stores import create_vector_store
        from rag.rag_agent import create_rag_agent, query_rag_agent
        
        test_docs = [
            Document(page_content="文档3更新日期：2026年5月7日。", metadata={"source": "文档1"}),
            Document(page_content="历史上第一个庆祝春节的地区是曲阜。", metadata={"source": "文档2"}),
        ]
        
        chunked_docs = split_document(test_docs, spliter_type="recursive", chunk_size=100, chunk_overlap=20)
        embeddings = get_embddings(backend="ollama", model="qwen3-embedding:0.6b")
        
        vector_store = create_vector_store(
            documents=chunked_docs,
            embeddings=embeddings,
            store_type="inmemory"
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.1})
        
        agent = create_rag_agent(
            retriever=retriever,
            model="qwen-plus",
            tool_name="knowledge_base",
        )
        
        result = query_rag_agent(agent=agent, query=query, return_sources=True)
        return result
        
    except Exception as e:
        logger.error(f"❌ RAG 查询失败: {e}")
        return {"error": str(e)}
    



