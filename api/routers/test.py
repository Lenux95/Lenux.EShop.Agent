"""
测试接口路由
"""

from fastapi import APIRouter
from config import get_logger

logger = get_logger("test_router")

router = APIRouter()

@router.get("/hello")
async def hello_world():
    """
    简单的 hello world 接口
    """
    logger.info("调用 hello_world 接口")
    return {"message": "Hello, World!"}

@router.get("/greet/{name}")
async def greet(name: str):
    """
    打招呼接口
    """
    logger.info(f"调用 greet 接口，name={name}")
    return {"message": f"Hello, {name}!"}

@router.get("/status")
async def get_status():
    """
    获取服务状态
    """
    logger.info("调用 get_status 接口")
    return {
        "status": "running",
        "service": "RAG API",
        "version": "0.1.0"
    }

@router.get("/echo/{text}")
async def echo(text: str):
    """
    回声接口
    """
    logger.info(f"调用 echo 接口，text={text}")
    return {"echo": text}

@router.post("/agent/query")
async def query_agent(data: dict):
    """
    RAG Agent 查询接口
    
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