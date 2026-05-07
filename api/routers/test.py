"""
测试接口路由
"""

from fastapi import APIRouter
from config import get_logger
from rag.rag_agent import create_rag_agent, query_rag_agent

logger = get_logger("test_router")

router = APIRouter()

@router.get("/hello")
async def hello_world():
    """
    简单的 hello world 接口
    返回字符串 "Hello, World!"
    """
    logger.info("调用 hello_world 接口")
    return {"message": "Hello, World!"}

@router.get("/greet/{name}")
async def greet(name: str):
    """
    打招呼接口
    根据输入的名字返回问候语
    
    Args:
        name: 姓名
    
    Returns:
        包含问候语的字典
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
    返回输入的文本
    
    Args:
        text: 要返回的文本
    
    Returns:
        包含输入文本的字典
    """
    logger.info(f"调用 echo 接口，text={text}")
    return {"echo": text}

@router.post("/agent/query")
async def query_agent(data: dict):
    """
    测试 RAG Agent 查询接口
    
    Args:
        data: 包含 query 的字典，例如 {"query": "你的问题"}
    
    Returns:
        包含回答的字典
    """
    query = data.get("query", "")
    if not query:
        return {"error": "缺少 query 参数"}
    
    logger.info(f"调用 query_agent 接口，query={query}")
    
    try:
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings as BaseEmbeddings
        from rag.vector_stores import create_vector_store
        from rag.embeddings import get_embddings
        
        test_docs = [
            Document(page_content="实时日期：2026年5月7日。", metadata={"source": "test"}),
            Document(page_content="春节是中国最重要的传统节日。", metadata={"source": "test"}),
        ]
        
        # class DummyEmbeddings(BaseEmbeddings):
        #     def embed_documents(self, texts):
        #         return [[0.1]*768 for _ in texts]
        #     def embed_query(self, text):
        #         return [0.1]*768
        embeddings = get_embddings(backend="ollama", model="qwen3-embedding:0.6b")
        
        vector_store = create_vector_store(
            documents=test_docs,
            embeddings=embeddings,
            store_type="inmemory"
        )
        retriever = vector_store.as_retriever()
        
        agent = create_rag_agent(retriever=retriever, model="qwen-plus")
        result = query_rag_agent(agent=agent, query=query)
        
        return result
        
    except Exception as e:
        logger.error(f"Agent 查询失败: {e}")
        return {"error": str(e)}