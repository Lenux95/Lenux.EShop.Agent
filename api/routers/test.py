"""
测试接口路由
"""

import os
from typing import Optional
from pathlib import Path
from fastapi import APIRouter
from rag import load_document, load_documents, split_document, get_embddings, create_rag_agent, query_rag_agent
from config import get_logger, settings

logger = get_logger("test_router")

router = APIRouter()

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent.resolve()

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

    logger.info(f"RAG 查询: {query}")

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
        logger.error(f"RAG 查询失败: {e}")
        return {"error": str(e)}

@router.post("/agent/create_index")
async def create_index(data: Optional[dict] = None):
    """
    将 data/documents 中的知识库文档进行向量持久化

    处理流程：
    1. 加载 data/documents 目录下的所有文档
    2. 文本分块
    3. 向量化（使用 Ollama 嵌入模型）
    4. 创建 Chroma 向量库并持久化到 data/indexes

    Args:
        data: 可选参数，支持自定义配置
            - documents_path: 文档目录路径（默认 data/documents）
            - index_path: 向量库存储路径（默认 data/indexes）
            - collection_name: Chroma 集合名称（默认 catalog_products）
            - chunk_size: 分块大小（默认 500）
            - chunk_overlap: 分块重叠（默认 100）
            - embedding_model: 嵌入模型名称（默认 qwen3-embedding:0.6b）

    Returns:
        包含索引创建结果的字典
    """
    if data is None:
        data = {}

    project_root = get_project_root()

    documents_path = data.get("documents_path", str(project_root / "data" / "documents"))
    index_path = data.get("index_path", str(project_root / settings.vector_store_path))
    collection_name = data.get("collection_name", "catalog_products")
    chunk_size = data.get("chunk_size", 500)
    chunk_overlap = data.get("chunk_overlap", 100)
    embedding_model = data.get("embedding_model", "qwen3-embedding:0.6b")

    logger.info(f"开始创建向量索引")
    logger.info(f"  项目根目录: {project_root}")
    logger.info(f"  文档路径: {documents_path}")
    logger.info(f"  索引路径: {index_path}")

    try:
        # 1. 检查文档目录
        docs_dir = Path(documents_path)
        if not docs_dir.exists():
            return {"error": f"文档目录不存在: {documents_path}"}

        # 2. 列出要加载的文件
        all_files = []
        for ext in [".txt", ".md", ".pdf"]:
            all_files.extend(docs_dir.glob(f"**/*{ext}"))

        logger.info(f"发现 {len(all_files)} 个文档文件")

        if not all_files:
            return {"error": f"未找到任何文档文件（支持 .txt, .md, .pdf）"}

        # 3. 加载文档
        all_docs = []
        for file_path in all_files:
            try:
                docs = load_document(str(file_path), add_metadata=True)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")
                continue

        if not all_docs:
            return {"error": "所有文档加载失败"}

        doc_count = len(all_docs)
        logger.info(f"加载了 {doc_count} 个文档块")

        # 4. 文本分块
        logger.info(f"文本分块: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        chunked_docs = split_document(
            all_docs,
            spliter_type="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunk_count = len(chunked_docs)
        logger.info(f"分块完成: {chunk_count} 个块")

        # 5. 向量化
        logger.info(f"加载嵌入模型: {embedding_model}")
        embeddings = get_embddings(backend="ollama", model=embedding_model)

        # 6. 确保索引目录存在
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        # 7. 创建 Chroma 向量库并持久化
        logger.info(f"创建 Chroma 向量库: persist_directory={index_path}, collection={collection_name}")
        from rag.vector_stores import create_vector_store

        vector_store = create_vector_store(
            documents=chunked_docs,
            embeddings=embeddings,
            store_type="chroma",
            persist_directory=str(index_dir),
            collection_name=collection_name,
        )

        # 8. 验证索引
        test_results = vector_store.similarity_search_with_score("智能手机", k=3)
        test_count = len(test_results)

        logger.info(f"向量索引创建成功: {doc_count} 个文档, {chunk_count} 个块")

        return {
            "status": "success",
            "documents_loaded": doc_count,
            "chunks_created": chunk_count,
            "index_path": str(index_dir),
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "test_search_results": test_count,
            "message": f"成功将 {doc_count} 个文档（{chunk_count} 个块）向量化并持久化到 {index_dir}"
        }

    except ImportError as e:
        logger.error(f"依赖缺失: {e}")
        return {"error": f"依赖缺失，请安装: pip install langchain-chroma chromadb"}
    except Exception as e:
        logger.error(f"创建索引失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return {"error": str(e)}

@router.post("/agent/search_knowledge")
async def search_knowledge(data: dict):
    """
    搜索已存在的知识库 catalog_products

    Args:
        data: 包含查询参数的字典
            - query: 搜索查询字符串（必需）
            - index_path: 向量库存储路径（默认 data/indexes）
            - collection_name: Chroma 集合名称（默认 catalog_products）
            - k: 返回结果数量（默认 5）
            - embedding_model: 嵌入模型名称（默认 qwen3-embedding:0.6b）
            - with_scores: 是否返回相似度分数（默认 true）

    Returns:
        包含搜索结果的字典
    """
    query = data.get("query", "")
    if not query:
        return {"error": "缺少 query 参数"}

    project_root = get_project_root()

    index_path = data.get("index_path", str(project_root / settings.vector_store_path))
    collection_name = data.get("collection_name", "catalog_products")
    k = data.get("k", 5)
    embedding_model = data.get("embedding_model", "qwen3-embedding:0.6b")
    with_scores = data.get("with_scores", True)

    logger.info(f"知识库搜索: query={query}, k={k}")

    try:
        # 1. 检查索引目录
        index_dir = Path(index_path)
        if not index_dir.exists():
            return {"error": f"索引目录不存在: {index_path}"}

        # 2. 加载嵌入模型
        logger.info(f"加载嵌入模型: {embedding_model}")
        embeddings = get_embddings(backend="ollama", model=embedding_model)

        # 3. 加载已持久化的向量库
        logger.info(f"加载向量库: {index_path}, collection={collection_name}")
        from rag.vector_stores import load_vector_store

        vector_store = load_vector_store(
            load_path=str(index_dir),
            embeddings=embeddings,
            store_type="chroma",
            collection_name=collection_name,
        )

        # 4. 执行搜索
        if with_scores:
            results = vector_store.similarity_search_with_score(query, k=k)
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "similarity": 1.0 - min(score, 1.0)  # 转换为相似度（0-1）
                })
        else:
            results = vector_store.similarity_search(query, k=k)
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })

        logger.info(f"搜索完成，找到 {len(formatted_results)} 个结果")

        return {
            "status": "success",
            "query": query,
            "index_path": str(index_dir),
            "collection_name": collection_name,
            "results_count": len(formatted_results),
            "results": formatted_results
        }

    except ImportError as e:
        logger.error(f"依赖缺失: {e}")
        return {"error": f"依赖缺失，请安装: pip install langchain-chroma chromadb"}
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return {"error": str(e)}

@router.post("/agent/query_with_knowledge")
async def query_with_knowledge(data: dict):
    """
    使用已存在的知识库 catalog_products 进行 RAG 问答

    Args:
        data: 包含查询参数的字典
            - query: 查询字符串（必需）
            - index_path: 向量库存储路径（默认 data/indexes）
            - collection_name: Chroma 集合名称（默认 catalog_products）
            - embedding_model: 嵌入模型名称（默认 qwen3-embedding:0.6b）
            - llm_model: LLM 模型名称（默认 qwen-plus）
            - k: 检索的文档数量（默认 3）
            - score_threshold: 相似度阈值（默认 0.1）

    Returns:
        包含回答的字典
    """
    query = data.get("query", "")
    if not query:
        return {"error": "缺少 query 参数"}

    project_root = get_project_root()

    index_path = data.get("index_path", str(project_root / settings.vector_store_path))
    collection_name = data.get("collection_name", "catalog_products")
    embedding_model = data.get("embedding_model", "qwen3-embedding:0.6b")
    llm_model = data.get("llm_model", "qwen-plus")
    k = data.get("k", 3)
    score_threshold = data.get("score_threshold", 0.1)

    logger.info(f"RAG 问答: query={query}")

    try:
        # 1. 检查索引目录
        index_dir = Path(index_path)
        if not index_dir.exists():
            return {"error": f"索引目录不存在: {index_path}"}

        # 2. 加载嵌入模型
        logger.info(f"加载嵌入模型: {embedding_model}")
        embeddings = get_embddings(backend="ollama", model=embedding_model)

        # 3. 加载已持久化的向量库
        logger.info(f"加载向量库: {index_path}, collection={collection_name}")
        from rag.vector_stores import load_vector_store

        vector_store = load_vector_store(
            load_path=str(index_dir),
            embeddings=embeddings,
            store_type="chroma",
            collection_name=collection_name,
        )

        # 4. 创建检索器
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        # 5. 创建 RAG Agent
        logger.info(f"创建 RAG Agent: model={llm_model}")
        agent = create_rag_agent(
            retriever=retriever,
            model=llm_model,
            tool_name="product_knowledge_base",
        )

        # 6. 执行查询
        result = query_rag_agent(agent=agent, query=query, return_sources=True)
        return result

    except ImportError as e:
        logger.error(f"依赖缺失: {e}")
        return {"error": f"依赖缺失，请安装: pip install langchain-chroma chromadb"}
    except Exception as e:
        logger.error(f"RAG 问答失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return {"error": str(e)}