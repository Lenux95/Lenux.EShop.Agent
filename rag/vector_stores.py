from typing import List,Optional,Literal
from pathlib import Path
import os
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore

try:
    from langchain_chroma import Chroma
    Chroma_AVAILABLE = True
except ImportError:
    Chroma_AVAILABLE = False

from config import get_logger,settings

logger=get_logger(__name__)

# 向量库类型
VectorStoreType = Literal["chroma", "inmemory"]

def create_vector_store(
        documents:List[Document],
        embeddings:Embeddings,
        store_type:Optional[VectorStoreType]="chroma",
        **kwargs,
)->VectorStore:
    """
    创建向量库实例
    支持内存、chroma本地
    -chroma可指定persist_directory、collection_name

    Args:
        documents (List[Document]): 文本列表
        embeddings (Embeddings): 向量化模型
        store_type (Optional[VectorStoreType], optional): 向量库类型. Defaults to "chroma".

    Raises:
        ImportError: _description_
        ValueError: _description_

    Returns:
        VectorStore: 向量库实例
    """    
    vector_store_type = store_type or settings.vector_store_type

    logger.info(f"向量库{vector_store_type}开始创建")
           
    try:
        if vector_store_type=="chroma":
            if not Chroma_AVAILABLE:
                raise ImportError(f"chroma未安装")
            
            # 从 kwargs 中获取 persist_directory，如果没有则使用默认值
            # chroma可以自动持久化
            persist_directory = kwargs.pop("persist_directory", None) or settings.vector_store_path
            collection_name = kwargs.pop("collection_name", "default")
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
                **kwargs
            )
        if vector_store_type=="inmemory":
            return InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的向量库类型：{vector_store_type}")
    except Exception as e:
        logger.error(f"创建向量库失败: {e}")
        raise

def save_vector_store(
        vector_store:VectorStore,
        save_path:str,
        embeddings:Embeddings,
):
    """
    向量库持久化

    inmemory不支持持久化，chroma创建时自动持久化，即无需此方法

    Args:
        vector_store (VectorStore): _description_
        save_path (str): _description_
        embeddings (Embeddings): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
    """    
    path = Path(save_path)
    
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始{vector_store}库本地持久化")
    
    try:
        if isinstance(vector_store,Chroma):
            # vector_store.persist() 新版自动持久化，该方法已废弃
            logger.info({"chroma保存成功"})
            logger.info(f"✅ Chroma 向量库已在创建时自动持久化")
        elif isinstance(vector_store,InMemoryVectorStore):
            raise ValueError("inmemory不支持持久化")
        else:
            logger.error(f"不支持的库类型{type(vector_store)}")
            raise ValueError(f"不支持的库类型{type(vector_store)}")
    except Exception as e:
        logger.error(f"保存向量库失败: {e}")
        raise   

def load_vector_store(
    load_path: str,
    embeddings: Embeddings,
    store_type: Optional[VectorStoreType] = None,
    **kwargs,
)->VectorStore:
    """
    从磁盘加载向量库
    
    -支持chroma

    Args:
        load_path (str): _description_
        embeddings (Embeddings): _description_
        store_type (Optional[VectorStoreType], optional):向量库类型. Defaults to None.
                                                        -chroma

    Raises:
        FileNotFoundError: _description_
        ImportError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        VectorStore: _description_
    """    
    load_path_obj=Path(load_path)
    if not load_path_obj.exists:
        raise FileNotFoundError(f"路径不存在：{load_path}")

    logger.info(f"正在从从路径{load_path}加载向量库{store_type}")

    try:
        if store_type=="chroma":
            if not Chroma_AVAILABLE:
                raise ImportError(f"chroma未安装")
            collection_name=kwargs.pop("collection_name", "default")
            vector_store = Chroma(
                    persist_directory=load_path,
                    embedding_function=embeddings,
                    collection_name=collection_name,
                    **kwargs,
                )
            logger.info("✅ chroma 向量库加载成功")
        elif store_type=="inmemory":
            raise ValueError("InMemoryVectorStore 不支持从磁盘加载")
        else:
            raise ValueError(f"不支持的向量库类型{store_type}")
        
        return vector_store
    except Exception as e:
        logger.error("加载向量库失败")
        raise

def add_documents_to_vector_store(
    vector_store: VectorStore,
    documents: List[Document],
)->None:
    """
    向量库添加文档

    Args:
        vector_store (VectorStore): _description_
        documents (List[Document]): _description_
    """
    if not documents:
        logger.warning("文档列表为空！")
        return
    logger.info(f"➕ 向向量库添加文档: {len(documents)} 个")
    
    try:
        vector_store.add_documents(documents)
        logger.info("文档添加成功")
        
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise

def search_vector_store(
        vector_store:VectorStore,
        query:str,
        k: int = 4,
        score_threshold: Optional[float] = None,
)->List[tuple[Document,float]]:
    """
    在向量库中搜索相似文档

    Args:
        vector_store (VectorStore): 向量库实例
        query (str): 查询文本
        k (int, optional): 返回的文档数量
        score_threshold (Optional[float], optional): 相似度阈值（可选）

    Returns:
        List[tuple[Document,float]]: _description_
    """
    logger.info(f"搜索向量库: query={query[:50]}...,k={k}")

    try:
        # 带得分搜索，可评估检索质量
        results=vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

        if score_threshold is not None:
            results=[
                (doc,score) for doc,score in results
                if score >= score_threshold
            ]

        logger.info(f"找到 {len(results)} 个相关文档")  # k与len(results)可能不同，因为数据库文档数量可能不足，或阈值过滤导致减少
        
        return results
    except Exception as e:
        logger.error(f"搜索失败：{e}")
        raise

def delete_vector_store(
        path:str
):
    """
    删除向量库文件

    Args:
        path (str): _description_
    """
    vector_store_path=Path(path)
    if not vector_store_path.exists:
        logger.warning("文件不存在")
        return
    
    logger.info(f"删除向量库：{path}")

    try:
        if vector_store_path.is_dir():
            import shutil
            shutil.rmtree(vector_store_path)
        else:
            vector_store_path.unlink()
        
        logger.info(f"向量库删除成功：{vector_store_path}")
    
    except Exception as e:
        logger.error(f"向量库删除失败：{e}")
        raise