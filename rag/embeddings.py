'''
向量化

统一的embedding接口
-支持local（）、huggingface（）

'''

from typing import Optional,Any
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from config import get_logger,settings

logger = get_logger(__name__)

def get_embddings(
        backend:str="ollama",
        model:Optional[str]=None,
        batch_size:Optional[int]=None,
        **kwargs,
)->Embeddings:
    """
    获取向量模型实例

    Args:
        backend (str, optional): 后端类型
                                    -ollma本地模型
                                    -huggingface云端模型
        model (Optional[str], optional): embddings模型
        batch_size (Optional[int], optional): 批处理大小

    Raises:
        ValueError: _description_

    Returns:
        Embeddings: _description_
    """    
    # todo 修改向量化默认配置项
    model=model or settings.embedding_model
    batch_size = batch_size or batch_size

    logger.info(
        f"向量化模型为{model}"
        f"batch_size={batch_size}"
    )
    try:
        if backend=="ollama":
            return OllamaEmbeddings(
                model=model,
                **kwargs,
            )
        # todo api访问向量化
        else:
            logger.error(f"不支持的后端{backend}")
            raise ValueError(f"不支持的后端{backend}")
    except Exception as e:
        logger.error(f"创建项链模型 {model} 失败")
        raise