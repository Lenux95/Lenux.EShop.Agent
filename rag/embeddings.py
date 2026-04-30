'''
向量化

统一的embedding接口
-支持local（ollama）

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
        Embeddings: 向量化模型实例
    """    
    # todo 修改向量化默认配置项
    model=model or settings.embedding_model
    batch_size = batch_size or settings.embedding_batch_size

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

def get_embedding_dimension(model:Optional[str]=None)->int:
    """
    获取对应模型的嵌入维度

    Args:
        model (Optional[str], optional): 模型

    Returns:
        int: 嵌入维度
    """    

    model=model or settings.embedding_model

    dimensions={
        "qwen3-embedding:0.6b":512,  #128-768
        # todo 补充推荐模型维度
    }

    if model in dimensions:
        return dimensions[model]
    else:
        return 768
    
