'''
文本分块

-

'''

from typing import List, Optional, Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)
from config import settings,get_logger

logger=get_logger(__name__)

#分块器类型
SpliterType=Literal["recursive","character","markdown","token"]

def get_text_spliter (
        spliter_type:str="recursive",
        chunk_size:Optional[int]=None,
        chunk_overlap:Optional[int]=None,
        **kwargs,
):
    """_summary_

    Args:
        spliter_type (str, optional): 分块器类型recursive, character, markdown, token
        chunk_size (Optional[int], optional): 分块大小
        chunk_overlap (Optional[int], optional): 分块重叠大小
    Raises:
        ValueError: _description_

    Returns:
        _type_: 文件分块器
    """    
    #分块大小及分块重叠大小为0及none时取默认值
    chunk_size=chunk_size or settings.chunk_size
    chunk_overlap=chunk_overlap or settings.chunk_overlap

    logger.info(f"获取分块器：{spliter_type}")

    #获取不同类型的分块器
    if spliter_type=="recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            lenth_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    elif spliter_type=="character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    elif spliter_type=="markdown":
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    elif spliter_type=="token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    else:
        logger.warning(f"不支持的文件分块类型{spliter_type}")
        logger.warning(f"支持的类型: recursive, character, markdown, token")
        raise ValueError(f"不支持的文件分块类型{spliter_type}"
                         f"支持的类型: recursive, character, markdown, token")
    
def split_document(
        documents:List[Document],
        spliter_type:str="recursive",
        chunk_size:Optional[int]=None,
        chunk_overlap:Optional[int]=None,
        **kwargs,
)->List[Document]:
    """
    document文档列表分块

    Args:
        documents (List[Document]): 文档
        spliter_type (str, optional): 分块器类型
        chunk_size (Optional[int], optional): 分块大小
        chunk_overlap (Optional[int], optional): 分块重叠大小

    Returns:
        List[Document]: 分块文档列表
    """    
    if not documents:
        logger.warning("没有文档需要分块")
        return []
    
    logger.info(f"开始分块：{len(documents)}个文档")

    spliter=get_text_spliter(
        spliter_type=spliter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    try:
        chunks=spliter.split_documents(documents)

        # 统计信息
        total_chars=sum(len(item.page_content) for item in chunks)
        avg_chunk_chars=total_chars/len(chunks) if chunks else 0

        logger.info(f"总字符数{total_chars}")
        logger.info(f"平均块大小{avg_chunk_chars}")

        return chunks
    except Exception as e:
        logger.error(f"❌ 分块失败: {e}")
        raise

def split_text(
        text:str,
        spliter_type:str="recursive",
        chunk_size:Optional[int]=None,
        chunk_overlap:Optional[int]=None,
        metadata:Optional[dict]=None,
        **kwargs,
)->List[Document]:
    """
    对文本进行分块

    Args:
        text (str): 文本
        spliter_type (str, optional): 分块器类型
        chunk_size (Optional[int], optional): 分块大小
        chunk_overlap (Optional[int], optional): 分块重叠大小
        metadata (Optional[dict], optional): 元数据
    Returns:
        List[Document]: 分块列表
    """    
    chunk_size=chunk_size or settings.chunk_size
    chunk_overlap=chunk_overlap or settings.chunk_overlap

    if not text:
        logger.warning("文本内容为空")
        return []
    
    logger.info(f"开始分块")

    # 获取分块器
    spliter=get_text_spliter(
        spliter_type=spliter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )

    # 分块
    try:
        metadatas=[metadata] if metadata else None
        chunks=spliter.create_documents(texts=[text],metadatas=metadatas)

        logger.info(f"分块完成，共{len(chunks)}个文本块")

        return chunks
    
    except Exception as e:
        logger.error(f"分块失败")
        raise

def get_optimal_chunk_size(
        document_type:str="general",
)->tuple[int,int]:
    """
    获取推荐分块大小

    Args:
        document_type (str, optional): 文档类型

    Returns:
        tuple[int,int]: (分块大小,分块重叠大小)
    """    
    #推荐块大小
    recommendations={
        "general": (1000, 200),      # 通用文档
        "code": (1500, 300),          # 代码需要更大的上下文
        "markdown": (800, 150),       # Markdown 通常结构清晰
        "academic": (1200, 250),      # 学术论文需要保持上下文
        "chat": (500, 50),            # 对话记录可以更小
    }

    if document_type not in recommendations:
        logger.info(f"{document_type}类型未配置分块大小")
        logger.info("已设置推荐分块大小的类型：genernal、code、markdown、academic、chat")
        return recommendations["general"]
    chunk_size,chunk_overlap=recommendations[document_type]

    logger.info(
        f"推荐的分块参数 ({document_type}): "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
    )
    
    return chunk_size, chunk_overlap

def analyze_chunks(chunks: List[Document]) -> dict:
    """
    分析分块结果的统计信息
    
    Args:
        chunks: 分块后的文档列表
        
    Returns:
        包含统计信息的字典
        
    Example:
        >>> chunks = split_documents(documents)
        >>> stats = analyze_chunks(chunks)
        >>> print(f"平均块大小: {stats['avg_chunk_size']}")
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
        }
    
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    total_chars = sum(chunk_sizes)
    
    stats = {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_size": total_chars / len(chunks),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
    }
    
    logger.info("📊 分块统计:")
    logger.info(f"   总块数: {stats['total_chunks']}")
    logger.info(f"   总字符数: {stats['total_chars']}")
    logger.info(f"   平均大小: {stats['avg_chunk_size']:.0f} 字符")
    logger.info(f"   最小块: {stats['min_chunk_size']} 字符")
    logger.info(f"   最大块: {stats['max_chunk_size']} 字符")
    
    return stats
