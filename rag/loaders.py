
from typing import Union
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

#todo 日志

'''
1.获取文件类型映射
2.获取当前文件所需要的loader
3.加载文档
4.批量加载
'''

SUPPORTED_EXTENSIONS={
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".mdx": "markdown",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
}

def get_support_extensions()->dict[str,str]:
    """获取支持的扩展名

    Returns:
        dict[str,str]: 扩展名到类型 映射字典
    """    
    return SUPPORTED_EXTENSIONS.copy();

def get_loader_for_file(file_path: str)->Optional[Any]:
    """
    根据该文件类型，选择合适的加载器

    Args:
        file_path (str): 文件地址

    Returns:
        Optional[any]: 对应文档类型的loader
    """    
    file_obj = Path(file_path)
    extension = file_obj.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("该文件类型不支持")
    
    #根据文件类型返回合适的loader
    file_type = SUPPORTED_EXTENSIONS[extension]
    
    try:
        # 根据文件类型选择对应的加载器
        if file_type == "pdf":
            return PyPDFLoader(str(file_path))
        elif file_type == "text":
            return TextLoader(str(file_path), encoding="utf-8")
        elif file_type == "markdown":
            return UnstructuredMarkdownLoader(str(file_path))
        elif file_type == "html":
            return UnstructuredHTMLLoader(str(file_path))
        elif file_type == "json":
            # JSON 加载器需要指定 jq_schema 来提取内容
            # 默认提取所有内容
            return JSONLoader(
                file_path=str(file_path),
                jq_schema=".",  # 提取所有内容
                text_content=False,  # 保持原始格式
            )
        else:
            raise ValueError("未实现该类型文档读取")
    
    except Exception as e:
        raise ValueError("加载文档出现错误")
    
def load_document(file_path:str,add_metadata:bool) ->List[Document]:
    """
    加载单个文件
    
    根据地址加载当个文件，并根据需要为文件添加元数据

    Args:
        file_path (str): 文件地址
        add_metadata (bool): 是否添加元数据（地址，文件名，类型）

    Raises:
        FileNotFoundError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        List[Document]: 文档加载内容 document
    """
    file_obj=Path(file_path)
    if not file_obj.exists():
        raise FileNotFoundError(f"文件不存在:{file_path}")
    if not file_obj.is_file():
        raise ValueError(f"不是文件:{file_path}")
    
    #获取合适的loader
    loader=get_loader_for_file(file_path)
    if loader is None:
        raise ValueError(f"文件类型不支持[{file_obj.suffix.lower()}]")
    
    #加载文件
    try:
        document=loader.load()
        if add_metadata:
            for item in document:
                if item.metadata is None:
                    item.metadata={}
                item.metadata.update({
                    "source":file_path,
                    "file_type":SUPPORTED_EXTENSIONS[f"{file_obj.suffix.lower}"],
                    "filename":file_obj.name
                })
        return document
    except Exception as e:
        raise ValueError(f"文件加载失败:{file_path}")
    
def load_documents(
        directory_path:str,
        recursive: bool,
        exclude_patterns: Optional[List[str]] = None,
        max_files:Optional[int]=None,
        )->List[Document]:
    """
    批量加载目录下的文件

    Args:
        directory_path (str): 目录地址
        recursive (bool): 是否递归加载文件
        exclude_patterns (Optional[List[str]], optional):排除模式
        max_files (Optional[int], optional): 最大文件数量

    Raises:
        FileNotFoundError: _description_
        ValueError: _description_

    Returns:
        List[Document]: 文档加载内容 document
    """
    directory_obj=Path(directory_path)
    if not directory_obj.exists:
        raise FileNotFoundError("目录不存在")
    if not directory_obj.is_dir:
        raise ValueError("该地址不是目录")
    
    #支持类型的所有文件对象
    all_files=[]
    for item in SUPPORTED_EXTENSIONS.keys():
        partten=f"**/.{item}" if recursive else "*/.{item}"
        files=list(directory_obj.glob(partten))
        all_files.append(files)

    #排除文件
    if exclude_patterns:
        filtered_files = []
        for item in all_files:
            is_include=True
            for pattern in exclude_patterns:
                if item.match(partten):
                    is_include=False
                    break
            if is_include:
                filtered_files.append(item)
        all_files=filtered_files

    #限制文件数量
    if max_files is not None and len(all_files)>max_files:
        #logger.warning(f"⚠️  文件数量 ({len(all_files)}) 超过限制 ({max_files})，只加载前 {max_files} 个")
        all_files = all_files[:max_files]

    #加载文件
    all_docs=[]
    success_count = 0  # 成功加载的文件数
    error_count = 0    # 加载失败的文件数
    for doc in all_files:
        try:
            docs=load_document(str(doc),True)
            all_docs.append(docs)
            success_count += 1
        except Exception as e:
            error_count +=1
            continue

    return all_docs

# todo多文件路径加载