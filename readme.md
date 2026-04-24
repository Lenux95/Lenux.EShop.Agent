# Python & AI 开发学习笔记

## 学习阶段：RAG 系统开发 - 阶段二（进行中）

### 已完成模块：loaders、splitters、config

### 日期：2026-04-23

---

## 一、核心问题记录与分析

### 1.1 类型标注问题

**问题场景**：在 VS Code 中使用 Pylance 进行类型检查时，代码出现黄色警告，提示类型不一致。

**具体代码**：

```python
def get_loader_for_file(file_path: str) -> Optional[Any]:
    file_path = Path(file_path)  # 这里类型从 str 变成了 Path
```

**深度分析**：

- 这是 Python 类型标注中最容易混淆的地方之一
- 参数标注的是**输入类型**，但函数内部可能进行类型转换
- Pylance 检测到变量类型在赋值过程中发生变化

**解决方案**：

```python
# 方案1：使用联合类型（最推荐）
from typing import Union
def get_loader_for_file(file_path: Union[str, Path]) -> Optional[Any]:
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path

# 方案2：使用变量副本
def get_loader_for_file(file_path: str) -> Optional[Any]:
    path_obj = Path(file_path)  # 使用不同的变量名
```

---

### 1.2 模块导入问题

**问题场景**：尝试导入 `config` 模块时，出现 `ImportError: attempted relative import with no known parent package`

**原因分析**：

- `config` 目录缺少 `__init__.py` 文件
- Python 将没有 `__init__.py` 的目录视为普通文件夹，而非包
- 相对导入（`from .settings import settings`）需要包结构支撑

**解决方案**：

```python
# config/__init__.py
from .settings import settings
from .logging import setup_logging, get_logger

__all__ = ["settings", "setup_logging", "get_logger"]
```

**`__init__.py` 的核心作用**：

1. **标识包目录**：告诉 Python 这是一个包
2. **控制导出**：`__all__` 列表定义 `from package import *` 时可导入的内容
3. **初始化代码**：包首次导入时自动执行的逻辑

---

### 1.3 元数据处理问题

**问题场景**：`Document` 对象的 `metadata` 可能为 `None`，导致后续操作报错。

**具体代码**：

```python
doc.metadata.update({...})  # 如果 doc.metadata 是 None，会报错
```

**解决方案**：

```python
# 防御性编程
if doc.metadata is None:
    doc.metadata = {}
doc.metadata.update({...})

# 或者使用 or 操作符
doc.metadata = doc.metadata or {}
doc.metadata.update({...})
```

**核心知识点**：

- Python 的假值（falsy）概念：`None`, `0`, `''`, `[]`, `{}`, `False` 都是假值
- `or` 操作符的短路求值特性
- 防御性编程原则

---

### 1.4 异常处理策略问题

**问题场景**：在 `load_directory` 和 `split_documents` 中对空输入的处理不一致。

**代码对比**：

```python
# load_directory - 使用 raise（目录不存在是致命错误）
if not directory_path.exists():
    raise FileNotFoundError(f"目录不存在: {directory_path}")

# split_documents - 使用 return（空列表是合理的返回值）
if not documents:
    logger.warning("文档列表为空，无需分块")
    return []
```

**选择原则**：

| 场景         | 处理方式           | 原因                                   |
| ------------ | ------------------ | -------------------------------------- |
| 目录不存在   | `raise`            | 函数无法完成核心任务，属于致命错误     |
| 文档列表为空 | `return []`        | 空列表是合理的返回值，不影响调用方逻辑 |
| 参数验证失败 | `raise ValueError` | 无效参数导致无法继续执行               |

**`raise` 不带参数的含义**：

```python
except Exception as e:
    logger.error(f"错误: {e}")
    raise  # 重新抛出原始异常，保留完整堆栈信息
```

---

### 1.5 日志配置问题

**问题场景**：代码中有 `logger = get_logger(__name__)`，但日志没有正常输出。

**正确流程**：

```python
# config/__init__.py
from .logging import setup_logging, get_logger

setup_logging()  # 初始化日志系统

# 其他模块
from config import get_logger
logger = get_logger(__name__)  # 获取已配置的 logger
```

**`__name__` 的作用**：

- 在直接运行的模块中，`__name__` == `"__main__"`
- 在被导入的模块中，`__name__` == 模块的完整路径（如 `"rag.loaders"`）
- 用于在日志中标识日志来源

---

### 1.6 生成器表达式 vs 列表推导式

**问题场景**：计算所有分块总字符数时，选择哪种方式更优？

**代码对比**：

```python
# 生成器表达式（内存高效）- 惰性求值
total_chars = sum(len(chunk.page_content) for chunk in chunks)

# 列表推导式（内存占用大）
total_chars = sum([len(chunk.page_content) for chunk in chunks])
```

**核心区别**：

- 生成器表达式：逐个计算，不需要一次性存储所有结果
- 列表推导式：先创建完整列表，再进行求和
- 对于大数据集，生成器表达式内存效率显著更高
- 注意：生成器只能遍历一次

---

## 二、loaders.py 模块学习总结

### 2.1 文件功能

负责从多种文件格式加载文档，支持 PDF、TXT、Markdown、Word、CSV、Excel 等常见格式。

### 2.2 核心函数

| 函数                         | 作用                         | 返回值               |
| ---------------------------- | ---------------------------- | -------------------- |
| `get_supported_extensions()` | 获取支持的文件扩展名         | Dict[str, str]       |
| `get_loader_for_file()`      | 根据文件类型获取对应的加载器 | Optional[BaseLoader] |
| `load_document()`            | 加载单个文档                 | List[Document]       |
| `load_directory()`           | 批量加载目录下的文档         | List[Document]       |

### 2.3 关键设计模式

**工厂模式应用**：

```python
def get_loader_for_file(file_path: str) -> Optional[BaseLoader]:
    extension = Path(file_path).suffix.lower()
    loader_class = LOADER_MAPPING.get(extension)
    if loader_class:
        return loader_class()
    return None
```

**元数据增强**：

```python
if add_metadata:
    for doc in documents:
        doc.metadata = doc.metadata or {}
        doc.metadata.update({
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": SUPPORTED_EXTENSIONS[file_path.suffix.lower()],
        })
```

---

## 三、splitters.py 模块学习总结

### 3.1 文件功能

将长文档分割成较小的文本块，便于向量化存储和检索。

### 3.2 核心函数

| 函数                  | 作用             | 返回值         |
| --------------------- | ---------------- | -------------- |
| `get_text_splitter()` | 获取分块器实例   | TextSplitter   |
| `split_documents()`   | 分块文档列表     | List[Document] |
| `split_text()`        | 分块单个文本     | List[str]      |
| `analyze_chunks()`    | 分析分块结果统计 | dict           |

### 3.3 类型标注学习

**Literal 用于限制取值范围**：

```python
SplitterType = Literal["recursive", "character", "markdown", "token"]

def get_text_splitter(splitter_type: SplitterType = "recursive"):
    # IDE 会提供自动补全，只允许选择这4个值
```

**Optional 用于表示可能为 None**：

```python
def get_text_splitter(
    splitter_type: SplitterType = "recursive",
    chunk_size: Optional[int] = None,  # 可选参数
    chunk_overlap: Optional[int] = None,
) -> Any:
```

### 3.4 默认值处理

**使用 `or` 操作符提供默认值**：

```python
chunk_size = chunk_size or settings.chunk_size
```

**原理**：当 `chunk_size` 为 `None`（假值）时，使用 `settings.chunk_size` 作为默认值。

### 3.5 关键字参数解包

**`**kwargs` 将字典解包为关键字参数**：

```python
def get_text_splitter(..., **kwargs) -> Any:
    ...
    return splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs  # 额外的参数传递给分块器
    )
```

---

## 四、config 模块学习总结

### 4.1 文件结构

```
config/
├── __init__.py      # 包初始化，导出核心功能
├── settings.py      # 应用配置（Pydantic）
└── logging.py       # 日志配置
```

### 4.2 settings.py 核心知识点

**Pydantic Settings 用于配置管理**：

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

**特点**：

- 自动从环境变量或 `.env` 文件读取配置
- 提供类型验证和默认值
- 敏感信息（如 API Key）通过环境变量注入

### 4.3 logging.py 核心知识点

**日志配置流程**：

```python
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

**日志级别**：

- `DEBUG`：详细调试信息
- `INFO`：一般信息
- `WARNING`：警告信息
- `ERROR`：错误信息
- `CRITICAL`：严重错误

### 4.4 `__all__` 的作用

```python
# config/__init__.py
__all__ = ["settings", "setup_logging", "get_logger"]

# 控制 from config import * 时的导出内容
```

---

## 五、Python 开发关键概念

### 5.1 包和模块

- **模块（Module）**：一个 `.py` 文件
- **包（Package）**：包含 `__init__.py` 的目录
- `__init__.py`：标识目录为 Python 包，可选包含初始化代码

### 5.2 导入机制

```python
# 绝对导入
from config.settings import settings

# 相对导入（需要在包内使用）
from .settings import settings
from ..parent_module import something
```

### 5.3 常见类型提示

| 类型提示            | 含义                           |
| ------------------- | ------------------------------ |
| `int`               | 整数                           |
| `str`               | 字符串                         |
| `bool`              | 布尔值                         |
| `List[int]`         | 整数列表                       |
| `Dict[str, Any]`    | 键为字符串，值为任意类型的字典 |
| `Optional[int]`     | 可以是整数或 None              |
| `Union[int, str]`   | 可以是整数或字符串             |
| `Literal["a", "b"]` | 只能是 "a" 或 "b"              |
| `Any`               | 任意类型                       |

### 5.4 元组解包

```python
# 元组定义
recommendations = {
    "general": (1000, 200),  # (chunk_size, overlap)
}

# 元组解包
chunk_size, overlap = recommendations["general"]
# chunk_size = 1000, overlap = 200
```

### 5.5 列表操作

```python
# extend：合并列表（在原列表末尾添加另一个列表的所有元素）
all_files = []
files = [file1, file2]
all_files.extend(files)  # all_files = [file1, file2]

# append：添加单个元素
all_files.append(file3)  # all_files = [file1, file2, file3]
```

### 5.6 enumerate 枚举

```python
# 同时获取索引和值
for i, file_path in enumerate(all_files, 1):  # 从 1 开始计数
    print(f"{i}. {file_path}")
```

---

## 六、项目结构（参考）

```
backend/
├── config/
│   ├── __init__.py
│   ├── settings.py          # 配置管理
│   └── logging.py           # 日志配置
├── rag/
│   ├── __init__.py
│   ├── loaders.py           # 文档加载 ✓ 已学习
│   ├── splitters.py         # 文本分块 ✓ 已学习
│   ├── embeddings.py        # 向量化（待学习）
│   ├── vector_stores.py     # 向量存储（待学习）
│   ├── retrievers.py        # 检索器（待学习）
│   └── rag_agent.py         # RAG Agent（待学习）
├── api/
│   ├── __init__.py
│   └── http_server.py       # FastAPI 主入口
└── main.py
```

---

*本笔记将持续更新...*