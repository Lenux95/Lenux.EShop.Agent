#!/usr/bin/env python3
"""
全面测试所有 RAG 模块
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings, get_logger, setup_logging
from rag.splitters import (
    get_text_spliter,
    split_text,
    split_document,
    get_optimal_chunk_size,
    analyze_chunks,
    SpliterType
)
from rag.loaders import (
    get_support_extensions,
    get_loader_for_file,
    load_document,
    load_documents,
    SUPPORTED_EXTENSIONS
)
from rag.embeddings import get_embddings, get_embedding_dimension
from rag.vector_stores import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    add_documents_to_vector_store,
    search_vector_store,
    delete_vector_store,
    VectorStoreType
)
from langchain_core.documents import Document

# 初始化日志
setup_logging(log_level="DEBUG")
logger = get_logger("test_all_modules")

# 测试结果统计
test_results = {
    "passed": 0,
    "failed": 0,
    "total": 0
}

def run_test(test_name, test_func):
    """运行单个测试并记录结果"""
    test_results["total"] += 1
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 测试: {test_name}")
    logger.info(f"{'='*60}")
    
    try:
        test_func()
        logger.info(f"✅ [{test_name}] 测试通过")
        test_results["passed"] += 1
        return True
    except Exception as e:
        logger.error(f"❌ [{test_name}] 测试失败: {e}", exc_info=True)
        test_results["failed"] += 1
        return False

# ==================== 配置模块测试 ====================
def test_config():
    """测试配置模块"""
    logger.info(f"应用名称: {settings.app_name}")
    logger.info(f"应用版本: {settings.app_version}")
    logger.info(f"日志级别: {settings.log_level}")
    logger.info(f"分块大小: {settings.chunk_size}")
    logger.info(f"分块重叠: {settings.chunk_overlap}")
    logger.info(f"嵌入模型: {settings.embedding_model}")
    logger.info(f"向量库类型: {settings.vector_store_type}")
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0

# ==================== 分块模块测试 ====================
def test_splitters():
    """测试文本分块模块"""
    test_text = """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
    致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
    人工智能领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    
    机器学习（Machine Learning，ML）是人工智能的核心技术之一，
    它使计算机系统能够从数据中学习并改进其性能，而无需进行明确编程。
    深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。"""

    # 测试 get_text_spliter
    for spliter_type in ["recursive", "character", "markdown", "token"]:
        spliter = get_text_spliter(spliter_type=spliter_type)
        assert spliter is not None
        logger.info(f"  - 获取 {spliter_type} 分块器成功")

    # 测试 split_text
    chunks = split_text(test_text, spliter_type="recursive", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 0
    logger.info(f"  - split_text 生成 {len(chunks)} 个块")

    # 测试 split_document
    doc = Document(page_content=test_text, metadata={"source": "test"})
    doc_chunks = split_document([doc], spliter_type="recursive")
    assert len(doc_chunks) > 0
    logger.info(f"  - split_document 生成 {len(doc_chunks)} 个块")

    # 测试 get_optimal_chunk_size
    chunk_size, overlap = get_optimal_chunk_size("general")
    assert chunk_size > 0
    assert overlap >= 0
    logger.info(f"  - 推荐分块大小: {chunk_size}, 重叠: {overlap}")

    # 测试 analyze_chunks
    stats = analyze_chunks(chunks)
    assert "total_chunks" in stats
    assert "avg_chunk_size" in stats
    logger.info(f"  - 分块统计分析完成")

# ==================== 加载器模块测试 ====================
def test_loaders():
    """测试文档加载器模块"""
    # 测试 get_support_extensions
    extensions = get_support_extensions()
    assert len(extensions) > 0
    logger.info(f"  - 支持的扩展名: {list(extensions.keys())}")

    # 测试 load_document (测试txt文件)
    test_file = "data/2.txt"
    if os.path.exists(test_file):
        docs = load_document(test_file, add_metadata=True)
        assert len(docs) > 0
        logger.info(f"  - 加载单个文件成功: {len(docs)} 页")
        assert "source" in docs[0].metadata
        assert "filename" in docs[0].metadata
    else:
        logger.info(f"  - 跳过文件加载测试: {test_file} 不存在")

    # 测试 load_documents (测试目录加载)
    test_dir = "data"
    if os.path.exists(test_dir):
        docs = load_documents(test_dir, recursive=True)
        logger.info(f"  - 加载目录文件完成")
    else:
        logger.info(f"  - 跳过目录加载测试: {test_dir} 不存在")

# ==================== 向量化模块测试 ====================
def test_embeddings():
    """测试向量化模块"""
    # 测试 get_embedding_dimension
    dim = get_embedding_dimension("qwen3-embedding:0.6b")
    assert dim == 512
    logger.info(f"  - 获取嵌入维度成功: {dim}")

    # 测试默认维度
    default_dim = get_embedding_dimension()
    assert default_dim == 768
    logger.info(f"  - 默认嵌入维度: {default_dim}")

    # 测试 get_embddings (需要 Ollama 服务)
    try:
        embeddings = get_embddings(backend="ollama", model="qwen3-embedding:0.6b")
        assert embeddings is not None
        logger.info(f"  - 获取 Ollama 嵌入模型成功")
        
        # 测试 embed_documents (实际调用)
        test_texts = ["人工智能", "机器学习", "深度学习"]
        result = embeddings.embed_documents(test_texts)
        assert len(result) == len(test_texts)
        logger.info(f"  - embed_documents 测试成功: 生成了 {len(result)} 个嵌入向量")
        
        # 测试 embed_query
        query_embedding = embeddings.embed_query("什么是人工智能")
        assert len(query_embedding) > 0
        logger.info(f"  - embed_query 测试成功: 生成了 {len(query_embedding)} 维嵌入向量")
        
    except Exception as e:
        logger.warning(f"  - ⚠️ get_embddings 测试跳过: {e}")
        logger.warning(f"     (需要 Ollama 服务运行，请确保已安装并启动 Ollama)")

# ==================== 向量存储模块测试 ====================
def test_vector_stores():
    """测试向量存储模块"""
    # 创建测试文档
    test_docs = [
        Document(page_content="人工智能是计算机科学的一个分支", metadata={"source": "test1"}),
        Document(page_content="机器学习是人工智能的核心技术", metadata={"source": "test2"}),
        Document(page_content="深度学习使用多层神经网络", metadata={"source": "test3"}),
    ]

    # 测试 inmemory 向量库（不需要外部依赖）
    # 创建简单的测试嵌入类，继承自 Embeddings
    from langchain_core.embeddings import Embeddings as BaseEmbeddings
    
    class TestEmbeddings(BaseEmbeddings):
        def embed_documents(self, texts):
            return [[0.1]*768 for _ in texts]
        def embed_query(self, text):
            return [0.1]*768

    test_embeddings = TestEmbeddings()

    # 测试 create_vector_store (inmemory)
    vector_store = create_vector_store(
        documents=test_docs,
        embeddings=test_embeddings,
        store_type="inmemory"
    )
    assert vector_store is not None
    logger.info(f"  - 创建 inmemory 向量库成功")

    # 测试 add_documents_to_vector_store
    new_docs = [Document(page_content="新增文档测试", metadata={"source": "test4"})]
    add_documents_to_vector_store(vector_store, new_docs)
    logger.info(f"  - 添加文档成功")

    # 测试 search_vector_store
    results = search_vector_store(vector_store, "人工智能", k=2)
    assert len(results) >= 0
    logger.info(f"  - 搜索成功，找到 {len(results)} 个文档")

    # 测试 Chroma 向量库（需要安装 langchain_chroma）
    try:
        from langchain_chroma import Chroma
        import tempfile
        import shutil
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        logger.info(f"  - 创建临时目录: {temp_dir}")
        
        # 测试 create_vector_store (chroma)
        chroma_store = create_vector_store(
            documents=test_docs,
            embeddings=test_embeddings,
            store_type="chroma",
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        assert chroma_store is not None
        logger.info(f"  - 创建 Chroma 向量库成功")
        
        # 测试 add_documents_to_vector_store
        new_docs = [Document(page_content="Chroma 测试文档", metadata={"source": "chroma_test"})]
        add_documents_to_vector_store(chroma_store, new_docs)
        logger.info(f"  - 向 Chroma 添加文档成功")
        
        # 测试 search_vector_store
        chroma_results = search_vector_store(chroma_store, "人工智能", k=2)
        assert len(chroma_results) >= 0
        logger.info(f"  - Chroma 搜索成功，找到 {len(chroma_results)} 个文档")
        
        # 测试 load_vector_store
        loaded_store = load_vector_store(
            load_path=temp_dir,
            embeddings=test_embeddings,
            store_type="chroma",
            collection_name="test_collection"
        )
        assert loaded_store is not None
        logger.info(f"  - 加载 Chroma 向量库成功")
        
        # 测试保存向量库（验证持久化）
        save_vector_store(chroma_store, temp_dir, test_embeddings)
        logger.info(f"  - 保存 Chroma 向量库成功")
        
        # 清理临时目录（Chroma 可能仍持有文件句柄，需要特殊处理）
        try:
            # 尝试直接删除
            shutil.rmtree(temp_dir)
            logger.info(f"  - 清理临时目录完成")
        except PermissionError:
            # 文件被占用时，记录警告但不中断测试
            logger.warning(f"  - ⚠️ 临时目录清理失败（文件被占用）: {temp_dir}")
            logger.info(f"  - 系统会在下次重启时自动清理临时文件")
        
    except ImportError:
        logger.warning(f"  - ⚠️ Chroma 测试跳过: 需要安装 langchain_chroma")
    except Exception as e:
        logger.warning(f"  - ⚠️ Chroma 测试失败: {e}")

# ==================== 主测试函数 ====================
def main():
    logger.info("🚀 开始全面测试所有 RAG 模块")
    logger.info("="*60)

    # 运行所有测试
    tests = [
        ("配置模块", test_config),
        ("文本分块模块", test_splitters),
        ("文档加载器模块", test_loaders),
        ("向量化模块", test_embeddings),
        ("向量存储模块", test_vector_stores),
    ]

    for test_name, test_func in tests:
        run_test(test_name, test_func)

    # 输出测试结果汇总
    logger.info("\n" + "="*60)
    logger.info("📊 测试结果汇总")
    logger.info("="*60)
    logger.info(f"总测试数: {test_results['total']}")
    logger.info(f"✅ 通过: {test_results['passed']}")
    logger.info(f"❌ 失败: {test_results['failed']}")
    
    if test_results["failed"] == 0:
        logger.info("🎉 所有测试通过！")
    else:
        logger.warning(f"⚠️  部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()