#!/usr/bin/env python3
"""
测试 rag_agent.py 中的方法（不修改原文件）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from config import setup_logging, get_logger
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as BaseEmbeddings

# 初始化日志
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# ==================== 1. 创建测试数据 ====================
logger.info("=" * 60)
logger.info("📋 步骤1：创建测试文档")
logger.info("=" * 60)

test_docs = [
    Document(
        page_content="元旦节（New Year's Day）是公历新年的第一天，即每年的1月1日。",
        metadata={"source": "元旦节日介绍"}
    ),
    Document(
        page_content="春节是中国最重要的传统节日之一，又称农历新年。",
        metadata={"source": "春节节日介绍"}
    ),
]

logger.info(f"✅ 创建了 {len(test_docs)} 个测试文档")

# ==================== 2. 创建测试嵌入 ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤2：创建测试嵌入")
logger.info("=" * 60)

class DummyEmbeddings(BaseEmbeddings):
    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 768

test_embeddings = DummyEmbeddings()
logger.info("✅ 创建了测试嵌入模型")

# ==================== 3. 创建向量存储 ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤3：创建向量存储")
logger.info("=" * 60)

from rag.vector_stores import create_vector_store

vector_store = create_vector_store(
    documents=test_docs,
    embeddings=test_embeddings,
    store_type="inmemory"
)
logger.info("✅ 创建了 inmemory 向量存储")

# ==================== 4. 创建检索器 ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤4：创建检索器")
logger.info("=" * 60)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
logger.info("✅ 创建了检索器")

# ==================== 5. 测试 DEFAULT_RAG_SYSTEM_PROMPT ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤5：测试 DEFAULT_RAG_SYSTEM_PROMPT")
logger.info("=" * 60)

from rag.rag_agent import DEFAULT_RAG_SYSTEM_PROMPT

logger.info(f"✅ 系统提示词长度: {len(DEFAULT_RAG_SYSTEM_PROMPT)}")
logger.info(f"✅ 系统提示词预览: {DEFAULT_RAG_SYSTEM_PROMPT[:50]}...")

# ==================== 6. 测试 create_rag_agent ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤6：测试 create_rag_agent")
logger.info("=" * 60)

from rag.rag_agent import create_rag_agent

try:
    agent = create_rag_agent(
        retriever=retriever,
        model="qwen-plus",
        tool_name="knowledge_base",
    )
    logger.info("✅ RAG Agent 创建成功")
except Exception as e:
    logger.error(f"❌ RAG Agent 创建失败: {e}")
    import traceback
    logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
    sys.exit(1)

# ==================== 7. 测试 query_rag_agent ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤7：测试 query_rag_agent")
logger.info("=" * 60)

from rag.rag_agent import query_rag_agent

try:
    result = query_rag_agent(
        agent=agent,
        query="元旦节是哪一天？",
        return_sources=True
    )
    logger.info("✅ query_rag_agent 调用成功")
    
    if isinstance(result, dict) and "answer" in result:
        logger.info(f"🤖 回答: {result}")
    else:
        logger.info(f"📤 返回结果: {result}")
        
except Exception as e:
    logger.error(f"❌ query_rag_agent 调用失败: {e}")
    import traceback
    logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")

# ==================== 8. 测试多个查询 ====================
logger.info("\n" + "=" * 60)
logger.info("📋 步骤8：测试多个查询")
logger.info("=" * 60)

test_queries = [
    "元旦节是哪一天？",
    "春节是什么时候？",
]

for query in test_queries:
    try:
        result = query_rag_agent(agent=agent, query=query)
        logger.info(f"\n❓ 问题: {query}")
        if isinstance(result, dict) and "answer" in result:
            logger.info(f"🤖 回答: {result}")
        else:
            logger.info(f"📤 结果: {result}")
    except Exception as e:
        logger.error(f"❌ 查询失败 [{query}]: {e}")

# ==================== 完成 ====================
logger.info("\n" + "=" * 60)
logger.info("🎉 测试完成")
logger.info("=" * 60)