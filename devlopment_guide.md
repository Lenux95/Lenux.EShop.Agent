# EShop.Agent

## rag

1.文档加载 loaders.py
2.文本分块 splitters.py
3.向量化embeddings.py
4.向量存储和索引管理 vector_stores.py/index_manager.py
5.检索器 retrievers.py
6.实现RAG rag_agent.py

EShop.Agent
├─ data
│  ├─ 2.txt
│  └─ documents
│     └─ 1.txt
├─ rag                  RAG 模块
│  ├─ embeddings.py     向量化
│  ├─ loaders.py        文档加载
│  ├─ splitters.py      文本分块
│  └─ vector_stores.py  向量存储
├─ readme.md
└─ requirements.txt
