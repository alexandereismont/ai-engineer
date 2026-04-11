from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    # Embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_prefix: str = "Represent this sentence for searching relevant passages: "

    # Chunking
    chunking_strategy: str = "recursive"  # fixed, recursive, sentence, semantic
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    initial_retrieval_k: int = 20
    final_top_k: int = 5
    use_bm25: bool = True
    use_reranking: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "pension_docs"

    # LLM
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0

    # Evaluation
    eval_dataset_path: str = "../evaluation/golden_dataset.json"
