import pandas as pd
from rag_local_lab.chuncker import LocalChunker
from rag_local_lab.embedder import LocalEmbedder
from rag_local_lab.db import PostgresVectorStore

def ingest_pipeline():
    splits = {'train': 'data/train-00000-of-00001-9df3a936e1f63191.parquet', 'test': 'data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'}
    df = pd.read_parquet("hf://datasets/neural-bridge/rag-dataset-12000/" + splits["train"])
    df_50 = df.sample(n=10, random_state=42)

    chunker = LocalChunker(chunk_size=1000, overlap=200)
    embedder = LocalEmbedder()
    db = PostgresVectorStore(dbname="rag_local_lab", user="postgres", password="postgres", host="localhost", port=5432)
    db.init_schema(dim=768)

    chunks_to_insert = []
    texts_to_embed = []
    for doc_idx, doc in df_50.itterows():
        content = doc["context"]
        question = doc["question"]
        answer = doc["answer"]

        chunks = chunker.chunk_text(content)
        for chunk_idx, chunk in enumerate(chunks):
            chunks_to_insert.append({
                "doc_id": doc_idx,
                "chunk_index": chunk_idx,
                "chunk_text": chunk,
                "gold_question": question,
                "gold_answer": answer
            })
            texts_to_embed.append(chunk)

    print(f"Generated {len(chunks_to_insert)} chunks. Now embedding...")

    embeddings = embedder.embed(texts_to_embed)

    db.ingest_data(chunks_to_insert, embeddings)
