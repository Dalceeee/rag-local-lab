import argparse
import sys

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import pandas as pd

from rag_local_lab.chuncker import LocalChunker
from rag_local_lab.embedder import LocalEmbedder
from rag_local_lab.generator import LocalGenerator
from rag_local_lab.db import PostgresVectorStore
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, LM_STUDIO_URL, EMBEDDING_MODEL, EMBEDDING_DIM, GENERATION_MODEL

def ingest_pipeline(chunker: LocalChunker, embedder: LocalEmbedder, db: PostgresVectorStore):
    splits = {'train': 'data/train-00000-of-00001-9df3a936e1f63191.parquet', 'test': 'data/test-00000-of-00001-af2a9f454ad1b8a3.parquet'}
    df = pd.read_parquet("hf://datasets/neural-bridge/rag-dataset-12000/" + splits["train"])
    df_10 = df.sample(n=10, random_state=42)

    db.init_schema(dim=EMBEDDING_DIM)
    print("Database initialized.")

    chunks_to_insert = []
    texts_to_embed = []
    for doc_idx, doc in df_10.iterrows():
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
    print("Ingestion completed.")

def retrieve_and_generate(
    query: str, 
    embedder: LocalEmbedder, 
    db: PostgresVectorStore, 
    generator: LocalGenerator, 
    top_k: int = 5
) -> str:
    query_embedding = embedder.embed_query(query)
    top_k_chunks = db.search_similar(query_embedding, top_k=top_k)

    formatted_chunks = format_chunks_for_generation(top_k_chunks)
    return generator.generate(query, formatted_chunks)

def format_chunks_for_generation(chunks: list[dict]) -> str:
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append(f"Chunk {chunk['chunk_index']} (Doc ID: {chunk['doc_id']}):\n{chunk['chunk_text']}\n")
    return "\n".join(formatted_chunks)

def main():
    parser = argparse.ArgumentParser(description="Local RAG Lab CLI")
    parser.add_argument("--ingest", action="store_true", help="Only ingest data into the database")
    parser.add_argument("--query", type=str, default=None, help="Execute a query on the RAG")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top-k results to retrieve")
    args = parser.parse_args()

    # Database connection
    db = PostgresVectorStore(
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST, port=POSTGRES_PORT
    )

    try:
        if args.ingest:
            # Chunker is instantiated only when ingesting data
            chunker = LocalChunker(chunk_size=1000, overlap=200)
            embedder = LocalEmbedder(model_name=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
            ingest_pipeline(chunker, embedder, db)

        elif args.query:
            embedder = LocalEmbedder(model_name=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
            generator = LocalGenerator(model_name=GENERATION_MODEL, base_url=LM_STUDIO_URL)
            
            ans = retrieve_and_generate(args.query, embedder, db, generator, top_k=args.top_k)
            print(f"\nGenerated Answer:\n{ans}")

        else:
            parser.print_help()
    finally:
        db.close()


if __name__ == "__main__":
    # query = "What was the main issue in the 2006 lawsuit filed by ASU Students for Life against Arizona State University?"
    # gold_answer = "The main issue in the 2006 lawsuit was that the university discriminated against students by imposing an insurance requirement for on-campus events."

    main()