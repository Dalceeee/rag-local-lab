import numpy as np
import psycopg2
from psycopg2.extras import execute_values

class PostgresVectorStore:
    def __init__(self, dbname: str = "rag_lab", user: str = "postgres", password: str = "postgres", host: str = "localhost", port: int = 5432):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def init_schema(self, dim: int = 768):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id INT NOT NULL,            -- original document ID to which this chunk belongs
                    chunk_index INT NOT NULL,       -- Position of the chunk within the original document
                    chunk_text TEXT NOT NULL,       -- The actual text on which we calculate the embedding
                    gold_question TEXT,             -- To track the evaluation
                    gold_answer TEXT,
                    embedding vector({dim})
                );
            """)
            # HNSW index for efficient similarity search on the embedding column
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """)
        self.conn.commit()

    def ingest_data(self, chunks: list[dict], embeddings: np.ndarray):

        records = []

        for chunk, embedding in zip(chunks, embeddings):
            record = (
                chunk["doc_id"],
                chunk["chunk_index"],
                chunk["chunk_text"],
                chunk.get("gold_question"),
                chunk.get("gold_answer"),
                embedding.tolist(),
            )
            records.append(record)
        
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks;")
            
            query = """
                INSERT INTO chunks (doc_id, chunk_index, chunk_text, gold_question, gold_answer, embedding)
                VALUES %s
            """
            execute_values(cur, query, records, template="(%s, %s, %s, %s, %s, %s::vector)")
        self.conn.commit()
        print(f"Successfully ingested {len(records)} records into the database.")

    def search_similar(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        query_vector_list = np.asarray(query_vector).reshape(-1).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, doc_id, chunk_index, chunk_text, gold_question, gold_answer, embedding <-> %s::vector AS similarity
                FROM chunks
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """, (query_vector_list, query_vector_list, top_k))
            results = cur.fetchall()

        return [
            {
                "id": row[0],
                "doc_id": row[1],
                "chunk_index": row[2],
                "chunk_text": row[3],
                "gold_question": row[4],
                "gold_answer": row[5],
                "similarity": row[6]
            }
            for row in results
        ]

    def close(self):
        self.conn.close()