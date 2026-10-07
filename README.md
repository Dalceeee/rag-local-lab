# rag-local-lab

Local Retrieval-Augmented Generation (RAG) lab using:

- `sentence-transformers` and EmbeddingGemma for embeddings;
- PostgreSQL with the `pgvector` extension for vector storage and search;
- LangChain text splitters for document chunking;
- LM Studio for local answer generation.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop with Docker Compose
- LM Studio, if you want to run the generation step

The first model load downloads `google/embeddinggemma-2` from Hugging Face.
The dataset is also downloaded from Hugging Face during ingestion.

## Initialization

Clone the repository and enter its directory:

```powershell
git clone <repository-url>
cd rag-local-lab
```

Create the virtual environment and install the locked dependencies:

```powershell
uv sync
```

Create a local `.env` file in the project root. Its PostgreSQL values must
match the values in `docker-compose.yml`:

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<the-password-used-by-docker-compose>

LM_STUDIO_URL=http://localhost:1234/v1
EMBEDDING_MODEL=google/embeddinggemma-2
EMBEDDING_DIM=768
GENERATION_MODEL=liquid/lfm2.5-1.2b
```

## Database setup with Docker

Start PostgreSQL with pgvector:

```powershell
docker compose up -d
```

Check that the container is running:

```powershell
docker compose ps
docker compose logs postgres
```

The database is exposed on `localhost:5433`. The container listens on its
internal port `5432`; the port mapping is defined in `docker-compose.yml`.

The application creates the `vector` extension, the `chunks` table, and the
HNSW index automatically during ingestion. No manual SQL setup is required.

To stop the database while preserving its data:

```powershell
docker compose stop
```

To stop it and remove the container:

```powershell
docker compose down
```

To remove the database volume as well (this deletes all ingested data):

```powershell
docker compose down -v
```

## Execution

### Ingestion

Start Docker first, then ingest the sample documents:

```powershell
uv run python -m rag_local_lab.main --ingest
```

The ingestion pipeline:

1. downloads the training split of `neural-bridge/rag-dataset-12000`;
2. samples 10 documents;
3. splits the contexts into chunks;
4. generates 768-dimensional embeddings;
5. stores the chunks and embeddings in PostgreSQL.

The database table is truncated before ingestion, so running `--ingest`
replaces the existing contents of `chunks`.

### Query

To generate an answer, start a compatible model in LM Studio and expose its
OpenAI-compatible API at `http://localhost:1234/v1` (or update
`LM_STUDIO_URL` in `.env`).

Then run:

```powershell
uv run python -m rag_local_lab.main --query "What was the main issue in the 2006 lawsuit filed by ASU Students for Life against Arizona State University?" --top-k 5
```

The application embeds the query, retrieves the top-k nearest chunks from
pgvector, and sends the resulting context to the configured LM Studio model.

To see all available CLI options:

```powershell
uv run python -m rag_local_lab.main --help
```

## Troubleshooting

### PostgreSQL authentication failed

Make sure `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PORT` in `.env` match `docker-compose.yml`. If the Docker volume
was created with different credentials, changing the environment variables
does not change the existing database password. Recreate the volume only if
you are allowed to delete its data:

```powershell
docker compose down -v
docker compose up -d
```

### Hugging Face warnings

Unauthenticated Hugging Face access works with lower rate limits. Set
`HF_TOKEN` in your environment if downloads are rate-limited:

```powershell
$env:HF_TOKEN = "<your-token>"
```

### pgvector DLL errors on native Windows PostgreSQL

Use the provided Docker Compose setup instead of a native PostgreSQL
installation. It includes a PostgreSQL image with pgvector already installed
and avoids Windows Code Integrity restrictions on unsigned native extension
DLLs.
