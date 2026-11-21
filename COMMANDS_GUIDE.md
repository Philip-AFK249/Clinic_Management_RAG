# Clinic RAG System - Terminal & Database Commands Guide

## 1. Docker & PostgreSQL / pgvector Operations

### Start the Database Stack

Run from the directory containing `docker-compose.yml`:

```bash
# Start container in detached mode
docker compose up -d

# Verify container status and mapped ports (Port 5433 -> 5432)
docker compose ps
```

### Access psql Console inside the Container

```bash
# Direct access via container name
docker exec -it clinic_postgres psql -U postgres -d clinic_rag

# Or access via docker compose service
docker compose exec postgres psql -U postgres -d clinic_rag
```

### Essential SQL Commands in psql

```sql
-- Check installed extensions
\dx

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- List all tables
\dt

-- Check table schema
\d clinic_knowledge_nodes

-- Count total chunks grouped by specialty
SELECT specialty, doc_id, COUNT(*) AS chunk_count 
FROM clinic_knowledge_nodes 
GROUP BY specialty, doc_id;

-- Exit psql
\q
```

### Run Schema Migrations

```bash
# Create table, GIN index, HNSW vector index (idempotent)
python -m app.database.schema
```

### Tear Down and Reset Volume (Wipe Database)

```bash
# Stop and remove containers along with persistent volumes
docker compose down -v

# Recreate a fresh database container
docker compose up -d
```

## 2. Process Management (Windows / Git Bash)

### Terminate Hanging Python Background Processes

```bash
# Git Bash syntax
taskkill //F //IM python.exe

# Native CMD fallback
cmd.exe /c "taskkill /F /IM python.exe"
```

## 3. Document Parsing (PDF to Markdown)

### Fast Parsing with Docling (Local CPU, OCR Disabled)

```bash
python -m app.parsers.docling_parser
```

### Layout-Preserving Parsing with LlamaParse (Cloud Vision GPU)

```bash
python -m app.parsers.llamaparse_parser
```

## 4. Vector Ingestion & Database Verification

### Run Ingestion Pipeline (Chunking + BGE-M3 1024d Embeddings)

```bash
# Ingest a single document
python scripts/run_ingest.py --file data/parsed_markdown/phac_do_dieu_tri_noi_khoa.md --specialty noi_khoa --doc-id QD_BYT_NOI_KHOA

# Bulk-ingest every document under data/parsed_markdown/
python scripts/run_ingest.py --all
```

### Check Database Health & Summary

```bash
python scripts/check_db.py
```

## 5. Retrieval & RAG Verification

### Run Semantic Vector Search Test

```bash
python scripts/run_search_test.py

# Custom query / specialty filter
python scripts/run_search_test.py --query "Cách xử trí IABP?" --specialty tim_mach --top-k 5

# Hybrid (full-text + dense) search
python scripts/run_search_test.py --query "nong van hai la" --hybrid
```

### Test Complete Groq LLM Inference (Console)

```bash
python -m app.rag.generator
```

### Launch Interactive Streamlit Web Interface

```bash
# Start web UI on default port 8501
streamlit run app/ui/streamlit_app.py

# Custom port if 8501 is occupied
streamlit run app/ui/streamlit_app.py --server.port 8502
```

## 6. Configuration

All environment variables live in `.env` at the project root:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | Mapped host port for pgvector | `5433` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `mysecretpassword` |
| `DB_NAME` | Database name | `clinic_rag` |
| `GROQ_API_KEY` | Groq LLM API key | *(required)* |
| `LLAMA_CLOUD_API_KEY` | LlamaParse API key | *(required for LlamaParse)* |