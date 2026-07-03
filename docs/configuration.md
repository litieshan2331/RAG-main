# Configuration Checklist

This project reads runtime configuration from `.env`.

For your current local Docker setup, start from:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`.

## Required Services

| Capability | Your Container / Service | Required | `.env` key |
| --- | --- | --- | --- |
| Relational metadata and Text2SQL | Local MySQL | Yes | `MYSQL_URL` |
| Cache for parent chunk lookup | `dodo-redis` on `6379` | Yes | `REDIS_URL` |
| File/object storage | `minio-server` on `9000` | Yes | `MINIO_*` |
| Vector and full-text retrieval | `es-node` on `9200` | Yes | `ELASTICSEARCH_*` |
| Graph query / Text2Cypher | `neo4j` on `7687` | Yes | `NEO4J_*` |
| Chat, query rewrite, routing, embedding | DashScope OpenAI-compatible API | Yes | `OPENAI_*`, model keys |
| PDF/Word to Markdown parsing | MinerU parse service | Required for PDF/Word; optional for `.md`/`.txt` | `MINERU_*` |
| Public web fallback | Tavily Search API | Optional | `WEB_SEARCH_ENABLED`, `TAVILY_*` |

CSV and Excel ingestion uses `STRUCTURED_TABLE` and writes rows to MySQL plus `table_meta`; graph-shaped CSV/Excel/JSON uses `GRAPH_DATA` and writes relationships to Neo4j. Excel parsing requires the `openpyxl` dependency from `pyproject.toml`, so run `uv sync --extra dev` after pulling these changes.

`kibana` is useful for inspecting Elasticsearch, but the app does not call it.

`pgvector` is not used by the current main pipeline because vectors are stored in Elasticsearch.

Your Elasticsearch container is 8.x, so the Python dependency is pinned to `elasticsearch>=8.14,<9`. After changing dependencies, run `uv sync --extra dev` so `uv.lock` does not keep a 9.x client.

## Suggested `.env` Values For Your Docker Ports

```env
MYSQL_URL=mysql+pymysql://root:YOUR_MYSQL_PASSWORD@localhost:3306/know_engine?charset=utf8mb4

# If Redis has no password:
REDIS_URL=redis://localhost:6379/0
# If Redis has a password:
# REDIS_URL=redis://:YOUR_REDIS_PASSWORD@localhost:6379/0

ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=know-engine-vector

MINIO_ENDPOINT=localhost:9000
MINIO_PUBLIC_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=know-engine
MINIO_SECURE=false

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
NEO4J_DATABASE=neo4j

OPENAI_API_KEY=YOUR_DASHSCOPE_KEY
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-max-latest
STREAMING_CHAT_MODEL=qwen-max-latest
VISION_MODEL=qwen3-vl-plus
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=9

MINERU_PARSE_API_URL=http://localhost:8000
MINERU_CONNECT_TIMEOUT_SECONDS=30
MINERU_RESPONSE_TIMEOUT_SECONDS=300
MANAGE_MINERU_PROCESS=false
MINERU_API_COMMAND=mineru-api
MINERU_API_HOST=127.0.0.1
MINERU_API_PORT=8000
MINERU_STARTUP_TIMEOUT_SECONDS=120
MINERU_SHUTDOWN_TIMEOUT_SECONDS=15
MINERU_LOG_FILE=.runtime/mineru-api.log
MINERU_FAIL_FAST_ON_STARTUP=false
MINERU_GENERATE_IMAGE_DESCRIPTIONS=true
MINERU_IMAGE_DESCRIPTION_MAX_BYTES=5000000

HYBRID_CANDIDATE_K=30
HYBRID_RRF_K=60
RERANKER_ENABLED=true
RERANKER_MODEL=qwen3-rerank
RERANKER_API_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
RERANKER_TIMEOUT_SECONDS=30
RERANKER_INSTRUCT=根据用户问题判断文档片段相关性，用于 RAG 问答召回重排。
RERANKER_MAX_INPUT_CHARS=900

WEB_SEARCH_ENABLED=true
TAVILY_API_KEY=YOUR_TAVILY_KEY
TAVILY_API_URL=https://api.tavily.com/search
TAVILY_SEARCH_DEPTH=basic
TAVILY_MAX_RESULTS=5
TAVILY_TIMEOUT_SECONDS=20
TAVILY_INCLUDE_ANSWER=true

REACT_AGENT_ENABLED=true
REACT_AGENT_MAX_STEPS=5
REACT_AGENT_CONFIDENCE_THRESHOLD=0.55
REACT_CONTEXT_MAX_CHARS=12000
REACT_OBSERVATION_MAX_CHARS=2500
REACT_CONTEXT_MAX_TOKENS=6000
REACT_CONTEXT_TRIGGER_RATIO=0.8
REACT_CONTEXT_RESERVE_RATIO=0.35
REACT_TOOL_RESULT_MAX_TOKENS=1200
REACT_SUMMARY_MAX_TOKENS=1000
REACT_CONTEXT_LLM_SUMMARY_ENABLED=true

TEXT2SQL_MAX_ROWS=100
TEXT2SQL_EMPTY_RESULT_TRIGGERS_FALLBACK=true
TEXT2CYPHER_MAX_ROWS=100
TEXT2CYPHER_EMPTY_RESULT_TRIGGERS_FALLBACK=true
TEXT2CYPHER_SCHEMA_SAMPLE_LIMIT=5

CONVERSATION_HISTORY_MAX_MESSAGES=10
CONVERSATION_HISTORY_MAX_CHARS=8000
CONTEXT_DEPENDENCY_THRESHOLD=0.6
```

## Online RAG Orchestration

The online chat entrypoint `/api/chat/rag` is implemented with LangGraph:

```text
rewrite -> route -> hybrid_retrieval -> answer
              |-> text2sql         -> answer
              |-> text2cypher      -> answer
              |-> web_search       -> answer
              |-> react_agent      -> answer
```

Before `route`, multi-turn requests pass through:

```text
load_history -> contextualize_query -> route
```

The contextualizer decides history dependency and produces the standalone query in one structured LLM call. Independent questions are forced to keep the original query; invalid or low-confidence decisions also use the original query, so prior intent is not injected into the router.

If a selected simple tool fails or returns no usable result, the graph can enter `react_agent` before continuing with static fallback. Low-confidence routing can also enter `react_agent` directly.

The current explicit tool nodes are:

- `hybrid_retrieval`: Elasticsearch hybrid retrieval over ingested document chunks.
- `text2sql`: Text2SQL over MySQL static schema plus `table_meta`.
- `text2cypher`: Text2Cypher over Neo4j.
- `web_search`: Tavily Search API for public web or time-sensitive questions.
- `react_agent`: Hand-rolled ReAct loop with max 5 steps by default. It reuses the same tool wrapper layer and compresses observations through `ContextCompressor`.

If `document_id` is provided, routing is forced to `knowledge_base`; ReAct can only use the knowledge-base tool and `web_search` is not called.

Text2SQL is schema-aware through static SQL resources plus dynamic `table_meta.create_sql`, `description`, and `columns_info`. Text2Cypher builds a compact Neo4j schema with labels, properties, relationship directions, and relationship property samples. Empty SQL/Cypher results can trigger fallback instead of being treated as sufficient evidence.

## MinerU Requirement

MinerU is needed when the uploaded source is PDF, Word, or another format that must be converted to Markdown.

The app expects a MinerU-compatible HTTP service:

- Base URL from `MINERU_PARSE_API_URL`
- Endpoint: `POST /file_parse`
- Multipart field: `files`
- Form options used by the app:
  - `backend=pipeline`
  - `response_format_zip=true`
  - `return_images=true`
  - `return_model_output=false`
  - `return_middle_json=false`
- Response: ZIP bytes containing at least one Markdown file and optional images

If you only upload `.md`, `.markdown`, or `.txt`, conversion skips MinerU and just cleans/uploads Markdown.

If MinerU is installed in the same uv environment as this project, the FastAPI app can manage it:

```env
MANAGE_MINERU_PROCESS=true
MINERU_PARSE_API_URL=http://localhost:8000
MINERU_API_COMMAND=mineru-api
MINERU_API_HOST=127.0.0.1
MINERU_API_PORT=8000
```

With this enabled, app startup checks `MINERU_PARSE_API_URL/health`. If MinerU is already healthy, the app leaves that external process alone. If it is not running, the app starts `mineru-api --host 127.0.0.1 --port 8000` and stops only that child process when FastAPI shuts down.

Set `MINERU_RESPONSE_TIMEOUT_SECONDS=0` to wait indefinitely for MinerU parsing responses. Connection and pool timeouts remain finite, so a missing MinerU service still fails instead of hanging forever.

If `MINERU_FAIL_FAST_ON_STARTUP=false`, FastAPI continues to start even when MinerU is slow or unhealthy. `/health/ready` will show MinerU as degraded until `MINERU_PARSE_API_URL/health` becomes healthy.

When `MINERU_GENERATE_IMAGE_DESCRIPTIONS=true`, images extracted by MinerU are sent to `VISION_MODEL` as base64 data URLs before the Markdown is uploaded. This writes semantic alt text into Markdown image tags and does not require the local MinIO URL to be publicly reachable.

## MySQL Initialization

Create database:

```sql
CREATE DATABASE IF NOT EXISTS know_engine CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Then execute:

```text
app/resources/sql/tables.sql
```

## MinIO Initialization

The app can create the bucket automatically if the configured MinIO user has permission.

Default bucket:

```env
MINIO_BUCKET=know-engine
```

## First Readiness Check

After `.env` is complete and the service is running:

```text
http://127.0.0.1:8009/health/ready
```

Expected behavior before MinerU is configured:

- Markdown/TXT-only testing can continue if `mineru` is failed but other required services are ok.
- PDF/Word testing should wait until `mineru` is ok.
