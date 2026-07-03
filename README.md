# RAG-main 知识库问答系统

这是一个基于 Python 的 RAG 知识库问答项目，使用 FastAPI 提供文档上传、异步入库、混合检索、Text2SQL、Text2Cypher、联网检索、SSE 流式问答和离线评测接口。项目目标是把文档知识、结构化数据和图谱数据统一接入同一套问答链路，并保持后续扩展足够清晰。

## 主要能力

- 支持 PDF、Word、Markdown、TXT、CSV、Excel、JSON 等多类型文件接入。
- PDF / Word 通过 MinerU 解析为 Markdown 和图片资源，Markdown / TXT 可直接清洗入库。
- 支持图片资源上传 MinIO，并调用视觉模型生成图片描述，回填到 Markdown 中提升可检索性。
- 基于 Markdown 标题和段落进行语义分块，超出 `chunk_size` 后生成父块和子块。
- 分块元数据写入 MySQL，非父块内容写入 Elasticsearch 全文和向量索引。
- Elasticsearch 全文检索和向量检索双路召回，使用 RRF 融合结果，并接入 `qwen3-rerank` 精排。
- 检索命中子块后先合并相邻子块，再按 `parentChunkId` 展开父块，提升上下文完整性。
- CSV / Excel 支持结构识别、MySQL 动态建表、数据入表和 `table_meta` 登记。
- CSV / Excel / JSON 支持实体关系抽取并写入 Neo4j。
- 在线问答使用 LangGraph 编排，包含问题上下文化、意图路由、工具调用、答案生成和历史持久化。
- 支持 hybrid retrieval、Text2SQL、Text2Cypher、web search 和 ReAct Agent 多分支工具链路。
- 手写 ReAct Agent，默认最多 5 步，配合上下文压缩器控制工具观察结果和历史上下文长度。
- 支持 SSE 推送思考状态、路由结果、工具调用、回答 token、引用溯源和完成事件。
- 支持 RAGAS 的 Context Recall、Context Precision、Faithfulness 三项离线评测。

## 系统架构

### 离线入库

```mermaid
flowchart LR
    A["上传文件"] --> B{"知识库类型"}
    B -->|DOCUMENT_SEARCH| C["MinerU / Markdown 清洗"]
    C --> D["图片描述回填"]
    D --> E["Markdown 分块"]
    E --> F["父子分块"]
    F --> G["MySQL 文档与分块元数据"]
    F --> H["Embedding"]
    H --> I["Elasticsearch 全文与向量索引"]
    B -->|STRUCTURED_TABLE| J["表结构识别"]
    J --> K["MySQL 建表与 table_meta"]
    B -->|GRAPH_DATA| L["实体关系抽取"]
    L --> M["Neo4j"]
```

### 在线问答

```mermaid
flowchart LR
    A["加载历史"] --> B["意图识别 + 问题改写"]
    B --> D["Hybrid Retrieval"]
    B --> E["Text2SQL"]
    B --> F["Text2Cypher"]
    B --> G["Web Search"]
    B --> H["ReAct Agent"]
    D --> I["答案生成与引用溯源"]
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J["保存会话"]
```

## 技术栈

- Python 3.12、FastAPI、Pydantic、asyncio
- LangGraph、LangChain
- MySQL、Redis、Elasticsearch 8.x、Neo4j、MinIO
- MinerU
- DashScope OpenAI-compatible API
- Tavily Search API，可选
- RAGAS，可选评测依赖
- uv

## 环境要求

启动应用前，需要准备以下服务：

| 服务 | 用途 | 默认端口 | 是否必需 |
| --- | --- | --- | --- |
| MySQL | 文档、分块、会话、业务表、Text2SQL | `3306` | 是 |
| Redis | 父块缓存和任务辅助 | `6379` | 是 |
| Elasticsearch 8.x | 全文与向量混合检索 | `9200` | 是 |
| MinIO | 原始文件、图片和解析资源 | `9000` | 是 |
| Neo4j | 图数据和 Text2Cypher | `7687` | 是 |
| MinerU API | PDF / Word 解析 | `8000` | PDF / Word 必需 |
| DashScope | Chat、Embedding、VL、Reranker | HTTPS | 是 |
| Tavily | 联网检索 | HTTPS | 可选 |

Kibana 只用于查看 Elasticsearch 数据，项目本身不会直接依赖。当前主链路也不依赖 pgvector。

## 快速开始

### 1. 创建环境

```powershell
cd F:\LLM\PythonProject\RAG-main

$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

如果需要运行 RAGAS 评测，再安装评测依赖：

```powershell
uv sync --extra dev --extra evaluation
```

MinerU 体积较大，建议安装到当前 uv 环境中：

```powershell
uv pip install mineru
uv run mineru-api --help
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少确认以下配置：

```env
MYSQL_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/know_engine?charset=utf8mb4
REDIS_URL=redis://localhost:6379/0

OPENAI_API_KEY=YOUR_DASHSCOPE_API_KEY
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-max-latest
STREAMING_CHAT_MODEL=qwen-max-latest
EMBEDDING_MODEL=text-embedding-v4

RERANKER_ENABLED=true
RERANKER_MODEL=qwen3-rerank

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=YOUR_MINIO_ACCESS_KEY
MINIO_SECRET_KEY=YOUR_MINIO_SECRET_KEY
MINIO_BUCKET=know-engine

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD

WEB_SEARCH_ENABLED=true
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
```

完整配置说明见 [docs/configuration.md](docs/configuration.md)。`.env` 已被 Git 忽略，不要把真实密钥写入 `.env.example`。

### 3. 初始化 MySQL

先创建数据库：

```sql
CREATE DATABASE IF NOT EXISTS know_engine
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

然后在 `know_engine` schema 中执行：

```text
app/resources/sql/tables.sql
```

MinIO Bucket 默认名为 `know-engine`。如果配置账号有建桶权限，应用启动或入库时会自动创建。

### 4. 启动 MinerU

推荐先单独启动 MinerU，等模型下载和服务健康后再启动主应用：

```powershell
.venv\Scripts\Activate.ps1
uv run mineru-api --host 127.0.0.1 --port 8000
```

也可以让主应用管理 MinerU 进程：

```env
MANAGE_MINERU_PROCESS=true
MINERU_PARSE_API_URL=http://127.0.0.1:8000
MINERU_API_COMMAND=mineru-api
MINERU_API_HOST=127.0.0.1
MINERU_API_PORT=8000
MINERU_FAIL_FAST_ON_STARTUP=false
```

如果只测试 Markdown / TXT，可以暂时不启动 MinerU。`MINERU_RESPONSE_TIMEOUT_SECONDS=0` 表示解析响应不设置读写超时。

### 5. 启动 FastAPI

```powershell
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8009
```

启动后访问：

- 前端页面：<http://127.0.0.1:8009/>
- Swagger API：<http://127.0.0.1:8009/docs>
- 健康检查：<http://127.0.0.1:8009/health/ready>

## 文档入库

大文件建议使用异步入库接口，避免浏览器或代理等待 MinerU 解析时超时：

```powershell
curl.exe -X POST "http://127.0.0.1:8009/api/documents/upload-and-ingest-async" `
  -F "file=@C:\path\manual.pdf" `
  -F "title=manual.pdf" `
  -F "upload_user=admin" `
  -F "knowledge_base_type=DOCUMENT_SEARCH" `
  -F "chunk_size=1000" `
  -F "overlap=80"
```

接口会立即返回 `task_id`。查询进度：

```powershell
curl.exe "http://127.0.0.1:8009/api/ingestion-tasks/{task_id}"
```

取消任务：

```powershell
curl.exe -X POST "http://127.0.0.1:8009/api/ingestion-tasks/{task_id}/cancel"
```

入库类型：

| `knowledge_base_type` | 文件类型 | 处理方式 |
| --- | --- | --- |
| `DOCUMENT_SEARCH` | PDF / Word / MD / TXT | Markdown 清洗、图片描述、父子分块，写入 MySQL 和 Elasticsearch |
| `STRUCTURED_TABLE` | CSV / Excel | 识别表结构，在 MySQL 建表并写入 `table_meta` |
| `GRAPH_DATA` | CSV / Excel / JSON | 抽取实体关系并写入 Neo4j，可在 MySQL 留档 |

## 问答接口

普通问答：

```powershell
curl.exe -X POST "http://127.0.0.1:8009/api/chat/rag" `
  -H "Content-Type: application/json" `
  -d '{"query":"后排常用部件有什么？","user_id":"demo","top_k":5}'
```

SSE 流式问答：

```http
POST /api/chat/rag/stream
Content-Type: application/json
Accept: text/event-stream

{"query":"后排常用部件有什么？","user_id":"demo","top_k":5}
```

主要 SSE 事件：

| 事件 | 含义 |
| --- | --- |
| `started` | 开始处理问题 |
| `route` | 意图识别和路由结果 |
| `tool_started` | 开始调用检索、SQL、Cypher 或联网工具 |
| `tool_completed` | 工具调用结束 |
| `token` | 回答 token |
| `citations` | 文档或网页引用 |
| `completed` | 完整回答、上下文和工具轨迹 |
| `error` | 处理失败 |

回答中的 `[1]`、`[2]` 与返回体里的 `contexts`、`citations` 顺序一致。知识库引用会保留文档、分块、父块展开、召回分数和重排分数；联网引用会保留网页标题和 URL。

## 检索与路由

文档召回会从 Elasticsearch 分别执行全文检索和向量检索，使用 RRF 融合候选结果，再通过 `qwen3-rerank` 重排并返回 Top K。命中的子块会先按相邻关系合并，再按 `parentChunkId` 展开父块，用较短召回结果换取更完整的回答证据。

在线图中的显式分支为：

- `hybrid_retrieval`：文档知识、产品手册、说明书和一般知识库问答。
- `text2sql`：结构化表数据查询。
- `text2cypher`：实体关系、路径、依赖和图谱查询。
- `web_search`：最新公开网络信息。
- `react_agent`：需要多个工具协同的复杂问题，默认最多 5 步。

当请求传入 `document_id` 时，系统会优先限制在对应文档知识库内问答，不让上一轮历史意图污染当前路由。

## 上下文压缩

项目内置上下文压缩器，用于控制多轮对话和 ReAct 工具观察结果的长度：

- 按 token 预算决定是否触发压缩。
- 保留近期用户问题、模型回答和关键工具观察。
- 对较早历史做增量摘要。
- 对大型工具结果进行裁剪和摘要。
- 当模型摘要不可用时，使用确定性降级摘要保证链路可继续运行。

这部分主要服务于多轮问答、工具调用链路和后续更复杂的 Agent 编排。

## RAGAS 评测

评测接口会运行真实 RAG 链路，得到答案和召回上下文后，再计算三项指标：

- `context_recall`：参考答案中的事实有多少能被召回上下文支持。
- `context_precision`：召回上下文按顺序看与问题和参考答案的相关程度。
- `faithfulness`：生成答案中的事实陈述有多少能被召回上下文支持。

测试集格式：

```json
{
  "samples": [
    {
      "sample_id": "manual-001",
      "question": "智界 R7 使用说明书的版本号和出版日期是什么？",
      "reference": "版本号为 R7OM25C1，出版日期为 2025 年 1 月。",
      "user_id": "ragas-eval",
      "document_id": 4,
      "top_k": 5
    }
  ]
}
```

调用接口：

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8009/api/evaluation/ragas" `
  -ContentType "application/json; charset=utf-8" `
  -InFile "evaluation\datasets\ragas_manual_eval.json" `
  -OutFile "evaluation\reports\ragas_manual_eval_report.json" `
  -TimeoutSec 7200
```

评测模型默认使用 `RAGAS_JUDGE_MODEL`，为空时复用 `CHAT_MODEL`。评测会产生额外模型调用，建议先用小规模人工标注集建立基线。

## 测试

```powershell
uv run pytest -q
```

## 上传前检查

提交到 GitHub 前建议执行：

```powershell
git status --short
git check-ignore -v .env .venv .runtime output evaluation/reports
```

确认以下内容没有进入暂存区：

- `.env` 和真实 API Key。
- `.venv/`、`.uv-cache/`。
- `.runtime/` 日志。
- `output/` MinerU 解析结果。
- `evaluation/reports/` 评测报告。
- 本地上传的 PDF、Word、Excel、CSV 等业务文件。
- 数据库导出文件和模型权重。

常用提交命令：

```powershell
git status
git add .
git status --short
git commit -m "docs: update project readme"
git push -u origin main
```

## 当前阶段

当前版本聚焦 RAG 主流程、在线工具编排、异步入库、引用溯源、流式输出和基础评测。后续可以继续补充权限体系、任务队列、生产部署脚本、更细粒度的评测看板和业务侧工具节点。
