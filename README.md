# KnowEngine Python RAG

一个全 Python 的 RAG 工程主流程实现，使用 FastAPI 提供文档入库、混合检索、Text2SQL、Text2Cypher、联网搜索和流式问答接口。Java 参考项目不参与运行，也不包含在本项目依赖中。

## 主要能力

- PDF / Word 通过 MinerU 转换为 Markdown，Markdown / TXT 可直接清洗入库
- 基于 Markdown 标题的语义分块；超过 `chunk_size` 时生成父块与子块
- 原始文件和解析图片存储到 MinIO，文档与分块元数据存储到 MySQL
- Elasticsearch 全文检索与向量检索，使用 RRF 融合并接入 `qwen3-rerank`
- 图片通过视觉模型生成 Markdown alt 描述
- CSV / Excel 表结构推断、MySQL 动态建表与 `table_meta`
- CSV / Excel / JSON 图数据抽取与 Neo4j 入库
- Text2SQL 与 Text2Cypher，支持动态读取数据库 Schema
- LangGraph 在线编排、手写 ReAct、多工具回退和上下文压缩
- 多轮问题依赖判断与独立问题重写，避免历史意图污染路由
- Tavily 联网搜索工具及内部 RAG 无结果时的回退
- SSE 推送思考、路由、工具调用、回答 Token 和引用溯源事件
- FastAPI 自带的文档上传与对话页面

## 系统架构

### 离线入库

```mermaid
flowchart LR
    A["上传文件"] --> B{"知识库类型"}
    B -->|DOCUMENT_SEARCH| C["MinerU / Markdown 清洗"]
    C --> D["Markdown 分块"]
    D --> E["父子分块"]
    E --> F["MySQL 文档与分块"]
    E --> G["Embedding"]
    G --> H["Elasticsearch 全文与向量索引"]
    B -->|STRUCTURED_TABLE| I["表结构识别"]
    I --> J["MySQL 建表与 table_meta"]
    B -->|GRAPH_DATA| K["实体关系抽取"]
    K --> L["Neo4j"]
```

### 在线问答

```mermaid
flowchart LR
    A["加载历史"] --> B["上下文依赖判断"]
    B --> C["追问改写 / 使用原问题"]
    C --> D["意图识别与路由"]
    D --> E["混合检索"]
    D --> F["Text2SQL"]
    D --> G["Text2Cypher"]
    D --> H["Web Search"]
    D --> I["ReAct Agent"]
    E --> J["生成回答与引用"]
    F --> J
    G --> J
    H --> J
    I --> J
```

## 技术栈

- Python 3.12、FastAPI、Pydantic、asyncio
- LangGraph、LangChain
- MySQL、Redis、Elasticsearch 8.x、Neo4j、MinIO
- MinerU
- DashScope OpenAI-compatible API
- Tavily Search API（可选）
- uv

## 环境要求

启动应用前，需要准备以下服务：

| 服务 | 用途 | 默认端口 | 是否必需 |
| --- | --- | --- | --- |
| MySQL | 文档、分块、会话、业务表、Text2SQL | `3306` | 是 |
| Redis | 父块缓存 | `6379` | 是 |
| Elasticsearch 8.x | 全文与向量混合检索 | `9200` | 是 |
| MinIO | 原始文件和解析资源 | `9000` | 是 |
| Neo4j | 图数据与 Text2Cypher | `7687` | 是 |
| DashScope | Chat、Embedding、VL、Reranker | HTTPS | 是 |
| MinerU API | PDF / Word 转 Markdown | `8000` | PDF / Word 必需 |
| Tavily | 公开网络搜索 | HTTPS | 可选 |

Kibana 仅用于查看 Elasticsearch，本项目不会直接调用。当前主流程也不使用 pgvector。

## 快速开始

### 1. 创建 Python 环境

```powershell
git clone https://github.com/<YOUR_GITHUB_NAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>

$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

`uv.lock` 应提交到 Git，用于复现项目依赖。

MinerU 属于体积较大的可选运行时，因此没有写入主项目依赖。需要解析 PDF / Word 时，将 MinerU 安装到当前虚拟环境，并确认以下命令可用：

```powershell
uv pip install mineru
uv run mineru-api --help
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少替换以下内容：

```env
MYSQL_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/know_engine?charset=utf8mb4
REDIS_URL=redis://localhost:6379/0

OPENAI_API_KEY=YOUR_DASHSCOPE_API_KEY
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
MINIO_ACCESS_KEY=YOUR_MINIO_ACCESS_KEY
MINIO_SECRET_KEY=YOUR_MINIO_SECRET_KEY

WEB_SEARCH_ENABLED=true
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
```

完整配置说明见 [docs/configuration.md](docs/configuration.md)。`.env` 已被 Git 忽略，禁止把真实 API Key 或数据库密码写进 `.env.example`。

### 3. 初始化 MySQL

先创建数据库：

```sql
CREATE DATABASE IF NOT EXISTS know_engine
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

然后在 `know_engine` Schema 中执行：

```text
app/resources/sql/tables.sql
```

MinIO Bucket 默认名为 `know-engine`。配置用户有建桶权限时，应用会自动创建。

### 4. 启动 MinerU

有两种方式，选择一种即可。

应用自动管理 MinerU：

```env
MANAGE_MINERU_PROCESS=true
MINERU_PARSE_API_URL=http://127.0.0.1:8000
MINERU_API_COMMAND=mineru-api
MINERU_FAIL_FAST_ON_STARTUP=false
```

或者在单独的 PowerShell 窗口手动启动：

```powershell
.venv\Scripts\Activate.ps1
uv run mineru-api --host 127.0.0.1 --port 8000
```

如果只测试 Markdown / TXT，可以暂时不启动 MinerU。设置 `MINERU_RESPONSE_TIMEOUT_SECONDS=0` 表示解析响应不设置读写超时。

### 5. 启动 FastAPI

```powershell
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8009
```

启动后访问：

- 对话与上传页面：<http://127.0.0.1:8009/>
- Swagger API：<http://127.0.0.1:8009/docs>
- 依赖健康检查：<http://127.0.0.1:8009/health/ready>

## 文档入库

大文件建议使用后台异步入库接口，避免浏览器或代理等待 MinerU 时超时：

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
| `DOCUMENT_SEARCH` | PDF / Word / MD / TXT | Markdown 清洗与父子分块，写入 MySQL 和 Elasticsearch |
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
| `route` | 意图识别与路由结果 |
| `tool_started` | 开始调用检索、SQL、Cypher 或联网工具 |
| `tool_completed` | 工具调用结束 |
| `token` | 回答 Token |
| `citations` | 文档或网页引用 |
| `completed` | 完整回答、上下文和工具轨迹 |
| `error` | 处理失败 |

回答中的 `[1]`、`[2]` 与 `contexts`、`citations` 的顺序一致。知识库引用会保留文档、分块、父块展开、召回分数与重排分数；联网引用会保留网页标题和 URL。

## 检索与路由

文档召回先从 Elasticsearch 分别执行全文检索和向量检索，使用 RRF 融合候选结果，再通过 `qwen3-rerank` 重排并返回 Top K。命中的子块会先按相邻关系合并，再按 `parent_id` 展开父块。

在线图的显式分支为：

- `hybrid_retrieval`：文档知识与产品手册
- `text2sql`：结构化关系数据
- `text2cypher`：实体关系、路径与图网络
- `web_search`：最新公开网络信息
- `react_agent`：需要多个工具协同的复杂问题，默认最多 5 步

当请求传入 `document_id` 时，路由会固定为对应文档知识库，不调用其他数据源。

## 测试

```powershell
uv run pytest -q
```

## 发布安全

提交前务必运行：

```powershell
git status --short
git check-ignore -v .env output .runtime r7-product-manual-20250123.pdf
```

确认以下内容没有进入暂存区：

- `.env` 和真实 API Key
- `.venv/`、`.uv-cache/`
- `.runtime/` 日志
- `output/` MinerU 解析结果
- 本地上传的 PDF、Word、Excel 和 CSV
- 数据库导出文件和模型权重

## 当前阶段

当前仓库聚焦 RAG 主流程和在线工具编排。RAGAS、完整权限体系、生产级任务队列与分布式部署将在后续迭代中补充。
