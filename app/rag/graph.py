from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.rag.conversation import (
    ContextDependencyDecision,
    ContextualizedQuery,
    ConversationContextService,
    ConversationRepository,
    ConversationTurn,
)
from app.rag.query_router import QueryRouteDecision, QueryRouterService, Strategy
from app.rag.react_agent import ReactAgent
from app.rag.tools import OnlineToolExecutor
from app.schemas import RetrievedDocument


TOOL_NODE_BY_STRATEGY: dict[Strategy, str] = {
    "knowledge_base": "hybrid_retrieval",
    "relational_db": "text2sql",
    "graph_db": "text2cypher",
    "web_search": "web_search",
    "react_agent": "react_agent",
}
TOOL_EDGE_MAP = {
    "hybrid_retrieval": "hybrid_retrieval",
    "text2sql": "text2sql",
    "text2cypher": "text2cypher",
    "web_search": "web_search",
    "react_agent": "react_agent",
    "answer": "answer",
}
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "knowledge_base": "混合检索",
    "relational_db": "Text2SQL",
    "graph_db": "Text2Cypher",
    "web_search": "Tavily 联网搜索",
    "react_agent": "ReAct 多工具推理",
}


class RagGraphState(TypedDict, total=False):
    query: str
    conversation_id: str
    user_id: str
    top_k: int
    document_id: int | None
    history: list[ConversationTurn]
    context_decision: ContextDependencyDecision
    depends_on_history: bool
    rewritten_query: str
    execution_query: str
    requires_decomposition: bool
    sub_queries: list[dict[str, Any]]
    force_knowledge_base: bool
    route_decision: QueryRouteDecision
    fallback_routes: list[Strategy]
    attempt_index: int
    route: Strategy
    tool_result: str
    contexts: list[RetrievedDocument]
    tool_trace: list[dict[str, Any]]
    errors: list[str]
    react_agent_attempted: bool
    answer: str


class OnlineRagGraph:
    def __init__(self, db: Session) -> None:
        from app.core.llm import get_chat_model

        self.llm = get_chat_model(temperature=0.2, streaming=True)
        self.settings = get_settings()
        self.router = QueryRouterService()
        self.conversations = ConversationRepository(db, self.settings)
        self.context_service = ConversationContextService(llm=get_chat_model(temperature=0), settings=self.settings)
        self.tools = OnlineToolExecutor(db)
        self.react_agent = ReactAgent(self.tools, llm=get_chat_model(temperature=0))
        self.graph = self._build_graph()

    def answer(
        self,
        query: str,
        user_id: str,
        top_k: int,
        document_id: int | None = None,
        conversation_id: str = "",
    ) -> dict:
        result = self.graph.invoke(self._initial_state(query, user_id, top_k, document_id, conversation_id))
        return self._result_payload(query, conversation_id, result)

    def stream_answer(
        self,
        query: str,
        user_id: str,
        top_k: int,
        document_id: int | None = None,
        conversation_id: str = "",
    ) -> Iterator[dict[str, Any]]:
        initial_state = self._initial_state(query, user_id, top_k, document_id, conversation_id)
        final_state: RagGraphState = initial_state
        streamed_answer = False

        yield {
            "event": "started",
            "data": {
                "conversation_id": conversation_id,
                "stage": "thinking",
                "message": "正在分析问题与对话上下文",
            },
        }

        for part in self.graph.stream(initial_state, stream_mode=["tasks", "updates", "messages", "values"], version="v2"):
            part_type = part.get("type")
            data = part.get("data")

            if part_type == "tasks" and isinstance(data, dict):
                task_event = self._event_for_task_start(data)
                if task_event:
                    yield task_event
                continue

            if part_type == "values" and isinstance(data, dict):
                final_state = data
                continue

            if part_type == "messages":
                token = self._answer_token(data)
                if token:
                    streamed_answer = True
                    yield {"event": "token", "data": {"content": token}}
                continue

            if part_type != "updates" or not isinstance(data, dict):
                continue

            for node_name, update in data.items():
                if not isinstance(update, dict):
                    continue
                yield from self._events_for_node(node_name, update)

        result = self._result_payload(query, conversation_id, final_state)
        if result["answer"] and not streamed_answer:
            yield {"event": "token", "data": {"content": result["answer"]}}
        if result["contexts"]:
            yield {"event": "citations", "data": {"items": result["contexts"]}}
        yield {"event": "completed", "data": result}

    @staticmethod
    def _initial_state(
        query: str,
        user_id: str,
        top_k: int,
        document_id: int | None,
        conversation_id: str,
    ) -> RagGraphState:
        return {
            "query": query,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "top_k": top_k,
            "document_id": document_id,
            "attempt_index": 0,
            "tool_result": "",
            "contexts": [],
            "tool_trace": [],
            "errors": [],
            "react_agent_attempted": False,
        }

    @staticmethod
    def _result_payload(query: str, conversation_id: str, result: RagGraphState) -> dict[str, Any]:
        route_decision = result.get("route_decision") or QueryRouterService.default_decision("未执行路由，降级到知识库。")
        return {
            "query": query,
            "conversation_id": conversation_id,
            "rewritten_query": result.get("rewritten_query") or query,
            "context_decision": (result.get("context_decision") or ContextDependencyDecision(False, "未执行依赖判断。", 0.0)).to_dict(),
            "route": result.get("route") or route_decision.strategy,
            "route_decision": route_decision.to_dict(),
            "answer": result.get("answer") or "",
            "contexts": result.get("contexts") or [],
            "tool_trace": result.get("tool_trace") or [],
        }

    @staticmethod
    def _answer_token(data: Any) -> str:
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            return ""
        message, metadata = data
        if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "answer":
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)

    @staticmethod
    def _event_for_task_start(data: dict[str, Any]) -> dict[str, Any] | None:
        if "result" in data or "error" in data:
            return None
        node_name = str(data.get("name") or "")
        strategy_by_node = {node: strategy for strategy, node in TOOL_NODE_BY_STRATEGY.items()}
        if node_name in strategy_by_node:
            strategy = strategy_by_node[node_name]
            tool = TOOL_DISPLAY_NAMES.get(strategy, strategy)
            return {
                "event": "tool_started",
                "data": {"strategy": strategy, "tool": tool, "node": node_name, "message": f"正在调用 {tool}"},
            }
        messages = {
            "load_history": "正在加载对话历史",
            "contextualize_query": "正在判断上下文依赖并生成独立问题",
            "route": "正在识别意图并选择工具",
            "answer": "正在生成回答",
            "persist_turn": "正在保存本轮对话",
        }
        if node_name not in messages:
            return None
        return {"event": "status", "data": {"stage": node_name, "node": node_name, "message": messages[node_name]}}

    @staticmethod
    def _events_for_node(node_name: str, update: RagGraphState) -> Iterator[dict[str, Any]]:
        if node_name == "contextualize_query":
            depends = bool(update.get("depends_on_history"))
            message = "追问已结合历史改写为独立问题" if depends else "问题可独立理解，正在选择检索路径"
            yield {"event": "status", "data": {"stage": "routing", "node": node_name, "message": message}}
            return

        if node_name == "route":
            decision = update.get("route_decision")
            routes = update.get("fallback_routes") or []
            strategy = routes[0] if routes else getattr(decision, "strategy", "knowledge_base")
            yield {"event": "route", "data": {"decision": decision, "strategy": strategy}}
            return

        if node_name in TOOL_NODE_BY_STRATEGY.values():
            trace = update.get("tool_trace") or []
            latest = trace[-1] if trace else {}
            strategy = str(latest.get("strategy") or update.get("route") or node_name)
            ok = bool(latest.get("ok"))
            yield {
                "event": "tool_finished" if ok else "tool_failed",
                "data": {
                    "strategy": strategy,
                    "tool": TOOL_DISPLAY_NAMES.get(strategy, strategy),
                    "ok": ok,
                    "metadata": latest.get("metadata") or {},
                    "error": latest.get("error"),
                    "context_count": len(update.get("contexts") or []),
                },
            }
            if ok:
                yield {"event": "status", "data": {"stage": "answering", "message": "工具调用完成，正在生成回答"}}
            return

        if node_name == "answer":
            yield {"event": "status", "data": {"stage": "persisting", "message": "回答生成完成，正在保存对话"}}

    def _build_graph(self):
        workflow = StateGraph(RagGraphState)
        workflow.add_node("load_history", self._load_history_node)
        workflow.add_node("contextualize_query", self._contextualize_query_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("hybrid_retrieval", self._hybrid_retrieval_node)
        workflow.add_node("text2sql", self._text2sql_node)
        workflow.add_node("text2cypher", self._text2cypher_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("react_agent", self._react_agent_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("persist_turn", self._persist_turn_node)

        workflow.set_entry_point("load_history")
        workflow.add_edge("load_history", "contextualize_query")
        workflow.add_edge("contextualize_query", "route")
        workflow.add_conditional_edges("route", self._next_tool_node, TOOL_EDGE_MAP)
        for node_name in TOOL_NODE_BY_STRATEGY.values():
            workflow.add_conditional_edges(node_name, self._after_tool, TOOL_EDGE_MAP)
        workflow.add_edge("answer", "persist_turn")
        workflow.add_edge("persist_turn", END)
        return workflow.compile()

    def _load_history_node(self, state: RagGraphState) -> RagGraphState:
        try:
            history = self.conversations.recent_turns(state.get("conversation_id") or "", state.get("user_id") or "unknown")
        except Exception as exc:
            self.conversations.db.rollback()
            return {
                "history": [],
                "errors": [*state.get("errors", []), f"load_history failed: {exc}"],
            }
        return {"history": history}

    def _contextualize_query_node(self, state: RagGraphState) -> RagGraphState:
        try:
            result = self.context_service.contextualize(state["query"], state.get("history") or [])
        except Exception as exc:
            decision = ContextDependencyDecision(False, "依赖判断失败，按独立问题处理。", 0.0)
            return {
                **self._query_payload(state["query"], state),
                "context_decision": decision,
                "depends_on_history": False,
                "errors": [*state.get("errors", []), f"contextualize_query failed: {exc}"],
            }
        return {
            **self._query_payload(result.standalone_query, state),
            "context_decision": result.decision,
            "depends_on_history": result.decision.depends_on_history,
            "requires_decomposition": result.decision.requires_decomposition,
            "sub_queries": result.decision.sub_queries,
        }

    @staticmethod
    def _query_payload(query: str, state: RagGraphState) -> RagGraphState:
        user_id = state.get("user_id") or "unknown"
        return {
            "rewritten_query": query,
            "execution_query": OnlineRagGraph._decorate_query(query, user_id),
            "force_knowledge_base": state.get("document_id") is not None,
        }

    def _route_node(self, state: RagGraphState) -> RagGraphState:
        force_knowledge_base = bool(state.get("force_knowledge_base"))
        errors = list(state.get("errors") or [])
        routing_query = state.get("rewritten_query") or state["query"]
        if state.get("requires_decomposition") and self.settings.react_agent_enabled:
            decision = QueryRouteDecision(
                "多跳问题规划与执行",
                "react_agent",
                "问题已拆分为存在依赖关系的多个步骤，需要按计划逐步调用工具并汇总证据。",
                0.98,
                slots={"sub_queries": state.get("sub_queries") or []},
            )
        elif not force_knowledge_base and self._is_explicit_web_query(routing_query):
            decision = QueryRouteDecision(
                "实时公开信息查询",
                "web_search",
                "问题包含明确的实时或联网查询信号，直接调用 web_search。",
                0.95,
            )
        else:
            try:
                decision = self.router.route(state["execution_query"], force_knowledge_base=force_knowledge_base)
            except Exception as exc:
                decision = self._fallback_route_decision(routing_query, force_knowledge_base=force_knowledge_base)
                errors.append(f"route LLM failed: {exc}")
        if decision.strategy == "react_agent":
            fallback_routes = ["react_agent"]
        else:
            fallback_routes = self.router.fallback_order(decision, force_knowledge_base=force_knowledge_base)
        if self._should_enter_react(decision, force_knowledge_base=force_knowledge_base):
            fallback_routes = ["react_agent"]
        return {
            "route_decision": decision,
            "fallback_routes": fallback_routes,
            "errors": errors,
        }

    def _hybrid_retrieval_node(self, state: RagGraphState) -> RagGraphState:
        return self._run_tool_node(state, "knowledge_base")

    def _text2sql_node(self, state: RagGraphState) -> RagGraphState:
        return self._run_tool_node(state, "relational_db")

    def _text2cypher_node(self, state: RagGraphState) -> RagGraphState:
        return self._run_tool_node(state, "graph_db")

    def _web_search_node(self, state: RagGraphState) -> RagGraphState:
        return self._run_tool_node(state, "web_search")

    def _react_agent_node(self, state: RagGraphState) -> RagGraphState:
        trace = list(state.get("tool_trace") or [])
        try:
            result = self.react_agent.run(
                state["execution_query"],
                top_k=state.get("top_k") or 5,
                document_id=state.get("document_id"),
                prior_errors=state.get("errors") or [],
                query_plan=state.get("sub_queries") or [],
            )
        except Exception as exc:
            trace.append(
                {
                    "strategy": "react_agent",
                    "node": "react_agent",
                    "ok": False,
                    "error": str(exc),
                }
            )
            return {
                "route": "react_agent",
                "tool_result": f"ReAct agent failed: {exc}",
                "tool_trace": trace,
                "react_agent_attempted": True,
                "errors": [*state.get("errors", []), f"react_agent failed: {exc}"],
            }

        trace.append(
            {
                "strategy": "react_agent",
                "node": "react_agent",
                "tool": result.tool_name,
                "ok": True,
                "metadata": result.metadata,
            }
        )
        return {
            "route": "react_agent",
            "tool_result": result.content,
            "contexts": result.contexts,
            "tool_trace": trace,
            "react_agent_attempted": True,
        }

    def _run_tool_node(self, state: RagGraphState, expected_strategy: Strategy) -> RagGraphState:
        routes = state.get("fallback_routes") or [expected_strategy]
        attempt_index = state.get("attempt_index", 0)
        if attempt_index >= len(routes):
            return {"tool_result": ""}

        strategy = routes[attempt_index]
        trace = list(state.get("tool_trace") or [])
        try:
            result = self.tools.run(
                strategy,
                state["execution_query"],
                top_k=state.get("top_k") or 5,
                document_id=state.get("document_id"),
            )
        except Exception as exc:
            trace.append(
                {
                    "strategy": strategy,
                    "node": TOOL_NODE_BY_STRATEGY.get(strategy, expected_strategy),
                    "ok": False,
                    "error": str(exc),
                }
            )
            return {
                "route": strategy,
                "attempt_index": attempt_index + 1,
                "tool_trace": trace,
                "errors": [*state.get("errors", []), f"{strategy} failed: {exc}"],
            }

        trace.append(
            {
                "strategy": strategy,
                "node": TOOL_NODE_BY_STRATEGY.get(strategy, expected_strategy),
                "tool": result.tool_name,
                "ok": True,
                "metadata": result.metadata,
            }
        )
        return {
            "route": strategy,
            "attempt_index": attempt_index + 1,
            "tool_result": result.content,
            "contexts": result.contexts,
            "tool_trace": trace,
        }

    @staticmethod
    def _next_tool_node(state: RagGraphState) -> str:
        routes = state.get("fallback_routes") or ["knowledge_base"]
        attempt_index = state.get("attempt_index", 0)
        if attempt_index >= len(routes):
            return "answer"
        return TOOL_NODE_BY_STRATEGY.get(routes[attempt_index], "hybrid_retrieval")

    def _after_tool(self, state: RagGraphState) -> str:
        if state.get("tool_result"):
            return "answer"
        if self._should_try_react_after_tool(state):
            return "react_agent"
        return OnlineRagGraph._next_tool_node(state)

    def _should_enter_react(self, decision: QueryRouteDecision, *, force_knowledge_base: bool) -> bool:
        if force_knowledge_base or not self.settings.react_agent_enabled:
            return False
        return decision.strategy == "react_agent" or decision.confidence < self.settings.react_agent_confidence_threshold

    def _should_try_react_after_tool(self, state: RagGraphState) -> bool:
        if state.get("force_knowledge_base") or state.get("react_agent_attempted"):
            return False
        return bool(self.settings.react_agent_enabled)

    def _answer_node(self, state: RagGraphState) -> RagGraphState:
        context = state.get("tool_result") or "未检索到可用依据。"
        route = state.get("route") or "knowledge_base"
        prompt = self._answer_prompt(
            original_query=state["query"],
            rewritten_query=state.get("rewritten_query") or state["query"],
            route=route,
            context=context,
            errors=state.get("errors") or [],
        )
        try:
            response = self.llm.invoke(prompt).content
            return {"answer": str(response).strip()}
        except Exception as exc:
            return {
                "answer": self._fallback_answer(route, context),
                "errors": [*state.get("errors", []), f"answer LLM failed: {exc}"],
            }

    @staticmethod
    def _fallback_route_decision(query: str, *, force_knowledge_base: bool) -> QueryRouteDecision:
        if force_knowledge_base:
            return QueryRouteDecision("知识库问答", "knowledge_base", "指定了文档范围。", 1.0)
        if OnlineRagGraph._is_explicit_web_query(query):
            return QueryRouteDecision("公开网络信息查询", "web_search", "LLM 路由不可用，按实时信息关键词降级到联网搜索。", 0.7)
        return QueryRouteDecision("知识库问答", "knowledge_base", "LLM 路由不可用，降级到内部知识库检索。", 0.7)

    @staticmethod
    def _is_explicit_web_query(query: str) -> bool:
        normalized = query.lower()
        web_hints = (
            "今天", "今日", "日期", "星期", "天气", "最新新闻", "实时信息", "联网搜索", "网络搜索", "网上搜索", "互联网搜索",
            "today", "current date", "day of the week", "weather", "latest news", "web search", "search the web", "internet search",
        )
        return any(hint in normalized for hint in web_hints)

    @staticmethod
    def _fallback_answer(route: str, context: str) -> str:
        if route == "web_search":
            return f"回答模型暂时不可用，以下是联网搜索结果：\n\n{context}"
        return f"回答模型暂时不可用，以下是工具检索结果：\n\n{context}"

    def _persist_turn_node(self, state: RagGraphState) -> RagGraphState:
        try:
            self.conversations.save_turn(
                conversation_id=state["conversation_id"],
                user_id=state.get("user_id") or "unknown",
                query=state["query"],
                rewritten_query=state.get("rewritten_query") or state["query"],
                answer=state.get("answer") or "",
                route=state.get("route") or "knowledge_base",
                contexts=state.get("contexts") or [],
                context_decision=(state.get("context_decision") or ContextDependencyDecision(False, "", 0.0)).to_dict(),
            )
        except Exception as exc:
            self.conversations.db.rollback()
            return {"errors": [*state.get("errors", []), f"persist_turn failed: {exc}"]}
        return {}

    @staticmethod
    def _decorate_query(query: str, user_id: str) -> str:
        return (
            f"用户问题：{query}\n"
            f"用户ID：{user_id}\n"
            f"当前时间：{datetime.now().isoformat(timespec='seconds')}"
        )

    @staticmethod
    def _answer_prompt(
        *,
        original_query: str,
        rewritten_query: str,
        route: str,
        context: str,
        errors: list[str],
    ) -> str:
        error_text = "\n".join(errors) if errors else "无"
        return f"""
你是汽车领域知识助手。请基于工具结果回答用户问题。

要求：
1. 优先使用工具结果，不要编造。
2. 如果工具结果不足，明确说明缺少依据。
3. 如果 route 是 web_search，说明依据来自外部网络搜索，并尽量保留关键 URL。
4. 回答简洁、清晰，必要时分步骤。
5. 工具结果中存在 [1]、[2] 等编号来源时，在对应事实后使用相同编号标注引用；不得引用不存在的编号。

用户原问题：
{original_query}

检索问题：
{rewritten_query}

已采用 route：
{route}

工具失败记录：
{error_text}

工具结果：
{context}
"""
