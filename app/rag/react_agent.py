from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.json_utils import fix_json
from app.rag.context_compressor import CompressionState, ContextCompressor
from app.rag.query_router import Strategy
from app.rag.tools import OnlineToolExecutor, RagToolResult
from app.schemas import RetrievedDocument


REACT_TOOL_STRATEGIES: tuple[Strategy, ...] = ("knowledge_base", "relational_db", "graph_db", "web_search")


@dataclass
class ReactRunResult:
    content: str
    contexts: list[RetrievedDocument] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReactAgent:
    def __init__(
        self,
        tools: OnlineToolExecutor,
        *,
        llm=None,
        settings: Settings | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.tools = tools
        self.settings = settings or get_settings()
        if llm is None:
            from app.core.llm import get_chat_model

            llm = get_chat_model(temperature=0)
        self.llm = llm
        self.compressor = compressor or ContextCompressor(
            max_total_chars=self.settings.react_context_max_chars,
            max_item_chars=self.settings.react_observation_max_chars,
            max_context_tokens=getattr(self.settings, "react_context_max_tokens", 6_000),
            trigger_ratio=getattr(self.settings, "react_context_trigger_ratio", 0.8),
            reserve_ratio=getattr(self.settings, "react_context_reserve_ratio", 0.35),
            tool_result_max_tokens=getattr(self.settings, "react_tool_result_max_tokens", 1_200),
            summary_max_tokens=getattr(self.settings, "react_summary_max_tokens", 1_000),
            summary_model=llm if getattr(self.settings, "react_context_llm_summary_enabled", True) else None,
            token_counter=getattr(llm, "get_num_tokens", None),
        )

    def run(
        self,
        query: str,
        *,
        top_k: int,
        document_id: int | None = None,
        prior_errors: list[str] | None = None,
        query_plan: list[dict[str, Any]] | None = None,
    ) -> RagToolResult:
        allowed_tools = self._allowed_tools(document_id=document_id)
        observations: list[dict[str, Any]] = []
        contexts: list[RetrievedDocument] = []
        final_answer = ""
        compression_state = CompressionState()

        for step in range(1, self.settings.react_agent_max_steps + 1):
            action = self._plan_next_action(
                query=query,
                step=step,
                allowed_tools=allowed_tools,
                observations=observations,
                prior_errors=prior_errors or [],
                compression_state=compression_state,
                query_plan=query_plan or [],
            )
            thought = str(action.get("thought") or "").strip()
            action_name = self._normalize_action(action.get("action"), allowed_tools)
            action_input = str(action.get("action_input") or query).strip() or query

            if action_name == "final_answer":
                final_answer = str(action.get("final_answer") or "").strip()
                observations.append(
                    {
                        "step": step,
                        "thought": thought,
                        "action": "final_answer",
                        "action_input": action_input,
                        "ok": True,
                        "observation": final_answer,
                    }
                )
                break

            try:
                result = self.tools.run(action_name, action_input, top_k=top_k, document_id=document_id)
                observation_text = self.compressor.compress_tool_content(
                    tool_name=result.tool_name,
                    content=result.content,
                    metadata=result.metadata,
                    state=compression_state,
                )
                contexts.extend(result.contexts)
                observations.append(
                    {
                        "step": step,
                        "thought": thought,
                        "action": action_name,
                        "action_input": action_input,
                        "ok": True,
                        "observation": observation_text,
                        "metadata": result.metadata,
                    }
                )
            except Exception as exc:
                observations.append(
                    {
                        "step": step,
                        "thought": thought,
                        "action": action_name,
                        "action_input": action_input,
                        "ok": False,
                        "error": str(exc),
                        "observation": "",
                    }
                )

        compressed_observations = self.compressor.compress_observations(observations, compression_state)
        if not final_answer:
            final_answer = self._finalize_from_observations(query, compressed_observations)

        content = f"ReAct final answer draft:\n{final_answer}\n\nReAct observations:\n{compressed_observations}"
        return RagToolResult(
            strategy="react_agent",
            tool_name="react_agent",
            content=content,
            contexts=contexts,
            metadata={
                "max_steps": self.settings.react_agent_max_steps,
                "steps_used": len(observations),
                "allowed_tools": list(allowed_tools),
                "query_plan": query_plan or [],
                "finished_with_final_action": any(item.get("action") == "final_answer" for item in observations),
                "trace": observations,
                "compression": {
                    "summary": compression_state.summary,
                    "summarized_steps": sorted(compression_state.summarized_steps),
                    "compression_count": compression_state.compression_count,
                    "offloaded_tool_result_keys": sorted(compression_state.offloaded_tool_results),
                    "last_report": compression_state.report(),
                },
            },
        )

    def _plan_next_action(
        self,
        *,
        query: str,
        step: int,
        allowed_tools: tuple[Strategy, ...],
        observations: list[dict[str, Any]],
        prior_errors: list[str],
        compression_state: CompressionState,
        query_plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self._action_prompt(
            query=query,
            step=step,
            allowed_tools=allowed_tools,
            observations=self.compressor.compress_observations(observations, compression_state),
            prior_errors=prior_errors,
            query_plan=query_plan,
        )
        response = self.llm.invoke(prompt).content
        return self._parse_action(str(response), allowed_tools)

    def _finalize_from_observations(self, query: str, observations: str) -> str:
        prompt = f"""
你是 ReAct 代理的最终整理器。请只基于下面的工具观察结果，给出一个简洁的答案草稿。
如果证据不足，请明确说明缺少依据。不要编造。

用户问题：
{query}

工具观察结果：
{observations}
"""
        try:
            return str(self.llm.invoke(prompt).content).strip()
        except Exception:
            return observations

    def _allowed_tools(self, *, document_id: int | None) -> tuple[Strategy, ...]:
        if document_id is not None:
            return ("knowledge_base",)
        return REACT_TOOL_STRATEGIES

    def _parse_action(self, response: str, allowed_tools: tuple[Strategy, ...]) -> dict[str, Any]:
        try:
            payload = json.loads(fix_json(response))
        except Exception:
            return self._default_action(allowed_tools)

        if not isinstance(payload, dict):
            return self._default_action(allowed_tools)

        payload["action"] = self._normalize_action(payload.get("action"), allowed_tools)
        return payload

    @staticmethod
    def _normalize_action(value: object, allowed_tools: tuple[Strategy, ...]) -> Strategy | str:
        action = str(value or "").strip().lower()
        aliases = {
            "hybrid_retrieval": "knowledge_base",
            "hybrid": "knowledge_base",
            "kb": "knowledge_base",
            "document": "knowledge_base",
            "sql": "relational_db",
            "text2sql": "relational_db",
            "mysql": "relational_db",
            "cypher": "graph_db",
            "text2cypher": "graph_db",
            "neo4j": "graph_db",
            "graph": "graph_db",
            "tavily": "web_search",
            "web": "web_search",
            "search": "web_search",
            "final": "final_answer",
            "finish": "final_answer",
            "answer": "final_answer",
        }
        action = aliases.get(action, action)
        if action == "final_answer":
            return action
        if action in allowed_tools:
            return action  # type: ignore[return-value]
        return allowed_tools[0]

    @staticmethod
    def _default_action(allowed_tools: tuple[Strategy, ...]) -> dict[str, Any]:
        return {
            "thought": "模型没有返回合法动作，先使用最保守的可用工具。",
            "action": allowed_tools[0],
            "action_input": "",
            "final_answer": "",
        }

    @staticmethod
    def _action_prompt(
        *,
        query: str,
        step: int,
        allowed_tools: tuple[Strategy, ...],
        observations: str,
        prior_errors: list[str],
        query_plan: list[dict[str, Any]],
    ) -> str:
        prior_error_text = "\n".join(prior_errors) if prior_errors else "无"
        query_plan_text = json.dumps(query_plan, ensure_ascii=False) if query_plan else "无预先拆分计划"
        return f"""
你是一个手搓 ReAct 多工具代理。你需要通过 Thought -> Action -> Observation 的方式逐步解决用户问题。

当前是第 {step} 步。你只能从 allowed_tools 中选择一个工具，或者在证据足够时选择 final_answer。

allowed_tools:
{", ".join(allowed_tools)}

工具说明：
- knowledge_base：查询内部文档、PDF、Markdown、技术手册、故障排查知识。
- relational_db：查询 MySQL 结构化数据，例如用户车辆、订单、保险、保养、统计字段。
- graph_db：查询 Neo4j 图谱关系，例如部件关系、路径、影响链、层级网络。
- web_search：查询公开网络信息、实时信息、新闻、政策和外部资料。

约束：
1. 只输出 JSON，不要 Markdown。
2. action 必须是 allowed_tools 之一，或 final_answer。
3. 如果已有观察结果足够回答，选择 final_answer。
4. 不要重复调用已经失败且没有新价值的工具。
5. action_input 应是给该工具的独立检索问题。
6. 如果提供了 query_plan，优先按依赖顺序完成其中尚未获得证据的步骤；依赖未满足时不要提前执行后续步骤。
7. query_plan 是初始计划，可以根据工具观察结果调整，但不得遗漏用户要求的目标。

query_plan:
{query_plan_text}

prior_errors:
{prior_error_text}

observations:
{observations}

用户问题：
{query}

输出格式：
{{
  "thought": "你为什么选择下一步",
  "action": "knowledge_base",
  "action_input": "给工具的检索问题",
  "final_answer": ""
}}
"""
