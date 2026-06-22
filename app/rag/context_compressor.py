from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.core.json_utils import fix_json


SUMMARY_FIELDS = (
    "task_overview",
    "completed_actions",
    "key_findings",
    "failures_and_constraints",
    "next_step",
    "context_to_preserve",
)


@dataclass(frozen=True)
class CompressionReport:
    triggered: bool
    before_tokens: int
    after_tokens: int
    summarized_steps: int
    reserved_steps: int
    compression_count: int
    offloaded_tool_results: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompressionState:
    summary: str = ""
    summarized_steps: set[str] = field(default_factory=set)
    offloaded_tool_results: dict[str, str] = field(default_factory=dict)
    compression_count: int = 0
    last_report: CompressionReport | None = None

    def report(self) -> dict[str, Any]:
        return self.last_report.to_dict() if self.last_report else {}


class ContextCompressor:
    """AgentScope-inspired context manager for the hand-rolled ReAct loop."""

    def __init__(
        self,
        *,
        max_total_chars: int = 12_000,
        max_item_chars: int = 2_500,
        max_context_tokens: int = 6_000,
        trigger_ratio: float = 0.8,
        reserve_ratio: float = 0.35,
        tool_result_max_tokens: int = 1_200,
        summary_max_tokens: int = 1_000,
        summary_model: Any | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if not 0 < reserve_ratio < trigger_ratio < 1:
            raise ValueError("reserve_ratio must be smaller than trigger_ratio and both must be between 0 and 1")
        self.max_total_chars = max(200, max_total_chars)
        self.max_item_chars = max(120, max_item_chars)
        self.max_context_tokens = max_context_tokens
        self.trigger_ratio = trigger_ratio
        self.reserve_ratio = reserve_ratio
        self.tool_result_max_tokens = max(64, tool_result_max_tokens)
        self.summary_max_tokens = max(128, summary_max_tokens)
        self.summary_model = summary_model
        self.token_counter = token_counter or self.estimate_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        value = str(text or "")
        if not value:
            return 0
        cjk_count = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", value))
        non_cjk_count = max(0, len(value) - cjk_count)
        return max(1, cjk_count + math.ceil(non_cjk_count / 4))

    def truncate_text(self, text: str, limit: int | None = None) -> str:
        limit = limit or self.max_item_chars
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        if limit <= 120:
            return value[:limit]

        head_size = int(limit * 0.7)
        tail_size = max(40, limit - head_size - 80)
        omitted = len(value) - head_size - tail_size
        return f"{value[:head_size]}\n...[compressed {omitted} chars]...\n{value[-tail_size:]}"

    def compress_tool_content(
        self,
        *,
        tool_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        state: CompressionState | None = None,
    ) -> str:
        value = str(content or "").strip()
        original_tokens = self.token_counter(value)
        over_limit = original_tokens > self.tool_result_max_tokens or len(value) > self.max_item_chars
        offload_key = ""
        if over_limit:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            offload_key = f"tool-result:{tool_name}:{digest}"
            if state is not None:
                state.offloaded_tool_results[offload_key] = value
            value = self._truncate_to_budget(
                value,
                token_limit=self.tool_result_max_tokens,
                char_limit=self.max_item_chars,
            )

        metadata_text = self._format_metadata(metadata or {})
        parts = [f"Tool: {tool_name}"]
        if metadata_text:
            parts.append(f"Metadata: {metadata_text}")
        if offload_key:
            parts.append(
                f"Compression: kept a bounded excerpt from {original_tokens} tokens; offload_key={offload_key}"
            )
        parts.append(f"Observation:\n{value}")
        return "\n".join(parts)

    def compress_observations(
        self,
        observations: list[dict[str, Any]],
        state: CompressionState | None = None,
    ) -> str:
        state = state or CompressionState()
        if not observations:
            return state.summary or "No observations yet."

        active = [
            (self._observation_key(item), item, self._format_observation(item))
            for item in observations
            if self._observation_key(item) not in state.summarized_steps
        ]
        rendered = self._render_context(state.summary, [item[2] for item in active])
        before_tokens = self.token_counter(rendered)
        trigger_tokens = int(self.max_context_tokens * self.trigger_ratio)

        if before_tokens <= trigger_tokens and len(rendered) <= self.max_total_chars:
            state.last_report = CompressionReport(
                triggered=False,
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                summarized_steps=0,
                reserved_steps=len(active),
                compression_count=state.compression_count,
                offloaded_tool_results=len(state.offloaded_tool_results),
            )
            return rendered

        to_compress, to_reserve = self._split_for_compression(active)
        if to_compress:
            state.summary = self._summarize(state.summary, [item[1] for item in to_compress])
            state.summary = self._truncate_to_budget(
                state.summary,
                token_limit=self.summary_max_tokens,
                char_limit=max(600, self.max_total_chars // 2),
            )
            state.summarized_steps.update(item[0] for item in to_compress)
            state.compression_count += 1

        reserved_texts = [item[2] for item in to_reserve]
        result = self._fit_context(state.summary, reserved_texts)
        after_tokens = self.token_counter(result)
        state.last_report = CompressionReport(
            triggered=True,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summarized_steps=len(to_compress),
            reserved_steps=len(to_reserve),
            compression_count=state.compression_count,
            offloaded_tool_results=len(state.offloaded_tool_results),
        )
        return result

    def _split_for_compression(
        self,
        active: list[tuple[str, dict[str, Any], str]],
    ) -> tuple[list[tuple[str, dict[str, Any], str]], list[tuple[str, dict[str, Any], str]]]:
        if not active:
            return [], []
        reserve_budget = max(64, int(self.max_context_tokens * self.reserve_ratio))
        reserved: list[tuple[str, dict[str, Any], str]] = []
        used_tokens = 0
        split_index = len(active)

        for index in range(len(active) - 1, -1, -1):
            key, observation, text = active[index]
            item_tokens = self.token_counter(text)
            if not reserved and item_tokens > reserve_budget:
                text = self._truncate_to_budget(text, reserve_budget, self.max_item_chars)
                reserved.append((key, observation, text))
                split_index = index
                break
            if reserved and used_tokens + item_tokens > reserve_budget:
                split_index = index + 1
                break
            reserved.append((key, observation, text))
            used_tokens += item_tokens
            split_index = index

        reserved.reverse()
        to_compress = active[:split_index]
        if not to_compress and len(active) > 1:
            to_compress = active[:1]
            reserved = active[1:]
        return to_compress, reserved

    def _summarize(self, previous_summary: str, observations: list[dict[str, Any]]) -> str:
        if self.summary_model is not None:
            try:
                response = self.summary_model.invoke(self._summary_prompt(previous_summary, observations)).content
                payload = json.loads(fix_json(str(response)))
                if isinstance(payload, dict):
                    return self._render_summary(payload)
            except Exception:
                pass
        return self._fallback_summary(previous_summary, observations)

    def _fit_context(self, summary: str, reserved_texts: list[str]) -> str:
        selected = list(reserved_texts)
        rendered = self._render_context(summary, selected)
        while len(selected) > 1 and (
            self.token_counter(rendered) > self.max_context_tokens or len(rendered) > self.max_total_chars
        ):
            selected.pop(0)
            rendered = self._render_context(summary, selected)
        if self.token_counter(rendered) <= self.max_context_tokens and len(rendered) <= self.max_total_chars:
            return rendered
        return self._truncate_to_budget(rendered, self.max_context_tokens, self.max_total_chars)

    def _truncate_to_budget(self, text: str, token_limit: int, char_limit: int) -> str:
        value = str(text or "").strip()
        tokens = self.token_counter(value)
        if tokens <= token_limit and len(value) <= char_limit:
            return value
        ratio = min(1.0, token_limit / max(1, tokens))
        target_chars = max(80, min(char_limit, int(len(value) * ratio * 0.9)))
        if target_chars >= len(value):
            target_chars = min(char_limit, len(value))
        return self.truncate_text(value, target_chars)

    def _format_observation(self, observation: dict[str, Any]) -> str:
        step = observation.get("step", "?")
        action = observation.get("action") or ""
        action_input = observation.get("action_input") or ""
        ok = bool(observation.get("ok"))
        content = str(observation.get("observation") or "").strip()
        error = observation.get("error")
        metadata = self._format_metadata(observation.get("metadata") or {})
        status = "ok" if ok else "failed"

        parts = [f"Step {step} [{status}]", f"Action: {action}", f"Action Input: {action_input}"]
        if metadata:
            parts.append(f"Metadata: {metadata}")
        if error:
            parts.append(f"Error: {error}")
        if content:
            parts.append(f"Observation:\n{content}")
        return "\n".join(parts)

    @staticmethod
    def _observation_key(observation: dict[str, Any]) -> str:
        return str(observation.get("step", "?"))

    @staticmethod
    def _render_context(summary: str, observations: list[str]) -> str:
        parts = []
        if summary:
            parts.append(summary)
        if observations:
            parts.append("<recent-observations>\n" + "\n\n".join(observations) + "\n</recent-observations>")
        return "\n\n".join(parts) or "No observations yet."

    @staticmethod
    def _summary_prompt(previous_summary: str, observations: list[dict[str, Any]]) -> str:
        return f"""
你是 ReAct Agent 的上下文压缩器。请把较早的工具调用记录合并为可继续执行任务的结构化摘要。

约束：
1. 只保留事实、工具动作、关键结果、失败原因、约束和下一步，不保留冗余推理过程。
2. 不得编造工具没有返回的信息。
3. previous_summary 非空时必须增量合并，不能丢失仍然有效的约束。
4. 严格只输出 JSON。

JSON 字段：
{{
  "task_overview": "任务目标与成功标准",
  "completed_actions": "已经执行的工具动作",
  "key_findings": "可用于后续回答的关键证据",
  "failures_and_constraints": "失败、限制和不能重复的无效尝试",
  "next_step": "接下来最合理的动作",
  "context_to_preserve": "必须持续保留的实体、参数、用户约束和引用"
}}

previous_summary:
{previous_summary or "无"}

observations:
{json.dumps(observations, ensure_ascii=False, default=str)}
"""

    @staticmethod
    def _render_summary(payload: dict[str, Any]) -> str:
        values = {field: str(payload.get(field) or "无").strip() for field in SUMMARY_FIELDS}
        return (
            "<compressed-context>\n"
            f"# Task Overview\n{values['task_overview']}\n\n"
            f"# Completed Actions\n{values['completed_actions']}\n\n"
            f"# Key Findings\n{values['key_findings']}\n\n"
            f"# Failures and Constraints\n{values['failures_and_constraints']}\n\n"
            f"# Next Step\n{values['next_step']}\n\n"
            f"# Context to Preserve\n{values['context_to_preserve']}\n"
            "</compressed-context>"
        )

    def _fallback_summary(self, previous_summary: str, observations: list[dict[str, Any]]) -> str:
        actions = []
        findings = []
        failures = []
        preserved = []
        for item in observations:
            step = item.get("step", "?")
            action = str(item.get("action") or "unknown")
            action_input = str(item.get("action_input") or "")
            actions.append(f"step {step}: {action}({action_input})")
            if item.get("ok") and item.get("observation"):
                findings.append(self._truncate_to_budget(str(item["observation"]), 180, 720))
            if not item.get("ok"):
                failures.append(f"step {step}: {item.get('error') or 'tool failed'}")
            if item.get("metadata"):
                preserved.append(self._format_metadata(item["metadata"]))

        payload = {
            "task_overview": previous_summary or "Continue solving the current user request with available tools.",
            "completed_actions": "; ".join(actions) or "无",
            "key_findings": "\n".join(findings) or "无",
            "failures_and_constraints": "; ".join(failures) or "无",
            "next_step": "Use the retained recent observations to choose the next non-redundant tool action.",
            "context_to_preserve": "; ".join(preserved) or "无",
        }
        return self._render_summary(payload)

    @staticmethod
    def _format_metadata(metadata: dict[str, Any]) -> str:
        if not metadata:
            return ""
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
