"""Supervisor multi-agent system for intelligent route planning.

Uses LangGraph StateGraph to orchestrate multiple specialized sub-agents:
- IntentAgent:     意图解析 + 用户画像
- PlanningAgent:   POI 检索 + 路线规划 + 重规划
- ExplanationAgent: 路线解释生成

A Supervisor LLM decides which agent to call next based on the current state.
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from models.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """你是路线规划系统的主管 Agent（Supervisor）。你的职责是协调以下专业 Agent 完成用户请求：

## 可用 Agent

1. **intent_agent** — 意图分析专家
   - 职责：解析用户需求、获取用户画像、查询城市信息
   - 何时调用：收到新用户请求时，必须首先调用

2. **planning_agent** — 路线规划专家
   - 职责：检索候选地点、规划路线方案、根据反馈重新规划
   - 何时调用：意图解析完成后，需要规划路线时

3. **explanation_agent** — 解释生成专家
   - 职责：为路线方案生成友好解释
   - 何时调用：路线规划完成后，需要生成最终解释时

## 工作流程

对于新路线规划请求：
intent_agent → planning_agent → explanation_agent → FINISH

对于重新规划请求：
planning_agent → explanation_agent → FINISH

## 规则
- 每次规划必须生成 3 条差异化路线
- 每条路线至少包含 3 个地点，其中必须有 1 个餐饮类
- 所有 Agent 工作完成后，必须调用 FINISH
- 回复用户时使用亲切自然的中文"""

SUPERVISOR_TOOL_PROMPT = """你是一个任务调度系统。根据当前对话状态，决定下一步应该由哪个 Agent 处理。

可选的 Agent：
- intent_agent: 意图分析专家，负责解析需求和获取用户画像
- planning_agent: 路线规划专家，负责检索POI和规划路线
- explanation_agent: 解释生成专家，负责生成路线解释
- FINISH: 所有工作完成，结束流程

请只返回一个 JSON：{"next": "agent_name"}"""


def _create_llm(timeout: float = 30.0):
    provider = settings.llm_provider.lower().strip()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.mimo_model,
            api_key=settings.mimo_api_key or None,
            base_url=settings.mimo_base_url or None,
            temperature=settings.mimo_temperature,
            max_tokens=settings.mimo_max_tokens,
            timeout=timeout,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.mimo_model,
            api_key=settings.mimo_api_key or None,
            base_url=settings.mimo_base_url or None,
            temperature=settings.mimo_temperature,
            max_tokens=settings.mimo_max_tokens,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Multi-agent graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str


def _build_multi_agent_graph():
    """Build the Supervisor multi-agent graph."""
    from core.sub_agents import (
        create_explanation_agent,
        create_intent_agent,
        create_planning_agent,
    )

    intent_agent = create_intent_agent()
    planning_agent = create_planning_agent()
    explanation_agent = create_explanation_agent()

    def supervisor_node(state: AgentState) -> dict:
        """Supervisor decides which agent to call next.

        Uses deterministic routing based on message history:
        - No intent yet  → intent_agent
        - Has intent but no routes → planning_agent
        - Has routes but no explanation → explanation_agent
        - Has both → FINISH
        """
        messages = state["messages"]

        has_intent = any(
            isinstance(m, ToolMessage) and getattr(m, "name", "") == "parse_intent"
            for m in messages
        )
        has_routes = any(
            isinstance(m, ToolMessage) and getattr(m, "name", "") == "plan_routes"
            for m in messages
        )
        has_explanation = any(
            isinstance(m, ToolMessage) and getattr(m, "name", "") == "explain_routes"
            for m in messages
        )

        if not has_intent:
            next_agent = "intent_agent"
        elif not has_routes:
            next_agent = "planning_agent"
        elif not has_explanation:
            next_agent = "explanation_agent"
        else:
            next_agent = "FINISH"

        logger.info("Supervisor routing: intent=%s routes=%s explanation=%s → %s",
                     has_intent, has_routes, has_explanation, next_agent)
        return {"next_agent": next_agent}

    def intent_node(state: AgentState) -> dict:
        """Intent agent processes the request."""
        messages = state["messages"]
        result = intent_agent.invoke({"messages": messages})
        # Return only new messages (skip the input messages)
        new_messages = result["messages"][len(messages):]
        return {"messages": new_messages}

    def planning_node(state: AgentState) -> dict:
        """Planning agent processes the request."""
        messages = state["messages"]
        result = planning_agent.invoke({"messages": messages})
        new_messages = result["messages"][len(messages):]
        return {"messages": new_messages}

    def explanation_node(state: AgentState) -> dict:
        """Explanation agent processes the request."""
        messages = state["messages"]
        result = explanation_agent.invoke({"messages": messages})
        new_messages = result["messages"][len(messages):]
        return {"messages": new_messages}

    def route_to_agent(state: AgentState) -> str:
        """Route to the next agent or FINISH."""
        next_agent = state.get("next_agent", "FINISH")
        if next_agent == "FINISH":
            return END
        return next_agent

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("intent_agent", intent_node)
    graph.add_node("planning_agent", planning_node)
    graph.add_node("explanation_agent", explanation_node)

    # Add edges: all agents return to supervisor
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent, {
        "intent_agent": "intent_agent",
        "planning_agent": "planning_agent",
        "explanation_agent": "explanation_agent",
        END: END,
    })
    graph.add_edge("intent_agent", "supervisor")
    graph.add_edge("planning_agent", "supervisor")
    graph.add_edge("explanation_agent", "supervisor")

    return graph.compile()


# Lazy-initialized singleton
_multi_agent = None


def get_multi_agent():
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = _build_multi_agent_graph()
    return _multi_agent


# ---------------------------------------------------------------------------
# Public API — used by FastAPI endpoints
# ---------------------------------------------------------------------------


def plan_with_agent(user_query: str, user_id: str = "u001") -> dict[str, Any]:
    """Run the multi-agent system to plan routes."""
    agent = get_multi_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=f"用户ID: {user_id}\n用户需求: {user_query}")],
        "next_agent": "",
    })
    parsed = _parse_multi_agent_result(result, user_query, user_id)

    # If agent flow returned no routes, call planning tools directly
    if not parsed.get("routes"):
        logger.warning("Agent returned no routes, calling planning tools directly")
        try:
            from core.agent_tools import plan_routes as _plan_routes_tool, parse_intent as _parse_intent_tool
            intent_str = _parse_intent_tool.invoke({"user_query": user_query})
            try:
                intent_data = json.loads(intent_str) if isinstance(intent_str, str) else intent_str
            except (json.JSONDecodeError, TypeError):
                intent_data = {}
            parsed["intent"] = intent_data
            routes_str = _plan_routes_tool.invoke({"intent_json": json.dumps(intent_data, ensure_ascii=False), "user_id": user_id})
            try:
                routes_data = json.loads(routes_str) if isinstance(routes_str, str) else routes_str
            except (json.JSONDecodeError, TypeError):
                routes_data = {}
            parsed["routes"] = routes_data.get("routes", []) if isinstance(routes_data, dict) else []
            parsed["meta"] = {**parsed.get("meta", {}), **(routes_data.get("meta", {}) if isinstance(routes_data, dict) else {})}
            if not parsed.get("explanation") and parsed["routes"]:
                from core.explanation import generate_explanation
                parsed["explanation"] = generate_explanation(parsed["routes"], intent_data)
        except Exception as exc:
            logger.error("Fallback planning failed: %s", exc, exc_info=True)

    return parsed


def replan_with_agent(
    previous_intent: dict[str, Any],
    feedback: str,
    user_id: str = "u001",
) -> dict[str, Any]:
    """Run the multi-agent system to replan routes."""
    agent = get_multi_agent()
    intent_json = json.dumps(previous_intent, ensure_ascii=False)
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            f"用户ID: {user_id}\n"
            f"之前的意图: {intent_json}\n"
            f"用户反馈: {feedback}\n"
            f"请根据反馈重新规划路线。"
        ))],
        "next_agent": "",
    })
    parsed = _parse_multi_agent_replan_result(result, previous_intent, feedback, user_id)

    # If agent flow returned no routes, call replan tool directly
    if not parsed.get("routes"):
        logger.warning("Agent replan returned no routes, calling replan tool directly")
        try:
            from core.agent_tools import replan_routes as _replan_routes_tool
            replan_str = _replan_routes_tool.invoke({
                "intent_json": intent_json, "feedback": feedback, "user_id": user_id,
            })
            try:
                replan_data = json.loads(replan_str) if isinstance(replan_str, str) else replan_str
            except (json.JSONDecodeError, TypeError):
                replan_data = {}
            parsed["routes"] = replan_data.get("routes", []) if isinstance(replan_data, dict) else []
            parsed["intent"] = replan_data.get("intent", previous_intent) if isinstance(replan_data, dict) else previous_intent
            parsed["changes"] = replan_data.get("changes", []) if isinstance(replan_data, dict) else []
            parsed["warnings"] = replan_data.get("warnings", []) if isinstance(replan_data, dict) else []
            if not parsed.get("explanation"):
                parsed["explanation"] = replan_data.get("explanation", "") if isinstance(replan_data, dict) else ""
        except Exception as exc:
            logger.error("Fallback replan failed: %s", exc, exc_info=True)

    return parsed


def plan_with_agent_stream(user_query: str, user_id: str = "u001"):
    """Generator that yields progress events for streaming."""
    import time

    t0 = time.time()

    yield {"event": "progress", "data": {"status": "Supervisor 正在协调多 Agent 规划路线..."}}
    agent = get_multi_agent()
    yield {"event": "progress", "data": {"status": "IntentAgent 正在解析需求..."}}

    result = agent.invoke({
        "messages": [HumanMessage(content=f"用户ID: {user_id}\n用户需求: {user_query}")],
        "next_agent": "",
    })

    parsed = _parse_multi_agent_result(result, user_query, user_id)

    logger.info("Plan result — intent keys: %s, routes count: %d, explanation len: %d",
                list(parsed.get("intent", {}).keys()), len(parsed.get("routes", [])), len(parsed.get("explanation", "")))

    # If agent flow returned no routes, call planning tools directly
    if not parsed.get("routes"):
        logger.warning("Agent returned no routes, calling planning tools directly")
        yield {"event": "progress", "data": {"status": "Agent 未返回路线，直接调用规划工具..."}}
        try:
            from core.agent_tools import plan_routes as _plan_routes_tool, parse_intent as _parse_intent_tool
            intent_str = _parse_intent_tool.invoke({"user_query": user_query})
            logger.info("Fallback parse_intent result type: %s", type(intent_str).__name__)
            try:
                intent_data = json.loads(intent_str) if isinstance(intent_str, str) else intent_str
            except (json.JSONDecodeError, TypeError):
                intent_data = {}
            parsed["intent"] = intent_data

            routes_str = _plan_routes_tool.invoke({"intent_json": json.dumps(intent_data, ensure_ascii=False), "user_id": user_id})
            logger.info("Fallback plan_routes result type: %s, len: %d", type(routes_str).__name__, len(str(routes_str)))
            try:
                routes_data = json.loads(routes_str) if isinstance(routes_str, str) else routes_str
            except (json.JSONDecodeError, TypeError):
                routes_data = {}
            parsed["routes"] = routes_data.get("routes", []) if isinstance(routes_data, dict) else []
            parsed["meta"] = {**parsed.get("meta", {}), **(routes_data.get("meta", {}) if isinstance(routes_data, dict) else {})}

            if not parsed.get("explanation"):
                # plan_routes tool returns explanation at top level
                parsed["explanation"] = routes_data.get("explanation", "") if isinstance(routes_data, dict) else ""
        except Exception as exc:
            logger.error("Fallback planning failed: %s", exc, exc_info=True)

    yield {"event": "intent", "data": parsed.get("intent", {})}
    yield {"event": "routes", "data": parsed.get("routes", [])}
    yield {"event": "explanation_chunk", "data": {"text": parsed.get("explanation", "")}}

    elapsed = round(time.time() - t0, 2)
    yield {"event": "done", "data": {"elapsed": elapsed, "meta": parsed.get("meta", {})}}


def replan_with_agent_stream(
    previous_intent: dict[str, Any],
    feedback: str,
    user_id: str = "u001",
):
    """Generator for streaming replan with multi-agent."""
    import time

    t0 = time.time()

    yield {"event": "progress", "data": {"status": "Supervisor 正在协调多 Agent 重新规划..."}}
    agent = get_multi_agent()
    intent_json = json.dumps(previous_intent, ensure_ascii=False)
    yield {"event": "progress", "data": {"status": "PlanningAgent 正在重新规划路线..."}}

    result = agent.invoke({
        "messages": [HumanMessage(content=(
            f"用户ID: {user_id}\n"
            f"之前的意图: {intent_json}\n"
            f"用户反馈: {feedback}\n"
            f"请根据反馈重新规划路线。"
        ))],
        "next_agent": "",
    })

    parsed = _parse_multi_agent_replan_result(result, previous_intent, feedback, user_id)

    # If agent flow returned no routes, call replan tool directly
    if not parsed.get("routes"):
        logger.warning("Agent replan returned no routes, calling replan tool directly")
        yield {"event": "progress", "data": {"status": "Agent 未返回路线，直接调用重规划工具..."}}
        try:
            from core.agent_tools import replan_routes as _replan_routes_tool
            replan_str = _replan_routes_tool.invoke({
                "intent_json": intent_json, "feedback": feedback, "user_id": user_id,
            })
            try:
                replan_data = json.loads(replan_str) if isinstance(replan_str, str) else replan_str
            except (json.JSONDecodeError, TypeError):
                replan_data = {}
            parsed["routes"] = replan_data.get("routes", []) if isinstance(replan_data, dict) else []
            parsed["intent"] = replan_data.get("intent", previous_intent) if isinstance(replan_data, dict) else previous_intent
            parsed["changes"] = replan_data.get("changes", []) if isinstance(replan_data, dict) else []
            parsed["warnings"] = replan_data.get("warnings", []) if isinstance(replan_data, dict) else []
            if not parsed.get("explanation"):
                parsed["explanation"] = replan_data.get("explanation", "") if isinstance(replan_data, dict) else ""
        except Exception as exc:
            logger.error("Fallback replan failed: %s", exc, exc_info=True)

    yield {"event": "intent", "data": parsed.get("intent", {})}
    yield {"event": "routes", "data": parsed.get("routes", [])}
    if parsed.get("changes"):
        yield {"event": "changes", "data": parsed["changes"]}
    yield {"event": "explanation_chunk", "data": {"text": parsed.get("explanation", "")}}

    elapsed = round(time.time() - t0, 2)
    yield {"event": "done", "data": {"elapsed": elapsed}}


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _extract_tool_results(messages: list) -> dict[str, Any]:
    results: dict[str, Any] = {}
    tool_names_seen = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = msg.name or ""
            tool_names_seen.append(tool_name)
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except (json.JSONDecodeError, TypeError):
                data = None
            if tool_name == "parse_intent" and isinstance(data, dict):
                results["intent"] = data
            elif tool_name == "get_user_profile" and isinstance(data, dict):
                results["user_profile"] = data
            elif tool_name == "plan_routes" and isinstance(data, dict):
                results["routes"] = data.get("routes", [])
                results["plan_meta"] = data.get("meta", {})
                if data.get("explanation"):
                    results["explanation"] = data["explanation"]
            elif tool_name == "replan_routes" and isinstance(data, dict):
                results["routes"] = data.get("routes", [])
                results["intent"] = data.get("intent", results.get("intent", {}))
                results["replan_explanation"] = data.get("explanation", "")
                results["changes"] = data.get("changes", [])
                results["warnings"] = data.get("warnings", [])
            elif tool_name == "explain_routes" and isinstance(data, dict):
                results["explanation"] = data.get("explanation", "")
    logger.info("Tool messages seen: %s, extracted keys: %s", tool_names_seen, list(results.keys()))
    return results


def _get_final_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return ""


def _parse_multi_agent_result(result: dict[str, Any], user_query: str, user_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    tool_results = _extract_tool_results(messages)
    final_answer = _get_final_answer(messages)

    intent = tool_results.get("intent", {})
    routes = tool_results.get("routes", [])
    explanation = tool_results.get("explanation", "") or final_answer
    meta = {"user_id": user_id, "agent_used": True, "architecture": "multi_agent_supervisor"}
    if "plan_meta" in tool_results:
        meta.update(tool_results["plan_meta"])

    return {
        "intent": intent,
        "routes": routes,
        "explanation": explanation,
        "meta": meta,
    }


def _parse_multi_agent_replan_result(
    result: dict[str, Any],
    previous_intent: dict[str, Any],
    feedback: str,
    user_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    tool_results = _extract_tool_results(messages)
    final_answer = _get_final_answer(messages)

    intent = tool_results.get("intent", previous_intent)
    routes = tool_results.get("routes", [])
    explanation = tool_results.get("replan_explanation", "") or tool_results.get("explanation", "") or final_answer
    changes = tool_results.get("changes", [])
    warnings = tool_results.get("warnings", [])
    meta = {"user_id": user_id, "agent_used": True, "architecture": "multi_agent_supervisor"}

    return {
        "intent": intent,
        "routes": routes,
        "explanation": explanation,
        "changes": changes,
        "warnings": warnings,
        "meta": meta,
    }
