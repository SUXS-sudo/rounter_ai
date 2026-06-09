"""Sub-agent definitions for the multi-agent route-planning system.

Each sub-agent is a specialized LangGraph ReAct agent with its own tools
and system prompt. The Supervisor orchestrates these agents.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from core.agent_tools import (
    EXPLANATION_TOOLS,
    INTENT_TOOLS,
    PLANNING_TOOLS,
)

# ---------------------------------------------------------------------------
# Sub-agent system prompts
# ---------------------------------------------------------------------------

INTENT_AGENT_PROMPT = """你是意图分析专家。你的职责是：
1. 解析用户的中文出行需求，提取结构化意图（城市、时间、预算、偏好等）
2. 获取用户画像信息
3. 查询支持的城市信息

工作完成后，将解析结果以 JSON 格式返回给主管 Agent。
不要尝试规划路线或生成解释，那是其他专家的工作。"""

PLANNING_AGENT_PROMPT = """你是路线规划专家。你的职责是：
1. 根据意图检索候选 POI（兴趣点）
2. 规划 3 条差异化路线方案（综合最优、少排队优先、低预算优先）
3. 根据用户反馈重新规划路线

工作完成后，将路线方案以 JSON 格式返回给主管 Agent。
不要尝试解析用户意图或生成解释，那是其他专家的工作。"""

EXPLANATION_AGENT_PROMPT = """你是解释生成专家。你的职责是：
1. 为已生成的路线方案生成友好、自然的中文解释
2. 包含路线亮点、行程安排、预算分析、风险提示
3. 使用亲切自然的语气，适当使用 emoji

工作完成后，将解释文本返回给主管 Agent。
不要尝试解析意图或规划路线，那是其他专家的工作。"""


def _create_llm(timeout: float = 30.0):
    """Create the LLM instance based on settings."""
    from models.config import settings

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


def create_intent_agent():
    """Create the Intent sub-agent."""
    from langgraph.prebuilt import create_react_agent

    llm = _create_llm()
    return create_react_agent(
        llm,
        INTENT_TOOLS,
        prompt=SystemMessage(content=INTENT_AGENT_PROMPT),
        name="intent_agent",
    )


def create_planning_agent():
    """Create the Planning sub-agent."""
    from langgraph.prebuilt import create_react_agent

    llm = _create_llm()
    return create_react_agent(
        llm,
        PLANNING_TOOLS,
        prompt=SystemMessage(content=PLANNING_AGENT_PROMPT),
        name="planning_agent",
    )


def create_explanation_agent():
    """Create the Explanation sub-agent."""
    from langgraph.prebuilt import create_react_agent

    llm = _create_llm()
    return create_react_agent(
        llm,
        EXPLANATION_TOOLS,
        prompt=SystemMessage(content=EXPLANATION_AGENT_PROMPT),
        name="explanation_agent",
    )
