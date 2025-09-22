from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agents import analyst, copywriter, finance, ideator, technician

AgentHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class AgentConfig:
    """Описание агентской роли, доступной в оркестрации."""

    id: str
    title: str
    owner: str
    description: str
    handler: AgentHandler
    result_key: str
    default_selected: bool = True
    default_in_summary: bool = True


_AGENT_CONFIGS: List[AgentConfig] = [
    AgentConfig(
        id="analyst_icp",
        title="Analyst (ICP)",
        owner="Analyst",
        description="Researches the market and forms a portrait of the target audience.",
        handler=analyst.icp,
        result_key="icp",
    ),
    AgentConfig(
        id="ideator_concepts",
        title="Ideator",
        owner="Ideator",
        description="Prepares creative concepts for the campaign and messages for communications.",
        handler=ideator.concepts,
        result_key="concepts",
    ),
    AgentConfig(
        id="copywriter_texts",
        title="Copywriter",
        owner="Copywriter",
        description="Creates texts for promotional materials taking into account the selected concept.",
        handler=copywriter.texts,
        result_key="copy",
    ),
    AgentConfig(
        id="finance_assessment",
        title="Finance Director",
        owner="Finance",
        description="Evaluates the budget, risks, and growth opportunities of the project.",
        handler=finance.assessment,
        result_key="finance",
    ),
    AgentConfig(
        id="technician_blueprint",
        title="Technical Director",
        owner="Technician",
        description="Develops the architecture and technical implementation plan.",
        handler=technician.blueprint,
        result_key="tech_plan",
    ),
]

AGENT_REGISTRY: Dict[str, AgentConfig] = {cfg.id: cfg for cfg in _AGENT_CONFIGS}
DEFAULT_AGENT_SEQUENCE: List[str] = [cfg.id for cfg in _AGENT_CONFIGS if cfg.default_selected]
DEFAULT_SUMMARY_AGENTS: List[str] = [
    cfg.id for cfg in _AGENT_CONFIGS if cfg.default_in_summary
]


__all__ = [
    "AgentConfig",
    "AgentHandler",
    "AGENT_REGISTRY",
    "DEFAULT_AGENT_SEQUENCE",
    "DEFAULT_SUMMARY_AGENTS",
]
