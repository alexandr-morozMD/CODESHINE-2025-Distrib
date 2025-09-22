# from __future__ import annotations

# from copy import deepcopy
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List

# from jinja2 import Environment, FileSystemLoader, select_autoescape

# from agents._llm import DEFAULT_MODEL, chat
# from memory.rag import get_context, save_artifact

# BASE_DIR = Path(__file__).resolve().parent.parent
# PROMPT_PATH = BASE_DIR / "prompts" / "manager.md"
# TEMPLATE_ENV = Environment(
#     loader=FileSystemLoader(str(BASE_DIR / "templates")),
#     autoescape=select_autoescape(enabled_extensions=("md", "j2"), default=False),
# )
# SUMMARY_TEMPLATE = TEMPLATE_ENV.get_template("campaign_summary.md.j2")
# PROMPT_TEXT = PROMPT_PATH.read_text(encoding="utf-8")


# def _append_artifact(state: Dict[str, Any], artifact_path: Path) -> List[str]:
#     artifacts = list(state.get("artifacts", []))
#     artifacts.append(str(artifact_path))
#     return artifacts


# def plan(state: Dict[str, Any]) -> Dict[str, Any]:
#     brief: str = state["brief"]
#     run_id: str = state["run_id"]
#     model: str = state.get("model", DEFAULT_MODEL)

#     context = get_context(run_id, k=5)
#     plan_markdown = chat(
#         [
#             {"role": "system", "content": PROMPT_TEXT},
#             {
#                 "role": "user",
#                 "content": (
#                     "Тебе нужно превратить бриф в пошаговый план."
#                     "\nИспользуй формат с колонками Backlog/Doing/Done"
#                     " и добавь критерии приёмки."
#                     f"\nБриф:\n{brief}\n\nКонтекст:\n{context}"
#                 ),
#             },
#         ],
#         model=model,
#     )

#     artifact = save_artifact(run_id, "manager_plan.md", plan_markdown)

#     board = {
#         "manager_plan": {
#             "title": "Планирование",
#             "owner": "Manager",
#             "status": "Done",
#             "notes": plan_markdown,
#         },
#         "analyst_icp": {
#             "title": "ICP и аналитика",
#             "owner": "Analyst",
#             "status": "Backlog",
#         },
#         "ideator_concepts": {
#             "title": "Креативные концепции",
#             "owner": "Ideator",
#             "status": "Backlog",
#         },
#         "copywriter_texts": {
#             "title": "Копирайтинг",
#             "owner": "Copywriter",
#             "status": "Backlog",
#         },
#         "manager_summary": {
#             "title": "Финальная сборка",
#             "owner": "Manager",
#             "status": "Backlog",
#         },
#     }

#     return {
#         "plan": plan_markdown,
#         "board": board,
#         "artifacts": _append_artifact(state, artifact),
#     }


# def _update_board(state: Dict[str, Any], task_id: str, *, status: str, notes: str) -> Dict[str, Any]:
#     board = deepcopy(state.get("board", {}))
#     board.setdefault(task_id, {})
#     board[task_id].update({"status": status, "notes": notes})
#     return board


# def assemble(state: Dict[str, Any]) -> Dict[str, Any]:
#     run_id: str = state["run_id"]
#     model: str = state.get("model", DEFAULT_MODEL)

#     icp = state.get("icp", "")
#     concepts = state.get("concepts", "")
#     copy = state.get("copy", "")
#     plan_md = state.get("plan", "")
#     brief = state["brief"]

#     validation_prompt = chat(
#         [
#             {"role": "system", "content": PROMPT_TEXT},
#             {
#                 "role": "user",
#                 "content": (
#                     "Проверь, что у нас есть все артефакты: ICP, концепции, тексты."
#                     " Если чего-то не хватает, перечисли проблемы."
#                     " Затем сформируй чек-лист из 5 задач внедрения."
#                     f"\nICP:\n{icp}\n\nКонцепции:\n{concepts}\n\nТексты:\n{copy}"
#                 ),
#             },
#         ],
#         model=model,
#     )

#     todo_items: List[Dict[str, Any]] = []
#     for line in validation_prompt.splitlines():
#         stripped = line.strip()
#         if stripped.startswith(("- ", "* ", "• ")):
#             text = stripped[2:].strip()
#             if text:
#                 todo_items.append({"text": text, "done": False})

#     if not todo_items:
#         todo_items = [
#             {"text": "Уточнить детали кампании с клиентом", "done": False},
#             {"text": "Подготовить визуальные материалы", "done": False},
#             {"text": "Сверстать лендинг и посадочные страницы", "done": False},
#             {"text": "Запланировать публикации", "done": False},
#             {"text": "Настроить аналитику и отчётность", "done": False},
#         ]

#     summary_markdown = SUMMARY_TEMPLATE.render(
#         project_title=state.get("project_title", "Маркетинговая кампания"),
#         brief=brief,
#         plan_markdown=plan_md,
#         icp_markdown=icp,
#         concepts_markdown=concepts,
#         copy_markdown=copy,
#         todo_items=todo_items,
#         timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
#     )

#     artifact = save_artifact(run_id, "campaign_summary.md", summary_markdown)

#     board = _update_board(state, "manager_summary", status="Done", notes=summary_markdown)

#     return {
#         "summary": summary_markdown,
#         "artifacts": _append_artifact(state, artifact),
#         "board": board,
#         "todo": todo_items,
#     }


# __all__ = ["plan", "assemble"]



from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agents._llm import chat
from agents.registry import AGENT_REGISTRY, DEFAULT_AGENT_SEQUENCE
from memory.rag import get_context, save_artifact

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "manager.md"
TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=select_autoescape(enabled_extensions=("md", "j2"), default=False),
)
SUMMARY_TEMPLATE = TEMPLATE_ENV.get_template("campaign_summary.md.j2")
PROMPT_TEXT = PROMPT_PATH.read_text(encoding="utf-8")


def _append_artifact(state: Dict[str, Any], artifact_path: Path) -> List[str]:
    artifacts = list(state.get("artifacts", []))
    artifacts.append(str(artifact_path))
    return artifacts


def plan(state: Dict[str, Any]) -> Dict[str, Any]:
    brief: str = state["brief"]
    run_id: str = state["run_id"]
    model: str | None = state.get("model")
    selected_agents = state.get("selected_agents") or DEFAULT_AGENT_SEQUENCE

    context = get_context(run_id, k=5)
    plan_markdown = chat(
        [
            {"role": "system", "content": PROMPT_TEXT},
            {
                "role": "user",
                "content": (
                    "You need to turn the brief into a step-by-step plan."
                    "\nUse the format with columns Backlog/Doing/Done"
                    " and add acceptance criteria."
                    f"\nBrief:\n{brief}\n\nContext:\n{context}"
                ),
            },
        ],
        model=model,
    )

    artifact = save_artifact(run_id, "manager_plan.md", plan_markdown)

    board = {
        "manager_plan": {
            "title": "Planning",
            "owner": "Manager",
            "status": "Done",
            "notes": plan_markdown,
        }
    }

    for agent_id in selected_agents:
        config = AGENT_REGISTRY.get(agent_id)
        if not config:
            continue
        board[agent_id] = {
            "title": config.title,
            "owner": config.owner,
            "status": "Backlog",
        }

    board["manager_summary"] = {
        "title": "Final Assembly",
        "owner": "Manager",
        "status": "Backlog",
    }

    return {
        "plan": plan_markdown,
        "board": board,
        "artifacts": _append_artifact(state, artifact),
    }


def _update_board(state: Dict[str, Any], task_id: str, *, status: str, notes: str) -> Dict[str, Any]:
    board = deepcopy(state.get("board", {}))
    board.setdefault(task_id, {})
    board[task_id].update({"status": status, "notes": notes})
    return board


def assemble(state: Dict[str, Any]) -> Dict[str, Any]:
    run_id: str = state["run_id"]
    model: str | None = state.get("model")

    icp = state.get("icp", "")
    concepts = state.get("concepts", "")
    copy = state.get("copy", "")
    plan_md = state.get("plan", "")
    brief = state["brief"]
    meeting_summary = state.get("meeting_summary", "")
    normalized_transcript = state.get("transcript_clean", "")
    audio_path = state.get("audio_path")
    selected_agents = state.get("selected_agents") or DEFAULT_AGENT_SEQUENCE
    agents_for_summary = state.get("agents_for_summary") or selected_agents

    validation_prompt = chat(
        [
            {"role": "system", "content": PROMPT_TEXT},
            {
                "role": "user",
                "content": (
                    "Verify that we have all the artifacts: ICP, concepts, texts."
                    " If anything is missing, list the issues."
                    " Then create a checklist of 5 implementation tasks."
                    f"\nICP:\n{icp}\n\nConcepts:\n{concepts}\n\nTexts:\n{copy}"
                ),
            },
        ],
        model=model,
    )

    todo_items: List[Dict[str, Any]] = []
    for line in validation_prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            text = stripped[2:].strip()
            if text:
                todo_items.append({"text": text, "done": False})

    if not todo_items:
        todo_items = [
            {"text": "Clarify campaign details with the client", "done": False},
            {"text": "Prepare visual materials", "done": False},
            {"text": "Layout the landing and landing pages", "done": False},
            {"text": "Plan publications", "done": False},
            {"text": "Set up analytics and reporting", "done": False},
        ]

    agent_sections = []
    for agent_id in agents_for_summary:
        config = AGENT_REGISTRY.get(agent_id)
        if not config:
            continue
        content = state.get(config.result_key, "")
        if content:
            agent_sections.append({"id": agent_id, "title": config.title, "body": content})

    summary_markdown = SUMMARY_TEMPLATE.render(
        project_title=state.get("project_title", "Marketing Campaign"),
        brief=brief,
        plan_markdown=plan_md,
        icp_markdown=icp,
        concepts_markdown=concepts,
        copy_markdown=copy,
        meeting_summary=meeting_summary,
        normalized_transcript=normalized_transcript,
        audio_path=audio_path,
        agent_sections=agent_sections,
        todo_items=todo_items,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    artifact = save_artifact(run_id, "campaign_summary.md", summary_markdown)

    board = _update_board(state, "manager_summary", status="Done", notes=summary_markdown)

    return {
        "summary": summary_markdown,
        "artifacts": _append_artifact(state, artifact),
        "board": board,
        "todo": todo_items,
    }


__all__ = ["plan", "assemble"]
