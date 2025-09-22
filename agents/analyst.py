# from __future__ import annotations

# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict

# # from agents._llm import DEFAULT_MODEL, chat
# from agents._llm import chat
# from memory.rag import get_context, save_artifact

# BASE_DIR = Path(__file__).resolve().parent.parent
# PROMPT_TEXT = (BASE_DIR / "prompts" / "analyst.md").read_text(encoding="utf-8")


# def icp(state: Dict[str, Any]) -> Dict[str, Any]:
#     brief: str = state["brief"]
#     run_id: str = state["run_id"]
#     model: str = state.get("model", DEFAULT_MODEL)

#     context = get_context(run_id, query=brief, k=5)
#     response = chat(
#         [
#             {"role": "system", "content": PROMPT_TEXT},
#             {
#                 "role": "user",
#                 "content": f"Бриф:\n{brief}\n\nКонтекст:\n{context}\n\nСформируй выводы.",
#             },
#         ],
#         model=model,
#     )

#     artifact = save_artifact(run_id, "analyst_icp.md", response)

#     board = deepcopy(state.get("board", {}))
#     board.setdefault("analyst_icp", {})
#     board["analyst_icp"].update({"status": "Done", "notes": response})

#     artifacts = list(state.get("artifacts", []))
#     artifacts.append(str(artifact))

#     return {
#         "icp": response,
#         "artifacts": artifacts,
#         "board": board,
#     }


# __all__ = ["icp"]
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from agents._llm import chat
from memory.rag import get_context, save_artifact

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_TEXT = (BASE_DIR / "prompts" / "analyst.md").read_text(encoding="utf-8")


def icp(state: Dict[str, Any]) -> Dict[str, Any]:
    brief: str = state["brief"]
    run_id: str = state["run_id"]
    model: str | None = state.get("model")  # если нет, chat() сам выберет дефолт

    context = get_context(run_id, query=brief, k=5)
    response = chat(
        [
            {"role": "system", "content": PROMPT_TEXT},
            {
                "role": "user",
                "content": f"Brief:\n{brief}\n\nContext:\n{context}\n\nSummarize the findings.",
            },
        ],
        model=model,
    )

    artifact = save_artifact(run_id, "analyst_icp.md", response)
    board = deepcopy(state.get("board", {}))
    board.setdefault("analyst_icp", {})
    board["analyst_icp"].update({"status": "Done", "notes": response})

    artifacts = list(state.get("artifacts", []))
    artifacts.append(str(artifact))

    return {
        "icp": response,
        "artifacts": artifacts,
        "board": board,
    }


__all__ = ["icp"]
