# from __future__ import annotations

# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict

# from agents._llm import DEFAULT_MODEL, chat
# from memory.rag import get_context, save_artifact

# BASE_DIR = Path(__file__).resolve().parent.parent
# PROMPT_TEXT = (BASE_DIR / "prompts" / "ideator.md").read_text(encoding="utf-8")


# def concepts(state: Dict[str, Any]) -> Dict[str, Any]:
#     brief: str = state["brief"]
#     run_id: str = state["run_id"]
#     icp: str = state.get("icp", "")
#     model: str = state.get("model", DEFAULT_MODEL)

#     context = get_context(run_id, query=icp or brief, k=5)

#     response = chat(
#         [
#             {"role": "system", "content": PROMPT_TEXT},
#             {
#                 "role": "user",
#                 "content": (
#                     f"Бриф:\n{brief}\n\nICP:\n{icp}\n\n"
#                     f"Контекст из памяти:\n{context}\n\nПредложи концепции."
#                 ),
#             },
#         ],
#         model=model,
#     )

#     artifact = save_artifact(run_id, "ideator_concepts.md", response)

#     board = deepcopy(state.get("board", {}))
#     board.setdefault("ideator_concepts", {})
#     board["ideator_concepts"].update({"status": "Done", "notes": response})

#     artifacts = list(state.get("artifacts", []))
#     artifacts.append(str(artifact))

#     return {
#         "concepts": response,
#         "artifacts": artifacts,
#         "board": board,
#     }


# __all__ = ["concepts"]



from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from agents._llm import chat
from memory.rag import get_context, save_artifact

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_TEXT = (BASE_DIR / "prompts" / "ideator.md").read_text(encoding="utf-8")


def concepts(state: Dict[str, Any]) -> Dict[str, Any]:
    brief: str = state["brief"]
    run_id: str = state["run_id"]
    icp: str = state.get("icp", "")
    model: str | None = state.get("model")

    context = get_context(run_id, query=icp or brief, k=5)

    response = chat(
        [
            {"role": "system", "content": PROMPT_TEXT},
            {
                "role": "user",
                "content": (
                    f"Brief:\n{brief}\n\nICP:\n{icp}\n\n"
                    f"Memory Context:\n{context}\n\nSuggest concepts."
                ),
            },
        ],
        model=model,
    )

    artifact = save_artifact(run_id, "ideator_concepts.md", response)

    board = deepcopy(state.get("board", {}))
    board.setdefault("ideator_concepts", {})
    board["ideator_concepts"].update({"status": "Done", "notes": response})

    artifacts = list(state.get("artifacts", []))
    artifacts.append(str(artifact))

    return {
        "concepts": response,
        "artifacts": artifacts,
        "board": board,
    }


__all__ = ["concepts"]
