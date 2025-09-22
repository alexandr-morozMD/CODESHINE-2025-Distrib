# from __future__ import annotations

# from copy import deepcopy
# from pathlib import Path
# from typing import Any, Dict

# from agents._llm import DEFAULT_MODEL, chat
# from memory.rag import get_context, save_artifact

# BASE_DIR = Path(__file__).resolve().parent.parent
# PROMPT_TEXT = (BASE_DIR / "prompts" / "copywriter.md").read_text(encoding="utf-8")


# def texts(state: Dict[str, Any]) -> Dict[str, Any]:
#     brief: str = state["brief"]
#     run_id: str = state["run_id"]
#     icp: str = state.get("icp", "")
#     concepts: str = state.get("concepts", "")
#     model: str = state.get("model", DEFAULT_MODEL)

#     context = get_context(run_id, query=concepts or icp or brief, k=5)

#     response = chat(
#         [
#             {"role": "system", "content": PROMPT_TEXT},
#             {
#                 "role": "user",
#                 "content": (
#                     f"Бриф:\n{brief}\n\nICP:\n{icp}\n\nКонцепции:\n{concepts}\n\n"
#                     f"Контекст памяти:\n{context}\n\n"
#                     "Используй лучший из предложенных концептов и создай тексты."
#                 ),
#             },
#         ],
#         model=model,
#     )

#     artifact = save_artifact(run_id, "copywriter_texts.md", response)

#     board = deepcopy(state.get("board", {}))
#     board.setdefault("copywriter_texts", {})
#     board["copywriter_texts"].update({"status": "Done", "notes": response})

#     artifacts = list(state.get("artifacts", []))
#     artifacts.append(str(artifact))

#     return {
#         "copy": response,
#         "artifacts": artifacts,
#         "board": board,
#     }


# __all__ = ["texts"]




from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from agents._llm import chat
from memory.rag import get_context, save_artifact

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_TEXT = (BASE_DIR / "prompts" / "copywriter.md").read_text(encoding="utf-8")


def texts(state: Dict[str, Any]) -> Dict[str, Any]:
    brief: str = state["brief"]
    run_id: str = state["run_id"]
    icp: str = state.get("icp", "")
    concepts: str = state.get("concepts", "")
    model: str | None = state.get("model")

    context = get_context(run_id, query=concepts or icp or brief, k=5)

    response = chat(
        [
            {"role": "system", "content": PROMPT_TEXT},
            {
                "role": "user",
                "content": (
                    f"Brief:\n{brief}\n\nICP:\n{icp}\n\nConcepts:\n{concepts}\n\n"
                    f"Memory Context:\n{context}\n\n"
                    "Use the best of the proposed concepts and create texts."
                ),
            },
        ],
        model=model,
    )

    artifact = save_artifact(run_id, "copywriter_texts.md", response)
    board = deepcopy(state.get("board", {}))
    board.setdefault("copywriter_texts", {})
    board["copywriter_texts"].update({"status": "Done", "notes": response})

    artifacts = list(state.get("artifacts", []))
    artifacts.append(str(artifact))

    return {
        "copy": response,
        "artifacts": artifacts,
        "board": board,
    }


__all__ = ["texts"]
