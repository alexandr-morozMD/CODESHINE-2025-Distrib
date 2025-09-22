# from __future__ import annotations

# import os
# import uuid
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional

# try:
#     import chromadb
#     from chromadb.utils import embedding_functions
# except ImportError:  # pragma: no cover - fallback for dev environments
#     chromadb = None  # type: ignore
#     embedding_functions = None  # type: ignore

# try:
#     from sentence_transformers import SentenceTransformer  # noqa: F401
# except ImportError:  # pragma: no cover - handled by Chroma embedding function loader
#     SentenceTransformer = None  # type: ignore

# BASE_DIR = Path(__file__).resolve().parent.parent
# ARTIFACT_ROOT = BASE_DIR / "artifacts"
# VECTOR_ROOT = ARTIFACT_ROOT / "vectorstore"
# VECTOR_ROOT.mkdir(parents=True, exist_ok=True)

# EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# _IN_MEMORY_STORE: Dict[str, List[str]] = {}

# _client = None
# _embedder = None


# @dataclass
# class ArtifactRecord:
#     path: Path
#     title: str


# def _get_client():
#     global _client
#     if _client is None and chromadb is not None:
#         _client = chromadb.PersistentClient(path=str(VECTOR_ROOT))
#     return _client


# def _get_embedding_function():
#     global _embedder
#     if _embedder is None and embedding_functions is not None:
#         _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name=EMBED_MODEL_NAME
#         )
#     return _embedder


# def get_collection(run_id: str):
#     client = _get_client()
#     embed_fn = _get_embedding_function()
#     if client is None:
#         return None
#     return client.get_or_create_collection(
#         name=f"campaign_{run_id}", embedding_function=embed_fn
#     )


# def init_project(run_id: str, brief: str, extra_documents: Optional[Dict[str, str]] = None) -> None:
#     """Initialise storage for the campaign run and index provided documents."""
#     ARTIFACT_ROOT.joinpath(run_id).mkdir(parents=True, exist_ok=True)

#     store = _IN_MEMORY_STORE.setdefault(run_id, [])
#     store.clear()
#     store.append(brief)

#     collection = get_collection(run_id)
#     if collection is not None:
#         docs = [brief]
#         ids = [f"brief-{run_id}"]
#         metadatas = [{"source": "brief"}]
#         if extra_documents:
#             for key, value in extra_documents.items():
#                 docs.append(value)
#                 ids.append(f"{key}-{uuid.uuid4().hex}")
#                 metadatas.append({"source": key})
#         collection.upsert(documents=docs, ids=ids, metadatas=metadatas)
#     elif extra_documents:
#         store.extend(extra_documents.values())


# def upsert_document(run_id: str, content: str, *, source: str) -> None:
#     collection = get_collection(run_id)
#     doc_id = f"{source}-{uuid.uuid4().hex}"
#     if collection is not None:
#         collection.upsert(
#             documents=[content],
#             ids=[doc_id],
#             metadatas=[{"source": source}],
#         )
#     else:
#         _IN_MEMORY_STORE.setdefault(run_id, []).append(content)


# def get_context(run_id: str, query: Optional[str] = None, k: int = 5) -> str:
#     """Retrieve a small slice of context for agents."""
#     collection = get_collection(run_id)
#     seed_query = query or (_IN_MEMORY_STORE.get(run_id, [""]) or [""])[0]

#     if collection is None:
#         docs = _IN_MEMORY_STORE.get(run_id, [])
#         return "\n---\n".join(docs[:k])

#     results = collection.query(query_texts=[seed_query], n_results=k)  # type: ignore[arg-type]
#     documents = results.get("documents") or []
#     if not documents:
#         return ""
#     flattened: List[str] = []
#     for batch in documents:
#         flattened.extend(batch)
#     return "\n---\n".join(flattened)


# def save_artifact(run_id: str, filename: str, content: str, *, add_to_memory: bool = True) -> Path:
#     run_dir = ARTIFACT_ROOT / run_id
#     run_dir.mkdir(parents=True, exist_ok=True)
#     target = run_dir / filename
#     target.write_text(content, encoding="utf-8")
#     if add_to_memory:
#         upsert_document(run_id, content, source=filename)
#     return target


# def list_artifacts(run_id: str) -> List[ArtifactRecord]:
#     run_dir = ARTIFACT_ROOT / run_id
#     if not run_dir.exists():
#         return []
#     records: List[ArtifactRecord] = []
#     for path in sorted(run_dir.glob("*.md")):
#         records.append(ArtifactRecord(path=path, title=path.stem.replace("_", " ").title()))
#     return records


# __all__ = [
#     "init_project",
#     "get_context",
#     "save_artifact",
#     "list_artifacts",
#     "upsert_document",
# ]






# from __future__ import annotations

# import os
# import uuid
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional

# try:
#     import chromadb
#     from chromadb.utils import embedding_functions
# except ImportError:  # pragma: no cover - fallback
#     chromadb = None
#     embedding_functions = None

# BASE_DIR = Path(__file__).resolve().parent.parent
# ARTIFACT_ROOT = BASE_DIR / "artifacts"
# VECTOR_ROOT = ARTIFACT_ROOT / "vectorstore"
# VECTOR_ROOT.mkdir(parents=True, exist_ok=True)

# EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# _IN_MEMORY_STORE: Dict[str, List[str]] = {}

# _client = None
# _embedder = None


# @dataclass
# class ArtifactRecord:
#     path: Path
#     title: str


# # def _get_client():
# #     """Create Chroma client depending on mode (cloud or local)."""
# #     global _client
# #     if _client is None and chromadb is not None:
# #         mode = os.getenv("CHROMA_MODE", "local").lower()
# #         if mode == "cloud":
# #             _client = chromadb.CloudClient(
# #                 api_key=os.getenv("CHROMA_API_KEY"),
# #                 tenant=os.getenv("CHROMA_TENANT"),
# #                 database=os.getenv("CHROMA_DATABASE", "Stagging")
# #             )
# #         else:
# #             _client = chromadb.PersistentClient(path=str(VECTOR_ROOT))
# #     # return _client
# #     return None

# def _get_client():
#     global _client
#     if _client is None and chromadb is not None:
#         _client = chromadb.CloudClient(
#             api_key=os.getenv("CHROMA_API_KEY"),
#             tenant=os.getenv("CHROMA_TENANT"),
#             database=os.getenv("CHROMA_DATABASE", "Stagging")
#         )
#     return _client



# def _get_embedding_function():
#     global _embedder
#     if _embedder is None and embedding_functions is not None:
#         _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name=EMBED_MODEL_NAME
#         )
#     return _embedder


# def get_collection(run_id: str):
#     client = _get_client()
#     embed_fn = _get_embedding_function()
#     if client is None:
#         return None
#     return client.get_or_create_collection(
#         name=f"campaign_{run_id}", embedding_function=embed_fn
#     )


# def init_project(run_id: str, brief: str, extra_documents: Optional[Dict[str, str]] = None) -> None:
#     ARTIFACT_ROOT.joinpath(run_id).mkdir(parents=True, exist_ok=True)
#     store = _IN_MEMORY_STORE.setdefault(run_id, [])
#     store.clear()
#     store.append(brief)

#     collection = get_collection(run_id)
#     if collection is not None:
#         docs = [brief]
#         ids = [f"brief-{run_id}"]
#         metadatas = [{"source": "brief"}]
#         if extra_documents:
#             for key, value in extra_documents.items():
#                 docs.append(value)
#                 ids.append(f"{key}-{uuid.uuid4().hex}")
#                 metadatas.append({"source": key})
#         collection.upsert(documents=docs, ids=ids, metadatas=metadatas)
#     elif extra_documents:
#         store.extend(extra_documents.values())


# def upsert_document(run_id: str, content: str, *, source: str) -> None:
#     collection = get_collection(run_id)
#     doc_id = f"{source}-{uuid.uuid4().hex}"
#     if collection is not None:
#         collection.upsert(
#             documents=[content],
#             ids=[doc_id],
#             metadatas=[{"source": source}],
#         )
#     else:
#         _IN_MEMORY_STORE.setdefault(run_id, []).append(content)


# def get_context(run_id: str, query: Optional[str] = None, k: int = 5) -> str:
#     collection = get_collection(run_id)
#     seed_query = query or (_IN_MEMORY_STORE.get(run_id, [""]) or [""])[0]

#     if collection is None:
#         docs = _IN_MEMORY_STORE.get(run_id, [])
#         return "\n---\n".join(docs[:k])

#     results = collection.query(query_texts=[seed_query], n_results=k)  # type: ignore
#     documents = results.get("documents") or []
#     if not documents:
#         return ""
#     flattened: List[str] = []
#     for batch in documents:
#         flattened.extend(batch)
#     return "\n---\n".join(flattened)


# def save_artifact(run_id: str, filename: str, content: str, *, add_to_memory: bool = True) -> Path:
#     run_dir = ARTIFACT_ROOT / run_id
#     run_dir.mkdir(parents=True, exist_ok=True)
#     target = run_dir / filename
#     target.write_text(content, encoding="utf-8")
#     if add_to_memory:
#         upsert_document(run_id, content, source=filename)
#     return target


# def list_artifacts(run_id: str) -> List[ArtifactRecord]:
#     run_dir = ARTIFACT_ROOT / run_id
#     if not run_dir.exists():
#         return []
#     records: List[ArtifactRecord] = []
#     for path in sorted(run_dir.glob("*.md")):
#         records.append(ArtifactRecord(path=path, title=path.stem.replace("_", " ").title()))
#     return records


# __all__ = [
#     "init_project",
#     "get_context",
#     "save_artifact",
#     "list_artifacts",
#     "upsert_document",
# ]


from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = BASE_DIR / "artifacts"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

# Простое хранилище в памяти
_IN_MEMORY_STORE: Dict[str, List[str]] = {}


@dataclass
class ArtifactRecord:
    path: Path
    title: str


def init_project(run_id: str, brief: str, extra_documents: Dict[str, str] | None = None) -> None:
    """Инициализация проекта: сохраняем бриф и доп. документы в память"""
    ARTIFACT_ROOT.joinpath(run_id).mkdir(parents=True, exist_ok=True)
    store = _IN_MEMORY_STORE.setdefault(run_id, [])
    store.clear()
    store.append(brief)
    if extra_documents:
        store.extend(extra_documents.values())


def upsert_document(run_id: str, content: str, *, source: str) -> None:
    """Добавить документ в память"""
    _IN_MEMORY_STORE.setdefault(run_id, []).append(content)


def get_context(run_id: str, query: str | None = None, k: int = 5) -> str:
    """Вернуть последние k документов из памяти"""
    docs = _IN_MEMORY_STORE.get(run_id, [])
    return "\n---\n".join(docs[-k:])


def save_artifact(run_id: str, filename: str, content: str, *, add_to_memory: bool = True) -> Path:
    """Сохранить артефакт на диск и в память"""
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / filename
    target.write_text(content, encoding="utf-8")
    if add_to_memory:
        upsert_document(run_id, content, source=filename)
    return target


def list_artifacts(run_id: str) -> List[ArtifactRecord]:
    """Список артефактов для проекта"""
    run_dir = ARTIFACT_ROOT / run_id
    if not run_dir.exists():
        return []
    records: List[ArtifactRecord] = []
    for path in sorted(run_dir.glob("*.md")):
        records.append(
            ArtifactRecord(path=path, title=path.stem.replace("_", " ").title())
        )
    return records


__all__ = [
    "init_project",
    "get_context",
    "save_artifact",
    "list_artifacts",
    "upsert_document",
]
