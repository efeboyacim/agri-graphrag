"""LanceDB + local sentence-transformers embeddings (shared by the seed script and retrieval)."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import lancedb

from app import config

log = logging.getLogger("agrigraphrag.vector")


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def _db():
    return lancedb.connect(config.LANCEDB_PATH)


def write_chunks(rows: list[dict[str, Any]]) -> int:
    """(Re)create the notes table. Each row needs 'text'; a 'vector' column is added."""
    vectors = embed([row["text"] for row in rows])
    data = [{**row, "vector": vector} for row, vector in zip(rows, vectors)]
    table = _db().create_table(config.LANCEDB_TABLE, data=data, mode="overwrite")
    return table.count_rows()


def _open_table():
    try:
        return _db().open_table(config.LANCEDB_TABLE)
    except Exception:  # table missing -> not seeded yet
        return None


def count() -> int:
    table = _open_table()
    return table.count_rows() if table is not None else 0


def search(text: str, k: int = 3) -> list[dict[str, Any]]:
    """Top-k chunks by cosine distance (0 = identical). Returns [] if the table isn't seeded."""
    table = _open_table()
    if table is None:
        log.warning("LanceDB table %r not found at %s; run seed/seed_lancedb.py", config.LANCEDB_TABLE, config.LANCEDB_PATH)
        return []
    hits = table.search(embed([text])[0]).distance_type("cosine").limit(k).to_list()
    return [{key: value for key, value in hit.items() if key != "vector"} for hit in hits]
