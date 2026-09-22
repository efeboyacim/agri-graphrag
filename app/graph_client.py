"""Neo4j connection + small helpers shared by the agents and the seed script."""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Any

from neo4j import Driver, GraphDatabase

from app import config

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            notifications_min_severity="OFF",  # no "label does not exist" chatter on a fresh DB
        )
    return _driver


def run(cypher: str, **params: Any) -> list[dict[str, Any]]:
    records, _, _ = get_driver().execute_query(cypher, params, database_="neo4j")
    return [record.data() for record in records]


def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
    known_regions.cache_clear()


def normalize(text: str) -> str:
    """Accent/case-insensitive key. Needed because 'İzmir'.lower() == 'i̇zmir' (combining dot)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().strip()


@lru_cache(maxsize=1)
def known_regions() -> dict[str, str]:
    """normalized name -> canonical Region.name as stored in the graph."""
    return {normalize(row["name"]): row["name"] for row in run("MATCH (r:Region) RETURN r.name AS name")}


def canonical_region(candidate: str | None, fallback_text: str = "") -> str | None:
    """Map an LLM-extracted region (or, failing that, a mention in the raw query) to a graph Region."""
    regions = known_regions()
    if candidate and normalize(candidate) in regions:
        return regions[normalize(candidate)]
    haystack = normalize(f"{candidate or ''} {fallback_text}")
    for key, name in regions.items():
        if key and key in haystack:
            return name
    return None
