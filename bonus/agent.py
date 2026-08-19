"""HybridMemoryAgent — episodic memory (Qdrant) + user profile (Feast).

Design decisions documented in bonus/ARCHITECTURE.md:
  - Chunking: per-message with 1-message overlap
  - Feature schema: tabular (topic_affinity, preferred_language, queries_last_hour)
  - Freshness: streaming push for episodic, batch for profile

Self-contained: uses Qdrant :memory: so `python bonus/demo.py` works without
any running service. Feast is optional — gracefully degrades to profile-free mode.
"""
from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make sure app/ is importable when running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.embeddings import Embedder
from app.search import Searcher

# ── constants ────────────────────────────────────────────────────────────
COLLECTION = "hybrid_memory"
BM25_DEPTH = 50          # how many BM25 hits to pull before RRF
RRF_K = 60               # standard RRF constant


# ── helpers ──────────────────────────────────────────────────────────────
@dataclass
class MemoryChunk:
    chunk_id: str
    user_id: str
    text: str
    timestamp: float = field(default_factory=time.time)


def _chunk_text(text: str, prev_chunk: str | None = None) -> list[str]:
    """Per-message chunking with 1-message overlap (see ARCHITECTURE.md §1).

    Each call to remember() is treated as one message. We prepend the
    previous chunk as context overlap so that anaphoric references
    ("cái đó", "it") in the current message can be resolved at recall time.
    """
    chunk = text.strip()
    if not chunk:
        return []
    if prev_chunk:
        # overlap: prepend truncated previous chunk (max 120 chars)
        overlap = prev_chunk[-120:].strip()
        return [overlap + " | " + chunk, chunk]
    return [chunk]


def _rrf_merge(
    bm25_ids: list[str],
    vec_ids: list[str],
    meta: dict[str, dict],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """Reciprocal Rank Fusion over two ranked lists of chunk_ids."""
    scores: dict[str, float] = {}
    for ranked in (bm25_ids, vec_ids):
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda c: -scores[c])[:top_k]
    return [{"chunk_id": c, "score": scores[c], **meta[c]} for c in ordered]


# ── main class ───────────────────────────────────────────────────────────
class HybridMemoryAgent:
    """Episodic memory agent combining Qdrant vector search and Feast features.

    Parameters
    ----------
    feast_store : optional
        A materialised Feast FeatureStore object. If None, profile lookup is
        skipped and recall() degrades gracefully to vector-only context.
    top_k : int
        Number of memory chunks to return per recall().
    """

    def __init__(
        self,
        feast_store: Any | None = None,
        top_k: int = 5,
    ) -> None:
        self.feast_store = feast_store
        self.top_k = top_k

        # Shared embedder (fastembed bge-small-en-v1.5, 384d by default)
        self.embedder = Embedder()

        # In-memory Qdrant — self-contained, no server needed
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

        # BM25 is rebuilt on every remember() call (tiny corpus in POC)
        # In production: incremental index update or separate BM25 service
        self._chunks: list[MemoryChunk] = []
        self._bm25: Any | None = None
        self._prev_text: dict[str, str] = {}   # user_id → last chunk text

    # ── memory ingestion ─────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user.

        Steps:
          1. Chunk text (per-message + overlap with previous message)
          2. Embed each chunk with fastembed
          3. Upsert to Qdrant with user_id payload for per-user filtering
          4. Rebuild BM25 index (lightweight for POC corpus size)
        """
        prev = self._prev_text.get(user_id)
        chunks = _chunk_text(text, prev)
        if not chunks:
            return

        points: list[PointStruct] = []
        for chunk_text in chunks:
            vec = next(self.embedder.embed([chunk_text]))
            cid = str(uuid.uuid4())
            self._chunks.append(MemoryChunk(
                chunk_id=cid, user_id=user_id,
                text=chunk_text, timestamp=time.time(),
            ))
            points.append(PointStruct(
                id=cid,
                vector=vec.tolist(),
                payload={
                    "user_id": user_id,
                    "text": chunk_text,
                    "timestamp": time.time(),
                },
            ))

        self.client.upsert(collection_name=COLLECTION, points=points)
        self._prev_text[user_id] = text

        # Rebuild BM25 (acceptable for small in-session corpus)
        from rank_bm25 import BM25Okapi
        user_chunks = [c for c in self._chunks if c.user_id == user_id]
        tokenized = [c.text.lower().split() for c in user_chunks]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    # ── memory retrieval ─────────────────────────────────────────────────
    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features → assembled context.

        Steps:
          1. Get user profile from Feast online store (tabular features)
          2. Hybrid search Qdrant filtered by user_id (BM25 + vector via RRF)
          3. Assemble context string
        """
        # Step 1: User profile from Feast (optional, graceful fallback)
        profile = self._get_profile(user_id)

        # Step 2: Hybrid search filtered by user_id
        user_chunks = [c for c in self._chunks if c.user_id == user_id]
        if not user_chunks:
            top_memories: list[str] = []
        else:
            top_memories = self._hybrid_search(query, user_id, user_chunks)

        # Step 3: Assemble context string
        return self._assemble_context(user_id, query, profile, top_memories)

    # ── private helpers ──────────────────────────────────────────────────
    def _get_profile(self, user_id: str) -> dict[str, Any]:
        """Fetch user profile from Feast. Returns empty dict on any failure."""
        if self.feast_store is None:
            return {}
        try:
            raw = self.feast_store.get_online_features(
                features=[
                    "user_profile_features:topic_affinity",
                    "user_profile_features:preferred_language",
                    "user_profile_features:reading_speed_wpm",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: (v[0] if v else None) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            return {}

    def _hybrid_search(
        self, query: str, user_id: str, user_chunks: list[MemoryChunk]
    ) -> list[str]:
        """BM25 + vector hybrid search, filtered to this user's memories."""
        chunk_ids = [c.chunk_id for c in user_chunks]
        meta = {c.chunk_id: {"text": c.text, "timestamp": c.timestamp}
                for c in user_chunks}

        # Vector search — Qdrant filter by user_id payload
        qvec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ])
        vec_results = self.client.query_points(
            collection_name=COLLECTION,
            query=qvec,
            query_filter=user_filter,
            limit=min(BM25_DEPTH, len(user_chunks)),
        ).points
        vec_ids = [p.id for p in vec_results]

        # BM25 search over user's chunks
        bm25_ids: list[str] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            bm25_ids = [chunk_ids[i] for i in ranked[:BM25_DEPTH]
                        if i < len(chunk_ids)]

        # RRF merge
        hits = _rrf_merge(bm25_ids, vec_ids, meta, top_k=self.top_k)
        return [h["text"] for h in hits]

    def _assemble_context(
        self,
        user_id: str,
        query: str,
        profile: dict[str, Any],
        memories: list[str],
    ) -> str:
        """Build the context string handed to the LLM (or returned to demo)."""
        lines: list[str] = [f"=== Memory Context for {user_id} ==="]

        # Profile block
        if profile:
            affinity = profile.get("topic_affinity", "unknown")
            lang = profile.get("preferred_language", "vi")
            speed = profile.get("reading_speed_wpm")
            speed_str = f"{speed:.0f} wpm" if speed else "unknown"
            lines.append(f"User profile: topic_affinity={affinity}, "
                         f"language={lang}, reading_speed={speed_str}")
        else:
            lines.append("User profile: (Feast not available — profile-free mode)")

        # Recent memories block
        lines.append(f"\nQuery: {query}")
        if memories:
            lines.append(f"\nTop-{len(memories)} episodic memories:")
            for i, mem in enumerate(memories, 1):
                preview = mem[:200].replace("\n", " ")
                lines.append(f"  [{i}] {preview}{'...' if len(mem) > 200 else ''}")
        else:
            lines.append("\nNo episodic memories found for this user yet.")

        lines.append("=" * 40)
        return "\n".join(lines)
