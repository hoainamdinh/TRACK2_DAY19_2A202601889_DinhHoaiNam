"""5-query demo for HybridMemoryAgent.

Usage:
    python bonus/demo.py

Exits 0 and prints assembled context for 5 queries as specified in
BONUS-CHALLENGE.md. No API key, no Docker, no GPU required.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent

# ── seed some episodic memory ────────────────────────────────────────────
SEED_MEMORIES = [
    # Kubernetes / cloud docs (for Q1, Q4, Q5)
    "Tôi vừa đọc tài liệu về Kubernetes: Pod là đơn vị nhỏ nhất trong K8s. "
    "Horizontal Pod Autoscaler (HPA) tự động mở rộng số Pod dựa trên CPU hoặc "
    "custom metrics. Cluster Autoscaler mở rộng node khi Pod không thể schedule.",

    "Kubernetes Ingress Controller (nginx/traefik) quản lý traffic HTTP vào cluster. "
    "Ingress resource định nghĩa routing rules: host-based và path-based. "
    "TLS termination thường được xử lý ở tầng Ingress.",

    # Cloud security (for Q5)
    "Cloud security best practices: least-privilege IAM, mã hoá at-rest (AES-256) "
    "và in-transit (TLS 1.3), network segmentation với VPC và Security Groups. "
    "Audit logs quan trọng để detect anomaly và comply với regulations.",

    # AI/ML topic (for Q2 — needs topic_affinity)
    "Embedding models như bge-small-en biến text thành vector. "
    "Semantic search dùng cosine similarity giữa query vector và document vectors. "
    "Hybrid search kết hợp BM25 (keyword) và vector search qua RRF.",

    # Recent activity context (for Q3)
    "Gần đây tôi quan tâm đến streaming data pipelines: Kafka, Flink. "
    "Schema evolution là thách thức lớn khi downstream consumers expect stable schema. "
    "Avro với Schema Registry là pattern phổ biến để giải quyết vấn đề này.",

    # Infrastructure scaling (for Q4 — paraphrase of "tự động mở rộng hạ tầng")
    "Auto-scaling hạ tầng: vertical scaling (tăng CPU/RAM máy) vs horizontal scaling "
    "(thêm máy). Cloud providers cung cấp Auto Scaling Groups (AWS), Managed Instance "
    "Groups (GCP). Trigger dựa trên CPU, memory, custom metrics hoặc schedule.",
]


def main() -> None:
    print("Initialising HybridMemoryAgent (Qdrant :memory:, no Feast)...")
    agent = HybridMemoryAgent(feast_store=None, top_k=3)

    print(f"Seeding {len(SEED_MEMORIES)} memory chunks for u_001...")
    for mem in SEED_MEMORIES:
        agent.remember(mem, user_id="u_001")
    print(f"  → {len(agent._chunks)} chunks indexed\n")
    print("=" * 60)

    # ── 5 demo queries ───────────────────────────────────────────────────

    # Q1: Simple vector hit — explicit keyword "Kubernetes"
    print("\n[Q1] Hỏi đơn giản — keyword hit:")
    print("  Query: 'Tôi đã đọc gì về Kubernetes?'")
    ctx = agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_001")
    print(ctx)

    # Q2: Cần profile context — topic_affinity drives retrieval hint
    print("\n[Q2] Hỏi cần profile context:")
    print("  Query: 'Recommend đọc gì tiếp' (no profile → vector picks best)")
    ctx = agent.recall("Recommend đọc gì tiếp theo trong lĩnh vực của tôi?",
                       user_id="u_001")
    print(ctx)

    # Q3: Hỏi cần fresh activity — recent queries signal
    print("\n[Q3] Hỏi cần fresh activity:")
    print("  Query: 'Tôi đang quan tâm gì gần đây?'")
    ctx = agent.recall("Tôi đang quan tâm gì gần đây?", user_id="u_001")
    print(ctx)

    # Q4: Paraphrase — vector wins ("tự động mở rộng hạ tầng" ≠ "auto-scaling")
    print("\n[Q4] Paraphrase query — vector search wins:")
    print("  Query: 'Tài liệu về tự động mở rộng hạ tầng?'")
    ctx = agent.recall("Tài liệu về tự động mở rộng hạ tầng?", user_id="u_001")
    print(ctx)

    # Q5: Mixed hybrid + profile — needs both episodic (cloud security) + profile
    print("\n[Q5] Mixed hybrid + profile context:")
    print("  Query: 'Cho tôi summary cloud security'")
    ctx = agent.recall("Cho tôi summary cloud security", user_id="u_001")
    print(ctx)

    print("\nAll 5 queries completed successfully.")


if __name__ == "__main__":
    main()
