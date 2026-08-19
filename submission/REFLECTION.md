# Reflection — Lab 19

**Tên:** Đinh Hoài Nam  
**MSSV:** 2A202601889  
**Path:** lite

---

## Mode nào thắng ở loại query nào?

Trên golden set 50 queries (15 exact / 15 paraphrase / 20 mixed):

| Mode | Exact | Paraphrase | Mixed |
|------|------:|----------:|------:|
| BM25 | **96.7%** | 33.3% | 80.0% |
| Semantic | 53.3% | 24.0% | 75.0% |
| Hybrid (RRF) | **96.7%** | 32.0% | **100%** |

- **Exact → BM25 = Hybrid**: keyword chính xác → term matching thắng tuyệt đối.
- **Paraphrase → cả ba đều thấp**: `bge-small-en` không được train trên tiếng Việt nên embedding kém; BM25 thậm chí nhỉnh hơn vector.
- **Mixed → Hybrid**: RRF kết hợp được cả term signal lẫn semantic signal, đạt 100%.

## Khi nào KHÔNG dùng hybrid?

1. **Dùng pure BM25** khi: query là keyword chính xác, cần latency thấp (P99 ~10 ms vs ~54 ms của hybrid), hoặc corpus chưa có embedding tốt.
2. **Dùng pure vector** khi: corpus có multilingual model chất lượng cao và query là câu ngữ nghĩa phức tạp, không có keyword literal.

## Điều ngạc nhiên nhất

Semantic (vector) lại **thua BM25** ở cả hai loại exact lẫn paraphrase — ngược hoàn toàn kỳ vọng. Nguyên nhân: `bge-small-en` không được fine-tune cho tiếng Việt. Đổi sang `bge-m3` hoặc `multilingual-e5-large` sẽ cải thiện đáng kể nhóm paraphrase.

---

## Bonus Challenge

Đã hoàn thành thư mục `bonus/` với 3 deliverables:

- **`bonus/ARCHITECTURE.md`** (~1 435 words) — Mermaid diagram + 3 architecture decisions với explicit tradeoff (chunking strategy, feature schema, freshness strategy) + Vietnamese-context considerations (code-switching, pyvi vs underthesea, phonetic typo)
- **`bonus/agent.py`** — `HybridMemoryAgent` class với `remember()` và `recall()`: episodic memory qua Qdrant in-memory, user profile qua Feast (graceful fallback nếu chưa apply)
- **`bonus/demo.py`** — 5 queries minh hoạ, `python bonus/demo.py` exits 0

