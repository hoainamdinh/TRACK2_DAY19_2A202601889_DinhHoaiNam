# Kiến trúc AI Memory — Trợ lý cá nhân tiếng Việt

**Tác giả:** Đinh Hoài Nam (2A202601889)
**Lab:** Day 19 — Vector Store & Feature Store
**Bonus Challenge:** Build Your Own AI Memory

---

## Tổng quan

Bài toán: xây dựng hệ thống memory cho một trợ lý AI cá nhân người dùng
Việt Nam. Trợ lý phải "nhớ" ba loại thông tin với chu kỳ cập nhật khác nhau:

| Loại memory | Nơi lưu | Chu kỳ cập nhật |
|---|---|---|
| **Episodic** — hội thoại, tài liệu đã đọc, ghi chú | Qdrant (vector store) | Realtime (mỗi message) |
| **Stable user profile** — ngôn ngữ, lĩnh vực quan tâm, tốc độ đọc | Feast (feature store) | Daily / weekly |
| **Recent activity** — query 1h qua, topic đang theo dõi | Feast streaming view | 5-minute batch hoặc streaming push |

---

## Sơ đồ kiến trúc

```mermaid
flowchart TD
    U([User / Client]) -->|"query: string"| AG[HybridMemoryAgent]

    AG -->|"remember(text, user_id)"| CH[Chunker]
    CH -->|chunks: list[str]| EM["Embedder\nbge-small-en / multilingual-e5"]
    EM -->|vectors| QD[("Qdrant\nEpisodic Memory\nfiltered by user_id")]

    AG -->|"recall(query, user_id)"| PF["Feast Online Store\nuser profile + recent activity"]
    AG -->|"hybrid search\nfiltered by user_id"| QD

    PF -->|"topic_affinity\npreferred_language\nqueries_last_hour"| CTX[Context Assembler]
    QD -->|top-K memory chunks| CTX

    CTX -->|assembled context string| LLM["LLM / Response Generator\n(optional — not in POC)"]
    LLM -->|final answer| U

    style QD fill:#4a90d9,color:#fff
    style PF fill:#e8a838,color:#fff
    style LLM fill:#888,color:#fff
```

**Data flow:**
1. `remember()`: text → chunk → embed → upsert Qdrant (payload: user_id, timestamp)
2. `recall()`: (a) Feast lấy profile, (b) hybrid BM25+vector search Qdrant filter user_id → assemble context string

---

## Quyết định kiến trúc 1: Chunking Strategy

**Vấn đề:** Episodic memory chunk thế nào? Per-message? Per-conversation? Semantic break?

### Các lựa chọn đã xem xét

| Chiến lược | Retrieval quality | Storage cost | Context window |
|---|---|---|---|
| Per-conversation (1 doc) | Thấp — khó pinpoint | Thấp | Tiêu tốn nhiều token |
| **Per-message + 1-message overlap** (CHỌN) | Cao — granular | Trung bình | Kiểm soát được |
| Semantic break (NLP sentence boundary) | Rất cao | Cao | Tốt |
| Fixed 256-token window | Trung bình | Thấp | Ổn định |

### Quyết định: Per-message với 1-message overlap

**Lý do chọn per-message + overlap:**
- Mỗi message của user là một đơn vị ngữ nghĩa tự nhiên — không tốn CPU cho sentence boundary detection
- 1-message overlap giữ context liền mạch: nếu user nói "cái đó" ở message kế, vector của message trước vẫn có thể bị recall
- Storage cost chấp nhận được: 1.000 messages ≈ vài MB trong Qdrant memory

**Lý do bác bỏ semantic break:**
Xem xét dùng `underthesea.sent_tokenize()` để cắt theo câu tiếng Việt. Tuy nhiên:
- `underthesea` tốn thêm ~150 MB install, vi phạm tinh thần "lite path"
- Boundary detection tiếng Việt kém hơn tiếng Anh vì thiếu punctuation rõ ràng
- Per-message đã đủ granular cho chat memory use-case

---

## Quyết định kiến trúc 2: Feature Schema — Tabular vs Embedding Features

**Vấn đề:** User profile lưu features gì? Tabular (topic_affinity: str) hay embedding features (latent preference vector từ history)?

**Option A — Tabular features (CHỌN):**
```
Entity: user_id
Features:
  topic_affinity: str         # "cloud", "ai_ml", ...   TTL: 7 days
  preferred_language: str     # "vi", "en", "mix"        TTL: 30 days
  reading_speed_wpm: float    # ước tính từ session data TTL: 7 days
  queries_last_hour: int      # windowed count           TTL: 1 hour
  last_active_hour: int       # 0-23                     TTL: 1 day
```

**Option B — Embedding features:**
Lưu vector 384d biểu diễn "latent preference" từ lịch sử tìm kiếm.

### Quyết định: Tabular features

**Lý do chọn tabular:**
1. **Latency:** Online lookup Feast SQLite < 1 ms vs phải tính cosine giữa preference vector và query vector (thêm ~5 ms)
2. **Interpretability:** `topic_affinity = "cloud"` rõ ràng hơn vector 384d khi debug tại sao trợ lý recommend sai
3. **TTL rõ ràng:** Mỗi tabular feature có TTL riêng — `queries_last_hour` expire sau 1h, `topic_affinity` expire sau 7 ngày. Embedding feature TTL phức tạp hơn: khi nào re-embed?
4. **Feast native:** Feast được thiết kế cho tabular features; embedding features phải workaround với BLOB column

**Lý do bác bỏ embedding features:**
Xem xét lưu episodic memory TRONG feature store như một embedding feature view.
Bác bỏ vì re-index cycle hoàn toàn khác nhau: episodic memory cập nhật mỗi giờ
(mỗi lần user chat), còn user profile cập nhật theo tuần. Trộn hai chu kỳ này vào
một feature store làm phức tạp materialization schedule mà không có lợi ích rõ ràng.
Tách riêng vector store (Qdrant) cho episodic và feature store (Feast) cho profile
là đúng về trách nhiệm.

---

## Quyết định kiến trúc 3: Freshness Strategy

**Vấn đề:** Sau khi user đọc xong 1 tài liệu mới, bao lâu thì `recall()` phản ánh tài liệu đó?

### Ba use case với freshness requirement khác nhau

| Use case | Freshness requirement | Chiến lược | Chi phí |
|---|---|---|---|
| Chat assistant (conversational) | < 1 giây | **Streaming push** — upsert Qdrant ngay sau `remember()` | CPU + network mỗi message |
| Daily digest / summary | 5–15 phút | **Batch refresh** — cron 5-minute upsert batch | Thấp; acceptable lag |
| User analytics / recommendations | 1–24 giờ | **Daily materialization** (Feast `materialize-incremental`) | Rất thấp |

### Quyết định: Hybrid freshness — streaming cho episodic, batch cho profile

- **Episodic memory (Qdrant):** streaming push — `remember()` upsert ngay lập tức. Vì Qdrant in-memory upsert < 10 ms, không cần queue.
- **User profile (Feast):** batch 5-minute cho `queries_last_hour`; daily cho `topic_affinity` và `reading_speed_wpm`. Lý do: profile thay đổi chậm, over-engineering realtime profile update tốn infra cost.

**Lý do không dùng daily-only:**
`queries_last_hour` mà stale 24h thì vô nghĩa — feature này chỉ có giá trị khi fresh.
Cần ít nhất batch 5-minute.

**Lý do không dùng full-streaming cho Feast:**
Feast streaming (Kafka source) cần Kafka infra, tăng complexity đáng kể. Với personal
assistant 1 user, batch 5-min là đủ và maintenance cost thấp hơn nhiều.

---

## Vietnamese-Context Considerations

### 1. Code-switching (vi/en mix)
Người dùng Việt Nam thường trộn tiếng Việt và tiếng Anh trong cùng một câu:
"Tôi muốn đọc về kubernetes networking và cách configure ingress controller".

**Ảnh hưởng đến retrieval:**
- BM25 tokenize trên whitespace — OK cho code-switching vì từ kỹ thuật tiếng Anh ("kubernetes", "ingress") không cần tách
- Vector search với `bge-small-en` gặp khó với phần tiếng Việt thuần túy; `multilingual-e5-large` xử lý tốt hơn cả hai ngôn ngữ

**Quyết định:** Default dùng `bge-small-en` (lite, không cần GPU). Nếu user profile có `preferred_language = "vi"`, agent nên warn về giới hạn này và recommend đổi `EMBEDDING_BACKEND=multilingual`.

### 2. Tokenizer choice — whitespace vs pyvi/underthesea

Tiếng Việt là ngôn ngữ đơn âm tiết: "học sinh" là 2 từ nhưng 1 khái niệm.
Whitespace tokenizer tách thành ["học", "sinh"] — BM25 sẽ match riêng lẻ.

| Tokenizer | Chất lượng BM25 (VN) | Install size | Dependency |
|---|---|---|---|
| Whitespace (hiện tại) | Baseline | 0 MB | None |
| `pyvi` | +10-15% recall | ~5 MB | Nhẹ |
| `underthesea` | +20-25% recall | ~150 MB | Nặng |

**Quyết định POC:** Giữ whitespace — đủ cho demo, không break lite setup.
Production recommendation: dùng `pyvi` cho balance giữa quality và install size.

### 3. Phonetic typo — đặc thù VN
VD: "bảo mặt" thay vì "bảo mật", "khái niệm" viết sai dấu.
BM25 sẽ miss hoàn toàn; vector search chịu đựng tốt hơn.
Hybrid search đặc biệt quan trọng cho corpus tiếng Việt chính vì lý do này.

---

## Limitations của POC này

1. **Privacy isolation:** Filter bằng `user_id` payload trong Qdrant — không có encryption per-user. Production cần per-user collection hoặc encryption at rest.
2. **Memory CRUD:** Không có `forget()` method. Production cần soft-delete (`deleted=True` payload) và periodic hard purge.
3. **Multi-device sync:** In-memory Qdrant không persist across restarts. Production cần Qdrant server + WAL hoặc snapshot.
4. **Context window budget:** `recall()` return raw string không đếm token. Với LLM 4K context, cần truncation logic.
5. **No LLM call:** POC chỉ assemble context, không gọi LLM thật. Integration với OpenAI/Gemini là bước tiếp theo.

---

## Vibe Coding Log

**Prompt hiệu quả nhất:** "Dùng lại `app.search.Searcher` từ codebase lab19, wrap thành `HybridMemoryAgent` với thêm user_id filter payload cho Qdrant, graceful fallback khi Feast chưa được apply."

**Prompt fail:** "Tạo một full production-grade memory system với Redis cache, Kafka streaming, và encryption" — quá rộng, output code không chạy được. Lesson: specify constraints rõ (lite path, no extra dependencies, exits 0).
