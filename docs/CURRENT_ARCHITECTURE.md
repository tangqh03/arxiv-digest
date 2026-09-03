# Current Architecture

- Source: `generate_digest.py` reads topic and preference JSON, then collects arXiv topic search, alphaxiv trending IDs, and Hugging Face Daily Paper IDs.
- Fetch: all HTTP goes through a retrying `curl` subprocess wrapper; alphaxiv/HF IDs are enriched in batches through the arXiv Atom API.
- Fetch recovery: `memory/.fetch_progress.json` stores per-source results for the current date and allows partial retries.
- Parsing: arXiv Atom entries become legacy dictionaries containing ID, title, summary, dates, authors, categories, comment, and links.
- Dedup: collection first deduplicates by title; `build_raw_data` merges again by arXiv ID (or title fallback) and annotates source names.
- Raw JSON: `--raw` pre-ranks merged papers with topic/history metadata and writes `memory/daily_raw.json` plus an LLM prompt-ready schema.
- LLM scores: the program does not call an LLM; an external step must write `memory/llm_scores.json`, keyed by arXiv ID.
- Filter: when rerank JSON is supplied, missing scores default to 5.0 and configured score-threshold filtering is applied independently to each source list.
- Ranking: a local heuristic uses substring topic matches, recency, comments, author count, and recommendation count; LLM scores are displayed/filtered but do not drive final ordering.
- Heat: optional `heat_signals.py` fetches external signals after filtering and before rendering.
- Archive: accepted papers update legacy recommendation history, per-paper JSON archives, the paper index, and heat/evaluation timelines.
- Digest: Markdown contains overview plus topic, alphaxiv, and HF sections; it is written to an optional output path and `memory/digests/YYYY-MM-DD.md`.
- Log: a short link/header is appended to `memory/RESEARCH_LOG.md`; callback detection runs after digest generation.
- Delivery: `deliver_digest.sh` orchestrates raw collection and legacy digest generation but still depends on an externally created score file.
- Resend: `resend_daily_digest.sh` sends an existing dated Markdown digest to a hard-coded QQBot target through `openclaw`.
