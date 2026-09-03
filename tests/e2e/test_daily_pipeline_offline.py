import json
import re
from collections import Counter
from pathlib import Path

import httpx

from arxiv_digest.delivery.feishu import FeishuDelivery
from arxiv_digest.filtering import KeywordFilter
from arxiv_digest.fulltext.reader import FullTextReader
from arxiv_digest.llm.deep_summary import DeepSummarizer, DeepSummaryCache
from arxiv_digest.llm.screening import PaperScreener, ScreeningCache
from arxiv_digest.models import Paper
from arxiv_digest.pipeline import DailyPaperPipeline
from arxiv_digest.selection import DeepReadSelector


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakeHFSource:
    def __init__(self, papers):
        self.templates = papers
        self.calls = 0

    def list_daily(self, date, limit):
        self.calls += 1
        return [
            Paper.from_legacy_dict(paper.to_legacy_dict())
            for paper in self.templates[:limit]
        ]


class FakeLLM:
    model = "offline-e2e-model"

    def __init__(self, scores, deep_result):
        self.scores = scores
        self.deep_result = deep_result
        self.screen_calls = Counter()
        self.deep_calls = 0

    def chat_json(self, messages):
        payload = json.loads(messages[1]["content"])
        if "stage" in payload:
            if payload["stage"] == "chunk_notes":
                self.deep_calls += 1
                return {"method": "supported evidence", "limitations": []}
            self.deep_calls += 1
            return self.deep_result
        paper_id = payload["paper"]["arxiv_id"]
        self.screen_calls[paper_id] += 1
        relevance = self.scores[paper_id]
        return {
            "relevance": relevance,
            "novelty": 8,
            "impact": 7,
            "matched_topic": "Video Reasoning",
            "relevance_reason": f"{paper_id} directly addresses the configured topic.",
            "abstract_zh": f"{paper_id} 的忠实中文摘要。",
            "tldr_zh": f"{paper_id} 的简短 TLDR。",
            "worth_deep_reading": relevance >= 7,
        }


def make_paper(letter, index, *, title, abstract, hf_summary=""):
    paper_id = f"2609.{index:05d}"
    return Paper(
        arxiv_id=paper_id,
        title=f"Paper {letter}: {title}",
        abstract=abstract,
        hf_summary=hf_summary,
        sources=["huggingface-daily"],
        hf_url=f"https://huggingface.co/papers/{paper_id}",
        abs_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


def fulltext_for(paper_id, title):
    body = "grounded temporal evidence and method details " * 80
    return f"# {title}\n\n## Introduction\n\n{body}\n\n## Method\n\n{body}\n\n## Experiments\n\n{body}"


def test_daily_pipeline_offline(tmp_path):
    papers = [
        make_paper("A", 1, title="Video Reasoning Architecture", abstract="general work"),
        make_paper("B", 2, title="Temporal Model", abstract="We study video reasoning."),
        make_paper("C", 3, title="Video Reasoning Survey", abstract="keyword hit but irrelevant"),
        make_paper("D", 4, title="Database Index", abstract="Relational query optimization."),
        make_paper("E", 5, title="Grounded Agent", abstract="general", hf_summary="video reasoning agent"),
        make_paper("F", 6, title="Long Video Agent", abstract="video reasoning with tools"),
        make_paper("G", 7, title="Temporal Grounding", abstract="video reasoning benchmark"),
        make_paper("H", 8, title="Video Evidence", abstract="video reasoning verification"),
    ]
    scores = {
        "2609.00001": 10,
        "2609.00002": 9.5,
        "2609.00003": 3,
        "2609.00005": 9,
        "2609.00006": 8.5,
        "2609.00007": 8,
        "2609.00008": 7.5,
    }
    deep_result = json.loads(
        (FIXTURES / "llm_responses" / "deep_summary_valid.json").read_text(
            encoding="utf-8"
        )
    )
    llm = FakeLLM(scores, deep_result)
    source = FakeHFSource(papers)
    memory = tmp_path / "memory"
    hf_calls = Counter()
    html_calls = Counter()
    pdf_calls = Counter()

    titles = {paper.arxiv_id: paper.title for paper in papers}

    def hf_reader(paper_id):
        hf_calls[paper_id] += 1
        if paper_id in {"2609.00006", "2609.00007", "2609.00008"}:
            raise RuntimeError("HF markdown unavailable")
        return fulltext_for(paper_id, titles[paper_id])

    def html_fetcher(url):
        paper_id = re.search(r"(2609\.\d+)", url).group(1)
        html_calls[paper_id] += 1
        if paper_id in {"2609.00007", "2609.00008"}:
            raise RuntimeError("arXiv HTML unavailable")
        text = fulltext_for(paper_id, titles[paper_id])
        return "<article><h1>" + text.replace("\n\n## ", "</p><h2>").replace("\n\n", "</p><p>") + "</article>"

    def pdf_downloader(url):
        paper_id = re.search(r"(2609\.\d+)", url).group(1)
        pdf_calls[paper_id] += 1
        if paper_id == "2609.00008":
            raise RuntimeError("PDF unavailable")
        return f"pdf:{paper_id}".encode()

    def pdf_extractor(data):
        paper_id = data.decode().split(":", 1)[1]
        return fulltext_for(paper_id, titles[paper_id])

    feishu_http_calls = 0

    def feishu_handler(request):
        nonlocal feishu_http_calls
        feishu_http_calls += 1
        return httpx.Response(200, json={"code": 0})

    sender = FeishuDelivery(
        webhook_url="https://open.feishu.test/hook/offline",
        max_papers_per_message=1,
        receipt_root=memory / "delivery",
        transport=httpx.MockTransport(feishu_handler),
    )
    pipeline = DailyPaperPipeline(
        source=source,
        keyword_filter=KeywordFilter(
            [{"name": "Video Reasoning", "keywords": ["video reasoning"]}]
        ),
        screener=PaperScreener(
            llm,
            [{"name": "Video Reasoning", "keywords": ["video reasoning"]}],
            cache=ScreeningCache(memory / "analysis"),
            run_date="2026-09-03",
        ),
        selector=DeepReadSelector(top_k=6, min_relevance=7),
        fulltext_reader=FullTextReader(
            hf_reader=hf_reader,
            html_fetcher=html_fetcher,
            pdf_downloader=pdf_downloader,
            pdf_extractor=pdf_extractor,
            cache_root=memory / "fulltext",
        ),
        deep_summarizer=DeepSummarizer(
            llm,
            cache=DeepSummaryCache(memory / "deep_summaries"),
            max_fulltext_chars=20000,
        ),
        topics=[{"name": "Video Reasoning", "keywords": ["video reasoning"]}],
        delivery=sender,
        memory_root=memory,
    )

    run1 = pipeline.run(date="2026-09-03")

    assert run1.counts == {
        "hf_total": 8,
        "keyword_matched": 7,
        "llm_screened": 7,
        "accepted": 6,
        "deep_read": 6,
        "deep_read_success": 5,
    }
    assert run1.digest_path.exists()
    assert run1.run_path.exists()
    digest = run1.digest_path.read_text(encoding="utf-8")
    assert "Paper C" not in digest
    assert "Paper D" not in digest
    assert set(hf_calls) == {
        "2609.00001",
        "2609.00002",
        "2609.00005",
        "2609.00006",
        "2609.00007",
        "2609.00008",
    }
    assert html_calls["2609.00006"] == 1
    assert pdf_calls["2609.00007"] == 1
    assert pdf_calls["2609.00008"] == 1
    assert len(list((memory / "analysis" / "2026-09-03").glob("*.json"))) == 7
    assert len(list((memory / "deep_summaries").glob("*.json"))) == 5
    assert len(list((memory / "fulltext").glob("*/paper.md"))) == 5
    assert (memory / "delivery" / "2026-09-03" / "feishu.json").exists()
    assert feishu_http_calls == 7
    payload_text = json.dumps(sender.last_payloads, ensure_ascii=False)
    assert "Paper A" in payload_text
    assert "Paper C" not in payload_text

    screen_calls_after_first = sum(llm.screen_calls.values())
    deep_calls_after_first = llm.deep_calls
    run2 = pipeline.run(date="2026-09-03")

    assert sum(llm.screen_calls.values()) == screen_calls_after_first
    assert llm.deep_calls == deep_calls_after_first
    assert pipeline.screener.cache_hits == 7
    assert pipeline.deep_summarizer.cache_hits == 5
    assert pdf_calls["2609.00007"] == 1
    assert run2.delivery.skipped is True
    assert feishu_http_calls == 7
    run_record = json.loads(run2.run_path.read_text(encoding="utf-8"))
    assert run_record["llm"]["screen_calls"] == 0
    assert run_record["llm"]["screen_cache_hits"] == 7
    assert run_record["delivery"]["feishu"] == "skipped_duplicate"
