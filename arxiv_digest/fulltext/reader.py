from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from huggingface_hub import HfApi

from arxiv_digest.fulltext.pdf import extract_pdf_text
from arxiv_digest.models import Paper


DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "memory" / "fulltext"


@dataclass(frozen=True)
class FullTextResult:
    text: str
    source: str | None
    cached: bool
    error: str | None


class FullTextReader:
    def __init__(
        self,
        *,
        hf_reader: Callable[[str], str] | None = None,
        html_fetcher: Callable[[str], str] | None = None,
        pdf_downloader: Callable[[str], bytes] | None = None,
        pdf_extractor: Callable[[bytes], str] = extract_pdf_text,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        min_chars: int = 2000,
    ):
        hf_api = HfApi(token=os.environ.get("HF_TOKEN") or False)
        self._hf_reader = hf_reader or hf_api.read_paper
        self._html_fetcher = html_fetcher or _fetch_text
        self._pdf_downloader = pdf_downloader or _fetch_bytes
        self._pdf_extractor = pdf_extractor
        self.cache_root = cache_root
        self.min_chars = min_chars

    def read(self, paper: Paper) -> FullTextResult:
        cached = self._read_cache(paper)
        if cached:
            return cached

        errors = []
        attempts = (
            ("hf", lambda: self._hf_reader(paper.arxiv_id)),
            (
                "arxiv-html",
                lambda: html_to_text(
                    self._html_fetcher(f"https://arxiv.org/html/{paper.arxiv_id}")
                ),
            ),
            ("arxiv-pdf", lambda: self._read_pdf(paper)),
        )
        for source, fetch in attempts:
            try:
                text = fetch().strip()
                if not self._is_valid(text, paper):
                    errors.append(f"{source}: invalid or incomplete text")
                    continue
                self._write_cache(paper, text, source)
                return FullTextResult(text=text, source=source, cached=False, error=None)
            except Exception as error:
                errors.append(f"{source}: {type(error).__name__}")
        return FullTextResult(text="", source=None, cached=False, error="; ".join(errors))

    def _read_pdf(self, paper: Paper) -> str:
        url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}"
        data = self._pdf_downloader(url)
        directory = self._paper_dir(paper)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "paper.pdf.tmp"
        temporary.write_bytes(data)
        temporary.replace(directory / "paper.pdf")
        return self._pdf_extractor(data)

    def _read_cache(self, paper: Paper) -> FullTextResult | None:
        directory = self._paper_dir(paper)
        text_path = directory / "paper.md"
        try:
            text = text_path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not self._is_valid(text, paper):
            return None
        source = "cache"
        try:
            metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            source = metadata.get("source") or source
        except (OSError, ValueError):
            pass
        paper.fulltext_path = str(text_path)
        return FullTextResult(text=text, source=source, cached=True, error=None)

    def _write_cache(self, paper: Paper, text: str, source: str) -> None:
        directory = self._paper_dir(paper)
        directory.mkdir(parents=True, exist_ok=True)
        text_path = directory / "paper.md"
        text_tmp = directory / "paper.md.tmp"
        text_tmp.write_text(text, encoding="utf-8")
        text_tmp.replace(text_path)
        metadata = {
            "paper_id": paper.arxiv_id,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "characters": len(text),
        }
        meta_tmp = directory / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_tmp.replace(directory / "meta.json")
        paper.fulltext_path = str(text_path)

    def _is_valid(self, text: str, paper: Paper) -> bool:
        if len(text) < self.min_chars:
            return False
        normalized = _normalize(text)
        title_tokens = [token for token in _normalize(paper.title).split() if len(token) >= 5]
        identity_found = not title_tokens or any(token in normalized for token in title_tokens[:5])
        section_found = bool(
            re.search(
                r"(?im)^\s*(?:#{1,4}\s*)?(abstract|introduction|methods?|experiments?|conclusion|references)\b",
                text,
            )
        )
        return identity_found and section_found

    def _paper_dir(self, paper: Paper) -> Path:
        return self.cache_root / paper.arxiv_id.replace("/", "_")


class _ArticleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "nav"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in {"h1", "h2", "h3", "h4", "p", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "nav"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in {"h1", "h2", "h3", "h4", "p", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str):
        if not self._ignored_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _ArticleTextParser()
    parser.feed(html)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def _fetch_text(url: str) -> str:
    return _fetch_bytes(url).decode("utf-8")


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "arxiv-digest/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _normalize(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold())
