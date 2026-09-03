from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable

from arxiv_digest.models import Paper


ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "arxiv-digest/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


class ArxivPaperSource:
    def __init__(self, text_fetcher: Callable[[str], str] = fetch_text):
        self._text_fetcher = text_fetcher

    def fetch_by_ids(self, ids: Iterable[str]) -> list[Paper]:
        paper_ids = list(dict.fromkeys(ids))
        if not paper_ids:
            return []
        query = urllib.parse.urlencode(
            {"id_list": ",".join(paper_ids), "max_results": len(paper_ids)}
        )
        xml = self._text_fetcher(f"https://export.arxiv.org/api/query?{query}")
        return parse_arxiv_feed(xml)


def parse_arxiv_feed(xml: str) -> list[Paper]:
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        links = entry.findall("atom:link", ATOM_NS)
        abs_url = next(
            (
                link.get("href")
                for link in links
                if link.get("rel") == "alternate" and link.get("type") == "text/html"
            ),
            None,
        )
        pdf_url = next(
            (
                link.get("href")
                for link in links
                if link.get("type") == "application/pdf"
            ),
            None,
        )
        id_text = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        match = re.search(r"/(\d{4}\.\d+)(?:v\d+)?$", abs_url or id_text)
        papers.append(
            Paper(
                arxiv_id=match.group(1) if match else "",
                title=_clean(entry.findtext("atom:title", default="", namespaces=ATOM_NS)),
                abstract=_clean(
                    entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
                ),
                authors=[
                    name.text.strip()
                    for name in entry.findall("atom:author/atom:name", ATOM_NS)
                    if name.text and name.text.strip()
                ],
                published=_date(entry.findtext("atom:published", default="", namespaces=ATOM_NS)),
                updated=_date(entry.findtext("atom:updated", default="", namespaces=ATOM_NS)),
                categories=[
                    category.get("term", "")
                    for category in entry.findall("atom:category", ATOM_NS)
                    if category.get("term")
                ],
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def _clean(value: str) -> str:
    return " ".join(value.split())


def _date(value: str) -> str | None:
    return value[:10] or None
