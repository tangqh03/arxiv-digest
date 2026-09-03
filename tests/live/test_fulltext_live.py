import pytest

from arxiv_digest.fulltext.reader import FullTextReader
from arxiv_digest.sources.huggingface import HuggingFacePaperSource


pytestmark = pytest.mark.live


def test_fulltext_reader_live(tmp_path):
    papers = HuggingFacePaperSource().list_daily("today", limit=1)
    assert papers

    result = FullTextReader(cache_root=tmp_path).read(papers[0])

    assert result.source in {"hf", "arxiv-html", "arxiv-pdf"}
    assert len(result.text) > 2000
