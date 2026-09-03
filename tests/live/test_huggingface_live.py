import pytest

from arxiv_digest.sources.huggingface import HuggingFacePaperSource


pytestmark = pytest.mark.live


def test_huggingface_daily_live():
    papers = HuggingFacePaperSource().list_daily("today", limit=5)

    assert len(papers) >= 1
    assert papers[0].arxiv_id
    assert papers[0].title
