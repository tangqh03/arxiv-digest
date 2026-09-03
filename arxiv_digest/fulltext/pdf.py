from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


class PdfExtractionError(ValueError):
    """PDF data could not be parsed into text."""


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except Exception:
        raise PdfExtractionError("Unable to extract text from PDF") from None
    text = text.strip()
    if not text:
        raise PdfExtractionError("PDF contains no extractable text")
    return text
