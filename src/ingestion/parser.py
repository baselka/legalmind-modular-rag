"""
PDF/OCR parser.

Strategy:
1. Try PyMuPDF (fitz) first -- fast, high-quality text extraction for
   digitally-generated PDFs.
2. If extracted text is suspiciously short (likely a scanned image PDF),
   fall back to pytesseract OCR on each page rendered as a PIL image.

Why both?
  Law firms often work with both native PDFs (contracts signed electronically)
  and scanned documents (older case files).  Handling both keeps the pipeline
  robust without requiring a pre-processing step to classify each file.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


_OCR_DPI = 300
_MIN_TEXT_RATIO = 0.1  # chars per page below this triggers OCR fallback


def _page_to_pil(page: fitz.Page) -> Image.Image:
    """Render a single PDF page to a PIL image at _OCR_DPI."""
    mat = fitz.Matrix(_OCR_DPI / 72, _OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def extract_text_from_pdf(path: str | Path) -> str:
    """
    Extract full text from *path*.
    Returns a single string with page breaks preserved via double newlines.
    """
    doc = fitz.open(str(path))
    pages_text: list[str] = []

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)

    raw_text = "\n\n".join(pages_text)
    total_chars = len(raw_text.strip())
    total_pages = len(doc)

    # Heuristic: if avg chars-per-page is very low, the PDF is likely scanned.
    avg_chars = total_chars / max(total_pages, 1)
    if avg_chars < _MIN_TEXT_RATIO * 500:  # < 50 chars per page on average
        pages_text = []
        for page in doc:
            img = _page_to_pil(page)
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            pages_text.append(ocr_text)
        raw_text = "\n\n".join(pages_text)

    doc.close()
    return raw_text


def extract_text_from_bytes(content: bytes, filename: str = "document.pdf") -> str:
    """Parse a PDF from raw bytes (e.g. HTTP upload)."""
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text: list[str] = []

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)

    raw_text = "\n\n".join(pages_text)
    avg_chars = len(raw_text.strip()) / max(len(doc), 1)

    if avg_chars < 50:
        pages_text = []
        for page in doc:
            img = _page_to_pil(page)
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            pages_text.append(ocr_text)
        raw_text = "\n\n".join(pages_text)

    doc.close()
    return raw_text
