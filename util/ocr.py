"""OCR front/back 분리 유틸."""

import re
from typing import Optional, Tuple


PAGE3_SPLIT_MARKER = "<— Page 3 Split —>"
PAGE_SPLIT_MARKER = re.compile(
    r"<\s*[-—]+\s*Page\s*(\d+)\s*Split\s*[-—]+\s*>",
    re.IGNORECASE,
)


def _split_pages(text: str) -> list[str]:
    chunks: list[str] = []
    last = 0
    for match in PAGE_SPLIT_MARKER.finditer(text):
        chunk = text[last:match.start()].strip()
        if chunk:
            chunks.append(chunk)
        last = match.end()

    tail = text[last:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def split_ocr_text(text: str) -> Tuple[str, Optional[str]]:
    """
    - 마커 없음: front = 전체, back = None
    - `<— Page 3 Split —>` 있음: front = 앞 3페이지, back = 후면 OCR 구간
    - 페이지별 split 마커 있음: 53페이지 이상이면 앞 3페이지 + 뒤 50페이지
    """
    normalized = (text or "").strip()
    if not normalized:
        return "", None

    if PAGE3_SPLIT_MARKER in normalized:
        front_ocr, back_ocr = normalized.split(PAGE3_SPLIT_MARKER, 1)
        return front_ocr.strip(), back_ocr.strip() or None

    if PAGE_SPLIT_MARKER.search(normalized):
        page_chunks = _split_pages(normalized)
        if len(page_chunks) >= 53:
            front_ocr = "\n\n".join(page_chunks[:3]).strip()
            back_ocr = "\n\n".join(page_chunks[-50:]).strip()
            return front_ocr, back_ocr or None

    return normalized, None
