import re


PAGE_SPLIT_MARKER = re.compile(
    r"<\s*[-—]*\s*Page\s*(\d+)\s*Split\s*[-—]*\s*>",
    re.IGNORECASE,
)


def _split_pages(text: str) -> list[str]:
    """
    OCR 텍스트를 Page N Split 마커 기준으로 페이지 청크로 분해합니다.
    """
    if not text:
        return []

    chunks: list[str] = []
    last = 0
    for m in PAGE_SPLIT_MARKER.finditer(text):
        chunk = text[last:m.start()].strip()
        if chunk:
            chunks.append(chunk)
        last = m.end()

    tail = text[last:].strip()
    if tail:
        chunks.append(tail)

    return chunks


def split_ocr_text(text: str):
    """
    정책:
    - 53페이지 이상: 앞 3페이지 + 뒤 50페이지
    - 53페이지 미만: 전체를 front_ocr, back_ocr=None
    """
    normalized = (text or "").strip()
    if not normalized:
        return "", None

    if not PAGE_SPLIT_MARKER.search(normalized):
        return normalized, None

    page_chunks = _split_pages(normalized)
    if not page_chunks:
        return normalized, None

    total_pages = len(page_chunks)
    if total_pages >= 53:
        front_ocr = "\n\n".join(page_chunks[:3]).strip()
        back_ocr = "\n\n".join(page_chunks[-50:]).strip()
        return front_ocr, back_ocr

    front_ocr = normalized
    back_ocr = None

    return front_ocr, back_ocr
