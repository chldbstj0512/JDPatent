"""OCR 전·후면 분리. 마커 `<— Page 3 Split —>` 는 OCR 파이프라인(US 장문)이 앞 3페이지 직후에 삽입한다."""

from typing import Optional, Tuple

PAGE3_SPLIT_MARKER = "<— Page 3 Split —>"


def split_ocr_text(text: str) -> Tuple[str, Optional[str]]:
    """
    - 마커 없음(한국·단문 US 등): front = 전체, back = None
    - 마커 있음(US 장문): front = 마커 이전(앞 3페이지), back = 마커 이후(뒤에서 OCR된 구간)
    """
    if not text:
        return "", None

    marker = PAGE3_SPLIT_MARKER
    if marker in text:
        front_ocr, back_ocr = text.split(marker, 1)
        return front_ocr.strip(), back_ocr.strip() or None

    return text.strip(), None
