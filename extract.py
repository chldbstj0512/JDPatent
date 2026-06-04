"""
특허 OCR → 메타데이터/청구 파이프라인 (프로덕션 지향 리팩터).

원칙: INID(51) IPC는 Python으로 CPC 혼입을 걸러 정규화(`ipc_codes_normalized`)한 뒤, 메타·청구의 나머지는
리팩터 이전과 동일한 **풀 LLM 프롬프트**로 추출한다. LLM이 반환한 `ipc_info`는 버리고,
strict IPC 목록과만 병합해 `ipc_info`·`ipc_code_count_llm`을 맞춘다.

1) Stage 1 — 결정론적: strict IPC, (감사·보조용) 인용 줄 카운트·청구 통계 등 `extract_structured_metadata`.
2) Stage 2 — 레거시 메타 LLM: 서지·NAICS·field·IPC 설명 등 전체 JSON (`METADATA_LEGACY_MODEL`).
3) Stage 3 — 레거시 청구 LLM 후 **결정론 병합**: 인용 수·청구 개수·독립항 전문·**청구 계열 수(종속 인용 그래프)**·**계열 수 진단 문구(LLM)**는 OCR·규칙/보조 LLM 기준, `ipc_count` 등은 LLM·후처리 유지.

임베딩·Pinecone NAICS fallback은 앞부분 OCR만 사용 (`truncate_ocr_for_metadata_embed`).
"""
import os
import re
import json
import time
import ast
from typing import Any, Optional
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

METADATA_REASONING_MODEL = os.getenv("METADATA_REASONING_MODEL", "gpt-4.1-mini")
CLAIM_REASONING_MODEL = os.getenv("CLAIM_REASONING_MODEL", "gpt-4.1")
# 리팩터 이전 파이프라인과 동일한 풀 프롬프트 호출(기본 모델도 동일).
METADATA_LEGACY_MODEL = os.getenv("METADATA_LEGACY_MODEL", "gpt-4o-mini")
CLAIM_LEGACY_MODEL = os.getenv("CLAIM_LEGACY_MODEL", "gpt-4o")
# 청구 계열 수 «왜 이렇게 나왔는지» 짧은 진단용(기본 경량 모델).
CLAIM_FAMILY_REASON_MODEL = os.getenv("CLAIM_FAMILY_REASON_MODEL", "gpt-4o-mini")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")      # aws
PINECONE_REGION = os.getenv("PINECONE_REGION")    # us-east-1
INDEX_NAME = os.getenv("INDEX_NAME")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)

_BAYESIAN_PROB_TABLE = None

# 메타데이터 추출 LLM·임베딩(Pinecone fallback)에만 동일하게 적용하는 OCR 앞부분 상한(글자 수).
# INID(51·57 등)는 대부분 문서 앞쪽이라 앞에서 자름. 최소 필요 글자(~3500 등)보다 크게 두고,
# 부족하면 환경변수 MAX_OCR_CHARS_METADATA 로 올리면 됨 (예: 16000, 20000).
_MAX_OCR_CHARS_FOR_METADATA_EMBED = int(os.getenv("MAX_OCR_CHARS_METADATA", "12000"))


def truncate_ocr_for_metadata_embed(text: str) -> str:
    """메타데이터/임베딩용으로만 사용. 베이지안 IPC 추출에는 전체 OCR을 쓴다."""
    if not text:
        return text
    return text[:_MAX_OCR_CHARS_FOR_METADATA_EMBED]


def embed_patent_text(text: str) -> list:
    # Embedding API 입력·비용 상한. 메타데이터 LLM과 동일한 앞부분 기준을 쓴다.
    text = truncate_ocr_for_metadata_embed(text)

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    embedding = response.data[0].embedding

    assert len(embedding) == 3072
    return embedding

def retrieve_top_naics(
    query_vector: list,
    top_k: int = 15
) -> list:
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    naics_candidates = []

    for match in result["matches"]:
        meta = match["metadata"]
        naics_candidates.append({
            "code": meta["NAICS_code"],      # 6자리
            "title": meta["NAICS_title"],
            "desc": meta["original_description"],
            "score": match["score"]
        })

    return naics_candidates

def normalize_ipc_code(ipc: str) -> str:
    """
    IPC 코드를 공백 없이 표준 형태로 정규화합니다.
    예) "A61B 3/00" -> "A61B3/00"
        "a 61 b 3 / 00" -> "A61B3/00"
    """
    if ipc is None:
        return ""
    s = str(ipc).upper()
    s = re.sub(r"\s+", "", s)  # remove all whitespace
    # keep only plausible IPC characters
    s = re.sub(r"[^A-Z0-9/]", "", s)
    return s

def _normalize_ocr_literals(s: str) -> str:
    """일부 OCR 파일에 JSON 이스케이프 형태로 박힌 \\n 등을 실제 개행으로 바꿉니다."""
    if not s:
        return s
    return (
        s.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t")
    )


# IPC: A–H + 2 digits + subclass letter + main/subgroup (CPC 혼입 시 subgroup 길이로 1차 필터)
IPC_PATTERN = re.compile(
    r"\b([A-H])\s*(\d{2})\s*([A-Z])\s*([0-9]{1,4})\s*/\s*([0-9]{1,4})\b"
)


def looks_like_cpc(code: str) -> bool:
    if "/" not in code:
        return False
    subgroup = code.split("/")[-1]
    return len(subgroup) > 2


def extract_ipc_codes_strict(text: str) -> list[str]:
    """
    INID (51) 구간에서만 IPC를 추출하고, CPC 스타일(서브그룹 길이>2) 후보는 제외합니다.
    """
    text_upper = _normalize_ocr_literals(text or "").upper()
    section_match = re.search(
        r"\(51\).*?(?:\(52\)|\(54\)|\(57\)|U\.?S\.\s*CL\.|CPC)",
        text_upper,
        flags=re.DOTALL,
    )
    if not section_match:
        return []
    section = section_match.group(0)
    seen: set[str] = set()
    results: list[str] = []
    for m in IPC_PATTERN.finditer(section):
        code = normalize_ipc_code(
            f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}/{m.group(5)}"
        )
        if not code or looks_like_cpc(code):
            continue
        if code not in seen:
            seen.add(code)
            results.append(code)
    return results


# (56) / References Cited / 표 제목으로 인용표 시작을 잡는다.
_CITATION_WINDOW_OPEN = re.compile(
    r"\(\s*56\s*\)|REFERENCES\s+CITED|선행기술조사문헌|"
    r"PATENT\s+CITATIONS|CITATION\s+LIST|"
    r"U\.S\.\s+PATENT\s+DOCUMENTS|FOREIGN\s+PATENT\s+DOCUMENTS",
    re.IGNORECASE,
)
# 명세·도면·요약으로 넘어가면 중단. BRIEF/BACKGROUND/DETAILED는 표와 멀리 떨어져 있어도
# 중간에 (56) 연속표가 있으므로 여기서 끊으면 안 된다(과소·과대 모두 발생).
_CITATION_WINDOW_CLOSE = re.compile(
    r"\(\s*57\s*\)|\(57\)\s*[–-]?\s*ABSTRACT|##\s*\(57\)\s*ABSTRACT|"
    r"OTHER\s+PUBLICATIONS|International\s+Search\s+Report|Primary\s+Examiner|"
    r"What\s+is\s+claimed|The\s+invention\s+claimed|^\s*Claims\s*$|청구범위|"
    r"<[—-]{1,3}\s*Page\s+\d+\s+Split\s*[—-]{1,3}>|"
    r"##\s*FIG(?:URE)?\.?\s*\d|\bFIG\.\s*\d",
    re.IGNORECASE | re.MULTILINE,
)

# US 등록: `5,619,644 A2` / `5,619,644 A *` / `8,503,238 B1 *`
_US_GRANT_PATENT_NUM = re.compile(
    r"\b(\d{1,3},\d{3},\d{3})\s+([AB]\d?)(?:\s*\*)?(?=\s|$|[*…])",
    re.IGNORECASE,
)
# US 공개 출원: `2008/0244338 A1`
_US_APP_PUBLICATION = re.compile(r"\b(20\d{2}/\d{7})\s+[AB]\d?", re.IGNORECASE)
# WO: `WO 2013089715 A1` / `WO2013089715`
_WO_PUBLICATION = re.compile(
    r"\bWO\s*[/\s,_-]*(\d{4}\s*\d{6,8})\s+[A-Z]\d?",
    re.IGNORECASE,
)


def _forward_citation_keys_in_span(span: str) -> set[str]:
    """인용 표 한 덩어에서 특허·공개번호 단위 키를 수집한다."""
    keys: set[str] = set()
    for line in span.splitlines():
        s = line.strip()
        if not s or len(s) > 400:
            continue
        for m in _US_GRANT_PATENT_NUM.finditer(s):
            keys.add(f"USG:{m.group(1).replace(',', '')}")
        for m in _US_APP_PUBLICATION.finditer(s):
            keys.add(f"USP:{m.group(1).upper()}")
        for m in _WO_PUBLICATION.finditer(s):
            w = re.sub(r"\s+", "", m.group(1))
            keys.add(f"WO:{w}")
    return keys


def extract_forward_citation_count(text: str) -> int:
    """
    선행 특허·출원문헌 **건수**(동일 문헌은 1회). (56) 등 인용표가 여러 번 나오면 각 창을 스캔해 합친다.
    명세 본문의 느슨한 숫자/날짜 줄은 세지 않는다(US `N,NNN,NNN` + kind, US 공개 `YYYY/NNNNNNN`, WO).
    """
    t = _normalize_ocr_literals(text or "")
    if not t.strip():
        return 0
    t_up = t.upper()
    seen: set[str] = set()
    for m in _CITATION_WINDOW_OPEN.finditer(t_up):
        start = m.start()
        window = t[start : start + 18000]
        skip = min(40, len(window))
        stop_m = _CITATION_WINDOW_CLOSE.search(window[skip:])
        end_rel = skip + stop_m.start() if stop_m else len(window)
        section = window[:end_rel]
        seen |= _forward_citation_keys_in_span(section)
    return len(seen)


# 청구 시작 앵커(우선순위 순). PDF Ctrl+F와 동일한 실무 키워드.
_CLAIM_START_ANCHOR_SPECS: list[tuple[str, re.Pattern[str]]] = [
    ("what_is_claimed", re.compile(r"What\s+is\s+claimed\s+is\s*:?", re.IGNORECASE)),
    ("invention_claimed", re.compile(r"The\s+invention\s+claimed\s+is\s*:?", re.IGNORECASE)),
    ("claims_heading", re.compile(r"(?m)^\s*Claims\s*$", re.IGNORECASE)),
    ("청구범위", re.compile(r"청구범위")),
    ("1_method_comprising", re.compile(r"(?m)^\s*1\.\s+A\s+method\s+comprising\s*:?", re.IGNORECASE)),
    ("1_apparatus_comprising", re.compile(r"(?m)^\s*1\.\s+An\s+apparatus\s+comprising\s*:?", re.IGNORECASE)),
    ("1_system_comprising", re.compile(r"(?m)^\s*1\.\s+A\s+system\s+comprising\s*:?", re.IGNORECASE)),
    ("1_method", re.compile(r"(?m)^\s*1\.\s+A\s+method\b", re.IGNORECASE)),
    ("1_apparatus", re.compile(r"(?m)^\s*1\.\s+An\s+apparatus\b", re.IGNORECASE)),
    ("1_system", re.compile(r"(?m)^\s*1\.\s+A\s+system\b", re.IGNORECASE)),
    ("claim_1", re.compile(r"(?i)\bclaim\s+1\b")),
    (
        "1_comprising",
        re.compile(
            r"(?m)^\s*1\.\s+(?:A|An)\s+[\w-]+(?:\s+[\w-]+){0,6}\s+comprising\s*:?",
            re.IGNORECASE,
        ),
    ),
]

CLAIM_SEC_START = _CLAIM_START_ANCHOR_SPECS[0][1]  # 하위 호환

CLAIM_NUM_EN = re.compile(r"(?:^|\n)\s*(\d{1,3})\.\s+(?=\S)", re.MULTILINE)
CLAIM_NUM_KO = re.compile(r"(?:^|\n)\s*제\s*(\d{1,3})\s*항\s*[:\.]?\s*", re.MULTILINE)

DEPENDENT_PATTERNS = [
    re.compile(r"claim\s+\d+", re.IGNORECASE),
    re.compile(r"claims\s+\d+", re.IGNORECASE),
    re.compile(r"according\s+to\s+claim", re.IGNORECASE),
    re.compile(r"of\s+claim\s+\d+", re.IGNORECASE),
    re.compile(r"제\s*\d+\s*항에\s*있어서"),
    re.compile(r"제\s*\d+\s*항에\s*따라"),
]


def claim_ocr_for_parsing(
    country: Optional[str],
    front: str,
    back: Optional[str] = None,
) -> str:
    """
    청구 파싱용 OCR 선택.
    - KR: 앞(전면)부터 — front 우선, back이 있으면 이어 붙임.
    - US(및 기타): 뒷면(back) 우선 — 장문 US는 청구·인용이 후반부에 있음.
    """
    front = (front or "").strip()
    back = (back or "").strip() if back else ""
    c = (country or "").upper()
    if c == "US" and back:
        return back
    if back:
        return front + ("\n" + back if front else "")
    return front


def find_claim_section_start(text: str) -> tuple[Optional[int], Optional[str]]:
    """실무 검색 키워드 순으로 청구 구간 시작 위치·앵커 id 반환."""
    if not text or not text.strip():
        return None, None
    for anchor_id, pat in _CLAIM_START_ANCHOR_SPECS:
        m = pat.search(text)
        if m:
            return m.start(), anchor_id
    return None, None


def _extract_claim_section(text: str, max_len: int = 250_000) -> str:
    start, _ = find_claim_section_start(text)
    if start is None:
        return ""
    return text[start : start + max_len]


def split_claims(claim_text: str) -> list[tuple[int, str]]:
    if not claim_text:
        return []
    matches = list(CLAIM_NUM_EN.finditer(claim_text))
    if len(matches) < 2:
        ko = list(CLAIM_NUM_KO.finditer(claim_text))
        if len(ko) > len(matches):
            matches = ko
    claims: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(claim_text)
        try:
            claim_num = int(match.group(1))
        except ValueError:
            continue
        claims.append((claim_num, claim_text[start:end].strip()))
    return claims


def is_dependent_claim(claim_text: str) -> bool:
    tl = claim_text.lower()
    return any(p.search(tl) for p in DEPENDENT_PATTERNS)


def extract_claim_statistics(claim_section: str) -> dict[str, Any]:
    claims = split_claims(claim_section)
    independent: list[tuple[int, str]] = []
    dependent: list[tuple[int, str]] = []
    for num, txt in claims:
        if is_dependent_claim(txt):
            dependent.append((num, txt))
        else:
            independent.append((num, txt))
    first_indep_text = ""
    if independent:
        first_indep_text = independent[0][1]
    return {
        "claim_count": len(claims),
        "independent_claim_count": len(independent),
        "dependent_claim_count": len(dependent),
        "independent_claims": independent,
        "dependent_claims": dependent,
        "first_independent_claim_text": first_indep_text,
        "all_claims": claims,
    }


def extract_claim_reference_numbers(claim_text: str) -> set[int]:
    """종속항 본문에서 인용하는 청구 번호(1..N)를 뽑는다."""
    nums: set[int] = set()
    if not claim_text:
        return nums
    t = claim_text[:8000]
    pats = [
        r"(?i)according\s+to\s+claim\s+(\d{1,3})",
        r"(?i)as\s+recited\s+in\s+claim\s+(\d{1,3})",
        r"(?i)per\s+claim\s+(\d{1,3})",
        r"(?i)of\s+claim\s+(\d{1,3})",
        r"(?i)from\s+claim\s+(\d{1,3})",
        r"(?i)claim\s+(\d{1,3})\s*[,;]",
        r"(?i)dependent\s+on\s+claim\s+(\d{1,3})",
    ]
    for pat in pats:
        for m in re.finditer(pat, t):
            for g in m.groups():
                if g:
                    try:
                        nums.add(int(g))
                    except ValueError:
                        pass
    for m in re.finditer(r"제\s*(\d{1,3})\s*항", t):
        try:
            nums.add(int(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r"제\s*(\d{1,3})\s*항\s*(?:내지|및)\s*제\s*(\d{1,3})\s*항", t):
        for g in m.groups():
            if g:
                try:
                    nums.add(int(g))
                except ValueError:
                    pass
    return nums


def build_claim_dependency_edge_digest(all_claims: Any, max_lines: int = 120) -> str:
    """각 청구 번호별로 파서가 잡은 인용 번호 목록(요약). LLM 진단용."""
    if not isinstance(all_claims, list) or not all_claims:
        return ""
    lines: list[str] = []
    for item in all_claims:
        if not item:
            continue
        try:
            n = int(item[0])
        except (TypeError, ValueError, IndexError):
            continue
        txt = item[1] if len(item) > 1 else ""
        if not isinstance(txt, str):
            txt = str(txt)
        refs = sorted(extract_claim_reference_numbers(txt))
        if refs:
            lines.append(f"{n}: -> {refs}")
        else:
            lines.append(f"{n}: (인용 번호 없음)")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def run_claim_family_reason_llm(
    structured: dict[str, Any],
    graph_family_count: int,
    all_claims: list[Any],
    legacy_llm_claim_family_count: Optional[int] = None,
) -> str:
    """
    청구 계열 수 **최종값(그래프)**이 왜 그 숫자인지 항·인용으로 설명하고,
    레거시 청구 LLM이 JSON에 넣었던 `claim_family_count`가 있으면 그 값을 **왜 그렇게 냈을지** 추정한다.
    `SKIP_CLAIM_FAMILY_REASON_LLM=1` 이면 호출하지 않는다.
    """
    if os.getenv("SKIP_CLAIM_FAMILY_REASON_LLM", "").strip().lower() in ("1", "true", "yes", "on"):
        return ""
    det = structured.get("claim_statistics") or {}
    digest = build_claim_dependency_edge_digest(all_claims)
    legacy_part = ""
    if legacy_llm_claim_family_count is not None:
        legacy_part = (
            f"레거시 청구 추출 LLM이 JSON에 제시한 `claim_family_count`는 **{legacy_llm_claim_family_count}**였다.\n"
            "이 값을 그렇게 판단했을 만한 근거(예: 독립항 유형만 세었을 가능성 등)를 **추정**하고, "
            f"아래 최종값 **{graph_family_count}**와 달라지는 이유를 한두 문장으로 짚어라.\n"
        )
    else:
        legacy_part = (
            "레거시 청구 LLM의 `claim_family_count`는 없거나 파싱하지 못했다. "
            f"최종값 **{graph_family_count}**가 어떻게 나왔는지에만 집중하라.\n"
        )

    payload = {
        "deterministic_claim_count": det.get("claim_count"),
        "deterministic_independent_claim_count": det.get("independent_claim_count"),
        "deterministic_dependent_claim_count": det.get("dependent_claim_count"),
        "final_claim_family_count_graph_union_find": graph_family_count,
        "legacy_llm_json_claim_family_count": legacy_llm_claim_family_count,
        "per_claim_parsed_citation_numbers": digest,
        "claim_section_preview_excerpt": (structured.get("claim_section_preview") or "")[:8000],
    }
    prompt = f"""
당신은 특허 청구항을 읽는 실무 보조 역할이다.

{legacy_part}

**최종 파이프라인이 채택한 청구항 계열 수는 {graph_family_count}**이다.
이 숫자는 소프트웨어가 각 청구 본문에서 **타 청구 인용 번호만** 뽑아 무방향 그래프의 **연결요소 개수**로 계산했다.

**질문 — 한국어로 `claim_family_count_reason` 한 필드에만 답하라:**

1) **왜 이 공보에서 계열 수가 {graph_family_count}인지**  
   `per_claim_parsed_citation_numbers`와 필요하면 `claim_section_preview_excerpt`를 근거로,
   **어떤 항 번호들이 인용으로 서로 묶여** 몇 개의 덩어리(계열)가 되었는지 **가능하면 계열별로**
   (예: 계열 A: 항 1,2 / 계열 B: 항 3,4 …) 구체적으로 설명하라.

2) 위에 레거시 LLM 숫자가 제시된 경우에만: 그 LLM이 그 숫자를 냈을 **가능한 판단 논리**와,
   최종값 {graph_family_count}와의 **차이**를 짧게 정리하라.

4~18문장, 불필요한 사과·메타말은 줄이고 항 번호·인용을 우선한다.

반드시 아래 JSON 한 객체만 출력한다 (다른 텍스트 금지):
{{"claim_family_count_reason": "..."}}

STATS_JSON:
{json.dumps(payload, ensure_ascii=False)}
"""
    try:
        response = client.chat.completions.create(
            model=CLAIM_FAMILY_REASON_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You output compact JSON only. Korean prose inside claim_family_count_reason.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2200,
        )
        content = response.choices[0].message.content.strip()
        obj = _parse_json_object(content)
        s = obj.get("claim_family_count_reason")
        if isinstance(s, str) and s.strip():
            return s.strip()[:12000]
    except Exception:
        pass
    return ""


def claim_family_count_from_dependency_graph(all_claims: list[tuple[int, str]]) -> int:
    """
    청구 계열 수(실무 정의에 가깝게 근사):

    - **독립항 수**: 청구항 본문상 독립으로 분류된 항의 개수(별도 집계).
    - **청구 계열 수**: 타 청구를 인용해 연결된 **기술적 묶음(연결요소)**의 개수.
      한 계열 안에 독립항이 여럿 있을 수 있어, 계열 수 ≤ 전체 청구 수이며 독립항 수와
      같지 않을 수 있다(흔히 계열 수 < 독립항 수).

    구현: 각 청구 번호를 노드로 하고, 본문에서 추출한 인용 번호(종속 패턴에 한정하지 않음)로
    무방향 union-find 한 뒤 연결요소 개수를 센다. (다중 인용·「제 n항 내지 제 m항」 반영)
    """
    if not all_claims:
        return 0
    nums: list[int] = []
    for item in all_claims:
        if not item:
            continue
        try:
            n = int(item[0])  # tuple or list
        except (TypeError, ValueError, IndexError):
            continue
        nums.append(n)
    if not nums:
        return 0
    parent = {n: n for n in nums}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    present = set(nums)
    for item in all_claims:
        if not item:
            continue
        try:
            n = int(item[0])
        except (TypeError, ValueError, IndexError):
            continue
        txt = item[1] if len(item) > 1 else ""
        if not isinstance(txt, str):
            txt = str(txt)
        for r in extract_claim_reference_numbers(txt):
            if r in present and r != n:
                union(n, r)
    return len({find(n) for n in nums})


def _strip_claim_ocr_noise(s: str) -> str:
    """페이지/슬라이드 마커·단독 `#` 줄 등 청구 본문에 끼어든 OCR 잡음을 제거한다."""
    if not s:
        return ""
    out_lines: list[str] = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            out_lines.append("")
            continue
        if re.match(r"^#+\s*\d*\s*$", t):
            continue
        if re.search(r"<[—-]{1,3}\s*Page\s+\d+", t, re.I):
            continue
        out_lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def extract_first_independent_claim_ocr_slice(
    claim_section: str,
    fallback_from_split: str,
    llm_independent_claim: Optional[str] = None,
) -> str:
    """
    청구 OCR에서 첫 독립항 전문: (1) LLM이 준 앞부분 문자열이 OCR에 있으면 그 위치부터,
    (2) 아니면 `1.` / `제1항`부터 다음 `2.` / `제2항` 직전까지 슬라이스.
    """
    raw = _normalize_ocr_literals(claim_section or "")
    if not raw.strip():
        return _strip_claim_ocr_noise((fallback_from_split or "").strip())

    start_idx: Optional[int] = None
    if llm_independent_claim:
        lv = re.sub(r"\s+", " ", str(llm_independent_claim).strip())
        if len(lv) >= 30:
            needle = lv[: min(100, len(lv))]
            pos = raw.find(needle)
            if pos >= 0:
                start_idx = pos

    def slice_from(start: int) -> str:
        tail = raw[start:]
        m2 = re.search(r"(?ms)^\s*2\.\s+", tail)
        mk = re.search(r"(?ms)^\s*제\s*2\s*항\s*", tail)
        if m2:
            return _strip_claim_ocr_noise(raw[start : start + m2.start()].strip())
        if mk:
            return _strip_claim_ocr_noise(raw[start : start + mk.start()].strip())
        return _strip_claim_ocr_noise(raw[start : start + 200000].strip())

    if start_idx is not None:
        return slice_from(start_idx)

    for _aid, pat in _CLAIM_START_ANCHOR_SPECS:
        if _aid in ("what_is_claimed", "invention_claimed", "claims_heading", "청구범위"):
            continue
        m_anchor = pat.search(raw)
        if m_anchor:
            return slice_from(m_anchor.start())

    m1 = re.search(r"(?ms)^\s*1\.\s+", raw)
    if m1:
        return slice_from(m1.start())
    mk1 = re.search(r"(?ms)^\s*제\s*1\s*항\s*", raw)
    if mk1:
        return slice_from(mk1.start())
    return _strip_claim_ocr_noise((fallback_from_split or "").strip())


def finalize_claims_with_deterministic(structured: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    """
    레거시 청구 LLM JSON 위에 결정론을 덮어쓴다.
    - 인용: (56) 등 짧은 창에서 특허·공개번호 단위로 dedupe(extract_forward_citation_count).
    - 청구 수/독립·종속: split_claims + 종속 패턴.
    - 계열 수: **인용 관계 그래프**의 연결요소 수(`claim_family_count_from_dependency_graph`).
      독립항 수와 다를 수 있으며 LLM 값은 사용하지 않는다.
    - `claim_family_count_reason`: 최종 계열 수(그래프)가 **왜 그 숫자인지** 항·인용으로 설명하고,
      레거시 청구 LLM이 JSON에 넣은 `claim_family_count`가 있으면 **그 값을 왜 그렇게 냈을지** 추정·대비(비활성: `SKIP_CLAIM_FAMILY_REASON_LLM=1`).
    - 독립항 전문: OCR 청구 구간에서 1.~2.(또는 LLM 앞문구 앵커)로 슬라이스 + 잡음 제거.
    """
    if not isinstance(llm, dict) or llm.get("error"):
        return llm
    det = structured.get("claim_statistics") or {}
    fwd = int(structured.get("forward_citation_count") or 0)
    cc = int(det.get("claim_count") or 0)
    indep = int(det.get("independent_claim_count") or 0)
    dep = int(det.get("dependent_claim_count") or 0)
    allc = det.get("all_claims")
    if not isinstance(allc, list):
        allc = []

    fam = claim_family_count_from_dependency_graph(allc) if cc > 0 else 0
    if cc > 0:
        fam = max(1, min(fam, cc))

    if cc > 0:
        indep = min(max(indep, 0), cc)
        dep = min(max(dep, 0), cc)
        if indep + dep > cc:
            dep = max(0, cc - indep)

    claim_full = str(structured.get("claim_section_full") or "")
    first_text = extract_first_independent_claim_ocr_slice(
        claim_full,
        str(det.get("first_independent_claim_text") or "").strip(),
        llm.get("independent_claim"),
    )

    out = dict(llm)
    out["forward_citation_count"] = fwd
    out["forward_citation_count_self_check"] = fwd
    out["claim_count"] = cc
    out["independent_claim_count"] = indep
    out["dependent_claim_count"] = dep
    out["claim_family_count"] = fam if cc > 0 else 0
    out["independent_claim"] = first_text[:200000]
    out["independent_claim_word_count"] = len(first_text.split()) if first_text else 0
    legacy_llm_fam: Optional[int] = None
    try:
        _lv = llm.get("claim_family_count")
        if _lv is not None and str(_lv).strip() != "":
            legacy_llm_fam = int(float(str(_lv).strip()))
    except (TypeError, ValueError):
        legacy_llm_fam = None
    out["claim_family_count_reason"] = (
        run_claim_family_reason_llm(structured, fam, allc, legacy_llm_fam) if cc > 0 else ""
    )
    return out


def _parse_inid_blocks(ocr: str) -> dict[int, str]:
    """
    INID (NN) 마커별 블록. 마커와 같은 줄에 오는 값(예: (71) Applicant: …)은
    본문에 반드시 포함한다 — 이전 구현은 첫 줄을 헤더에서만 소비해 (71)(73)(21) 등이 비는 버그가 있었다.
    """
    if not ocr:
        return {}
    pat = re.compile(
        r"(?:^|\n)\s*#*\s*\(\s*(\d{1,3})\s*\)\s*(.*?)(?:\n|$)",
        re.MULTILINE,
    )
    matches = list(pat.finditer(ocr))
    blocks: dict[int, str] = {}
    for i, m in enumerate(matches):
        try:
            nid = int(m.group(1))
        except ValueError:
            continue
        opening = (m.group(2) or "").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ocr)
        body = ocr[start:end].strip()
        if opening and body:
            blocks[nid] = (opening + "\n" + body).strip()
        elif body:
            blocks[nid] = body
        else:
            blocks[nid] = opening
    return blocks


def _first_line(block: Optional[str]) -> Optional[str]:
    if not block:
        return None
    for ln in block.splitlines():
        s = ln.strip()
        if s:
            return s
    return None


def _strip_biblio_labels(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = re.sub(r"\s+", " ", s.strip())
    for prefix in (
        r"(?i)^applicant[s]?\s*:\s*",
        r"(?i)^assignee[s]?\s*:\s*",
        r"(?i)^appl\.?\s*no\.?\s*:\s*",
        r"(?i)^application\s+number\s*:\s*",
        r"(?i)^filed\s*:\s*",
        r"(?i)^date\s+of\s+patent\s*:\s*",
        r"(?i)^patent\s+no\.?\s*:\s*",
    ):
        t = re.sub(prefix, "", t).strip()
    return t or None


def _infer_country(blocks: dict[int, str], head: str) -> Optional[str]:
    b19 = _first_line(blocks.get(19))
    if b19:
        u = b19.upper()
        if "U.S." in u or "UNITED STATES" in u or "US " in u:
            return "US"
        if "KOREA" in u or "대한민국" in b19 or "한국" in b19:
            return "KR"
    h = (head or "")[:800].upper()
    if "UNITED STATES PATENT" in h or "US PATENT" in h:
        return "US"
    if "공개특허" in (head or "") or "특허공보" in (head or ""):
        return "KR"
    return None


def _ipc_display(norm: str) -> str:
    if not norm or "/" not in norm:
        return norm
    body, frac = norm.split("/", 1)
    return f"{body[:4]} {body[4:]}/{frac}" if len(body) > 4 else norm


def extract_structured_metadata(user_ocr: str, back_ocr: Optional[str] = None) -> dict[str, Any]:
    """Stage 1: INID·IPC(51)·인용·청구 통계 등 결정론적 구조."""
    front = _normalize_ocr_literals(user_ocr or "")
    back = _normalize_ocr_literals(back_ocr or "") if back_ocr else ""
    combined = front + ("\n" + back if back else "")
    blocks = _parse_inid_blocks(front[:80000])
    blocks_c = _parse_inid_blocks(combined[:400000])

    title_blk = blocks.get(54)
    title = _strip_biblio_labels(_first_line(title_blk))
    if not title and title_blk:
        lines = title_blk.splitlines()
        title = lines[0].strip() if lines else None

    ab = blocks.get(57) or blocks_c.get(57)
    abstract = ab.strip() if ab else None

    app_raw = _first_line(blocks.get(71)) or _first_line(blocks.get(73))
    applicant_name = _strip_biblio_labels(app_raw)
    appl_no = _strip_biblio_labels(_first_line(blocks.get(21)))
    filed = _strip_biblio_labels(_first_line(blocks.get(22)))
    g_line = _first_line(blocks.get(11)) or _first_line(blocks.get(10))
    grant_number = _strip_biblio_labels(g_line)
    grant_date = _strip_biblio_labels(_first_line(blocks.get(45)))

    country = _infer_country(blocks, front)
    ipc_norm_list = extract_ipc_codes_strict(combined)
    fwd = extract_forward_citation_count(combined)
    claim_ocr = claim_ocr_for_parsing(country, front, back if back else None)
    claim_start, claim_anchor = find_claim_section_start(claim_ocr)
    claim_sec = _extract_claim_section(claim_ocr)
    if not claim_sec.strip() and claim_ocr != combined:
        claim_start, claim_anchor = find_claim_section_start(combined)
        claim_sec = _extract_claim_section(combined)
    cstats = extract_claim_statistics(claim_sec)

    return {
        "country": country,
        "title": title,
        "applicant_name": applicant_name,
        "applicant_number": appl_no,
        "applicant_date": filed,
        "grant_number": grant_number,
        "grant_date": grant_date,
        "abstract": abstract,
        "ipc_codes_normalized": ipc_norm_list,
        "forward_citation_count": fwd,
        "claim_statistics": cstats,
        "claim_section_full": claim_sec,
        "claim_section_preview": (claim_sec or "")[:16000],
        "claim_parse_ocr_source": "back" if (country or "").upper() == "US" and back else "front",
        "claim_section_anchor": claim_anchor,
        "claim_section_start_offset": claim_start,
    }


def _fill_applicant_from_ocr(result_item: dict, full_ocr: str) -> None:
    """출원인이 비어 있으면 OCR 앞부분에서 보조 추출."""
    if result_item.get("applicant_name"):
        return
    text = _normalize_ocr_literals(full_ocr or "")[:16000]
    if not text.strip():
        return
    patterns = [
        r"(?:\(71\)|\(73\))\s*([^\n\r]+?)(?=\s*(?:\(72\)|\(54\)|\(57\)|\(51\)|\n\s*\(\d))",
        r"(?:출원인|특허권자|어플리칸트)\s*[:：]\s*([^\n\r]+)",
        r"(?:Applicant|Assignee)\s*[:#]\s*([^\n\r]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            name = m.group(1).strip()
            name = re.sub(r"\s+", " ", name)
            if len(name) >= 2:
                result_item["applicant_name"] = name[:500]
                return


def _build_parse_audit(
    result_item: dict,
    user_ocr_full: str,
    back_ocr: Optional[str],
    claims: dict,
    metadata_ocr_text: str,
) -> dict:
    """파이프라인 점검: OCR 길이, INID(51) IPC 수 vs ipc_info, 청구 합계 등."""
    ocr_front_len = len(user_ocr_full or "")
    ocr_back_len = len(back_ocr or "") if back_ocr else 0
    meta_len = len(metadata_ocr_text or "")
    combined = (user_ocr_full or "") + ("\n" + (back_ocr or "") if back_ocr else "")
    ipc_strict_full = extract_ipc_codes_strict(combined)
    ipc_strict_meta = extract_ipc_codes_strict(metadata_ocr_text or "")
    ipc_info = result_item.get("ipc_info") if isinstance(result_item.get("ipc_info"), list) else []
    ipc_n = len(ipc_info)

    title = (result_item.get("title") or "").strip()
    abstract = (result_item.get("abstract") or "").strip()
    uo = user_ocr_full or ""

    def _in_ocr(snippet: str, ocr: str) -> Optional[bool]:
        if not snippet or not ocr:
            return None
        return snippet[: min(400, len(snippet))] in ocr

    claims_audit: dict = {}
    if isinstance(claims, dict) and "error" not in claims:
        try:
            cc = int(claims.get("claim_count") or 0)
            icn = int(claims.get("independent_claim_count") or 0)
            dcn = int(claims.get("dependent_claim_count") or 0)
            claims_audit = {
                "forward_citation_count": claims.get("forward_citation_count"),
                "claim_count": cc,
                "independent_claim_count": icn,
                "dependent_claim_count": dcn,
                "independent_plus_dependent_le_claim_count": (icn + dcn) <= cc + 1 if cc else None,
                "ipc_count_claims": claims.get("ipc_count"),
            }
        except (TypeError, ValueError):
            claims_audit = {"error": "claims_count_audit_failed"}

    return {
        "ocr_lengths": {
            "front_ocr_chars": ocr_front_len,
            "back_ocr_chars": ocr_back_len,
            "metadata_prompt_ocr_chars": meta_len,
            "metadata_max_chars_setting": _MAX_OCR_CHARS_FOR_METADATA_EMBED,
            "metadata_truncated": ocr_front_len > meta_len,
        },
        "ipc_counts": {
            "strict_ipc_count_inid51_full_ocr": len(ipc_strict_full),
            "strict_ipc_count_inid51_metadata_slice": len(ipc_strict_meta),
            "ipc_info_count_after_pipeline": ipc_n,
            "delta_strict_full_minus_ipc_info": len(ipc_strict_full) - ipc_n,
        },
        "verbatim_checks": {
            "title_substring_in_front_ocr": _in_ocr(title, uo),
            "abstract_prefix_in_front_ocr": _in_ocr(abstract[:200], uo) if abstract else None,
            "independent_claim_400chars_prefix_in_full_ocr": (
                _in_ocr(str(claims.get("independent_claim") or "")[:400], combined)
                if isinstance(claims, dict) and "error" not in claims and claims.get("independent_claim")
                else None
            ),
        },
        "claims_audit": claims_audit,
    }


def extract_ipc_codes_from_text(text: str) -> list[str]:
    """하위 호환: 전체 OCR이 아닌 INID(51) 구간의 IPC만 반환합니다."""
    return extract_ipc_codes_strict(text)

def _build_bayesian_prob_table(bayesian_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    bayesian_df columns expected:
    - ipc_list: list[str] (or string repr of list)
    - naic_list: list[str/int] (or string repr of list)
    Returns: { ipc_code: [(naics_code, P(naics|ipc)), ...sorted desc...] }
    """
    from collections import Counter

    pair_counts: Counter[tuple[str, str]] = Counter()
    ipc_counts: Counter[str] = Counter()

    for _, row in bayesian_df.iterrows():
        ipc_list = row.get("ipc_list", [])
        naic_list = row.get("naic_list", [])

        if isinstance(ipc_list, str):
            try:
                ipc_list = ast.literal_eval(ipc_list)
            except Exception:
                ipc_list = []
        if isinstance(naic_list, str):
            try:
                naic_list = ast.literal_eval(naic_list)
            except Exception:
                naic_list = []

        # normalize
        ipc_list = [normalize_ipc_code(x) for x in ipc_list if str(x).strip()]
        naic_list = [str(x).strip().split(".")[0] for x in naic_list if str(x).strip()]

        for ipc in ipc_list:
            ipc_counts[ipc] += 1
            for naic in naic_list:
                pair_counts[(ipc, naic)] += 1

    prob_table: dict[str, list[tuple[str, float]]] = {}
    for (ipc, naic), cnt in pair_counts.items():
        denom = ipc_counts.get(ipc, 0)
        if denom <= 0:
            continue
        prob = cnt / denom
        prob_table.setdefault(ipc, []).append((naic, prob))

    for ipc in list(prob_table.keys()):
        prob_table[ipc] = sorted(prob_table[ipc], key=lambda x: -x[1])

    return prob_table

def get_bayesian_prob_table(
    bayesian_csv_path: str = None,
) -> dict[str, list[tuple[str, float]]]:
    global _BAYESIAN_PROB_TABLE
    if _BAYESIAN_PROB_TABLE is not None:
        return _BAYESIAN_PROB_TABLE

    if bayesian_csv_path is None:
        bayesian_csv_path = os.path.join(os.path.dirname(__file__), "data", "bayesian_df.csv")

    bayesian_df = pd.read_csv(bayesian_csv_path)
    _BAYESIAN_PROB_TABLE = _build_bayesian_prob_table(bayesian_df)
    return _BAYESIAN_PROB_TABLE

def recommend_naics_from_ipcs(
    ipc_input: list[str],
    prob_table: dict[str, list[tuple[str, float]]],
    top_k: int = 15,
) -> list[tuple[str, float]]:
    from collections import Counter

    scores: Counter[str] = Counter()
    for ipc in ipc_input:
        ipc_norm = normalize_ipc_code(ipc)
        if ipc_norm in prob_table:
            for naic, prob in prob_table[ipc_norm]:
                scores[naic] += prob
    return scores.most_common(top_k)

def build_naics_candidates_from_bayesian(
    patent_text: str,
    naic_map: dict,
    top_k: int = 15,
    bayesian_csv_path: str = None,
) -> list[dict]:
    ipcs = extract_ipc_codes_from_text(patent_text)
    if not ipcs:
        return []

    prob_table = get_bayesian_prob_table(bayesian_csv_path=bayesian_csv_path)
    ranked = recommend_naics_from_ipcs(ipcs, prob_table=prob_table, top_k=top_k)

    candidates: list[dict] = []
    for code, score in ranked:
        info = naic_map.get(str(code), {})
        candidates.append(
            {
                "code": str(code),
                "title": info.get("title"),
                "desc": info.get("description"),
                "score": float(score),
            }
        )
    return candidates

def build_naics_context_text(naics_candidates: list) -> str:
    naics_text = "\n".join(
        [
            f"- {str(c['code']).split('.')[0]}: {c['title']} — {c['desc']}"
            for c in naics_candidates
        ]
    )

    return naics_text


def parse_naics_codes_from_context_block(naics_context: str) -> list[str]:
    """
    build_naics_context_text()가 만든 블록에서 NAICS 코드만 순서대로 추출합니다.
    (프롬프트의 NAICS CANDIDATES 섹션에 동일 문자열이 삽입됨)
    """
    codes: list[str] = []
    for line in naics_context.splitlines():
        s = line.strip()
        if not s.startswith("-"):
            continue
        m = re.match(r"^\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*:", s)
        if m:
            codes.append(m.group(1).split(".")[0])
    return codes


def verify_naics_candidates_vs_context(
    naics_candidates: list,
    naics_context: str,
    log_path: Optional[str] = None,
    candidate_source: Optional[str] = None,
    candidate_source_detail: Optional[dict] = None,
) -> dict:
    """
    naics_candidates 리스트와 naics_context(프롬프트에 들어가는 후보 블록)의 코드 목록이
    동일한지 검증합니다. 터미널이 잘릴 수 있어 결과를 파일로도 남길 수 있습니다.

    candidate_source:
      - "bayesian_ipc": bayesian_df 기반 P(NAICS|IPC) 합산 후 상위 후보
      - "embedding_pinecone": OCR 임베딩 + Pinecone 유사 벡터 검색 후보
    """
    from_candidates = [str(c["code"]).split(".")[0] for c in naics_candidates]
    from_context = parse_naics_codes_from_context_block(naics_context)
    ok = from_candidates == from_context

    source_labels = {
        "bayesian_ipc": {
            "label_ko": "베이지안(IPC→NAICS, bayesian_df 공출현)",
            "label_en": "Bayesian co-occurrence (IPC -> NAICS) from bayesian_df.csv",
        },
        "embedding_pinecone": {
            "label_ko": "임베딩 비교(Pinecone 유사 벡터 검색)",
            "label_en": "text-embedding-3-large + Pinecone vector similarity",
        },
    }

    report = {
        "ok": ok,
        "candidate_source": candidate_source,
        "candidate_source_labels": source_labels.get(candidate_source or "", {}),
        "codes_from_candidates": from_candidates,
        "codes_from_context_block": from_context,
        "n_candidates": len(from_candidates),
        "n_parsed": len(from_context),
    }
    if candidate_source_detail:
        report["candidate_source_detail"] = candidate_source_detail
    if not ok:
        for i, (a, b) in enumerate(zip(from_candidates, from_context)):
            if a != b:
                report["first_mismatch_index"] = i
                report["first_mismatch"] = {"candidate": a, "parsed_from_context": b}
                break
        else:
            if len(from_candidates) != len(from_context):
                report["reason"] = "length_mismatch"

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, indent=2))

    return report

def _parse_json_object(content: str) -> dict:
    content = (content or "").strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No valid JSON object found in model response")
    return json.loads(content[start:end])


def extract_patent_metadata_llm_legacy(text: str, naics_context: str):
    prompt = f"""
You are a patent classification and information extraction system.

You will be given:
1) Raw OCR text extracted from a patent publication.
2) A list of candidate NAICS industry codes (with descriptions).

You must first determine whether the provided document is an official patent publication.

A document qualifies as a patent publication ONLY if it clearly contains at least one of the following:
- A patent publication number (e.g., US 2023/0123456 A1, EP 1234567 B1, KR 10-2023-0123456, WO 2023/123456)
- An explicit label such as "Patent", "Patent Application", "Patent Publication", "공개특허", "특허공보", "公開特許", etc.
- A structured patent format including sections such as Abstract, Claims, Description, Inventors, Assignee, Filing Date, Publication Date.

If the document does NOT clearly satisfy these conditions,
OR if there is any uncertainty,
OR if the format does not match an official patent publication structure,

you MUST return exactly the following JSON and nothing else:

{{"error": "not_a_patent_document"}}

The OCR text may contain OCR errors, duplicated lines, broken line breaks,
or reading-order issues. Do NOT attempt to fix or rewrite the text.

Your tasks are:
(A) Extract bibliographic and technical fields STRICTLY according to INID codes.
(B) Select the MOST APPROPRIATE NAICS industry code(s) from the GIVEN CANDIDATES ONLY.

----------------------------------------
NAICS SELECTION RULES (VERY IMPORTANT)
----------------------------------------
- Please note that the provided NAICS codes are already sorted in order of relevance, so take this into consideration.
- You MUST choose NAICS code(s) ONLY from the candidate list provided below.
- Select the code(s) that BEST match the patent's technical field and application.
- Base your decision primarily on:
  - IPC codes (INID 51)
  - Abstract (INID 57)
- If exactly one NAICS code is clearly the best fit, return ONLY one.
- If two or three codes are strongly relevant, you MAY return up to three.
- NEVER invent or infer NAICS codes not present in the candidate list.
- Return NAICS codes as a list of 6-digit strings.
- If returning multiple NAICS codes, order them by relevance,
  with the most relevant code appearing first in the list.
- You MUST return at least ONE NAICS code.

----------------------------------------
IMPORTANT CONSTRAINTS
----------------------------------------
- Do NOT hallucinate missing information.
- Do NOT infer beyond the text.
- Output MUST be valid JSON.
- Output ONLY the JSON object.
- Return all extracted text fields (including title and abstract) in the ORIGINAL LANGUAGE of the patent as indicated by the country code.
- **Verbatim rule (OCR vs 파이프라인 구분용): title, abstract, applicant_name, ipc_info[].code 및 모든 인용용 문자열은
  아래 [OCR TEXT]에 **실제로 나타난 문자열을 그대로** 복사한다. 띄어쓰기·오탈자를 "교정"하지 말 것.
  정규화가 필요한 경우(날짜 ISO 등)는 해당 필드 설명에 한정한다.**
- The "field" value MUST be assigned to EXACTLY ONE of the five allowed categories ("compu", "bio", "comm", "elec", "etc"); it MUST NOT be null under any circumstances.

----------------------------------------
OUTPUT FORMAT (JSON ONLY)
----------------------------------------
Return a JSON object with EXACTLY the following fields.
Use null if a field cannot be confidently extracted.

{{
  "country": "US",
  "title": "Example Title",
  "applicant_name": "Example Applicant",
  "applicant_number": "17/123,456",
  "applicant_date": "2021-02-22",
  "grant_number": null,
  "grant_date": null,
  "naics_code": ["325414"],
  "abstract": "Example abstract text...",
  "field": "bio",
  "ipc_code_count_llm": 1,
  "ipc_info": [{{
    "code": "A61K 39/05",
    "short_description": "백신기술",
    "long_description": "면역 치료용 백신 조성물 관련 기술"
  }}]
}}

----------------------------------------
FIELD EXTRACTION RULES
----------------------------------------

1. country
- Infer from INID (19).
- If missing, infer from document kind or header in INID (12).
- Return ISO 2-letter country code (e.g., "US", "KR").
- Allowed exceptions: "PCT", "EP".
- If uncertain, return null.

2. title
- Extract ONLY from INID (54).
- Use the original text verbatim.

3. applicant_name
- Extract verbatim (출원인/특허권자 정보 — 누락 금지).
- Priority:
  - INID (71) Applicant name (U.S. and many foreign publications).
  - INID (73) Applicant/assignee for Korean publications (공보 표기).
  - If (71)/(73) 블록이 끊기거나 OCR로 식별 어려우면, 동일 면의 "출원인", "특허권자", "Applicant", "Assignee" 라벨 직후 텍스트를 사용.
  - INID (72) Inventor는 출원인으로 사용하지 말 것 (발명자와 혼동 금지).

4. applicant_number
- Extract ONLY from INID (21).

5. applicant_date
- Extract ONLY from INID (22).
- Convert to ISO format YYYY-MM-DD.

6. grant_number
- Extract ONLY from INID (11).

7. grant_date
- Extract ONLY from INID (45).

8. abstract
- Extract FULL abstract from INID (57).
- Preserve verbatim text.

9. ipc_code
- Extract ALL IPC codes under INID (51) WITHOUT OMITTING ANY.
- If multiple IPC codes are listed, you MUST return every single one.
- Do NOT summarize or filter.
- Preserve original formatting exactly as written (verbatim from OCR).
- Return them as a list in the same order as they appear in the document.
- The number of returned IPC codes MUST exactly match the number found in INID (51).
- After extraction, set **ipc_code_count_llm** to the integer count of IPC codes you placed in ipc_info (must equal len(ipc_info)).

10. ipc_long_description
- For EACH IPC code extracted under INID (51),
  provide a clear technical/industry-oriented explanation.
- Do NOT quote legal definitions.
- Explain in practical terms suitable for M&A or industry analysis.
- Return as a dictionary mapping IPC → description.

11. ipc_short_description
- For EACH IPC code,
  provide a VERY SHORT summary (within 7 Korean characters).
- No legal wording.
- Practical and intuitive.
- Return as a dictionary mapping IPC → short label.

12. field
- Assign EXACTLY ONE of:
  "compu", "bio", "comm", "elec", "etc"
- Decide primarily from IPC codes.

----------------------------------------
NAICS CANDIDATES (CHOOSE FROM THIS LIST ONLY)
----------------------------------------
{naics_context}

----------------------------------------
OCR TEXT (metadata prompt; 길이={len(text)} chars, 앞부분만 잘린 경우 MAX_OCR_CHARS_METADATA 참고)
----------------------------------------
{text}
"""

    response = client.chat.completions.create(
        model=METADATA_LEGACY_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract patent metadata and select the most appropriate NAICS codes from provided candidates."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=5000,
    )

    content = response.choices[0].message.content.strip()
    # JSON 안전 추출
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found in model response")

    return json.loads(content[start:end])


def merge_legacy_metadata_with_strict_ipc(
    legacy: dict[str, Any], structured: dict[str, Any]
) -> dict[str, Any]:
    """레거시 LLM 메타를 유지하되, IPC 목록·코드 표시만 strict 정규화 결과로 덮어쓴다."""
    if not isinstance(legacy, dict):
        return {"error": "metadata_legacy_invalid"}
    if legacy.get("error"):
        return legacy

    ipc_norms: list[str] = list(structured.get("ipc_codes_normalized") or [])
    legacy_infos = legacy.get("ipc_info")
    legacy_by_norm: dict[str, dict[str, Any]] = {}
    if isinstance(legacy_infos, list):
        for entry in legacy_infos:
            if not isinstance(entry, dict):
                continue
            code = entry.get("code") or ""
            n = normalize_ipc_code(str(code))
            if n and n not in legacy_by_norm:
                legacy_by_norm[n] = entry

    ipc_info: list[dict[str, str]] = []
    for norm in ipc_norms:
        le = legacy_by_norm.get(norm)
        if le:
            ipc_info.append(
                {
                    "code": _ipc_display(norm),
                    "short_description": str(le.get("short_description") or "—"),
                    "long_description": str(le.get("long_description") or "—"),
                }
            )
        else:
            ipc_info.append(
                {
                    "code": _ipc_display(norm),
                    "short_description": "—",
                    "long_description": "—",
                }
            )

    out = dict(legacy)
    out["ipc_info"] = ipc_info
    out["ipc_code_count_llm"] = len(ipc_info)
    return out


def extract_patent_claims_llm_legacy(
    text_front: str,
    text_back: str = None,
    country: Optional[str] = None,
):
    """
    OCR 입력(업스트림):
    - 50페이지 미만: text_front = 전체, text_back = None
    - 53페이지 이상(US): text_front = 앞 3페이지, text_back = 뒤에서 OCR(예: 후 50페이지)

    청구 추출 시 country에 따라 파싱 대상 OCR 선택(KR=앞부터, US=뒷면 우선).
    """

    claim_ocr = claim_ocr_for_parsing(country, text_front or "", text_back)
    front_snip = (text_front or "")[:12000]
    if text_back and (country or "").upper() == "US":
        combined_text = f"""
        [BIBLIO_FRONT_OCR — reference only, claims are usually NOT here]
        {front_snip}

        [CLAIM_PARSE_OCR — extract all claim fields from THIS section]
        {claim_ocr}
        """
    elif text_back:
        combined_text = f"""
        [FRONT_PART_OCR]
        {text_front}

        [BACK_PART_OCR]
        {text_back}
        """
    else:
        combined_text = claim_ocr or text_front

    prompt = f"""
You are a patent claim extraction and structural analysis system.

You will be given:
1) OCR text extracted from a patent publication.
   - If the patent has fewer than 50 pages: full text is provided.
   - If the patent has 53 pages or more: 
     the first 3 pages and the last 50 pages are provided separately.

If patent claims cannot be identified or extracted from the document, 
return only {{"error": "claims_not_found"}} in JSON format without any additional explanation.

The OCR text may contain:
- OCR noise
- line break errors
- duplicated lines
- broken numbering

You must strictly extract structured claim information according to the rules below.

----------------------------------------
YOUR TASKS
----------------------------------------

(A) Extract IPC codes (INID 51)
(B) Count forward citations (INID 56)
(C) Extract full claim section
(D) Count total claims (excluding deleted claims)
(E) Count dependent claims
(F) Extract independent claims and compute:
    - independent claim count
    - independent claim word count
(G) Compute claim family count (technical category count)

----------------------------------------
CRITICAL RULES (VERY IMPORTANT)
----------------------------------------

- Do NOT hallucinate.
- Do NOT infer beyond text.
- Do NOT rewrite claims.
- Preserve original language (Korean or English).
- **Verbatim rule: `independent_claim` 본문은 OCR에 나온 청구항 문자를 그대로 복사한다. 요약·교정·번역 금지.**
- Output MUST be valid JSON.
- Output ONLY the JSON object.
- If uncertain, return null (NOT empty string).

----------------------------------------
1. IPC EXTRACTION RULE
----------------------------------------

- Extract ALL IPC codes under INID (51).
- Count them.
- Do NOT return the individual IPC code list.
- Return ONLY the total count as ipc_count.

----------------------------------------
2. FORWARD CITATION COUNT RULE
----------------------------------------

Find citation section:
- English: "References Cited"
- Korean: "선행기술조사문헌"

Count ALL listed references.
If none exist, return 0.
forward_citation_count must exactly match visible entries.

If the citation section includes indications such as "(Continued)", 
you MUST also examine subsequent pages and include all additional listed references 
in the total count.

----------------------------------------
3. CLAIM SECTION EXTRACTION RULE
----------------------------------------

Locate claim section start (search in this practical order):
- "claim 1"
- "1. A method" / "1. An apparatus" / "1. A system"
- "1. A method comprising:" (and similar … comprising — very common in US patents)
- "comprising" near claim 1 wording
- "What is claimed is:" / "The invention claimed is:" / "Claims"
- Korean: "청구범위" / "제1항"

For US patents, prioritize the [CLAIM_PARSE_OCR] block (back pages). For KR, read from the start of the document stream.

Extract entire claim section.
Exclude deleted claims.
Deleted claim indicators:
- English: "(canceled)"
- Korean: "삭제"

Extract entire claim section to analyze.
Do NOT return the full claim section in the output.

----------------------------------------
4. TOTAL CLAIM COUNT RULE
----------------------------------------

Count numbered claims in claim section.
Exclude deleted claims.

Cross-check:
- English front text may state: "N claims"
- Korean front text may state: "총 N 항"

If mismatch occurs, prioritize actual visible claims.

You MUST carefully read each claim individually and explicitly determine whether it is independent or dependent based on its full textual content.
Do NOT rely only on pattern matching.
Analyze the legal structure of each claim before classifying it.
Only after identifying all independent claims, determine independent_claim_count.

Return:
claim_count

----------------------------------------
5. DEPENDENT CLAIM RULE
----------------------------------------

Dependent claim indicators:

English:
- "according to claim"
- "of claim"

Korean:
- "제 n항에 있어서"

Count them strictly.

Return:
dependent_claim_count

----------------------------------------
6. INDEPENDENT CLAIM RULE
----------------------------------------

A claim is independent if it does NOT refer to any other claim.

A claim is dependent if it explicitly refers to another claim.

You MUST read all claims up to the total claim_count and review them completely before providing the final answer.

Step 1:
Read EVERY numbered claim in the claim section from 1 to claim_count.

Step 2:
For each claim:
- If it refers to another claim → classify as dependent.
- If it does NOT refer to any other claim → classify as independent.

Step 3:
Extract the FULL TEXT of all independent claims.
Include the original claim number (e.g., "1.", "12.", "제1항").
Do NOT remove numbering.
Do NOT summarize.

Step 4:
Set independent_claim_count equal to the actual number of independent claims extracted above.

Step 5:
Calculate independent_claim_word_count from the extracted independent_claim text.
Use whitespace splitting for both English and Korean.

----------------------------------------
7. CLAIM FAMILY COUNT (informational only for this LLM call)
----------------------------------------

The downstream pipeline computes **claim_family_count** deterministically as the number of
**connected components** in the undirected graph formed by explicit **claim-to-claim references**
(e.g., "according to claim 5", "of claim 10", "제3항에 있어서"). Multiple independent claims can
fall in the same component, so claim_family_count often differs from independent_claim_count.

You may output `claim_family_count` as **0**; it will be replaced by software. Do not spend effort
categorizing "method vs apparatus" for this field.

Return:
claim_family_count (optional; 0 is fine)

----------------------------------------
OUTPUT FORMAT (JSON ONLY)
----------------------------------------

Return EXACTLY:

{{
  "claim_count": 0,
  "independent_claim_count": 0,
  "independent_claim_word_count": 0,
  "dependent_claim_count": 0,
  "claim_family_count": 0,
  "independent_claim": "full independent claim text",
  "ipc_count": 0,
  "forward_citation_count": 0,
  "forward_citation_count_self_check": 0
}}

`forward_citation_count_self_check` MUST equal `forward_citation_count` (re-count mentally before output).

----------------------------------------
OCR TEXT
----------------------------------------
{combined_text}
"""

    response = client.chat.completions.create(
        model=CLAIM_LEGACY_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract structured patent claim information with strict rule-based accuracy."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=10000,
    )

    content = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in model response")

    json_str = match.group(0)
    return json.loads(json_str)

def run_metadata_reasoning_llm(structured: dict[str, Any], naics_context: str) -> dict[str, Any]:
    """
    Stage 2: NAICS·field·IPC 설명만 LLM. IPC 코드 목록은 변경하지 않음.
    """
    payload = {
        "deterministic": {
            "country": structured.get("country"),
            "title": structured.get("title"),
            "applicant_name": structured.get("applicant_name"),
            "applicant_number": structured.get("applicant_number"),
            "applicant_date": structured.get("applicant_date"),
            "grant_number": structured.get("grant_number"),
            "grant_date": structured.get("grant_date"),
            "abstract": structured.get("abstract"),
            "ipc_codes_normalized": structured.get("ipc_codes_normalized") or [],
        }
    }
    prompt = f"""
You are a patent industry classification system.

You are NOT responsible for extracting IPC codes.
IPC codes were already extracted deterministically (see JSON keys ipc_codes_normalized).

Your tasks:
1) Select the most appropriate NAICS codes ONLY from the candidate list (by code relevance).
2) For EACH ipc_codes_normalized entry, provide short_description and long_description.
3) Assign exactly one field category: compu | bio | comm | elec | etc

IMPORTANT:
- Do NOT modify, add, or remove IPC codes.
- Do NOT invent NAICS codes outside the candidate list.
- Do NOT re-extract title, abstract, or bibliographic fields; use them only for reasoning.
- If the document is clearly NOT a patent publication, return {{"error": "not_a_patent_document"}} only.

Return valid JSON only, with this shape:
{{
  "naics_code": ["123456"],
  "field": "compu",
  "ipc_short_description": {{"G06F11/00": "짧은요약"}},
  "ipc_long_description": {{"G06F11/00": "긴 설명"}}
}}

Use ipc_codes_normalized strings EXACTLY as keys in the two description maps (same spelling as in the list).

NAICS CANDIDATES:
{naics_context}

STRUCTURED_INPUT_JSON:
{json.dumps(payload, ensure_ascii=False)}
"""
    response = client.chat.completions.create(
        model=METADATA_REASONING_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify patents and describe IPC entries; you never change IPC codes.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=3500,
    )
    content = response.choices[0].message.content.strip()
    return _parse_json_object(content)


def merge_structured_with_reasoning(structured: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(llm, dict):
        return {"error": "metadata_reasoning_invalid"}
    if llm.get("error"):
        return llm

    ipc_norms: list[str] = list(structured.get("ipc_codes_normalized") or [])
    shortd = llm.get("ipc_short_description") if isinstance(llm.get("ipc_short_description"), dict) else {}
    longd = llm.get("ipc_long_description") if isinstance(llm.get("ipc_long_description"), dict) else {}

    def _lookup_desc(m: dict, norm: str) -> str:
        if norm in m:
            return str(m[norm] or "")
        for k, v in m.items():
            if normalize_ipc_code(str(k)) == norm:
                return str(v or "")
        return ""

    ipc_info: list[dict[str, str]] = []
    for norm in ipc_norms:
        ipc_info.append(
            {
                "code": _ipc_display(norm),
                "short_description": _lookup_desc(shortd, norm) or "—",
                "long_description": _lookup_desc(longd, norm) or "—",
            }
        )

    return {
        "country": structured.get("country"),
        "title": structured.get("title"),
        "applicant_name": structured.get("applicant_name"),
        "applicant_number": structured.get("applicant_number"),
        "applicant_date": structured.get("applicant_date"),
        "grant_number": structured.get("grant_number"),
        "grant_date": structured.get("grant_date"),
        "abstract": structured.get("abstract"),
        "naics_code": llm.get("naics_code") or [],
        "field": llm.get("field") or "etc",
        "ipc_info": ipc_info,
    }


def extract_patent_metadata(text: str, naics_context: str) -> dict[str, Any]:
    """하위 호환: 리팩터 이전 풀 메타 LLM + INID(51) strict IPC 병합."""
    st = extract_structured_metadata(text, None)
    meta_ocr = truncate_ocr_for_metadata_embed(text)
    legacy = extract_patent_metadata_llm_legacy(meta_ocr, naics_context)
    return merge_legacy_metadata_with_strict_ipc(legacy, st)


def run_claim_reasoning_llm(structured: dict[str, Any], claim_preview: str) -> dict[str, Any]:
    """Stage 3: 청구 구조 해석만 (개수 재산출 금지)."""
    det = structured.get("claim_statistics") or {}
    allc = det.get("all_claims")
    if not isinstance(allc, list):
        allc = []
    graph_fam = claim_family_count_from_dependency_graph(allc)
    cc = int(det.get("claim_count") or 0)
    if cc > 0:
        graph_fam = max(1, min(graph_fam, cc))
    payload = {
        "deterministic_claim_count": det.get("claim_count"),
        "deterministic_independent_claim_count": det.get("independent_claim_count"),
        "deterministic_dependent_claim_count": det.get("dependent_claim_count"),
        "deterministic_claim_family_count_union_find": graph_fam,
        "per_claim_parsed_citation_numbers": build_claim_dependency_edge_digest(allc),
        "deterministic_first_independent_excerpt": (det.get("first_independent_claim_text") or "")[:4000],
    }
    prompt = f"""
You are a patent claim structure analysis system.

The claim texts were already split and counted deterministically.
Do NOT change claim_count. Do NOT invent new claims.

Your tasks:
1) Verify or correct independent_claim_count and dependent_claim_count (must sum consistently with claim_count).
2) Set `claim_family_count` in your JSON **exactly** to **{graph_fam}** (same as `deterministic_claim_family_count_union_find` in STATS_JSON). In `claim_family_count_reason` (Korean, 5–16 sentences): explain **why this patent has {graph_fam} claim families** — use `per_claim_parsed_citation_numbers` and, if needed, CLAIM_SECTION_PREVIEW to walk through **which claim numbers link via citations** and form each group. If you might have chosen a different count before seeing the graph, briefly note **what reasoning could have produced that** and how it differs from the graph-based {graph_fam}.
3) Compute independent_claim_word_count from the primary independent claim text (whitespace token count; Korean OK).
4) Provide independent_claim: verbatim full text of the first independent claim from the excerpt/preview if possible; otherwise use deterministic excerpt.

Return JSON only:
{{
  "independent_claim_count": 0,
  "dependent_claim_count": 0,
  "claim_family_count": 0,
  "claim_family_count_reason": "",
  "independent_claim_word_count": 0,
  "independent_claim": "..."
}}

DETERMINISTIC_STATS_JSON:
{json.dumps(payload, ensure_ascii=False)}

CLAIM_SECTION_PREVIEW:
{claim_preview[:12000]}
"""
    response = client.chat.completions.create(
        model=CLAIM_REASONING_MODEL,
        messages=[
            {"role": "system", "content": "You analyze claim dependencies and families; you do not recount total claims."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=8000,
    )
    content = response.choices[0].message.content.strip()
    try:
        return _parse_json_object(content)
    except Exception:
        return {}


def merge_claim_reasoning(structured: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    det = structured.get("claim_statistics") or {}
    cc = int(det.get("claim_count") or 0)
    fwd = int(structured.get("forward_citation_count") or 0)
    ipc_n = len(structured.get("ipc_codes_normalized") or [])

    def _pick_int(key: str, fallback: int) -> int:
        if not isinstance(llm, dict):
            return fallback
        v = llm.get(key)
        if v is None:
            return fallback
        try:
            return int(v)
        except (TypeError, ValueError):
            return fallback

    indep = _pick_int("independent_claim_count", int(det.get("independent_claim_count") or 0))
    dep = _pick_int("dependent_claim_count", int(det.get("dependent_claim_count") or 0))
    if cc > 0:
        indep = min(max(indep, 0), cc)
        dep = min(max(dep, 0), cc)
        if indep + dep > cc:
            dep = max(cc - indep, 0)

    allc = det.get("all_claims")
    if not isinstance(allc, list):
        allc = []
    fam = claim_family_count_from_dependency_graph(allc) if cc > 0 else 0
    if cc > 0:
        fam = max(1, min(fam, cc))
    reason = ""
    if isinstance(llm, dict):
        r0 = llm.get("claim_family_count_reason")
        if isinstance(r0, str) and r0.strip():
            reason = r0.strip()[:12000]
    legacy_llm_fam: Optional[int] = None
    if isinstance(llm, dict):
        try:
            _lv = llm.get("claim_family_count")
            if _lv is not None and str(_lv).strip() != "":
                legacy_llm_fam = int(float(str(_lv).strip()))
        except (TypeError, ValueError):
            legacy_llm_fam = None
    if not reason and cc > 0:
        reason = run_claim_family_reason_llm(structured, fam, allc, legacy_llm_fam)

    if isinstance(llm, dict) and llm.get("independent_claim"):
        indep_text = str(llm["independent_claim"])
    else:
        indep_text = str(det.get("first_independent_claim_text") or "")

    iwc = _pick_int("independent_claim_word_count", len(indep_text.split()) if indep_text else 0)
    if isinstance(llm, dict) and llm.get("independent_claim_word_count") is not None:
        iwc = _pick_int("independent_claim_word_count", iwc)

    return {
        "claim_count": cc,
        "independent_claim_count": indep,
        "dependent_claim_count": dep,
        "claim_family_count": fam,
        "claim_family_count_reason": reason,
        "independent_claim_word_count": iwc,
        "independent_claim": indep_text[:20000],
        "ipc_count": ipc_n,
        "forward_citation_count": fwd,
    }


def extract_patent_claims(text_front: str, text_back: str = None) -> dict[str, Any]:
    """
    레거시 청구 LLM 호출 후, 인용·청구 수·첫 독립항 본문·**청구 계열 수(인용 그래프)**·**계열 수 진단(reason)**는 `extract_structured_metadata` 결정값·보조 LLM으로 병합한다.
    text_front / text_back 규칙은 `extract_patent_claims_llm_legacy` docstring 참고.
    결정론 파서가 청구를 찾았는데 LLM만 실패한 경우 결정론 결과만으로 완료한다.
    """
    structured = extract_structured_metadata(text_front, text_back)
    country = structured.get("country")
    llm = extract_patent_claims_llm_legacy(text_front, text_back, country=country)
    det_cc = int((structured.get("claim_statistics") or {}).get("claim_count") or 0)
    if not isinstance(llm, dict) or llm.get("error"):
        if det_cc > 0:
            return finalize_claims_with_deterministic(structured, {})
        return llm if isinstance(llm, dict) else {"error": "claims_not_found"}
    return finalize_claims_with_deterministic(structured, llm)

def run_NAIC_extract(
    naic_df,
    user_id,
    user_ocr,      # front OCR (기존 NAICS/metadata용)
    back_ocr=None  # claim 추출용
):
    results = []
    claims = None
    patent_meta = None

    # ---------------------------------
    # 0. NAICS 코드 매핑 (후보 생성에도 사용)
    # ---------------------------------
    naic_map = {
        str(code): {
            "title": title,
            "description": desc
        }
        for code, title, desc in zip(
            naic_df["naics_code"].astype(str),
            naic_df["naics_title"],
            naic_df["description"]
        )
    }

    # ---------------------------------
    # 1. NAICS / Metadata (베이지안 IPC는 전·후면 OCR 합본 기준, 임베딩은 전면만)
    # ---------------------------------
    patent_text = user_ocr
    metadata_ocr_text = truncate_ocr_for_metadata_embed(patent_text)
    combined_ocr = (user_ocr or "") + ("\n" + (back_ocr or "") if back_ocr else "")

    # 후보 생성: 베이지안 기반 (IPC -> NAICS)
    naics_candidates = build_naics_candidates_from_bayesian(
        patent_text=combined_ocr,
        naic_map=naic_map,
        top_k=15,
    )
    naics_candidate_source = "bayesian_ipc"
    verify_detail: dict = {
        "bayesian_attempted": True,
        "ipc_codes_used_for_bayesian": extract_ipc_codes_from_text(combined_ocr),
    }

    # IPC 추출 실패 등으로 후보가 없으면 기존 임베딩/Pinecone 방식으로 fallback
    if not naics_candidates:
        verify_detail["bayesian_yielded_candidates"] = False
        verify_detail["fallback_reason"] = (
            "no_ipc_extracted_from_ocr_or_no_hits_in_prob_table"
            if not verify_detail["ipc_codes_used_for_bayesian"]
            else "bayesian_ranking_empty"
        )
        query_vector = embed_patent_text(patent_text)
        naics_candidates = retrieve_top_naics(query_vector, top_k=15)
        naics_candidate_source = "embedding_pinecone"
        verify_detail["embedding_model"] = "text-embedding-3-large"
        verify_detail["pinecone_index"] = INDEX_NAME
    else:
        verify_detail["bayesian_yielded_candidates"] = True

    naics_context = build_naics_context_text(naics_candidates)

    if naics_candidates:
        verify_path = os.path.join(os.path.dirname(__file__), "naics_candidate_verify.json")
        vrep = verify_naics_candidates_vs_context(
            naics_candidates,
            naics_context,
            log_path=verify_path,
            candidate_source=naics_candidate_source,
            candidate_source_detail=verify_detail,
        )
        if vrep["ok"]:
            print(
                f"[NAICS verify] OK: {vrep['n_candidates']} codes "
                f"({naics_candidate_source}) — "
                f"prompt NAICS block matches candidates (log: {verify_path})",
                flush=True,
            )
        else:
            print(
                f"[NAICS verify] FAIL ({naics_candidate_source}): "
                f"prompt block != candidates — see {verify_path}",
                flush=True,
            )

    structured = extract_structured_metadata(user_ocr, back_ocr)

    try:
        llm_meta = extract_patent_metadata_llm_legacy(metadata_ocr_text, naics_context)
        patent_meta = merge_legacy_metadata_with_strict_ipc(llm_meta, structured)
        if isinstance(patent_meta, str):
            patent_meta = json.loads(patent_meta)
    except Exception:
        # return [], {"error": "metadata_extraction_failed"}
        results = []
        claims = {"error": "metadata_extraction_failed"}
        patent_meta = None

    if patent_meta is not None and isinstance(patent_meta, dict) and "error" in patent_meta:
        # return [], patent_meta
        results = []
        claims = patent_meta
        patent_meta = None

    if patent_meta is not None:
        result_item = {
            "pdf_name": user_id,
            **patent_meta
        }

        # ---------------------------------
        # 3. NAICS fallback 처리
        # ---------------------------------
        codes = result_item.get("naics_code", [])

        if not codes and naics_candidates:
            fallback_code = str(naics_candidates[0]["code"])
            codes = [fallback_code]
            result_item["naics_code"] = codes

        primary_code = codes[0] if codes else None
        candidate_codes = codes[1:] if len(codes) > 1 else []

        result_item["primary_naic_info"] = (
            {
                "code": str(primary_code),
                "title": naic_map.get(str(primary_code), {}).get("title"),
                "description": naic_map.get(str(primary_code), {}).get("description")
            }
            if primary_code else None
        )

        result_item["candidate_naic_info"] = [
            {
                "code": str(code),
                "title": naic_map.get(str(code), {}).get("title"),
                "description": naic_map.get(str(code), {}).get("description")
            }
            for code in candidate_codes
        ]

        results.append(result_item)
        _fill_applicant_from_ocr(result_item, user_ocr)

        # ---------------------------------
        # 4. 청구 (레거시 LLM + 결정론 병합)
        # ---------------------------------
        try:
            claims = extract_patent_claims(
                text_front=user_ocr,
                text_back=back_ocr,
            )
            if isinstance(claims, str):
                claims = json.loads(claims)
        except Exception:
            # return results, {"error": "claim_extraction_failed"}
            claims = {"error": "claim_extraction_failed"}

        if isinstance(claims, dict) and "error" not in claims:
            ri = results[0]
            fc = claims.get("forward_citation_count")
            ri["forward_citation_count"] = int(fc) if fc is not None else 0
            claims["ipc_count"] = max(
                int(claims.get("ipc_count") or 0),
                len(ri.get("ipc_info") or []),
            )
            if claims.get("forward_citation_count") is None:
                claims["forward_citation_count"] = ri["forward_citation_count"]

        ri = results[0]
        ri["parse_audit"] = _build_parse_audit(ri, user_ocr, back_ocr, claims, metadata_ocr_text)

        # if isinstance(claims, dict) and "error" in claims:
        #     return results, claims

    return results, claims
