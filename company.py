import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

from extract import (
    get_bayesian_prob_table,
    normalize_ipc_code,
    recommend_naics_from_ipcs,
)


# ==========================================
# 초기화
# ==========================================

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

_SEARCH_TOP_K = 600
_IPC_SUBCLASS_LEN = 4  # e.g. "A61B"
_FOREIGN_MIX_RATIO = 0.4
_US_BOOST_TOP_K = 600
_MIN_US_SLOTS_FOR_KR = 2


# ==========================================
# 유틸
# ==========================================

def get_embedding(text: str, model: str = "text-embedding-3-large") -> Optional[List[float]]:
    text = text.replace("\n", " ").strip()
    if not text:
        return None
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def parse_ipc_codes(ipc_string: str) -> List[str]:
    """파이프(|) 구분 IPC 문자열에서 정규화된 코드 리스트 반환."""
    if not ipc_string:
        return []
    return [normalize_ipc_code(part) for part in ipc_string.split("|") if part.strip()]


def ipc_subclasses(ipc_codes: List[str]) -> set:
    """IPC 코드 리스트 → 서브클래스(앞 4자리) 집합. e.g. {'A61B', 'G06F'}"""
    result = set()
    for code in ipc_codes:
        norm = normalize_ipc_code(code)
        if len(norm) >= _IPC_SUBCLASS_LEN:
            result.add(norm[:_IPC_SUBCLASS_LEN])
    return result


@lru_cache(maxsize=1024)
def _estimate_naics_4digit_cached(ipc_tuple: tuple) -> frozenset:
    """IPC 코드 튜플 → 추정 NAICS 앞 4자리 (캐시 적용)."""
    if not ipc_tuple:
        return frozenset()
    prob_table = get_bayesian_prob_table()
    ranked = recommend_naics_from_ipcs(list(ipc_tuple), prob_table, top_k=10)
    return frozenset(code[:4] for code, _ in ranked if len(code) >= 4)


def estimate_naics_4digit(ipc_codes: List[str]) -> set:
    return set(_estimate_naics_4digit_cached(tuple(ipc_codes)))


# ==========================================
# 국가/언어 다양성 리랭킹 유틸
# ==========================================

def _normalize_country(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if not s:
        return None

    aliases = {
        "KOREA": "KR",
        "SOUTH KOREA": "KR",
        "REPUBLIC OF KOREA": "KR",
        "KOR": "KR",
        "대한민국": "KR",
        "한국": "KR",
        "USA": "US",
        "U.S.": "US",
        "U.S.A.": "US",
        "UNITED STATES": "US",
        "US": "US",
    }
    if s in aliases:
        return aliases[s]

    if re.fullmatch(r"[A-Z]{2}", s):
        return s
    return None


def _extract_country_from_metadata(meta: dict) -> str | None:
    if not isinstance(meta, dict):
        return None

    country_keys = [
        "country",
        "country_code",
        "patent_country",
        "publication_country",
        "nation",
        "origin_country",
    ]
    for key in country_keys:
        country = _normalize_country(meta.get(key))
        if country:
            return country

    number_keys = [
        "publication_number",
        "patent_number",
        "application_number",
        "grant_number",
        "patent_id",
        "doc_number",
    ]
    for key in number_keys:
        raw = meta.get(key)
        if not raw:
            continue
        raw_s = str(raw).strip()
        m = re.search(r"\b([A-Z]{2})\s*[\d/\-]", raw_s.upper())
        if m:
            country = _normalize_country(m.group(1))
            if country:
                return country
        # US 출원번호는 "17/123,456"처럼 국가코드 없이 slash 형식인 경우가 많다.
        if key in {"application_number", "patent_number"}:
            compact = re.sub(r"\s+", "", raw_s)
            if compact.startswith(("10", "20")) and re.fullmatch(r"\d{10,14}", compact):
                return "KR"
            if re.fullmatch(r"\d{2}/\d{3},\d{3}", compact) or re.fullmatch(r"\d{2}/\d{6}", compact):
                return "US"
            if re.fullmatch(r"\d{7,9}", compact):
                # US grant number(예: 11234567) 형태
                return "US"

    return None


def _mix_company_ranking_by_country(
    ranked: List[dict],
    user_country: str | None,
    top_k: int,
) -> List[dict]:
    if not ranked:
        return []
    if top_k <= 0:
        return []

    user_country_norm = _normalize_country(user_country)
    if not user_country_norm:
        return ranked[:top_k]

    domestic = [r for r in ranked if r.get("_dominant_country") == user_country_norm]
    foreign = [
        r
        for r in ranked
        if r.get("_dominant_country") and r.get("_dominant_country") != user_country_norm
    ]
    unknown = [r for r in ranked if not r.get("_dominant_country")]

    # KR 입력 시 US 특허가 상대적으로 묻히는 문제를 완화하기 위해 US를 우선 노출
    foreign_priority: List[dict]
    if user_country_norm == "KR":
        us_foreign = [r for r in foreign if r.get("_dominant_country") == "US"]
        non_us_foreign = [r for r in foreign if r.get("_dominant_country") != "US"]
        foreign_priority = us_foreign + non_us_foreign
    else:
        foreign_priority = foreign

    foreign_target = max(1, int(round(top_k * _FOREIGN_MIX_RATIO)))
    foreign_take = min(len(foreign_priority), foreign_target)
    domestic_take = min(len(domestic), max(0, top_k - foreign_take))

    mixed = domestic[:domestic_take] + foreign_priority[:foreign_take]

    if len(mixed) < top_k:
        used_ids = {id(x) for x in mixed}
        backfill = [r for r in (domestic + foreign_priority + unknown) if id(r) not in used_ids]
        mixed.extend(backfill[: top_k - len(mixed)])

    if user_country_norm == "KR":
        current_us = sum(1 for x in mixed if x.get("_dominant_country") == "US")
        required_us = min(_MIN_US_SLOTS_FOR_KR, top_k)
        if current_us < required_us:
            need = required_us - current_us
            us_pool = [
                r for r in foreign_priority
                if r.get("_dominant_country") == "US" and id(r) not in {id(x) for x in mixed}
            ]
            # US 슬롯 보장을 위해 가장 낮은 점수의 non-US 항목부터 교체
            replaceable = sorted(
                [i for i, r in enumerate(mixed) if r.get("_dominant_country") != "US"],
                key=lambda i: mixed[i]["avg_score"],
            )
            for i in range(min(need, len(us_pool), len(replaceable))):
                mixed[replaceable[i]] = us_pool[i]

    mixed.sort(key=lambda x: -x["avg_score"])
    return mixed[:top_k]


def _company_key(item: dict) -> str:
    return str(item.get("company_id") or item.get("company_name") or "")


def _merge_ranked_unique(primary: List[dict], supplement: List[dict]) -> List[dict]:
    merged: List[dict] = []
    seen = set()
    for row in primary + supplement:
        key = _company_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _query_matches(index, query_embedding: List[float], top_k: int, metadata_filter: dict | None = None) -> list:
    kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    result = index.query(**kwargs)
    return list(result.matches or [])


def _fetch_us_boosted_matches(index, query_embedding: List[float]) -> list:
    us_matches: list = []
    us_filters = [
        {"country": {"$eq": "US"}},
        {"country_code": {"$eq": "US"}},
        {"patent_country": {"$eq": "US"}},
        {"publication_country": {"$eq": "US"}},
        {"nation": {"$eq": "US"}},
    ]

    for f in us_filters:
        try:
            chunk = _query_matches(index, query_embedding, _US_BOOST_TOP_K, metadata_filter=f)
        except Exception:
            continue
        if chunk:
            us_matches.extend(chunk)

    # 메타 필터가 안 듣는 인덱스 대비 fallback: 깊은 조회 후 메타에서 US만 선별
    if not us_matches:
        try:
            deep_matches = _query_matches(index, query_embedding, _US_BOOST_TOP_K)
        except Exception:
            deep_matches = []
        us_matches = [
            m for m in deep_matches
            if _extract_country_from_metadata(getattr(m, "metadata", {}) or {}) == "US"
        ]

    dedup: Dict[str, object] = {}
    for m in us_matches:
        match_id = str(getattr(m, "id", "") or "")
        if not match_id:
            match_id = str(id(m))
        prev = dedup.get(match_id)
        if prev is None or getattr(m, "score", 0.0) > getattr(prev, "score", 0.0):
            dedup[match_id] = m

    merged = list(dedup.values())
    merged.sort(key=lambda x: -float(getattr(x, "score", 0.0)))
    return merged


# ==========================================
# 필터링 + 기업 그룹핑 (공통 로직)
# ==========================================

def _filter_and_group(
    matches: list,
    user_naics_4digit: str,
    user_subclasses: set,
    name_key: str,
    id_key: str,
    require_ipc_overlap: bool,
    require_naics_match: bool = True,
) -> list:
    """
    Pinecone 검색 결과를 필터링 → 기업 단위 그룹핑 → 유사도 평균 정렬.
    require_ipc_overlap=False 면 IPC 2차 필터를 건너뛴다 (완화 모드).
    """
    company_groups: Dict[str, dict] = defaultdict(lambda: {
        "scores": [],
        "patents": [],
        "company_name": "",
        "company_id": "",
        "country_counts": defaultdict(int),
    })

    for match in matches:
        meta = match.metadata
        patent_ipc_raw = meta.get("ipc_code", "") or ""
        patent_ipc_codes = parse_ipc_codes(patent_ipc_raw)

        # --- NAICS 1차 필터 (앞 4자리 기준) ---
        if require_naics_match and user_naics_4digit and patent_ipc_codes:
            patent_naics_4digits = estimate_naics_4digit(patent_ipc_codes)
            if not patent_naics_4digits:
                continue
            if user_naics_4digit not in patent_naics_4digits:
                continue

        # --- IPC 2차 필터 (서브클래스 겹침) ---
        if require_ipc_overlap and user_subclasses and patent_ipc_codes:
            patent_subclasses = ipc_subclasses(patent_ipc_codes)
            if not patent_subclasses & user_subclasses:
                continue

        key = meta.get(id_key, "") or meta.get(name_key, "")
        if not key:
            continue

        group = company_groups[key]
        group["scores"].append(match.score)
        group["patents"].append({"score": match.score, "metadata": meta})
        group["company_name"] = meta.get(name_key, "")
        group["company_id"] = meta.get(id_key, "")
        patent_country = _extract_country_from_metadata(meta)
        if patent_country:
            group["country_counts"][patent_country] += 1

    ranked = []
    for key, group in company_groups.items():
        avg_score = sum(group["scores"]) / len(group["scores"])
        dominant_country = None
        if group["country_counts"]:
            dominant_country = max(group["country_counts"].items(), key=lambda x: x[1])[0]
        raw_patents = sorted(group["patents"], key=lambda x: -float(x.get("score") or 0.0))
        deduped = []
        seen_key = set()
        for p in raw_patents:
            meta = p.get("metadata") or {}
            pid = str(meta.get("application_number") or meta.get("patent_id") or "").strip()
            title = str(meta.get("invention_name") or "").strip()[:80]
            dedup_k = (pid, title) if pid else (str(id(meta)),)
            if dedup_k in seen_key:
                continue
            seen_key.add(dedup_k)
            deduped.append(p)
            if len(deduped) >= 3:
                break
        ranked.append({
            "company_name": group["company_name"],
            "company_id": group["company_id"],
            "avg_score": avg_score,
            "matched_patent_count": len(group["patents"]),
            "patents": deduped,
            "_dominant_country": dominant_country,
        })

    ranked.sort(key=lambda x: -x["avg_score"])
    return ranked


def _search_and_rank_companies(
    index_name: str,
    query_embedding: List[float],
    user_naics_4digit: str,
    user_ipc_codes: List[str],
    user_country: str | None,
    company_top_k: int = 5,
) -> List[dict]:
    """
    1) 벡터 유사도로 특허 풀 확보 (top _SEARCH_TOP_K) — 임베딩은 외부에서 1회 생성
    2) NAICS 1차 필터 + IPC 2차 필터 → 기업 그룹핑
    3) KR 입력 시 US 후보를 별도 보강해 후보풀에 병합
    4) 결과 부족 또는 US 부족 시 IPC 필터 완화로 재랭킹
    """
    index = pc.Index(index_name)

    base_matches = _query_matches(index, query_embedding, _SEARCH_TOP_K)
    if not base_matches:
        return []

    user_subclasses = ipc_subclasses(user_ipc_codes)
    user_country_norm = _normalize_country(user_country)
    matches = list(base_matches)

    if user_country_norm == "KR":
        us_boosted = _fetch_us_boosted_matches(index, query_embedding)
        if us_boosted:
            by_id: Dict[str, object] = {}
            for m in matches + us_boosted:
                match_id = str(getattr(m, "id", "") or "")
                if not match_id:
                    match_id = str(id(m))
                prev = by_id.get(match_id)
                if prev is None or getattr(m, "score", 0.0) > getattr(prev, "score", 0.0):
                    by_id[match_id] = m
            matches = list(by_id.values())
            matches.sort(key=lambda x: -float(getattr(x, "score", 0.0)))

    is_target = index_name == "target"
    name_key = "target_short_name" if is_target else "acquiror_short_name"
    id_key = "target_id" if is_target else "acquiror_id"

    # 1차: NAICS + IPC 필터
    ranked = _filter_and_group(
        matches=matches,
        user_naics_4digit=user_naics_4digit,
        user_subclasses=user_subclasses,
        name_key=name_key,
        id_key=id_key,
        require_ipc_overlap=True,
        require_naics_match=True,
    )

    needs_relax = len(ranked) < company_top_k
    needs_us_topup = (
        user_country_norm == "KR"
        and sum(1 for r in ranked if r.get("_dominant_country") == "US")
        < min(_MIN_US_SLOTS_FOR_KR, company_top_k)
    )

    # 결과 부족 / US 부족 시: IPC 필터 완화 (같은 검색 결과 재사용)
    if (needs_relax or needs_us_topup) and user_subclasses:
        relaxed = _filter_and_group(
            matches=matches,
            user_naics_4digit=user_naics_4digit,
            user_subclasses=user_subclasses,
            name_key=name_key,
            id_key=id_key,
            require_ipc_overlap=False,
            require_naics_match=True,
        )
        if needs_relax:
            ranked = relaxed
        elif needs_us_topup:
            us_relaxed = [r for r in relaxed if r.get("_dominant_country") == "US"]
            if len(us_relaxed) < min(_MIN_US_SLOTS_FOR_KR, company_top_k):
                us_relaxed_wo_naics = _filter_and_group(
                    matches=matches,
                    user_naics_4digit=user_naics_4digit,
                    user_subclasses=user_subclasses,
                    name_key=name_key,
                    id_key=id_key,
                    require_ipc_overlap=False,
                    require_naics_match=False,
                )
                us_relaxed = _merge_ranked_unique(
                    us_relaxed,
                    [r for r in us_relaxed_wo_naics if r.get("_dominant_country") == "US"],
                )
            ranked = _merge_ranked_unique(ranked, us_relaxed)

    mixed = _mix_company_ranking_by_country(
        ranked=ranked,
        user_country=user_country,
        top_k=company_top_k,
    )
    for item in mixed:
        item.pop("_dominant_country", None)
    return mixed


# ==========================================
# 외부에서 호출할 단일 함수
# ==========================================

def run_company(
    user_info: list,
    top_k: int = 5,
) -> dict:
    info = user_info[0]

    title = info.get("title", "")
    abstract = info.get("abstract", "")
    query_text = f"{title} {abstract}".strip()

    # 사용자 IPC 코드
    ipc_info = info.get("ipc_info", [])
    user_ipc_codes = [ipc["code"] for ipc in ipc_info] if ipc_info else []

    # 사용자 NAICS 앞 4자리
    primary_naic = info.get("primary_naic_info") or {}
    naics_code = str(primary_naic.get("code", ""))
    user_naics_4digit = naics_code[:4] if len(naics_code) >= 4 else ""
    user_country = info.get("country")

    # 임베딩 1회만 생성
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return {"target_similar_companies": [], "acquiror_similar_companies": []}

    target_results = _search_and_rank_companies(
        index_name="target",
        query_embedding=query_embedding,
        user_naics_4digit=user_naics_4digit,
        user_ipc_codes=user_ipc_codes,
        user_country=user_country,
        company_top_k=top_k,
    )

    acquiror_results = _search_and_rank_companies(
        index_name="acquiror",
        query_embedding=query_embedding,
        user_naics_4digit=user_naics_4digit,
        user_ipc_codes=user_ipc_codes,
        user_country=user_country,
        company_top_k=top_k,
    )

    return {
        "target_similar_companies": target_results,
        "acquiror_similar_companies": acquiror_results,
    }
