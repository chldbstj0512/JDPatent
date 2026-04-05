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

_SEARCH_TOP_K = 200
_IPC_SUBCLASS_LEN = 4  # e.g. "A61B"


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
# 필터링 + 기업 그룹핑 (공통 로직)
# ==========================================

def _filter_and_group(
    matches: list,
    user_naics_4digit: str,
    user_subclasses: set,
    name_key: str,
    id_key: str,
    require_ipc_overlap: bool,
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
    })

    for match in matches:
        meta = match.metadata
        patent_ipc_raw = meta.get("ipc_code", "") or ""
        patent_ipc_codes = parse_ipc_codes(patent_ipc_raw)

        # --- NAICS 1차 필터 (앞 4자리 기준) ---
        if user_naics_4digit and patent_ipc_codes:
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

    ranked = []
    for key, group in company_groups.items():
        avg_score = sum(group["scores"]) / len(group["scores"])
        best = max(group["patents"], key=lambda x: x["score"])
        ranked.append({
            "company_name": group["company_name"],
            "company_id": group["company_id"],
            "avg_score": avg_score,
            "matched_patent_count": len(group["patents"]),
            "patents": sorted(group["patents"], key=lambda x: -x["score"])[:3],
        })

    ranked.sort(key=lambda x: -x["avg_score"])
    return ranked


def _search_and_rank_companies(
    index_name: str,
    query_embedding: List[float],
    user_naics_4digit: str,
    user_ipc_codes: List[str],
    company_top_k: int = 5,
) -> List[dict]:
    """
    1) 벡터 유사도로 특허 풀 확보 (top 200) — 임베딩은 외부에서 1회 생성
    2) NAICS 1차 필터 + IPC 2차 필터 → 기업 그룹핑
    3) 결과 부족 시 IPC 필터 제거하고 같은 결과에서 재필터 (Pinecone 재호출 없음)
    """
    index = pc.Index(index_name)

    results = index.query(
        vector=query_embedding,
        top_k=_SEARCH_TOP_K,
        include_metadata=True,
    )

    if not results.matches:
        return []

    user_subclasses = ipc_subclasses(user_ipc_codes)

    is_target = index_name == "target"
    name_key = "target_short_name" if is_target else "acquiror_short_name"
    id_key = "target_id" if is_target else "acquiror_id"

    # 1차: NAICS + IPC 필터
    ranked = _filter_and_group(
        matches=results.matches,
        user_naics_4digit=user_naics_4digit,
        user_subclasses=user_subclasses,
        name_key=name_key,
        id_key=id_key,
        require_ipc_overlap=True,
    )

    # 결과 부족 시: IPC 필터 완화 (같은 Pinecone 결과 재사용)
    if len(ranked) < company_top_k and user_subclasses:
        ranked = _filter_and_group(
            matches=results.matches,
            user_naics_4digit=user_naics_4digit,
            user_subclasses=user_subclasses,
            name_key=name_key,
            id_key=id_key,
            require_ipc_overlap=False,
        )

    return ranked[:company_top_k]


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

    # 임베딩 1회만 생성
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return {"target_similar_companies": [], "acquiror_similar_companies": []}

    target_results = _search_and_rank_companies(
        index_name="target",
        query_embedding=query_embedding,
        user_naics_4digit=user_naics_4digit,
        user_ipc_codes=user_ipc_codes,
        company_top_k=top_k,
    )

    acquiror_results = _search_and_rank_companies(
        index_name="acquiror",
        query_embedding=query_embedding,
        user_naics_4digit=user_naics_4digit,
        user_ipc_codes=user_ipc_codes,
        company_top_k=top_k,
    )

    return {
        "target_similar_companies": target_results,
        "acquiror_similar_companies": acquiror_results,
    }
