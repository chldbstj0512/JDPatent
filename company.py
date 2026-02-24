import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI


# ==========================================
# 초기화
# ==========================================

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ==========================================
# 유틸
# ==========================================

def get_embedding(text: str, model: str = "text-embedding-3-large") -> Optional[List[float]]:
    text = text.replace("\n", " ").strip()
    if not text:
        return None

    response = openai_client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def parse_ipc_codes(ipc_string: str) -> List[str]:
    if not ipc_string:
        return []
    pattern = r'([A-Z]\d{2}[A-Z])'
    return list(set(re.findall(pattern, ipc_string)))


def _ipc_overlap_count(query_ipc_codes: List[str], candidate_ipc: str) -> int:
    """Count shared IPC class prefixes between query and candidate."""
    if not query_ipc_codes:
        return 0
    candidate_codes = set(parse_ipc_codes(candidate_ipc))
    return len(set(query_ipc_codes) & candidate_codes)


def _detect_language_group(text: str) -> str:
    if not text:
        return "unk"
    if re.search(r"[\uac00-\ud7a3]", text):
        return "ko"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unk"


def _detect_country_group(metadata: dict) -> str:
    country = (metadata.get("country") or metadata.get("nation") or "").strip().upper()
    if len(country) == 2 and country.isalpha():
        return country

    app_no = str(metadata.get("application_number") or "").strip().upper()
    if re.match(r"^[A-Z]{2}", app_no):
        return app_no[:2]

    # Korean application numbers are often numeric and start with "10".
    if app_no.startswith("10") and app_no.replace("-", "").isdigit():
        return "KR"
    return "UNK"


def _dedupe_patents(patents: List[dict]) -> List[dict]:
    seen = set()
    deduped = []
    for p in patents:
        meta = p.get("metadata", {})
        app_no = str(meta.get("application_number") or "").strip()
        title = str(meta.get("invention_name") or "").strip().lower()
        key = app_no or title
        if not key:
            key = f"__idx_{len(deduped)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _diversity_rerank(patents: List[dict], top_k: int) -> List[dict]:
    pool = _dedupe_patents(patents)
    if len(pool) <= top_k:
        return pool

    selected = []
    lang_counts = Counter()
    country_counts = Counter()

    while pool and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")

        for i, p in enumerate(pool):
            base = float(p.get("score", 0.0))
            ipc_overlap = int(p.get("ipc_overlap", 0))
            lang = p.get("lang_group", "unk")
            country = p.get("country_group", "UNK")

            # Keep semantic score as primary signal.
            adjusted = base + min(ipc_overlap, 2) * 0.03

            # Encourage language/country variety in top-k.
            adjusted += 0.05 if lang_counts[lang] == 0 else -0.01 * lang_counts[lang]
            if country != "UNK":
                adjusted += 0.04 if country_counts[country] == 0 else -0.008 * country_counts[country]

            if adjusted > best_score:
                best_score = adjusted
                best_idx = i

        chosen = pool.pop(best_idx)
        selected.append(chosen)
        lang_counts[chosen.get("lang_group", "unk")] += 1
        country_counts[chosen.get("country_group", "UNK")] += 1

    return selected


# ==========================================
# 내부 검색 함수 (공통)
# ==========================================

def _search(index_name: str, query_text: str, ipc_code: str = None, top_k: int = 5):
    index = pc.Index(index_name)

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []

    # Fetch a much wider pool to reduce language/country bias.
    fetch_k = max(top_k * 40, 120)
    results = index.query(
        vector=query_embedding,
        top_k=fetch_k,
        include_metadata=True
    )

    ipc_classes = parse_ipc_codes(ipc_code) if ipc_code else []
    patents = []
    for match in results.matches:
        metadata = match.metadata or {}
        title = str(metadata.get("invention_name") or "")
        patents.append({
            "score": match.score,
            "metadata": metadata,
            "ipc_overlap": _ipc_overlap_count(ipc_classes, metadata.get("ipc_code", "")),
            "lang_group": _detect_language_group(title),
            "country_group": _detect_country_group(metadata),
        })

    patents.sort(key=lambda p: p["score"], reverse=True)
    return _diversity_rerank(patents, top_k)


# ==========================================
# 🔥 외부에서 호출할 단일 함수
# ==========================================
def run_company(
    user_info: dict,
    top_k: int = 5
) -> dict:

    user_info = user_info[0]    
    title = user_info.get("title", "")
    abstract = user_info.get("abstract", "")
    query_text = f"{title} {abstract}".strip()

    ipc_code = user_info.get("ipc_code")

    target_results = _search(
        index_name="target",
        query_text=query_text,
        ipc_code=ipc_code,
        top_k=top_k
    )

    acquiror_results = _search(
        index_name="acquiror",
        query_text=query_text,
        ipc_code=ipc_code,
        top_k=top_k
    )

    return {
        "target_similar_patents": target_results,
        "acquiror_similar_patents": acquiror_results
    }
