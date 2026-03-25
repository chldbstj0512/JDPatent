import os
import re
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


# ==========================================
# 내부 검색 함수 (공통)
# ==========================================

def _search(index_name: str, query_text: str, ipc_code: str = None, top_k: int = 5):
    index = pc.Index(index_name)

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    patents = []
    for match in results.matches:
        patents.append({
            "score": match.score,
            "metadata": match.metadata
        })

    # IPC 후처리
    if ipc_code:
        ipc_classes = parse_ipc_codes(ipc_code)
        patents = [
            p for p in patents
            if any(
                ipc in parse_ipc_codes(p["metadata"].get("ipc_code", ""))
                for ipc in ipc_classes
            )
        ]

    return patents


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