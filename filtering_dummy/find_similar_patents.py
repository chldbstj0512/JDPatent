"""
Pinecone을 사용한 유사 특허 검색
- Target 인덱스 검색
- Acquiror 인덱스 검색
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI


# ============================================================================
# 설정 및 초기화
# ============================================================================

# .env 파일 로드
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# 환경 변수에서 API 키 로드
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 클라이언트 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_embedding(text: str, model: str = "text-embedding-3-large") -> Optional[List[float]]:
    """OpenAI 임베딩 생성"""
    text = text.replace("\n", " ").strip()
    if not text:
        return None
    
    response = openai_client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def parse_ipc_codes(ipc_string: str) -> List[str]:
    """
    IPC 코드 문자열을 파싱하여 상위 클래스 리스트 반환
    예: "A61K 31/501|A61P 9/04" -> ["A61K", "A61P"]
    """
    if not ipc_string:
        return []
    
    # IPC 코드 패턴: 대문자+숫자+대문자 (예: A61K, B32B, E06B)
    pattern = r'([A-Z]\d{2}[A-Z])'
    matches = re.findall(pattern, ipc_string)
    
    # 중복 제거하고 리스트로 반환
    return list(set(matches))


def create_search_text(abstract: str, invention_name: str) -> str:
    """검색용 텍스트 생성 (초록 + 특허명 결합)"""
    parts = []
    if invention_name:
        parts.append(f"특허명: {invention_name}")
    if abstract:
        parts.append(f"초록: {abstract}")
    return " ".join(parts)


# ============================================================================
# Target 검색 함수
# ============================================================================

def search_target_patents(
    query_text: str,
    ipc_code: str = None
) -> List[Dict]:
    """
    Target 인덱스에서 유사 특허 검색
    
    Args:
        query_text: 검색할 텍스트 (초록 + 특허명)
        ipc_code: IPC 코드 (optional, 필터링용)
    
    Returns:
        유사 특허 리스트 (상위 5개)
    """
    index = pc.Index("target")
    
    # 1. 임베딩 생성
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("[오류] 임베딩 생성 실패")
        return []
    
    # 2. IPC 코드 파싱
    ipc_classes = []
    if ipc_code and use_ipc_filter:
        ipc_classes = parse_ipc_codes(ipc_code)
        print(f"[정보] 입력 IPC 클래스: {ipc_classes}")
    
    # 3. Pinecone 검색
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # 4. 결과 추출
    patents = []
    for match in results.matches:
        patents.append(match.metadata)
    
    # 5. IPC 필터 (후처리)
    if ipc_classes:
        filtered = []
        for patent in patents:
            patent_ipcs = parse_ipc_codes(patent.get('ipc_code', ''))
            if any(ipc in patent_ipcs for ipc in ipc_classes):
                filtered.append(patent)
        patents = filtered
        print(f"[검색] IPC 필터 적용됨: {ipc_classes} (후처리)")
    
    return patents


# ============================================================================
# Acquiror 검색 함수
# ============================================================================

def search_acquiror_patents(
    query_text: str,
    ipc_code: str = None
) -> List[Dict]:
    """
    Acquiror 인덱스에서 유사 특허 검색
    
    Args:
        query_text: 검색할 텍스트 (초록 + 특허명)
        ipc_code: IPC 코드 (optional, 필터링용)
    
    Returns:
        유사 특허 리스트 (상위 5개)
    """
    index = pc.Index("acquiror")
    
    # 1. 임베딩 생성
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("[오류] 임베딩 생성 실패")
        return []
    
    # 2. IPC 코드 파싱
    ipc_classes = []
    if ipc_code:
        ipc_classes = parse_ipc_codes(ipc_code)
        print(f"[정보] 입력 IPC 클래스: {ipc_classes}")
    
    # 3. Pinecone 검색
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # 4. 결과 추출
    patents = []
    for match in results.matches:
        patents.append(match.metadata)
    
    # 5. IPC 필터 (후처리)
    if ipc_classes:
        filtered = []
        for patent in patents:
            patent_ipcs = parse_ipc_codes(patent.get('ipc_code', ''))
            if any(ipc in patent_ipcs for ipc in ipc_classes):
                filtered.append(patent)
        patents = filtered
        print(f"[검색] IPC 필터 적용됨: {ipc_classes} (후처리)")
    
    return patents


# ============================================================================
# 결과 출력 함수
# ============================================================================

def print_target_results(patents: List[Dict], scores: List[float] = None):
    """Target 검색 결과 출력"""
    if not patents:
        print("[알림] 검색 결과 없음")
        return
    
    print(f"\n{'='*80}")
    print(f"[결과] Target 유사 특허 검색 결과 (상위 {len(patents)}개)")
    print(f"{'='*80}\n")
    
    for i, patent in enumerate(patents, 1):
        if scores and i <= len(scores):
            print(f"[{i}] 유사도: {scores[i-1]:.4f}")
        else:
            print(f"[{i}]")
        
        print(f"    특허명: {patent.get('invention_name', '')[:80]}...")
        print(f"    IPC: {patent.get('ipc_code', '')[:50]}...")
        print(f"    초록: {patent.get('abstract', '')[:150]}...")
        print(f"    출원번호: {patent.get('application_number', '')}")
        print(f"    출원인: {patent.get('applicant', '')}")
        print()


def print_acquiror_results(patents: List[Dict], scores: List[float] = None):
    """Acquiror 검색 결과 출력"""
    if not patents:
        print("[알림] 검색 결과 없음")
        return
    
    print(f"\n{'='*80}")
    print(f"[결과] Acquiror 유사 특허 검색 결과 (상위 {len(patents)}개)")
    print(f"{'='*80}\n")
    
    for i, patent in enumerate(patents, 1):
        if scores and i <= len(scores):
            print(f"[{i}] 유사도: {scores[i-1]:.4f}")
        else:
            print(f"[{i}]")
        
        print(f"    Acquiror: {patent.get('acquiror_short_name', '')}")
        print(f"    특허명: {patent.get('invention_name', '')[:80]}...")
        print(f"    IPC: {patent.get('ipc_code', '')[:50]}...")
        print(f"    초록: {patent.get('abstract', '')[:150]}...")
        print(f"    출원번호: {patent.get('application_number', '')}")
        print(f"    출원인: {patent.get('applicant', '')}")
        print()


# ============================================================================
# CLI 인터페이스
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pinecone 유사 특허 검색 (상위 5개 결과)')
    parser.add_argument(
        '--index',
        type=str,
        required=True,
        choices=['target', 'acquiror'],
        help='검색할 인덱스 (target 또는 acquiror)'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='검색 쿼리 텍스트'
    )
    parser.add_argument(
        '--ipc',
        type=str,
        default=None,
        help='IPC 코드 필터 (예: G06Q 30/02)'
    )
    
    args = parser.parse_args()
    
    # 검색 실행
    if args.index == 'target':
        print(f"\n[검색 시작] Target 인덱스")
        print(f"쿼리: {args.query[:100]}...")
        if args.ipc:
            print(f"IPC: {args.ipc}")
        
        results = search_target_patents(
            query_text=args.query,
            ipc_code=args.ipc
        )
        print_target_results(results)
        
    elif args.index == 'acquiror':
        print(f"\n[검색 시작] Acquiror 인덱스")
        print(f"쿼리: {args.query[:100]}...")
        if args.ipc:
            print(f"IPC: {args.ipc}")
        
        results = search_acquiror_patents(
            query_text=args.query,
            ipc_code=args.ipc
        )
        print_acquiror_results(results)


if __name__ == "__main__":
    main()
