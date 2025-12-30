"""
특허 출원인 매칭 알고리즘 (Patent Applicant Matching Algorithm)
=================================================================
영문 회사명(target_short_name)과 한글 출원인 리스트(applicants)를 매칭하는 알고리즘

주요 기능:
1. 영문-영문 직접 매칭 (정규화 + 유사도)
2. 영문-한글 음차(Phonetic) 매칭 (한글→로마자 변환 후 유사도 비교)

Author: Claude
Date: 2024
"""

import pandas as pd
import re
from difflib import SequenceMatcher
from typing import Tuple, List, Optional


# ============================================================================
# 1. 한글 → 로마자 변환 (Korean to Romanization)
# ============================================================================

# 한글 자모 → 로마자 매핑 테이블
CHOSUNG = ['g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's', 'ss', '', 'j', 'jj', 'ch', 'k', 't', 'p', 'h']
JUNGSUNG = ['a', 'ae', 'ya', 'yae', 'eo', 'e', 'yeo', 'ye', 'o', 'wa', 'wae', 'oe', 'yo', 'u', 'wo', 'we', 'wi', 'yu', 'eu', 'ui', 'i']
JONGSUNG = ['', 'k', 'kk', 'ks', 'n', 'nj', 'nh', 't', 'l', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 'm', 'p', 'ps', 's', 'ss', 'ng', 'j', 'ch', 'k', 't', 'p', 'h']


def korean_to_romanized(text: str) -> str:
    """
    한글 텍스트를 로마자(영문)로 변환
    
    예시:
        '레고리스' → 'regoriseu'
        '인덱스핑거' → 'indekseupinggeo'
        '삼성전자' → 'samseongjeonja'
    """
    result = []
    for char in str(text):
        if '가' <= char <= '힣':
            code = ord(char) - ord('가')
            cho = code // 588          # 초성
            jung = (code % 588) // 28  # 중성
            jong = code % 28           # 종성
            result.append(CHOSUNG[cho] + JUNGSUNG[jung] + JONGSUNG[jong])
        elif char.isalpha() or char.isdigit():
            result.append(char.lower())
        # 공백, 특수문자는 무시
    return ''.join(result)


# ============================================================================
# 2. 텍스트 정규화 (Text Normalization)
# ============================================================================

# 한글 법인형태 접미사
KOREAN_SUFFIXES = [
    '주식회사', '(주)', '유한회사', '유한책임회사', '합자회사', '합명회사',
    '사단법인', '재단법인', '산학협력단', '연구원', '대학교', '공사', '기술원',
    '피티이', '엘티디', '리미티드', '코포레이션', '컴퍼니', '인코포레이티드',
    '인크', '게엠베하', '아게', '엘엘씨'
]

# 영문 법인형태 접미사
ENGLISH_SUFFIXES = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd',
    'company', 'co', 'pte', 'llc', 'llp', 'gmbh', 'ag', 'sa', 'bv', 'nv',
    'holdings', 'holding', 'group', 'plc', 'pty'
}


def extract_korean_core(text: str) -> str:
    """
    한글 회사명에서 법인형태 제거하고 핵심 이름만 추출
    
    예시:
        '주식회사 레고리스' → '레고리스'
        '삼성전자주식회사' → '삼성전자'
    """
    result = str(text)
    for suffix in KOREAN_SUFFIXES:
        result = result.replace(suffix, '')
    return re.sub(r'\s+', '', result).strip()


def extract_english_core(text: str) -> str:
    """
    영문 회사명에서 법인형태 제거하고 핵심 이름만 추출
    
    예시:
        'REGOLITH Inc' → 'regolith'
        'Samsung Electronics Co Ltd' → 'samsungelectronics'
    """
    result = str(text).lower()
    result = re.sub(r'[^\w\s]', ' ', result)  # 특수문자 제거
    words = [w for w in result.split() if w not in ENGLISH_SUFFIXES]
    return ''.join(words)


def normalize_korean_roman(text: str) -> str:
    """
    한글 로마자에서 불필요한 모음 패턴 제거/단순화
    (한글 음차 특성상 '으(eu)' 같은 모음이 많이 추가됨)
    
    예시:
        'regoriseu' → 'regoris'
        'nekseoseukeonteurolseu' → 'neksoskontrols'
    """
    result = text
    result = result.replace('eu', '')      # 으 제거
    result = result.replace('eo', 'o')     # 어 → o
    result = result.replace('ae', 'e')     # 애 → e
    result = result.replace('kk', 'k')     # 쌍자음 단순화
    result = result.replace('tt', 't')
    result = result.replace('pp', 'p')
    result = result.replace('ss', 's')
    return result


def normalize_english(text: str) -> str:
    """
    영어 발음을 한글 음차 스타일로 변환
    (영어→한글 음차 시 발생하는 발음 변화 반영)
    
    예시:
        'regolith' → 'regolis' (th → s)
        'nexus' → 'neksus' (x → ks)
    """
    result = text.lower()
    result = result.replace('th', 's')       # th → ㅅ
    result = result.replace('x', 'ks')       # x → ㄱㅅ
    result = result.replace('f', 'p')        # f → ㅍ
    result = result.replace('v', 'b')        # v → ㅂ
    result = result.replace('z', 'j')        # z → ㅈ
    result = result.replace('ph', 'p')       # ph → ㅍ
    result = result.replace('tion', 'syon')  # tion → 션
    result = result.replace('sion', 'syon')  # sion → 션
    return result


# ============================================================================
# 3. 유사도 계산 (Similarity Computation)
# ============================================================================

def compute_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간의 유사도 계산 (SequenceMatcher 사용)
    """
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def compute_best_similarity(eng_core: str, kor_roman: str) -> float:
    """
    다양한 정규화 조합으로 유사도 계산 후 최고값 반환
    
    4가지 조합 비교:
    1. 원본 영문 vs 원본 한글로마자
    2. 정규화 영문 vs 정규화 한글로마자
    3. 정규화 영문 vs 원본 한글로마자
    4. 원본 영문 vs 정규화 한글로마자
    """
    if not eng_core or not kor_roman:
        return 0.0
    
    eng_norm = normalize_english(eng_core)
    kor_norm = normalize_korean_roman(kor_roman)
    
    similarities = [
        compute_similarity(eng_core, kor_roman),   # 원본 vs 원본
        compute_similarity(eng_norm, kor_norm),    # 정규화 vs 정규화
        compute_similarity(eng_norm, kor_roman),   # 정규화 영문 vs 원본 한글
        compute_similarity(eng_core, kor_norm),    # 원본 영문 vs 정규화 한글
    ]
    
    return max(similarities)


# ============================================================================
# 4. 영문-영문 직접 매칭 (English-English Direct Matching)
# ============================================================================

def english_direct_match(eng_name: str, applicants: str, threshold: float = 0.9) -> Tuple[Optional[str], float]:
    """
    영문 회사명과 영문 출원인 간의 직접 매칭
    
    Args:
        eng_name: 영문 회사명 (예: "Samsung Electronics")
        applicants: 출원인 리스트 (콤마로 구분, |로 공동출원인 구분)
        threshold: 매칭 임계값 (기본 0.9)
    
    Returns:
        (매칭된 출원인, 유사도 점수)
    """
    if pd.isna(applicants):
        return None, 0.0
    
    eng_core = extract_english_core(eng_name)
    if len(eng_core) < 2:
        return None, 0.0
    
    applicant_list = str(applicants).split(', ')
    best_match = None
    best_score = 0.0
    
    for app in applicant_list:
        # 공동출원인 처리
        for co_app in app.split('|'):
            # 한글이 포함된 경우 스킵 (영문 매칭만)
            if re.search('[가-힣]', co_app):
                continue
            
            app_core = extract_english_core(co_app)
            if len(app_core) < 2:
                continue
            
            # 길이 비율 체크
            len_ratio = min(len(eng_core), len(app_core)) / max(len(eng_core), len(app_core))
            if len_ratio < 0.5:
                continue
            
            # 유사도 계산
            sim = compute_similarity(eng_core, app_core)
            
            # 첫 글자 일치 보너스
            if eng_core and app_core and eng_core[0] == app_core[0]:
                sim += 0.02
            
            if sim > best_score:
                best_score = sim
                best_match = app
    
    return best_match, min(best_score, 1.0)


# ============================================================================
# 5. 영문-한글 음차 매칭 (English-Korean Phonetic Matching)
# ============================================================================

def phonetic_match(eng_name: str, applicants: str, threshold: float = 0.7) -> Tuple[Optional[str], float]:
    """
    영문 회사명과 한글 출원인 간의 음차(발음) 기반 매칭
    
    원리:
    1. 한글 출원인을 로마자로 변환
    2. 영문 회사명과 로마자 변환된 한글의 유사도 비교
    
    Args:
        eng_name: 영문 회사명 (예: "REGOLITH Inc")
        applicants: 출원인 리스트
        threshold: 매칭 임계값 (기본 0.7)
    
    Returns:
        (매칭된 출원인, 유사도 점수)
    
    예시:
        'REGOLITH Inc' vs '주식회사 레고리스'
        → 'regolith' vs 'regoriseu' → 정규화 후 비교 → 유사도 0.857
    """
    if pd.isna(applicants):
        return None, 0.0
    
    eng_core = extract_english_core(eng_name)
    if len(eng_core) < 3:  # 최소 3글자 필요
        return None, 0.0
    
    applicant_list = str(applicants).split(', ')
    best_match = None
    best_score = 0.0
    
    for app in applicant_list:
        for co_app in app.split('|'):
            # 한글이 없으면 스킵 (음차 매칭은 한글 대상)
            if not re.search('[가-힣]', co_app):
                continue
            
            kor_core = extract_korean_core(co_app)
            if len(kor_core) < 2:
                continue
            
            # 한글 → 로마자 변환
            kor_roman = korean_to_romanized(kor_core)
            
            # 길이 비율 체크 (너무 차이나면 스킵)
            len_ratio = min(len(eng_core), len(kor_roman)) / max(len(eng_core), len(kor_roman))
            if len_ratio < 0.4:
                continue
            
            # 다양한 정규화 조합으로 유사도 계산
            sim = compute_best_similarity(eng_core, kor_roman)
            
            # 첫 글자 일치 보너스
            eng_norm = normalize_english(eng_core)
            kor_norm = normalize_korean_roman(kor_roman)
            if eng_norm and kor_norm and eng_norm[0] == kor_norm[0]:
                sim += 0.02
            
            if sim > best_score:
                best_score = sim
                best_match = app
    
    return best_match, min(best_score, 1.0)


# ============================================================================
# 6. 통합 매칭 함수 (Combined Matching)
# ============================================================================

def match_applicant(
    target_name: str, 
    applicants: str, 
    eng_threshold: float = 0.9,
    phonetic_threshold: float = 0.7
) -> Tuple[Optional[str], float, str]:
    """
    영문-영문 매칭과 영문-한글 음차 매칭을 순차적으로 시도
    
    Args:
        target_name: 대상 회사명 (영문)
        applicants: 출원인 리스트
        eng_threshold: 영문 매칭 임계값
        phonetic_threshold: 음차 매칭 임계값
    
    Returns:
        (매칭된 출원인, 유사도 점수, 매칭 유형)
        매칭 유형: 'english', 'phonetic', 'none'
    """
    # 1. 영문-영문 직접 매칭 시도
    match, score = english_direct_match(target_name, applicants, eng_threshold)
    if score >= eng_threshold:
        return match, score, 'english'
    
    # 2. 영문-한글 음차 매칭 시도
    match, score = phonetic_match(target_name, applicants, phonetic_threshold)
    if score >= phonetic_threshold:
        return match, score, 'phonetic'
    
    return None, 0.0, 'none'


# ============================================================================
# 7. 배치 처리 함수 (Batch Processing)
# ============================================================================

def process_dataframe(
    df: pd.DataFrame,
    target_col: str = 'target_short_name',
    applicants_col: str = 'applicants',
    eng_threshold: float = 0.9,
    phonetic_threshold: float = 0.7
) -> pd.DataFrame:
    """
    DataFrame 전체에 대해 매칭 수행
    
    Args:
        df: 입력 DataFrame
        target_col: 대상 회사명 컬럼
        applicants_col: 출원인 리스트 컬럼
        eng_threshold: 영문 매칭 임계값
        phonetic_threshold: 음차 매칭 임계값
    
    Returns:
        매칭 결과가 추가된 DataFrame
    """
    results = []
    
    for idx, row in df.iterrows():
        target = row[target_col]
        applicants = row[applicants_col]
        
        match, score, match_type = match_applicant(
            target, applicants, eng_threshold, phonetic_threshold
        )
        
        results.append({
            'matched_applicant': match if match else '',
            'confidence_score': round(score, 3),
            'match_type': match_type
        })
    
    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


# ============================================================================
# 8. 사용 예시 (Usage Example)
# ============================================================================

if __name__ == '__main__':
    # 테스트 케이스
    test_cases = [
        # (영문 회사명, 출원인 리스트)
        ('REGOLITH Inc', '주식회사 레고리스'),
        ('Indexfinger Corp', '주식회사 인덱스핑거'),
        ('Nexus Controls LLC', '넥서스 컨트롤스 엘엘씨'),
        ('Samsung Electronics', 'Samsung Electronics Co., Ltd., 삼성전자주식회사'),
        ('Intel Corp', '인텔 코포레이션, Intel Corporation'),
        ('Melcon Co Ltd', '멜콘 주식회사, 멜콘 주식회사|김동헌'),
    ]
    
    print("=" * 70)
    print("특허 출원인 매칭 알고리즘 테스트")
    print("=" * 70)
    
    for eng_name, applicants in test_cases:
        match, score, match_type = match_applicant(eng_name, applicants)
        
        print(f"\n대상: {eng_name}")
        print(f"출원인: {applicants}")
        print(f"결과: {match} (유사도: {score:.3f}, 유형: {match_type})")
    
    print("\n" + "=" * 70)
    
    # DataFrame 처리 예시
    print("\n[DataFrame 처리 예시]")
    
    df = pd.DataFrame({
        'target_short_name': ['REGOLITH Inc', 'Intel Corp', 'Unknown Company'],
        'applicants': ['주식회사 레고리스', '인텔 코포레이션', '전혀 다른 회사']
    })
    
    result_df = process_dataframe(df)
    print(result_df.to_string())