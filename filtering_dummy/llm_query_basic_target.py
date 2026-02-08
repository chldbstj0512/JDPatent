"""
OpenAI API를 사용하여 특허 출원인(applicant)과 대상 기업(target)의 매칭을 검증하는 비동기 스크립트
"""

import os
import csv
import json
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

load_dotenv()

# 프롬프트 템플릿 로드
sys.path.insert(0, str(Path(__file__).parent))
from prompts.matching_rules import RULES

PROMPT_TEMPLATE = f"""당신은 회사명 매칭 검증 전문가입니다.

당신의 작업:
주어진 영문 회사명과 한글 회사명이
서로 올바르고 합리적인 대응 관계인지 판단하십시오.

## 규칙 및 예시(모두 준수):

{RULES}


## 판단 기준
1. Target 기업명과 Applicant가 동일한 기업을 가리키는지 확인
2. '|'로 구분된 공동 출원인의 경우, 하나라도 일치하면 True로 판단
3. 판단 결과가 확실하지 않다면(확신도 0.8미만인 경우) False로 판단

## 응답 형식 (JSON)
아래 2가지 중 하나로 "is_valid"를 선택하세요.
- True: 명확히 일치
- False: 명확히 불일치

{{
    "is_valid": true 또는 false,
    "confidence": 0.0 ~ 1.0 사이의 확신도
}}

반드시 JSON만 응답해주세요.

---

"""

# Langfuse 통합 (설정된 경우 자동 활성화)
try:
    from langfuse.openai import AsyncOpenAI
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))
    if LANGFUSE_ENABLED:
        print("✓ Langfuse 로깅 활성화")
except ImportError:
    from openai import AsyncOpenAI
    LANGFUSE_ENABLED = False

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 4o-mini 모델은 가장 저렴한 cost, 낮은 성능
#MODEL = "gpt-4o-mini"
MODEL = "gpt-5-mini"
# o4-mini 모델은 상대적으로 저렴한 cost, 중간 정도 성능
#MODEL = "o4-mini"

# gpt-5.2 모델은 가장 최신 모델, 높은 cost(o4-mini대비 약 3배 이상), 가장 높은 성능
#MODEL = "gpt-5.2"


INPUT_CSV_PATH = "./data/input/output_random_acquiror_3.csv"
OUTPUT_CSV_PATH = f"filtering_dummy/data/output/target_10_validated_{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
MAX_CONCURRENT_REQUESTS = 20


def get_model_params(model: str, max_tokens: int, temperature: float = 0) -> dict:
    """모델별 API 파라미터 반환 (GPT-5/o1/o3/o4는 max_completion_tokens 사용, reasoning 모델은 temperature 제외)"""
    new_token_models = ["gpt-5", "o1", "o3", "o4"]
    reasoning_models = ["o1", "o3", "o4"]
    
    params = {}
    
    if any(model.startswith(p) for p in new_token_models):
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
    
    if not any(model.startswith(p) for p in reasoning_models):
        params["temperature"] = temperature
    
    return params


async def validate_matching(
    target_short_name: str,
    target_nation: str,
    abstract: str,
    invention_name: str,
    applicant: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """OpenAI API를 사용하여 target과 applicant 매칭 검증"""
    
    # 동적 부분 (각 요청마다 다름)
    dynamic_part = f"""## 입력 정보
- **Target 기업명**: {target_short_name}
- **Target 국가**: {target_nation}
- **특허 초록(일부분)**: {abstract[:80] if abstract else "없음"}...
- **Invention Name**: {invention_name[:30] if invention_name else "없음"}...
- **출원인(Applicant)**: {applicant}"""
    
    prompt = PROMPT_TEMPLATE + dynamic_part

    messages = [
        {"role": "system", "content": "당신은 한국의 글로벌 특허 데이터 매칭 검증 전문가입니다. 각국의 기업과 사업 분야에 대해서 잘 알고 있습니다. 항상 유효한 JSON 형식으로만 응답합니다."},
        {"role": "user", "content": prompt}
    ]

    max_retries = 3
    retry_delay = 1.0  # 초
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                model_params = get_model_params(MODEL, max_tokens=1000, temperature=0)
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    **model_params
                )
                
                if not response.choices or not response.choices[0].message.content:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"is_valid": None, "confidence": 0.0}
                
                result_text = response.choices[0].message.content.strip()
                
                # JSON 파싱
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                result_text = result_text.strip()
                
                # 빈 문자열 체크
                if not result_text:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"is_valid": None, "confidence": 0.0}
                
                return json.loads(result_text)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                # 마지막 시도에서도 실패하면 원본 응답도 기록
                return {"is_valid": None, "confidence": 0.0}
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                return {"is_valid": None, "confidence": 0.0}
        
        return {"is_valid": None, "confidence": 0.0}


async def process_row(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """단일 행 처리"""
    validation_result = await validate_matching(
        target_short_name=row.get('target_short_name', ''),
        target_nation=row.get('target_nation', ''),
        abstract=row.get('abstract', ''),
        invention_name=row.get('invention_name', ''),
        applicant=row.get('applicant', ''),
        semaphore=semaphore
    )
    
    return {
        **row,
        'is_valid': validation_result.get('is_valid'),
        'confidence': validation_result.get('confidence', 0.0)
    }


async def process_csv_async(input_path: str, output_path: str):
    """비동기로 CSV 처리"""
    
    # CSV 파일 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"총 {len(rows)}개 행 처리 (동시 요청: {MAX_CONCURRENT_REQUESTS}개)")
    
    # 동시 요청 수 제한을 위한 세마포어
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 모든 행을 비동기로 처리
    tasks = [process_row(row, semaphore) for row in rows]
    results = await tqdm_asyncio.gather(*tasks, desc="검증 진행")
    
    # 통계 계산
    def normalize(val):
        if val is None:
            return None
        return str(val).lower() if val else None
    
    valid_count = sum(1 for r in results if normalize(r['is_valid']) == "true")
    invalid_count = sum(1 for r in results if normalize(r['is_valid']) == "false")
    uncertain_count = sum(1 for r in results if normalize(r['is_valid']) == "uncertain")
    error_count = sum(1 for r in results if r['is_valid'] is None)
    
    # 결과 CSV 저장
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n결과가 {output_path}에 저장되었습니다.")
        
        print(f"\n=== 검증 결과 통계 ===")
        print(f"총 처리: {len(results)}건")
        print(f"✅ 일치(True): {valid_count}건 ({valid_count/len(results)*100:.1f}%)")
        print(f"❌ 불일치(False): {invalid_count}건 ({invalid_count/len(results)*100:.1f}%)")
        print(f"❓ 불확실(Uncertain): {uncertain_count}건 ({uncertain_count/len(results)*100:.1f}%)")
        print(f"⚠️  오류(Error): {error_count}건 ({error_count/len(results)*100:.1f}%)")


def main():
    """메인 함수"""
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {INPUT_CSV_PATH}")
        return
    
    asyncio.run(process_csv_async(INPUT_CSV_PATH, OUTPUT_CSV_PATH))


if __name__ == "__main__":
    main()
