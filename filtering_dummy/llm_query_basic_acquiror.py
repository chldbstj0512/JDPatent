"""
OpenAI API를 사용하여 특허 출원인(applicant)과 인수 기업(acquiror)의 매칭을 검증하는 비동기 스크립트
- 배치 API가 아닌 일반 API 사용
- 비동기 처리로 빠른 실행
"""

import os
import csv
import json
import math
import asyncio
import pandas as pd
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
1. Acquiror 기업명과 Applicant가 동일한 기업을 가리키는지 확인
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

# 모델 설정
MODEL = "gpt-5-mini"
#MODEL = "gpt-5.2"

# 입력/출력 파일 경로
INPUT_PATH = "filtering_dummy/data/input/output_random_acquiror_10.csv"
OUTPUT_DIR = "filtering_dummy/data/output"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_PATH = f"{OUTPUT_DIR}/validated_acquiror_{MODEL}_{TIMESTAMP}.xlsx"

# 동시 요청 수 제한
MAX_CONCURRENT_REQUESTS = 20


def safe_str(value, default='') -> str:
    """NaN 등을 안전하게 문자열로 변환"""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return str(value)


def create_prompt(row: dict) -> str:
    """프롬프트 생성 (acquiror 기반) - 캐싱 최적화: 정적 부분을 앞에, 동적 부분을 뒤에 배치"""
    acquiror_short_name = safe_str(row.get('acquiror_short_name', ''))
    acquiror_nation = safe_str(row.get('acquiror_nation', ''))
    abstract = safe_str(row.get('abstract', ''))
    invention_name = safe_str(row.get('invention_name', ''))
    applicant = safe_str(row.get('applicant', ''))
    
    # 정적 부분 (캐싱됨) - 템플릿 파일에서 로드
    static_part = PROMPT_TEMPLATE
    
    # 동적 부분 (캐싱 안됨) - 각 요청마다 다름
    dynamic_part = f"""## 입력 정보
- **Acquiror 기업명**: {acquiror_short_name}
- **Acquiror 국가**: {acquiror_nation}
- **특허 초록(일부분)**: {abstract[:80] if abstract else "없음"}...
- **Invention Name**: {invention_name[:30] if invention_name else "없음"}...
- **출원인(Applicant)**: {applicant}"""
    
    return static_part + dynamic_part


async def validate_matching(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """OpenAI API를 사용하여 acquiror(인수 기업)과 applicant 매칭 검증"""
    
    prompt = create_prompt(row)
    
    messages = [
        {"role": "system", "content": "당신은 글로벌 특허 데이터 매칭 검증 전문가입니다. 각국의 기업과 사업 분야에 대해서 잘 알고 있습니다. 항상 유효한 JSON 형식으로만 응답합니다."},
        {"role": "user", "content": prompt}
    ]

    max_retries = 3
    retry_delay = 1.0
    raw_response = None
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # gpt-5-mini는 temperature를 지원하지 않음
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_completion_tokens=4000
                )
                
                if not response.choices or not response.choices[0].message.content:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"is_valid": None, "confidence": 0.0, "error_info": "빈 응답"}
                
                result_text = response.choices[0].message.content.strip()
                raw_response = result_text[:500]
                
                # JSON 파싱
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                result_text = result_text.strip()
                
                if not result_text:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"is_valid": None, "confidence": 0.0, "error_info": "빈 응답"}
                
                parsed = json.loads(result_text)
                return {
                    "is_valid": parsed.get("is_valid"),
                    "confidence": parsed.get("confidence", 0.0),
                    "error_info": None
                }
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                return {
                    "is_valid": None, 
                    "confidence": 0.0, 
                    "error_info": f"JSON 파싱 오류: {str(e)}",
                    "raw_response": raw_response
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                return {
                    "is_valid": None, 
                    "confidence": 0.0, 
                    "error_info": f"API 오류: {str(e)}"
                }
        
        return {"is_valid": None, "confidence": 0.0, "error_info": "최대 재시도 횟수 초과"}


async def process_row(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """단일 행 처리"""
    validation_result = await validate_matching(row, semaphore)
    
    return {
        'id': row.get('id', ''),
        'acquiror_short_name': row.get('acquiror_short_name', ''),
        'applicant': row.get('applicant', ''),
        'is_valid': validation_result.get('is_valid'),
        'confidence': validation_result.get('confidence', 0.0),
        'error_info': validation_result.get('error_info'),
        'raw_response': validation_result.get('raw_response')
    }


async def process_data_async(input_path: str, output_path: str):
    """비동기로 데이터 처리"""
    
    # 파일 읽기 (CSV 또는 XLSX)
    print(f"파일 로드 중: {input_path}")
    if input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path, engine='openpyxl')
        rows = df.to_dict('records')
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    print(f"총 {len(rows)}개 행 처리 (동시 요청: {MAX_CONCURRENT_REQUESTS}개)")
    print(f"모델: {MODEL}")
    print()
    
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
    
    # 결과 저장 (XLSX)
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"\n✓ 결과 저장: {output_path}")
        
        print(f"\n=== 검증 결과 통계 ===")
        print(f"총 처리: {len(results)}건")
        print(f"✅ 일치(True): {valid_count}건 ({valid_count/len(results)*100:.1f}%)")
        print(f"❌ 불일치(False): {invalid_count}건 ({invalid_count/len(results)*100:.1f}%)")
        print(f"❓ 불확실(Uncertain): {uncertain_count}건 ({uncertain_count/len(results)*100:.1f}%)")
        print(f"⚠️  오류(Error): {error_count}건 ({error_count/len(results)*100:.1f}%)")
        
        # 에러가 있으면 상세 출력
        if error_count > 0:
            print(f"\n=== 에러 상세 ===")
            for i, r in enumerate(results):
                if r['is_valid'] is None and r.get('error_info'):
                    print(f"  - {r.get('acquiror_short_name', 'N/A')}: {r['error_info']}")
                    if r.get('raw_response'):
                        print(f"    원본: {r['raw_response'][:100]}...")


def main():
    """메인 함수"""
    print("=" * 70)
    print("OpenAI API 특허 매칭 검증 - Acquiror 기반 (일반 API)")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
    
    if not os.path.exists(INPUT_PATH):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {INPUT_PATH}")
        return
    
    asyncio.run(process_data_async(INPUT_PATH, OUTPUT_PATH))


if __name__ == "__main__":
    main()
