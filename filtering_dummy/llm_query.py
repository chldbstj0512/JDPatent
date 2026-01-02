"""
OpenAI API를 사용하여 target_short_name, target_nation, abstract 정보를 기반으로
applicant가 올바르게 매칭되었는지 검증하는 스크립트 (비동기 버전 + Langfuse 로깅)
"""

import os
import csv
import json
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

# .env 파일에서 환경 변수 로드
load_dotenv()

# Langfuse OpenAI 통합 사용 (자동 로깅)
# LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST 환경변수가 설정되어 있으면 자동 활성화
try:
    from langfuse.openai import AsyncOpenAI
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))
    if LANGFUSE_ENABLED:
        print("✓ Langfuse 로깅이 활성화되었습니다.")
except ImportError:
    from openai import AsyncOpenAI
    LANGFUSE_ENABLED = False
    print("⚠ Langfuse가 설치되지 않았습니다. 기본 OpenAI 클라이언트를 사용합니다.")

# OpenAI API 클라이언트 초기화 (비동기)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 사용할 OpenAI 모델
MODEL = "o4-mini"
#MODEL = "gpt-5.2"

# 입력/출력 파일 경로
INPUT_CSV_PATH = "./data/input/output_random_6.csv"
OUTPUT_CSV_PATH = f"./data/output/random_6_validated_{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 동시 요청 수 제한 (Rate limit 고려)
MAX_CONCURRENT_REQUESTS = 20

# 세션 ID (Langfuse에서 그룹핑용)
SESSION_ID = f"patent_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_model_params(model: str, max_tokens: int, temperature: float = 0) -> dict:
    """
    모델 버전에 따라 적절한 API 파라미터를 반환합니다.
    
    - GPT-5 이상, o1, o3, o4 계열: max_completion_tokens 사용
    - o1, o3, o4 계열: temperature 설정 불가 (고정값 1)
    - 그 외 (GPT-4, GPT-4o 등): max_tokens + temperature 사용
    """
    # 새로운 토큰 파라미터를 사용하는 모델 (max_completion_tokens)
    new_token_models = ["gpt-5", "o1", "o3", "o4"]
    
    # temperature 설정이 불가능한 모델 (reasoning 모델)
    no_temperature_models = ["o1", "o3", "o4"]
    
    params = {}
    
    # 토큰 파라미터 설정
    use_new_token = any(model.startswith(prefix) for prefix in new_token_models)
    if use_new_token:
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
    
    # temperature 설정 (reasoning 모델은 제외)
    no_temp = any(model.startswith(prefix) for prefix in no_temperature_models)
    if not no_temp:
        params["temperature"] = temperature
    
    return params


async def validate_matching(
    row_id: str,
    target_short_name: str,
    target_nation: str,
    abstract: str,
    applicant: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """비동기로 매칭 검증 (Langfuse 자동 로깅)"""
    
    prompt = f"""당신은 특허 데이터의 매칭 검증 전문가입니다.

아래 정보를 분석하여 'applicant(출원인)'가 'target(대상 기업)'과 올바르게 매칭되었는지 판단해주세요.

단, 명확하게 true(일치)/false(불일치)로 판단이 어려울 경우, "불확실(uncertain)"을 선택할 수 있습니다. 즉, 총 3가지 선택지를 고려하여 판단해주세요.

## 입력 정보
- **Target 기업명**: {target_short_name}
- **Target 국가**: {target_nation}
- **특허 초록**: {abstract[:500] if abstract else "없음"}...
- **출원인(Applicant)**: {applicant}

## 판단 기준
1. Target 기업명과 Applicant가 동일한 기업을 가리키는지 확인
2. 기업명이 영어/한국어/일본어 등 다른 언어로 표기되었을 수 있음을 고려
3. 음차 표기(예: Sony → 소니), 약어, 또는 법인 형태(Inc, Ltd, 주식회사 등)의 차이 고려
4. 음차 표기가 일치하지 않더라도, 특허 초록의 기술 분야가 해당 기업의 사업 분야와 연관성이 있는지 참고
    
5. 다만 일부 데이터에서는 실제로는 불일치한 매칭이지만, 우연히 표기명과 음차명이 동일할 수 있기 때문에 반드시 Target 기업의 국적과 출원인의 국적 및 사업분야을 고려하여 판단 

## 응답 형식 (JSON)
아래 3가지 중 하나로 "is_valid"를 선택하세요.
- true: 명확히 일치한다고 판단됨
- false: 명확히 일치하지 않는다고 판단됨
- uncertain: 정보가 불충분하거나 판단이 어려움(글자수 최대 100자 이내로 설명)

## 예시
- Target 기업명이 'Prometheus Biosciences Inc'인 기업은 출원인 '프로메테우스 바이오사이언시즈, 인크.'와 일치한다고 판단
- Target 기업명: Samsung Electronics, Applicant: OO반도체 회사,  특허 초록: 반도체 장치 및 반도체 장치의 제조 방법 -> 이 경우 회사명은 다르지만, 사업분야가 유사하므로 일치한다고 판단

{{
    "is_valid": "True", "False", 또는 "Uncertain",
    "confidence": 0.0 ~ 1.0 사이의 확신도,
    "reason": "판단 이유를 간단히 설명"
}}

반드시 JSON만 응답해주세요."""

    messages = [
        {"role": "system", "content": "당신은 특허 데이터 매칭 검증 전문가입니다. 항상 유효한 JSON 형식으로만 응답합니다."},
        {"role": "user", "content": prompt}
    ]

    max_retries = 3
    retry_delay = 1.0  # 초
    
    async with semaphore:  # 동시 요청 수 제한
        for attempt in range(max_retries):
            try:
                # 모델 버전에 따라 파라미터 자동 설정
                # - GPT-5/o1/o3/o4: max_completion_tokens 사용
                # - o1/o3/o4 (reasoning 모델): temperature 설정 불가
                model_params = get_model_params(MODEL, max_tokens=1000, temperature=0)
                
                # Langfuse 자동 로깅 (환경변수 설정 시 자동 활성화)
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    **model_params
                )
                
                # 응답 검증
                if not response.choices or not response.choices[0].message.content:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"is_valid": None, "confidence": 0.0, "reason": "빈 응답"}
                
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
                    return {"is_valid": None, "confidence": 0.0, "reason": "빈 응답"}
                
                return json.loads(result_text)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                # 마지막 시도에서도 실패하면 원본 응답도 기록
                return {"is_valid": None, "confidence": 0.0, "reason": f"JSON 파싱 오류: {str(e)}", "raw_response": result_text[:200] if 'result_text' in locals() else "N/A"}
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                return {"is_valid": None, "confidence": 0.0, "reason": f"API 오류: {str(e)}"}
        
        return {"is_valid": None, "confidence": 0.0, "reason": "최대 재시도 횟수 초과"}


async def process_row(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """단일 행 처리"""
    row_id = row.get('id', 'unknown')
    
    validation_result = await validate_matching(
        row_id=row_id,
        target_short_name=row.get('target_short_name', ''),
        target_nation=row.get('target_nation', ''),
        abstract=row.get('abstract', ''),
        applicant=row.get('applicant', ''),
        semaphore=semaphore
    )
    
    return {
        **row,
        'is_valid': validation_result.get('is_valid'),
        'confidence': validation_result.get('confidence', 0.0),
        'validation_reason': validation_result.get('reason', '')
    }


async def process_csv_async(input_path: str, output_path: str):
    """비동기로 CSV 처리"""
    
    # CSV 파일 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"총 {len(rows)}개의 행을 처리합니다. (동시 요청: {MAX_CONCURRENT_REQUESTS}개)")
    if LANGFUSE_ENABLED:
        print(f"Langfuse 세션 ID: {SESSION_ID}")
    
    # 동시 요청 수 제한을 위한 세마포어
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 모든 행을 비동기로 처리
    tasks = [process_row(row, semaphore) for row in rows]
    results = await tqdm_asyncio.gather(*tasks, desc="검증 진행")
    
    # is_valid 값 정규화 (문자열 "true"/"false" → boolean 변환)
    for r in results:
        val = r['is_valid']
        if val is True or (isinstance(val, str) and val.lower() == "true"):
            r['is_valid'] = True
        elif val is False or (isinstance(val, str) and val.lower() == "false"):
            r['is_valid'] = False
        elif isinstance(val, str) and val.lower() == "uncertain":
            r['is_valid'] = "uncertain"
        # None이나 기타 값은 그대로 유지
    
    # 통계 계산 (true, false, uncertain, error)
    valid_count = sum(1 for r in results if r['is_valid'] is True)
    invalid_count = sum(1 for r in results if r['is_valid'] is False)
    uncertain_count = sum(1 for r in results if r['is_valid'] == "uncertain")
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
    
    if LANGFUSE_ENABLED:
        print(f"\n✓ Langfuse 로그가 자동 전송되었습니다. 세션: {SESSION_ID}")


def main():
    """메인 함수"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
    
    if not LANGFUSE_ENABLED:
        print("경고: Langfuse 키가 설정되지 않았습니다. 로깅이 비활성화됩니다.")
        print("  LANGFUSE_SECRET_KEY와 LANGFUSE_PUBLIC_KEY를 .env 파일에 추가해주세요.")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {INPUT_CSV_PATH}")
        return
    
    # 비동기 실행
    asyncio.run(process_csv_async(INPUT_CSV_PATH, OUTPUT_CSV_PATH))


if __name__ == "__main__":
    main()
