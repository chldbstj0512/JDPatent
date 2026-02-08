"""
OpenAI Batch API를 사용하여 특허 출원인(applicant)과 인수 기업(acquiror)의 매칭을 검증하는 스크립트
- 대량 데이터 처리에 적합 (50% 비용 절감)
- 24시간 내 처리 완료
- 50,000개씩 배치 분할하여 대용량 처리 지원
"""

import os
import csv
import json
import time
import sys
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# .env 파일 명시적 로드
load_dotenv(Path(__file__).parent / '.env')

from openai import OpenAI

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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#MODEL = "gpt-5-nano"
MODEL = "gpt-5-mini"

# o4-mini 모델은 상대적으로 저렴한 cost, 중간 정도 성능
#MODEL = "o4-mini"

# gpt-5.2 모델은 가장 최신 모델, 높은 cost(o4-mini대비 약 3배 이상), 가장 높은 성능
#MODEL = "gpt-5.2"

# OpenAI Batch API 제한: 50,000 requests per batch
BATCH_SIZE = 20000
#BATCH_SIZE = 3000

# 테스트 모드: 처음 N개 배치만 실행 (None이면 전체 실행)
# 예: 1 = 첫 번째 배치만, 2 = 처음 2개 배치, None = 전체
MAX_BATCHES = None # 테스트 시 1로 설정, 전체 실행 시 None으로 변경
#MAX_BATCHES = 1

# 에러 임계값: 실패 요청이 이 값 이상이면 배치 자동 취소
ERROR_THRESHOLD = 10

# 입력/출력 파일 경로
INPUT_CSV_PATH = "filtering_dummy/data/input/retry_acquiror_empty_20260206_022216.xlsx"
#INPUT_CSV_PATH = "filtering_dummy/data/input/output_random_acquiror_10.csv"
OUTPUT_DIR = "filtering_dummy/data/output"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_XLSX_PATH = f"{OUTPUT_DIR}/validated_acquiror_{MODEL}_batch_{TIMESTAMP}.xlsx"
BATCH_IDS_FILE = f"{OUTPUT_DIR}/batch_ids_acquiror_{TIMESTAMP}.json"


def safe_str(value, default='') -> str:
    """NaN 등을 안전하게 문자열로 변환"""
    import math
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
- **출원인(Applicant)**: {applicant}"""
    
    return static_part + dynamic_part


def create_batch_requests(rows: list, batch_num: int, start_idx: int) -> str:
    """Batch API용 JSONL 파일 생성"""
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    batch_file_path = f"{OUTPUT_DIR}/batch_requests_acquiror_{TIMESTAMP}_part{batch_num}.jsonl"
    
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        for local_idx, row in enumerate(rows):
            global_idx = start_idx + local_idx
            request = {
                "custom_id": f"request-{global_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 글로벌 특허 데이터 매칭 검증 전문가입니다. 각국의 기업과 사업 분야에 대해서 잘 알고 있습니다. 항상 유효한 JSON 형식으로만 응답합니다."
                        },
                        {
                            "role": "user",
                            "content": create_prompt(row)
                        }
                    ],
                    "max_completion_tokens": 4000,
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Batch 요청 파일 생성: {batch_file_path}")
    return batch_file_path


def submit_batch(file_path: str, batch_num: int) -> str:
    """Batch 작업 제출"""
    
    # 파일 업로드
    print(f"  파일 업로드 중...")
    with open(file_path, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    print(f"  ✓ 파일 업로드 완료: {batch_input_file.id}")
    
    # Batch 작업 생성
    print(f"  Batch 작업 생성 중...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Patent matching validation (acquiror) - Part {batch_num} - {TIMESTAMP}"
        }
    )
    print(f"  ✓ Batch 작업 생성 완료: {batch.id}")
    
    return batch.id


def save_batch_ids(batch_info: list):
    """batch_id 정보를 파일에 저장 (중간 복구용)"""
    with open(BATCH_IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)
    print(f"✓ Batch ID 저장: {BATCH_IDS_FILE}")


def load_batch_ids() -> list:
    """저장된 batch_id 정보 로드"""
    if os.path.exists(BATCH_IDS_FILE):
        with open(BATCH_IDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def cancel_batch(batch_id: str) -> bool:
    """배치 작업 취소"""
    try:
        client.batches.cancel(batch_id)
        print(f"  ✓ 배치 취소됨: {batch_id}")
        return True
    except Exception as e:
        print(f"  ✗ 배치 취소 실패: {e}")
        return False


def cancel_all_batches(batch_ids_file: str = None):
    """저장된 모든 배치 작업 취소"""
    file_path = batch_ids_file or BATCH_IDS_FILE
    
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_info_list = json.load(f)
    
    print(f"\n총 {len(batch_info_list)}개 배치 취소 시도...")
    
    for info in batch_info_list:
        batch_id = info['batch_id']
        print(f"  배치 {info['batch_num']} (ID: {batch_id}) 취소 중...")
        cancel_batch(batch_id)
    
    print("✓ 모든 배치 취소 완료")


def wait_for_batch(batch_id: str, batch_num: int, poll_interval: int = 30) -> dict:
    """Batch 작업 완료 대기 (에러 임계값 초과 시 자동 취소)"""
    
    print(f"\n[배치 {batch_num}] 작업 진행 중... (ID: {batch_id})")
    print("(Ctrl+C로 중단 가능, 나중에 resume_batches()로 재개 가능)")
    
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        # 진행 상황 출력
        completed = batch.request_counts.completed
        failed = batch.request_counts.failed
        total = batch.request_counts.total
        
        print(f"  상태: {status} | 완료: {completed}/{total} | 실패: {failed}    ", end='\r')
        
        # 에러 임계값 초과 시 자동 취소
        if failed >= ERROR_THRESHOLD:
            print(f"\n  ⚠️ 실패 요청 {failed}개 (임계값: {ERROR_THRESHOLD}) - 배치 자동 취소")
            cancel_batch(batch_id)
            batch = client.batches.retrieve(batch_id)  # 취소 후 상태 다시 조회
            return batch
        
        if status == "completed":
            print(f"\n  ✓ 배치 {batch_num} 완료!")
            return batch
        elif status in ["failed", "expired", "cancelled"]:
            print(f"\n  ✗ 배치 {batch_num} 실패: {status}")
            return batch
        
        time.sleep(poll_interval)


def download_results(batch: dict) -> list:
    """Batch 결과 다운로드"""
    
    if not batch.output_file_id:
        print("결과 파일이 없습니다.")
        return []
    
    print(f"  결과 다운로드 중... (file_id: {batch.output_file_id})")
    
    result_content = client.files.content(batch.output_file_id)
    results = []
    
    for line in result_content.text.strip().split('\n'):
        if line:
            results.append(json.loads(line))
    
    print(f"  ✓ {len(results)}개 결과 다운로드 완료")
    return results


def parse_results(batch_results: list, original_rows: list, start_idx: int = 0) -> list:
    """Batch 결과 파싱 및 원본 데이터와 병합
    
    Args:
        batch_results: 배치 API 응답 결과
        original_rows: 원본 데이터 행들
        start_idx: 이 배치의 시작 인덱스 (custom_id 매칭용)
    """
    
    # custom_id → result 매핑
    result_map = {}
    for result in batch_results:
        custom_id = result.get('custom_id', '')
        idx = int(custom_id.replace('request-', ''))
        result_map[idx] = result
    
    merged_results = []
    
    for local_idx, row in enumerate(original_rows):
        global_idx = start_idx + local_idx
        result = result_map.get(global_idx, {})
        
        # 응답 파싱
        validation = {"is_valid": None, "confidence": 0.0}
        
        if result.get('response', {}).get('status_code') == 200:
            try:
                content = result['response']['body']['choices'][0]['message']['content']
                content = content.strip()
                
                # JSON 파싱
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                
                validation = json.loads(content)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                validation = {"is_valid": None, "confidence": 0.0}
        elif result.get('error'):
            validation = {"is_valid": None, "confidence": 0.0}
        
        merged_results.append({
            'id': row.get('id', ''),
            'acquiror_short_name': row.get('acquiror_short_name', ''),
            'applicant': row.get('applicant', ''),
            'is_valid': validation.get('is_valid'),
            'confidence': validation.get('confidence', 0.0),
        })
    
    return merged_results


def save_results(results: list, output_path: str):
    """결과 XLSX 저장"""
    
    if not results:
        print("저장할 결과가 없습니다.")
        return
    
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"✓ 결과 저장: {output_path}")
    
    # 통계 출력
    print_statistics(results)


def print_statistics(results: list):
    """통계 출력"""
    def normalize(val):
        if val is None:
            return None
        return str(val).lower() if val else None
    
    valid_count = sum(1 for r in results if normalize(r['is_valid']) == "true")
    invalid_count = sum(1 for r in results if normalize(r['is_valid']) == "false")
    uncertain_count = sum(1 for r in results if normalize(r['is_valid']) == "uncertain")
    error_count = sum(1 for r in results if r['is_valid'] is None)
    
    print(f"\n=== 검증 결과 통계 ===")
    print(f"총 처리: {len(results)}건")
    print(f"✅ 일치(True): {valid_count}건 ({valid_count/len(results)*100:.1f}%)")
    print(f"❌ 불일치(False): {invalid_count}건 ({invalid_count/len(results)*100:.1f}%)")
    print(f"❓ 불확실(Uncertain): {uncertain_count}건 ({uncertain_count/len(results)*100:.1f}%)")
    print(f"⚠️  오류(Error): {error_count}건 ({error_count/len(results)*100:.1f}%)")


def check_batch_status(batch_id: str):
    """기존 Batch 작업 상태 확인"""
    batch = client.batches.retrieve(batch_id)
    print(f"Batch ID: {batch.id}")
    print(f"상태: {batch.status}")
    print(f"완료: {batch.request_counts.completed}/{batch.request_counts.total}")
    print(f"실패: {batch.request_counts.failed}")
    return batch


def resume_batches(batch_ids_file: str = None):
    """
    중단된 배치 작업 재개
    
    사용법:
        resume_batches('filtering_dummy/data/output/batch_ids_acquiror_20260114_123456.json')
    """
    if batch_ids_file is None:
        print("batch_ids 파일 경로를 입력하세요.")
        return
    
    if not os.path.exists(batch_ids_file):
        print(f"파일을 찾을 수 없습니다: {batch_ids_file}")
        return
    
    with open(batch_ids_file, 'r', encoding='utf-8') as f:
        batch_info = json.load(f)
    
    print(f"저장된 배치 정보 로드: {len(batch_info)}개 배치")
    
    # 원본 파일 로드 (CSV 또는 XLSX)
    input_path = batch_info[0].get('input_file') if batch_info else None
    if not input_path or not os.path.exists(input_path):
        print("원본 파일을 찾을 수 없습니다.")
        return
    
    if input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path, engine='openpyxl')
        all_rows = df.to_dict('records')
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    
    all_results = []
    
    for info in batch_info:
        batch_id = info['batch_id']
        batch_num = info['batch_num']
        start_idx = info['start_idx']
        end_idx = info['end_idx']
        
        print(f"\n=== 배치 {batch_num} 확인 중 ({start_idx+1}~{end_idx}) ===")
        
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            print(f"  상태: {batch.status} - 완료 대기 중...")
            batch = wait_for_batch(batch_id, batch_num)
        
        if batch.status == "completed":
            batch_results = download_results(batch)
            chunk_rows = all_rows[start_idx:end_idx]
            merged = parse_results(batch_results, chunk_rows, start_idx=start_idx)
            all_results.extend(merged)
        else:
            print(f"  ⚠️ 배치 {batch_num} 처리 실패")
    
    if all_results:
        output_path = batch_ids_file.replace('batch_ids_acquiror_', 'validated_acquiror_').replace('.json', '.xlsx')
        save_results(all_results, output_path)


def main():
    """메인 함수"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {INPUT_CSV_PATH}")
        return
    
    print("=" * 70)
    print("OpenAI Batch API 특허 매칭 검증 - Acquiror 기반 (대용량 지원)")
    print("=" * 70)
    print(f"모델: {MODEL}")
    print(f"배치 크기: {BATCH_SIZE:,}개/배치")
    if MAX_BATCHES:
        print(f"⚠️  테스트 모드: 처음 {MAX_BATCHES}개 배치만 실행")
    print(f"입력: {INPUT_CSV_PATH}")
    print()
    
    # 파일 읽기 (CSV 또는 XLSX)
    print("파일 로드 중...")
    if INPUT_CSV_PATH.endswith('.xlsx'):
        df = pd.read_excel(INPUT_CSV_PATH, engine='openpyxl')
        all_rows = df.to_dict('records')
    else:
        with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    
    total_rows = len(all_rows)
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    # 실행할 배치 수 결정
    num_batches = min(total_batches, MAX_BATCHES) if MAX_BATCHES else total_batches
    
    print(f"총 {total_rows:,}개 행")
    print(f"→ 전체 {total_batches}개 배치 중 {num_batches}개 실행 예정")
    print()
    
    try:
        # 배치 정보 저장용
        batch_info_list = []
        
        # Step 1: 모든 배치 제출
        print("=" * 50)
        print("[Step 1] 배치 제출")
        print("=" * 50)
        
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, total_rows)
            chunk = all_rows[start_idx:end_idx]
            
            print(f"\n--- 배치 {i+1}/{num_batches} ({start_idx+1:,}~{end_idx:,}) ---")
            
            batch_file = create_batch_requests(chunk, batch_num=i+1, start_idx=start_idx)
            batch_id = submit_batch(batch_file, batch_num=i+1)
            
            batch_info_list.append({
                "batch_num": i + 1,
                "batch_id": batch_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "count": len(chunk),
                "input_file": INPUT_CSV_PATH,
                "status": "submitted"
            })
        
        # batch_id 저장
        save_batch_ids(batch_info_list)
        print(f"\n⚠️  중요: batch_id가 {BATCH_IDS_FILE}에 저장되었습니다.")
        print("   터미널을 종료해도 나중에 resume_batches()로 결과를 받을 수 있습니다.")
        
        # Step 2: 모든 배치 완료 대기 및 결과 수집
        print("\n" + "=" * 50)
        print("[Step 2] 배치 완료 대기 및 결과 수집")
        print("=" * 50)
        
        all_results = []
        
        for i, info in enumerate(batch_info_list):
            batch_id = info['batch_id']
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            
            batch = wait_for_batch(batch_id, batch_num=i+1)
            
            if batch.status == "completed":
                batch_results = download_results(batch)
                
                chunk_rows = all_rows[start_idx:end_idx]
                merged = parse_results(batch_results, chunk_rows, start_idx=start_idx)
                all_results.extend(merged)
                
                # 상태 업데이트
                batch_info_list[i]['status'] = 'completed'
                save_batch_ids(batch_info_list)
            else:
                print(f"  ⚠️ 배치 {i+1} 처리 실패: {batch.status}")
                batch_info_list[i]['status'] = batch.status
                save_batch_ids(batch_info_list)
        
        # Step 3: 최종 결과 저장
        print("\n" + "=" * 50)
        print("[Step 3] 최종 결과 저장")
        print("=" * 50)
        
        if all_results:
            save_results(all_results, OUTPUT_XLSX_PATH)
            print(f"\n✅ 결과가 저장되었습니다: {OUTPUT_XLSX_PATH}")
        else:
            print("저장할 결과가 없습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
