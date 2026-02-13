## 더미데이터 필터링 작업

### Raw data

- raw data는 `target`, `acquiror`, `acquisition` 데이터 총 3가지로 이뤄져 있음
- `target` 데이터는 약 242만개, `acquiror` 데이터는 약 231만개의 행으로 이뤄져 있으며 `.csv` 형태로 export 했을 때 각각 7GB 이상 차지
- 따라서 github에 업로드하지 않고 구글 드라이브에 업로드 해놓음

**`target` 예시 데이터**

| id  | update_date      | target_id | target_short_name | target_nation | api_flag | applicant               | application_date | application_number | abstract                                    | invention_name | ipc_code                         | claim                     |
| --- | ---------------- | --------- | ----------------- | ------------- | -------- | ----------------------- | ---------------- | ------------------ | ------------------------------------------- | -------------- | -------------------------------- | ------------------------- |
| 1   | 2025.11.22 20:57 | 4         | +Automation Inc   | Japan         | KR       | 주식회사 인텍오토메이션 | 20160701         | 1.02016E+12        | 본 발명은 직교좌표 로봇에 있어서, ...(생략) | 직교좌표 로봇  | B25J 9/02\|B25J 9/10\|B25J 19/00 | 1. 구동수단(700)...(생략) |

**`acquiror` 예시 데이터**
| id | update_date | acquiror_id | acquiror_short_name | acquiror_nation | api_flag | applicant | application_date | application_number | abstract | invention_name | ipc_code | claim |
|----|----------------------|-------------|---------------------|-----------------|----------|-------------------------------|------------------|-------------------|---------------------------------------------------|----------------|-----------------------------------------------------------------------------------------|---------------------------------------|
| 3 | 2025-11-21 20:49:06 | 2 | & Vet | Japan | KR | 베링거잉겔하임베트메디카게엠베하 | 20180705 | 1020237039620 | 고혈압의 치료를 필요로 하는 고양이에서...(생략) | 고양이에서 전신 질환의 예방 또는 치료를 위한 안지오텐신 II 수용체 길항제 | A61D 7/00&#124;A61K 31/4184&#124;A61K 9/08&#124;A61P 9/12&#124;A61J 1/20&#124;A61M 5/31 | 1. 안지오텐신 II 수용체 1 길항제 ..(생략) |

**`acquisition` 예시 데이터**

| id  | year | deal_id    | deal_no | acquiror_code | acquiror_long_name                                                   | acquiror_short_name | acquiror_nation | acquiror_sic_code | target_code | target_long_name       | target_short_name | target_nation  | target_sic_code | deal_synopsis                                                          | deal_value | per_sought | sub_target_long_name | sub_target_nation | sub_target_ultimate_nation | sub_target_public_status | target_public_status | target_advisors | target_acquiror |
| --- | ---- | ---------- | ------- | ------------- | -------------------------------------------------------------------- | ------------------- | --------------- | ----------------- | ----------- | ---------------------- | ----------------- | -------------- | --------------- | ---------------------------------------------------------------------- | ---------- | ---------- | -------------------- | ----------------- | -------------------------- | ------------------------ | -------------------- | --------------- | --------------- |
| 1   | 2005 | 2034796-N1 | N1      | 9             | Hynix Semiconductor Inc(Wuxi)Wuxi Hynix Zhongying Electronics Co Ltd | Hynix Semicon...    | South Korea     | 3674              | 6           | Longcheer Holdings Ltd | Longcheer H...    | Cayman Islands | 3663            | 1/31/05 Hynix Semiconductor Inc agreed to acquire a 51% interest in... | 38.16      | 0.51       | Sub-Target...        | Cayman Islands    | Cayman Islands             | Target                   | Target Public Status | ...             | ...             |

## 해결 방법

### 방법 1. 음차 기반 매칭

- **target 기업**과 **출원인(applicants)** 에 등장하는 기업명을 비교
- 영어로 표기된 target 기업명을 토대로 영어 혹은 한국어로 음차표기했을 때 유사도가 가장 높은 applicants 추출 (인공지능 적용되지 않은 상태)
- 7170개는 매칭 되었으나, 3764개는 매칭 불가로 분류(확신도 낮음)
- 매칭 불가된 데이터들은 보통 target 기업의 표기명을 제대로 인지하지 못해서 발생
  - ex)Genexon Co Ltd → 주식회사 지넥슨
- 다만 target 기업과 applicants의 상호 연관성을 전혀 유추할 수 없는 사례도 등장
  - ex)target : Zing Dev Ltd → applicants: (주)무등종합개발
  - 이런 경우는 분류 불가능
- 또한 공동 출원인이 여러개 등장하는 경우 어떻게 처리해야 하는지 논의 필요
  - 예)Delta X Co Ltd → 주식회사 델타엑스|나노인텍 주식회사, 주식회사 델타엑스, 나노인텍 주식회사|주식회사 델타엑스
- `pronounce_based.py`참고

### 방법 2. OpenAI API로 쿼리

- `target_short_name` + `abstract`(혹은 `invention_name`) -> `applicant`가 올바르게 매칭되었는지 확인
- LLM을 활용하여 회사명 매칭의 정확도를 높임

## 스크립트 설명

### 1. 비동기 방식 (실시간 처리)

#### `llm_query_basic_acquiror.py`

- Acquiror 기업과 출원인 매칭 검증
- AsyncOpenAI를 사용한 비동기 처리
- 중소규모 데이터셋에 적합
- 실시간 응답이 필요한 경우 사용

#### `llm_query_basic_target.py`

- Target 기업과 출원인 매칭 검증
- AsyncOpenAI를 사용한 비동기 처리
- 중소규모 데이터셋에 적합

### 2. 배치 방식 (대용량 처리)

#### `llm_query_batch_acquiror.py`

- **OpenAI Batch API** 사용으로 **50% 비용 절감**
- 대량 데이터 처리 최적화 (20,000개씩 배치 분할)
- 24시간 내 처리 완료
- 배치 상태 모니터링 및 재개 기능
- 200만 건 이상의 데이터 처리 가능

**주요 설정:**

```python
MODEL = "gpt-5-mini"           # 모델 선택
BATCH_SIZE = 20000             # 배치 크기
MAX_BATCHES = None             # None: 전체 실행, 숫자: 테스트용
ERROR_THRESHOLD = 10           # 에러 임계값
```

**사용 예시:**

```bash
python llm_query_batch_acquiror.py
```

#### `llm_query_batch_target.py`

- Target 기업 매칭을 위한 Batch API 스크립트
- `llm_query_batch_acquiror.py`와 동일한 구조 및 기능

### 3. 배치 재개 스크립트

#### `resume_batch.py`

- 중단된 배치 작업을 재개
- 저장된 `batch_ids.json` 파일을 읽어 진행 상황 확인
- 완료되지 않은 배치만 처리

## 프롬프트 관리

### 중앙화된 규칙 관리

모든 매칭 검증 스크립트는 공통 규칙을 사용합니다:

```
prompts/
└── matching_rules.py    # 회사명 매칭 규칙 정의
```

**`matching_rules.py` 구조:**

- 규칙 1: 음역 또는 표준 표기의 합리성
- 규칙 2: 의미적 내용의 추가/누락 금지
- 규칙 3: 법인 형태 접미사 유연 처리
- 규칙 4: 브랜드 확장/약어 재해석 금지
- 규칙 5: 기호/숫자/특수 문자 합리적 반영

**사용 방법:**

```python
from prompts.matching_rules import RULES

PROMPT_TEMPLATE = f"""당신은 회사명 매칭 검증 전문가입니다.
...
{RULES}
...
"""
```

**장점:**

- ✅ 규칙 수정 시 한 곳만 변경하면 모든 스크립트에 반영
- ✅ 일관된 검증 기준 유지
- ✅ 유지보수 용이

## 비용 최적화

### OpenAI Batch API 활용

| 방식           | 비용         | 처리 시간   | 적합한 경우                |
| -------------- | ------------ | ----------- | -------------------------- |
| **실시간 API** | 기본 요금    | 즉시        | 소규모 데이터, 실시간 필요 |
| **Batch API**  | **50% 할인** | 24시간 이내 | 대량 데이터, 비용 중요     |

### 모델 선택 가이드

| 모델          | 상대적 비용 | 성능 | 추천 용도                |
| ------------- | ----------- | ---- | ------------------------ |
| `gpt-5-nano`  | 매우 낮음   | 낮음 | 간단한 매칭              |
| `gpt-5-mini`  | 낮음        | 중간 | **일반적인 매칭 (권장)** |
| `gpt-4o-mini` | 중간        | 중상 | 복잡한 매칭              |
| `gpt-5.2`     | 높음        | 최고 | 매우 복잡한 매칭         |

### 비용 추정 예시

**200만 건 데이터 처리 (gpt-5-mini + Batch API 기준):**

- 예상 비용: 약 $XXX (50% 할인 적용)
- 처리 시간: 24시간 이내
- 배치 개수: 100개 (20,000건씩)

## 유사 특허 검색

### `find_similar_patents.py`

Pinecone 벡터 DB를 활용한 유사 특허 검색 스크립트입니다.

**주요 기능:**
- Target/Acquiror 인덱스에서 유사 특허 검색
- OpenAI 임베딩 기반 의미적 유사도 검색
- IPC 코드 기반 필터링 (optional)
- 상위 5개 결과 반환

**사용 예시:**

```bash
# Target 인덱스 검색
python find_similar_patents.py \
  --index target \
  --query "온라인 쇼핑몰 관련 특허"

# IPC 필터와 함께 검색
python find_similar_patents.py \
  --index acquiror \
  --query "전자상거래 시스템" \
  --ipc "G06Q 30/02"
```

**CLI 옵션:**
- `--index`: 검색할 인덱스 (target 또는 acquiror) - 필수
- `--query`: 검색 쿼리 텍스트 - 필수
- `--ipc`: IPC 코드 필터 (선택)

## 환경 설정

### 필수 환경 변수 (.env)

```bash
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 API 키를 설정하세요.

### 의존성 설치

```bash
pip install openai python-dotenv pandas openpyxl tqdm pinecone-client
```

## 사용 워크플로우

### 1. 소규모 테스트 (실시간)

```bash
# Acquiror 매칭
python llm_query_basic_acquiror.py

# Target 매칭
python llm_query_basic_target.py
```

### 2. 대규모 처리 (배치)

```bash
# 1. 배치 제출 및 실행
python llm_query_batch_acquiror.py

# 2. (선택) 중단된 경우 재개
python resume_batch.py
```

### 3. 결과 확인

- 출력 파일: `data/output/validated_acquiror_YYYYMMDD_HHMMSS.xlsx`
- 컬럼: `id`, `acquiror_short_name`, `applicant`, `is_valid`, `confidence`

## 주요 특징

### ✅ 비용 효율성

- Batch API 활용으로 50% 비용 절감
- 필요에 따라 모델 선택 가능

### ✅ 확장성

- 20,000개씩 배치 분할로 대용량 처리
- 200만 건 이상 데이터 처리 가능

### ✅ 안정성

- 배치 상태 자동 모니터링
- 에러 임계값 설정으로 자동 취소
- 중단 시 재개 기능

### ✅ 유지보수성

- 중앙화된 프롬프트 관리
- 명확한 코드 구조
- 상세한 로그 출력

## 파일 구조

```
filtering_dummy/
├── README.md
├── .env.example
├── prompts/
│   └── matching_rules.py          # 공통 매칭 규칙
├── llm_query_basic_acquiror.py    # 실시간 Acquiror 매칭
├── llm_query_basic_target.py      # 실시간 Target 매칭
├── llm_query_batch_acquiror.py    # 배치 Acquiror 매칭
├── llm_query_batch_target.py      # 배치 Target 매칭
├── resume_batch.py                # 배치 재개
├── find_similar_patents.py        # Pinecone 유사 특허 검색
├── pronouce_based.py              # 음차 기반 매칭
├── extract_acquiror.py            # Acquiror 데이터 추출
└── extract_target.py              # Target 데이터 추출
```
