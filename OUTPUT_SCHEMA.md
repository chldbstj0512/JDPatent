# Final Output JSON 필드 설명

특허 M&A 분석 파이프라인의 **최종 산출물** 스키마 문서입니다.

| 구분 | 파일 |
|------|------|
| **현재 버전** | `final/final_output_{user_id}.json` (예: `final_output_0604.json`) |
| **Legacy 참조** | `final/output.json` |
| **조립 코드** | `final/output.py` → `build_final_output()` |

---

## 개요

| 항목 | 내용 |
|------|------|
| 형식 | UTF-8 JSON, `indent=2` |
| 사용자-facing 키 (6개) | `basic_info` → `naics` → `evaluation` → `ipc_descriptions` → `recommend_companies` → `ma_patterns` |
| QA 전용 키 (1개) | **`parse_audit`** — 동일 파일 **맨 마지막**. UI/API에서는 생략 가능 |

Legacy `output.json`과 **앞 6개 섹션의 키 이름·순서는 동일**합니다. 차이는 아래 [Legacy 대비 변경](#legacy-outputjson-대비-변경-사항) 참고.

### 점수 체계 (`evaluation`)

| 영역 | 필드 | 범위 | 비고 |
|------|------|------|------|
| 시장 매력도 | `ma_market_score` | 0 ~ 50 | 정량 앵커(42%) + LLM 4관점 합(58%), 합 상한 50 |
| 기술성 | `tech_score` | 0 ~ 25 | LLM 평가 |
| 권리성 | `rights_score` | 0 ~ 25 | LLM 평가 |
| **종합** | `final_score` | 0 ~ 100 | 위 세 점수의 **산술 합** |

---

## Legacy (`output.json`) 대비 변경 사항

### 요약

| 구분 | Legacy | 현재 | 호환 |
|------|--------|------|------|
| 최상위 키 개수 | 6 | 7 (`parse_audit` 추가) | Legacy 클라이언트는 `parse_audit` 무시하면 동일 |
| `basic_info` | 동일 구조 | 동일 + 출원인명·`field` 표시 정규화 | ✅ |
| `naics` | 동일 | 동일 | ✅ |
| `evaluation.ma_market_evaluation` | 3필드 | **세분화** (아래) | ⚠️ 하위 필드 확장 |
| `evaluation.tech/rights/final_result` | 동일 | 동일 | ✅ |
| `ipc_descriptions` | 동일 | 동일 | ✅ |
| `recommend_companies` | **특허 단위** | **기업 단위** | ⚠️ 구조 변경 |
| `ma_patterns` | 동일 | 동일 | ✅ |
| `parse_audit` | 없음 | **파일 맨 끝** | 🆕 QA 전용 |

### 1. `parse_audit` (신규)

- Legacy에는 없음.
- OCR·IPC·청구 추출 **품질 점검**용. `basic_info`와 분리해 최상위 **마지막**에만 위치.
- 피인용 수(`forward_citation_count`) 등은 여기 `claims_audit`에만 포함 (`basic_info.patent`에는 없음).

### 2. `evaluation.ma_market_evaluation` (세분화)

Legacy는 아래 3필드만 제공했습니다.

```json
{
  "ma_market_score": 23,
  "ma_market_evaluation": "...",
  "ma_market_reason": "..."
}
```

현재 버전은 **동일 3필드를 유지**하면서, 해석·UI용 하위 필드를 추가합니다.

| 필드 | Legacy | 현재 | 설명 |
|------|--------|------|------|
| `ma_market_score` | ✅ | ✅ | 최종 시장 매력도 (0~50) |
| `ma_market_evaluation` | ✅ | ✅ | 통합 서술문 (4관점 평가 연결) |
| `ma_market_reason` | ✅ | ✅ | 근거 전문 (섹션 제목 포함) |
| `ma_anchor_score` | — | 🆕 | 정량 앵커 단독 점수 |
| `ma_llm_four_perspectives_sum` | — | 🆕 | LLM 4관점 점수 합 (상한 50) |
| `ma_score_blending` | — | 🆕 | 블렌딩 가중치 (`anchor_weight` 0.42 등) |
| `ma_market_perspectives` | — | 🆕 | `field_trend`, `crossborder`, `ma_type`, `hightech` |
| `ma_market_anchor` | — | 🆕 | 앵커 요약·`anchor_reason` |

**마이그레이션:** Legacy UI는 기존처럼 `ma_market_score` / `ma_market_evaluation` / `ma_market_reason`만 사용하면 됩니다.

### 3. `recommend_companies` (특허 → 기업 단위)

| 항목 | Legacy (`output.json`) | 현재 |
|------|------------------------|------|
| 목록 키 | `target_similar_patents`, `acquiror_similar_patents` | `target_similar_companies`, `acquiror_similar_companies` |
| 한 행의 단위 | **특허 1건** | **기업 1곳** (유사 특허 `patents[]`로 묶음) |
| 유사도 | 행마다 `similarity` | 기업 `avg_similarity` + 특허별 `similarity` |
| 권리자 | `assignee: { name, id }` | `company_name`, `company_id` |
| 추가 필드 | — | `matched_patent_count`, `patent_country` (특허별) |

Legacy 특허 행 예:

```json
{
  "rank": 1,
  "similarity": 0.6385,
  "assignee": { "name": "Netlist Inc", "id": "38346" },
  "title": "...",
  "ipc": ["G06F11/00"],
  "abstract": "...",
  "application_number": "16517210",
  "application_date": "20190719"
}
```

현재 기업 행 예:

```json
{
  "rank": 1,
  "company_name": "Medtronic Inc",
  "company_id": "35311",
  "avg_similarity": 0.5312,
  "matched_patent_count": 2,
  "patents": [
    {
      "similarity": 0.5602,
      "title": "...",
      "ipc": ["A61N1/02"],
      "abstract": "...",
      "application_number": "15375722",
      "application_date": "20161212",
      "patent_country": "US"
    }
  ]
}
```

### 4. `basic_info` (미세 변경)

| 필드 | Legacy | 현재 |
|------|--------|------|
| `field` | 문자열 그대로 (예: `computer`) | 내부 코드 → **영문 풀네임** (`compu`→`computer`, `elec`→`electronic` 등) |
| `patent.applicant.name` | 법인명 | US 서지 `Corp, City, ST (US)` → **법인명만** (`Intel Corporation`) |
| `patent` 키 순서 | title → applicant → grant → ipc_code → abstract | 동일 |

### 5. 변경 없음 (구조 동일)

- `naics` (`primary`, `candidates`)
- `evaluation.tech_evaluation`, `rights_evaluation`, `final_result`
- `ipc_descriptions`
- `ma_patterns` (`target_naic`, `target_naic_title`, `patterns[]`)

---

## 1. `basic_info`

입력 특허(PDF/OCR) 메타데이터. Legacy `output.json`과 **동일한 키 구조**.

| 필드 | 타입 | 설명 |
|------|------|------|
| `pdf_name` | string | 입력 식별자 (`user_id`와 동일) |
| `country` | string \| null | ISO 3166-1 alpha-2 (`KR`, `US`) |
| `field` | string \| null | 기술 분야 표시명 (아래 매핑表) |
| `patent` | object | 특허 메타 (1.1) |

**`field` 내부 코드 → 출력값**

| 내부 | 출력 |
|------|------|
| `compu` | `computer` |
| `bio` | `biotechnology` |
| `comm` | `communications` |
| `elec` | `electronic` |
| `etc` | `other` |

### 1.1 `patent`

| 필드 | 타입 | 설명 |
|------|------|------|
| `title` | string | 발명의 명칭 |
| `applicant.name` | string | 출원인 법인명 (주소 접미 제거) |
| `applicant.number` | string | 출원번호 |
| `applicant.date` | string | 출원일 (`YYYY-MM-DD` 권장) |
| `grant.number` | string | 등록번호 |
| `grant.date` | string | 등록일 |
| `ipc_code` | string[] | IPC 코드 목록 |
| `abstract` | string | 초록 |

---

## 2. `naics`

Legacy와 동일.

| 필드 | 설명 |
|------|------|
| `primary.code` | 1순위 NAICS 6자리 |
| `primary.title` | 영문 산업명 |
| `primary.description` | 영문 산업 설명 |
| `candidates[]` | 후보 NAICS (`primary`와 동일 구조, 없으면 `[]`) |

---

## 3. `evaluation`

### 3.1 `ma_market_evaluation`

#### Legacy 호환 필드 (필수 소비)

| 필드 | 타입 | 설명 |
|------|------|------|
| `ma_market_score` | integer | 최종 시장 매력도 0~50 |
| `ma_market_evaluation` | string | 사용자용 통합 서술 |
| `ma_market_reason` | string | 지표별 근거 전문 |

#### 확장 필드 (현재 버전)

| 필드 | 설명 |
|------|------|
| `ma_anchor_score` | 정량 앵커 0~50 |
| `ma_llm_four_perspectives_sum` | LLM 4관점 합 (cap 50) |
| `ma_score_blending` | `anchor_weight` 0.42, `llm_four_perspectives_sum_weight` 0.58 |
| `ma_market_anchor` | `ma_market_anchor_evaluation`, `anchor_reason` |
| `ma_market_perspectives` | 4관점 객체 (3.1.1) |

**최종 점수:** `round(0.42 × ma_anchor_score + 0.58 × ma_llm_four_perspectives_sum)`, 0~50 클램프.

**앵커 구성 (max 50):**

| 항목 | 산식 | 상한 |
|------|------|------|
| 분야 M&A 종합 | `final_score × 22` | 22 |
| 동종 인수 비중 | 최근 3년 `B_ratio` 평균 × 16 | 16 |
| 국경간 비중 | `cross_border_ratio × 12` | 12 |
| 완전 인수 비율 | `full/total × 10` | 10 |

#### 3.1.1 `ma_market_perspectives` (관점별 공통 패턴)

각 관점: `{관점}_score`, `ma_market_{관점}_evaluation`, `ma_market_{관점}_evaluation_paragraphs[]`, `{관점}_reason`.

| 키 | 내용 |
|----|------|
| `field_trend` | 5개년 M&A 성장·회복·동종 인수 비중·집중도 |
| `crossborder` | 국경간 비중·inbound/outbound·연도별 건수 |
| `ma_type` | 전체 인수 vs 지분 인수·반복 지분 인수 |
| `hightech` | High-Tech 분류 적합성 |

### 3.2 `tech_evaluation` / 3.3 `rights_evaluation` / 3.4 `final_result`

Legacy와 **필드·구조 동일**.

| 블록 | 필드 |
|------|------|
| `tech_evaluation` | `tech_score`, `tech_evaluation`, `tech_reason` |
| `rights_evaluation` | `rights_score`, `rights_evaluation`, `rights_reason` |
| `final_result` | `final_score`, `evaluation`, `reason` |

---

## 4. `ipc_descriptions`

Legacy와 동일. `basic_info.patent.ipc_code`와 1:1 대응.

| 필드 | 설명 |
|------|------|
| `ipc_code` | IPC 코드 |
| `short_description` | 짧은 한글 라벨 |
| `long_description` | 상세 한글 설명 |

---

## 5. `recommend_companies`

Legacy **특허 단위** → 현재 **기업 단위** (위 [§3](#3-recommend_companies-특허--기업-단위) 참고).

| 필드 | 설명 |
|------|------|
| `target_similar_companies` | 피인수(타깃) 관점 유사 기업 |
| `acquiror_similar_companies` | 인수자 관점 유사 기업 |

**기업 객체:** `rank`, `company_name`, `company_id`, `avg_similarity`, `matched_patent_count`, `patents[]`

**`patents[]`:** `similarity`, `title`, `ipc[]`, `abstract`, `application_number`, `application_date`, `patent_country`

---

## 6. `ma_patterns`

Legacy와 동일.

| 필드 | 설명 |
|------|------|
| `target_naic` | 피인수 NAICS |
| `target_naic_title` | 피인수 NAICS 영문명 |
| `patterns[]` | 인수자 NAICS별 패턴 |

**`patterns[]`:** `acquirer_naic`, `acquirer_naic_title`, `ratio_percent`, `relation_label`, `one_line_description`, `reason`

---

## 7. `parse_audit` (현재 버전 전용 · 파일 맨 마지막)

QA·디버깅용. `final/extract.py` → `_build_parse_audit()`.

| 하위 블록 | 용도 |
|-----------|------|
| `ocr_lengths` | OCR 길이·메타데이터 truncation |
| `ipc_counts` | INID(51) strict 추출 vs `ipc_info` 개수 |
| `verbatim_checks` | 제목·초록·독립항 OCR 포함 여부 |
| `claims_audit` | 청구항 수, 피인용 수, IPC 수 등 |

---

## 부록

### A. JSON 트리 — Legacy (`output.json`)

```
basic_info → naics → evaluation → ipc_descriptions → recommend_companies → ma_patterns
(6 keys, parse_audit 없음)
```

### B. JSON 트리 — 현재 (`final_output_*.json`)

```
basic_info
naics
evaluation
ipc_descriptions
recommend_companies
ma_patterns
parse_audit          ← 맨 마지막만 추가
```

### C. 참고 파일

| 파일 | 역할 |
|------|------|
| `final/output.json` | Legacy 샘플 |
| `final/final_output_0604.json` | 현재 샘플 |
| `final/output.py` | JSON 조립 |
| `final/score.py` | 평가·블렌딩 |
| `final/extract.py` | 메타 추출, `parse_audit` |
| `final/company.py` | 유사 기업 (기업 단위) |
| `final/pattern.py` | M&A 패턴 |
| `final/main.py` | 실행·저장 |

---

*문서 버전: 2026-06 · Legacy 기준 `output.json`, 현재 기준 `final_output_0604.json`*
