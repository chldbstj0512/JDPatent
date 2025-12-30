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

### 방법 2. OpenAI API 로 쿼리

- `target_short_name` + `abstract`(혹은 `invention_name`) -> `applicant` 가 올바르게 매칭되었는지 확인

**프롬프트**
