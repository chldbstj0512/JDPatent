import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json

from extract import run_NAIC_extract
from score import run_score
from company import run_company
from pattern import run_pattern
from output import build_final_output

from util.ocr import split_ocr_text
from validate import log_ocr_parse_validation


def prepare_acquisitions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    대규모 M&A 테이블용 최소 품질 필터(수정요구: 추가 데이터 필터링의 기본선).
    권리·기술 기준은 컬럼 정의가 과제별로 달라질 수 있어, 여기서는 핵심 키 결측만 제거한다.
    """
    d = df.copy()
    key_cols = [
        "Target Primary NAIC Code 2022",
        "Year",
        "Acquiror Primary NAIC Code 2022",
    ]
    for c in key_cols:
        if c not in d.columns:
            return df
    d = d.dropna(subset=key_cols)
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d = d.dropna(subset=["Year"])
    return d


def main(
    user_id: str,
    front_ocr: str,
    back_ocr: str,
    acquisitions_df: pd.DataFrame,
    naic_df: pd.DataFrame,
    field_scores: dict,
    hightech_list: dict,
    user_prefer: str,
    user_prefer_nation: str,
    user_prefer_area: str,
    avg_claim_count: float,
    avg_ipc_count: float,
    avg_citation_count: float
):

    acquisitions_df = prepare_acquisitions_df(acquisitions_df)

    # ---------------------------
    # 1. NAIC + Claim 추출
    # ---------------------------
    user_info_list, claims = run_NAIC_extract(
        naic_df,
        user_id,
        front_ocr,
        back_ocr
    )

    if isinstance(claims, dict) and "error" in claims:
        try:
            log_ocr_parse_validation(
                user_id=user_id,
                front_ocr=front_ocr,
                back_ocr=back_ocr,
                user_info_list=user_info_list if user_info_list else None,
                claims=claims,
            )
        except Exception as e:
            print(f"[validation] log failed: {e}", flush=True)
        return {
            "status": "error",
            "reason": claims["error"]
        }

    try:
        log_ocr_parse_validation(
            user_id=user_id,
            front_ocr=front_ocr,
            back_ocr=back_ocr,
            user_info_list=user_info_list,
            claims=claims,
        )
    except Exception as e:
        print(f"[validation] log failed: {e}", flush=True)

    # ---------------------------
    # 2. 점수 평가 (row → claims)
    # ---------------------------
    user_score, _ = run_score(
        user_info_list=user_info_list,
        df=acquisitions_df,
        field_scores=field_scores,
        hightech_list=hightech_list,
        user_prefer=user_prefer,
        row=claims,  
        avg_claim_count=avg_claim_count,
        avg_ipc_count=avg_ipc_count,
        avg_citation_count=avg_citation_count,
        user_prefer_nation=user_prefer_nation,
        user_prefer_area=user_prefer_area
    )

    # ---------------------------
    # 3. 유사 기업 추천
    # ---------------------------
    user_similar_company = run_company(user_info_list)

    # ---------------------------
    # 4. M&A 패턴 분석
    # ---------------------------
    user_pattern = run_pattern(
        user_info_list=user_info_list,
        acquisitions_df=acquisitions_df,
        naic_df=naic_df
    )

    # ---------------------------
    # 5. 최종 포맷 빌드
    # ---------------------------
    final_result = build_final_output(
        user_info_list=user_info_list,
        user_score=user_score,
        user_similar_company=user_similar_company,
        user_pattern=user_pattern
        )

    return final_result

if __name__ == "__main__":
    user_id = "0604"

    # ----------------------------
    # OCR
    # ----------------------------
    text = open("./util/ocr_text.txt", "r", encoding="utf-8").read()

    front_ocr, back_ocr = split_ocr_text(text)
    # ----------------------------
    # 평가 옵션
    # ----------------------------
    user_prefer = "nation"
    user_prefer_nation = None
    user_prefer_area = None

    avg_claim_count = 7
    avg_ipc_count = 4.15
    avg_citation_count = 10

    # ----------------------------
    # DB 로드
    # ----------------------------
    acquisitions_df = pd.read_csv(
        './data/acquisitions_20260129.csv'
    )

    naic_df = pd.read_csv(
        './data/NAICS_descripition.csv'
    )

    with open("./data/field_scores.json", "r", encoding="utf-8") as f:
        field_scores = json.load(f)

    with open("./data/hightech_list.json", "r", encoding="utf-8") as f:
        hightech_list = json.load(f)

    # ----------------------------
    # main 호출
    # ----------------------------
    result = main(
        user_id=user_id,
        front_ocr=front_ocr,
        back_ocr=back_ocr,
        acquisitions_df=acquisitions_df,
        naic_df=naic_df,
        field_scores=field_scores,
        hightech_list=hightech_list,
        user_prefer=user_prefer,
        user_prefer_nation=user_prefer_nation,
        user_prefer_area=user_prefer_area,
        avg_claim_count=avg_claim_count,
        avg_ipc_count=avg_ipc_count,
        avg_citation_count=avg_citation_count
    )

    _final_dir = os.path.dirname(os.path.abspath(__file__))
    final_json_path = os.path.join(_final_dir, f"final_output_{user_id}.json")
    with open(final_json_path, "w", encoding="utf-8") as _f:
        json.dump(result, _f, ensure_ascii=False, indent=2)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {final_json_path}", flush=True)

