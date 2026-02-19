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

    # ---------------------------
    # 1. NAIC + Claim 추출
    # ---------------------------
    user_info_list, claims = run_NAIC_extract(
        naic_df,
        user_id,
        front_ocr,
        back_ocr
    )

    # ---------------------------
    # 2. 점수 평가 (row → claims)
    # ---------------------------
    user_score = run_score(
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

    user_id = "yunseo"

    # ----------------------------
    # OCR
    # ----------------------------
    text = open("./util/ocr_text.txt", "r", encoding="utf-8").read()

    front_ocr, back_ocr = split_ocr_text(text)
    # ----------------------------
    # 평가 옵션
    # ----------------------------
    user_prefer = "nation"
    user_prefer_nation = "South Korea"
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

    print(json.dumps(result, indent=2, ensure_ascii=False))
