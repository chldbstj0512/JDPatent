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

def main(
    user_id: str,
    user_ocr: str,
    row: pd.Series, 
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
    # 1. NAIC 추출
    # ---------------------------
    user_info_list = run_NAIC_extract(
        naic_df,
        user_id,
        user_ocr
    )

    # ---------------------------
    # 2. 점수 평가
    # ---------------------------
    user_score = run_score(
        user_info_list=user_info_list,
        df=acquisitions_df,
        field_scores=field_scores,
        hightech_list=hightech_list,
        user_prefer=user_prefer,
        row=row,
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

    # 1. 사용자 아이디 (구분자)
    user_id = "yunseo" 

    # 2. 사용자 OCR (1~3페이지)
    user_ocr = """
    --- Page 1 (ocr) ---
US 20240139302A1

a2) Patent Application Publication co) Pub. No.: US 2024/0139302 Al

as) United States

Selak et al.

(43) Pub. Date:          May 2, 2024

 

(54) PROPIONIBACTERIUM ACNES
PROPHYLACTIC AND THERAPEUTIC
IMMUNE TREATMENT

(71) Applicant: Origimm Biotechnology GmbH, Wien
(AT)

(72) Inventors: Sanja Selak, Wein (AT); Christine
Triska, Korneuburg (AT); Manfired
Schuster, Schrick (AT); Johannes
Séllner, Wien (AT); Bernhard
Roppenser, Wien (AT); Theresa
Weinhaupl, Wien (AT); Max Réssler,
Wien (AT)

(21) Appl. No.: — 17/801,099

(22) PCT Filed:      Feb. 22, 2021

 

 

 

 

 

 

(86) PCT No.:      PCT/EP2021/054346
8 371 (0001),
(2) Date:       Aug. 19, 2022
(30)            Foreign Application Priority Data
Feb. 21, 2020 (EP) wee cteeeeteneeeees 20158656.7
Feb. 21, 2020 (EP) ...             ... 20158659.1
Feb. 21, 2020 (EP) ...             ... 20158661.7
Feb. 21, 2020 (EP) .[시니늬늬니니이에에에에에아아아 20158662.5
100,000> (> IA1 (NCTC737)
| ES [A2 (P.aen31}
| ES] 1B (KPA171202)
| RSI IC (PV68)
| a tl (HLOSOPA2)
80,000- MHI {Asn12)
은
근 는 50.000-
ey
so
oy       |
o         |
sc
5 = 40,0004
®         |
20,000-
o-           fey ep

oh. oi dn

Publication Classification

(51) Int. CL
AGIK 39/05                 (2006.01)
AGIP 17/10                 (2006.01)
AGIP 31/04                 (2006.01)
CO7K 14/195             (2006.01)
(52) U.S. Cl
CPC .……………   A61K 39/05 (2013.01); A6IP 17/10
(2018.01); A61P 31/04 (2018.01); CO7K
14/195 (2013.01)
(57)                          ABSTRACT

The present invention discloses a vaccine comprising one or
more of Dermatan sulfate-binding adhesin 1 of P. acnes
(DsA1 polypeptide), Dermatan sulfate-binding adhesin 2 of
P. acnes (DsA2 polypeptide), and putative iron-transport
protein (PITP) polypeptide of P acnes, and/or a fragment
and/or derivative of DsAl and/or DsA2 and/or PITP,
wherein the DsAl polypeptide and the DsA2 polypeptide
comprise from N- to C-terminus an N-terminal swapping
region (“NSR”), a first conserved sub-domain (“CSD1”), a
first swapping region (“SR1”), a second conserved sub-
domain (“CSD2”), a second swapping region (“SR2”), a
third conserved sub-domain (“CSD3”), a Pro-Thr repeat
containing region (“PT repeat region”), and a C-terminal
region (“CTR”), and wherein the PITP polypeptide com-
prises from N- to C-terminus an extended neocarzinostatin
family domain (““ENFD”), a first swapping region (“SR1”),
a heme-binding domain (“HbD”), a second swapping region
(“SR2”) including the C-terminal LPXTG motif, and a
hydrophobic C-terminal region (““hLAR”).

Specification includes a Sequence Listing.

    

Soh

 

 

UA SARs

ERAN

 

 

on aon
ee &€ F&F EP SF EF EE S ¥

   

Immunization antigens

    
    """

    # 3. 사용자 특허청구항 정보 (3~, 청구항 상세내용 및 청구항 개수 등이 포함된 df)
    patent_df = pd.read_csv(
        "/home/ys0660/JDProject/final/data/user_patent.csv"
    )
    row = patent_df.iloc[0]

    # 4. 평가 옵션
    user_prefer = "nation"
    user_prefer_nation = "South Korea"
    user_prefer_area = None

    # 5. acquiror/target 각 DB에서 select하여 가지고 오는 값 (업데이트 주기 정해서 update.py에 작성하기)
    avg_claim_count = 7
    avg_ipc_count = 4.15
    avg_citation_count = 10

    # 6. DB 로드
    acquisitions_df = pd.read_csv(
        '/home/ys0660/JDProject/NAIC_with_RAG/src/final/DB/acquisitions_20260129.csv'
    )

    # 7. 이외 사전 데이터 로드
    naic_df = pd.read_csv(
        '/home/ys0660/JDProject/final/data/NAICS_descripition.csv'
    )

    with open("data/field_scores.json", "r", encoding="utf-8") as f:
        field_scores = json.load(f)

    with open("data/hightech_list.json", "r", encoding="utf-8") as f:
        hightech_list = json.load(f)

    ################# main 호출 #################

    result = main(
        user_id=user_id,
        user_ocr=user_ocr,
        row=row,                     
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
