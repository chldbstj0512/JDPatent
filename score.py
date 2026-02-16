import pandas as pd
import numpy as np
import ast
import json

from dotenv import load_dotenv
from openai import OpenAI 
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_field(field_scores, user_field):
    field_score = field_scores.get(user_field)

    if field_score is None:
        raise ValueError(f"Unknown field: {user_field}")

    llm_payload = {
        "field": user_field,
        "trend_metrics": field_score,
        "field_metric_definition": {
            "recent_growth": "최근 N년간 M&A 성장률",
            "recovery_ratio": "과거 최저점 대비 회복 수준",
            "recent_trend_slope": "최근 추세의 연간 기울기",
            "final_score": "성장성과 회복성을 가중 결합한 종합 점수"
        }
    }

    return llm_payload

def get_naics_trend_payload(
    acquisitions_df,
    user_code,
    start_year=2020,
    end_year=2025,
    year_col="Year",
    tgt_col="Target Primary NAIC Code 2022",
    acq_col="Acquiror Primary NAIC Code 2022"
):

    user_code = str(user_code)
    user_prefix = user_code[:4]
    years = list(range(start_year, end_year + 1))

    df_tgt = acquisitions_df[
        acquisitions_df[tgt_col].astype(str).eq(user_code) &
        acquisitions_df[year_col].isin(years)
    ].copy()

    df_tgt["is_internal"] = (
        df_tgt[acq_col]
        .astype(str)
        .str[:4]
        .eq(user_prefix)
    )

    result = (
        df_tgt
        .groupby(year_col)
        .agg(
            A_target_cnt=(tgt_col, "count"),
            B_internal_cnt=("is_internal", "sum")
        )
        .reset_index()
    )

    result["B_ratio"] = (
        result["B_internal_cnt"] / result["A_target_cnt"]
    ).round(3)

    result = (
        pd.DataFrame({year_col: years})
        .merge(result, on=year_col, how="left")
        .fillna(0)
    )

    payload = {
        "naics_code": user_code,
        "naics_trend_metric_definition": {
            "A_target_cnt": "연도별 Target NAICS 기업 M&A 건수",
            "B_internal_cnt": "동종 산업군 인수자의 인수 건수",
            "B_ratio": "동종 산업 내부 인수 비중 (B/A)"
        },
        "time_series": [
            {
                "year": int(row[year_col]),
                "A": int(row["A_target_cnt"]),
                "B": int(row["B_internal_cnt"]),
                "B_ratio": round(float(row["B_ratio"]), 3)
            }
            for _, row in result.iterrows()
        ]
    }

    return payload

def get_hightech(user_title, user_abstract, hightech_list):

    payload = {
        "hightech_task_definition": {
            "objective": (
                "특허의 제목과 요약을 분석하여 "
                "해당 특허가 High-Tech 기술 분야에 해당하는지 판단"
            ),
            "hightech_criteria": (
                "아래에 정의된 High-Tech 기술 분야 중 하나 이상과 "
                "기술적으로 명확한 관련성이 있는 경우 High-Tech로 판단"
            )
        },

        "patent_info": {
            "title": user_title,
            "abstract": user_abstract
        },

        "hightech_categories": hightech_list,

        "output_schema": {
            "is_hightech": {
                "type": "binary",
                "description": "High-Tech 해당 여부 (해당: 1, 비해당: 0)"
            },
            "matched_hightech": {
                "type": "list[string]",
                "description": "해당되는 High-Tech 기술 분야 목록 (없으면 빈 리스트)"
            },
            "explanation": {
                "type": "string",
                "description": "High-Tech로 판단한 근거에 대한 간단한 설명"
            }
        },

        "constraints": [
            "High-Tech 기술 분야는 반드시 제공된 리스트 내에서만 선택",
            "명확한 기술적 관련성이 없는 경우 is_hightech는 0으로 판단",
            "출력은 반드시 output_schema를 따를 것"
        ]
    }

    return payload

def analyze_user_preference_ma( # 여기에서의 df는 사용자 코드에 대한 df임
    df,
    user_prefer,                 # "nation" or "area"
    user_prefer_nation=None,     # ex) "USA"
    user_prefer_area=None        # ex) "Europe"
):

    df = df.copy()

    total_cnt = len(df)
    if total_cnt == 0:
        raise ValueError("입력 df에 데이터가 없습니다.")

    if user_prefer == "nation":
        cross_border_cnt = df["Cross_Border_Deal_Flag_Nation"].sum()
    elif user_prefer == "area":
        cross_border_cnt = df["Cross_Border_Deal_Flag_Area"].sum()
    else:
        raise ValueError("user_prefer는 'nation' 또는 'area'여야 합니다.")

    cross_border_majority = cross_border_cnt / total_cnt > 0.5

    if user_prefer == "nation":
        if user_prefer_nation is None:
            raise ValueError("user_prefer_nation이 필요합니다.")

        outbound_cnt = (
            (df["Acquiror Nation"] == user_prefer_nation) &
            (df["Target Nation"] != user_prefer_nation)
        ).sum()

        inbound_cnt = (
            (df["Acquiror Nation"] != user_prefer_nation) &
            (df["Target Nation"] == user_prefer_nation)
        ).sum()

    else:  # area
        if user_prefer_area is None:
            raise ValueError("user_prefer_area가 필요합니다.")

        outbound_cnt = (
            (df["Acquiror Primary Nation Region"] == user_prefer_area) &
            (df["Target Primary Nation Region"] != user_prefer_area)
        ).sum()

        inbound_cnt = (
            (df["Acquiror Primary Nation Region"] != user_prefer_area) &
            (df["Target Primary Nation Region"] == user_prefer_area)
        ).sum()

    if outbound_cnt > inbound_cnt:
        direction = "outbound"
    elif inbound_cnt > outbound_cnt:
        direction = "inbound"
    else:
        direction = "balanced"

    if user_prefer == "nation":
        prefer_acquiror_cnt = (df["Acquiror Nation"] == user_prefer_nation).sum()
    else:
        prefer_acquiror_cnt = (df["Acquiror Primary Nation Region"] == user_prefer_area).sum()

    prefer_acquiror_ratio = round(prefer_acquiror_cnt / total_cnt, 3)

    return {
        "total_deals": total_cnt,
        "cross_border_ratio": round(cross_border_cnt / total_cnt, 3),
        "cross_border_majority": cross_border_majority,
        "outbound_cnt": int(outbound_cnt),
        "inbound_cnt": int(inbound_cnt),
        "direction": direction,
        "prefer_acquiror_ratio": prefer_acquiror_ratio
    }

def get_crossborder(
    naics_code,
    crossborder_result,
    prefer_type="nation"  # "nation" or "area"
):

    payload = {
        "naics_code": str(naics_code),

        "crossborder_metric_definition": {
            "total_deals": "분석 대상 NAICS 코드의 전체 M&A 건수",

            "cross_border_ratio": (
                "전체 M&A 중 국경 간 거래 비중"
                if prefer_type == "nation"
                else "전체 M&A 중 지역 간 거래 비중"
            ),

            "cross_border_majority": (
                "국경(또는 지역) 간 거래가 과반을 차지하는지 여부"
            ),

            "outbound_cnt": (
                "사용자 선호 국가/지역이 인수자(Acquiror)인 cross-border 거래 건수 "
                "(기술·자본 유출)"
            ),

            "inbound_cnt": (
                "사용자 선호 국가/지역이 피인수자(Target)인 cross-border 거래 건수 "
                "(기술·자본 유입)"
            ),

            "direction": (
                "cross-border 거래의 방향성 "
                "(outbound: 유출 중심, inbound: 유입 중심, balanced: 균형)"
            ),

            "prefer_acquiror_ratio": (
                "전체 M&A 중 사용자 선호 국가/지역이 인수자인 비중"
            )
        },

        "crossborder_metrics": {
            "total_deals": int(crossborder_result["total_deals"]),
            "cross_border_ratio": float(crossborder_result["cross_border_ratio"]),
            "cross_border_majority": bool(crossborder_result["cross_border_majority"]),
            "outbound_cnt": int(crossborder_result["outbound_cnt"]),
            "inbound_cnt": int(crossborder_result["inbound_cnt"]),
            "direction": crossborder_result["direction"],
            "prefer_acquiror_ratio": float(crossborder_result["prefer_acquiror_ratio"])
        }
    }

    return payload

def get_MAtype(
    df,
    user_code,
    target_name_col="Target Full Name",
    stake_col="M&A_Is_Stake"
):

    total_deals = len(df)

    if total_deals == 0:
        return {
            "naics_code": str(user_code),
            "ma_type_definition": {},
            "ma_type_metrics": {},
            "note": "해당 NAICS 코드에 대한 M&A 데이터가 존재하지 않음"
        }
    
    stake_df = df[df[stake_col] == 1]
    full_df = df[df[stake_col] == 0]

    stake_target_counts = (
        stake_df[target_name_col]
        .value_counts()
    )

    stake_single_target_cnt = int((stake_target_counts == 1).sum())
    stake_multi_target_cnt = int((stake_target_counts >= 2).sum())
    full_acquisition_cnt = int(len(full_df))

    payload = {
        "naics_code": str(user_code),

        "ma_type_definition": {
            "stake_single_target_cnt": (
                "주식 일부 인수(Stake Purchase) 중 "
                "동일 Target 기업에 대해 1회만 발생한 경우 "
                "(탐색적·옵션형 투자 성격)"
            ),
            "stake_multi_target_cnt": (
                "주식 일부 인수(Stake Purchase) 중 "
                "동일 Target 기업에 대해 2회 이상 발생한 경우 "
                "(단계적 인수 또는 지배력 강화 전략)"
            ),
            "full_acquisition_cnt": (
                "주식 전체 인수(Full Acquisition) 건수 "
                "(기술·시장 즉시 확보 목적)"
            ),
            "total_deals": "해당 NAICS 코드의 전체 M&A 건수"
        },

        "ma_type_metrics": {
            "total_deals": total_deals,
            "stake_single_target_cnt": stake_single_target_cnt,
            "stake_multi_target_cnt": stake_multi_target_cnt,
            "full_acquisition_cnt": full_acquisition_cnt
        }
    }

    return payload

def get_one_prompt(
    user_info_list,     
    df,                 # 전체 M&A df
    field_scores,
    hightech_list,
    user_prefer,
    user_prefer_nation=None,
    user_prefer_area=None
):

    user_info = user_info_list[0]   # 현재 구조는 단일 특허

    user_nation = user_info.get("country")
    user_title = user_info.get("title")
    user_abstract = user_info.get("abstract")
    user_field = user_info.get("field")

    # NAICS는 리스트 구조
    user_code = user_info.get("primary_naic_info", {}).get("code")

    df_tgt = df[
        df['Target Primary NAIC Code 2022'].astype(str) == str(user_code)
    ].copy()

    field_result = get_field(
        field_scores=field_scores,
        user_field=user_field
    )

    trend = get_naics_trend_payload(
        acquisitions_df=df,
        user_code=user_code
    )

    hightech_prompt = get_hightech(
        user_title=user_title,
        user_abstract=user_abstract,
        hightech_list=hightech_list
    )

    crossborder_result = analyze_user_preference_ma(
        df=df_tgt,
        user_prefer=user_prefer,
        user_prefer_nation=user_prefer_nation,
        user_prefer_area=user_prefer_area
    )

    crossborder_payload = get_crossborder(
        naics_code=user_code,
        crossborder_result=crossborder_result,
        prefer_type=user_prefer
    )

    ma_type_payload = get_MAtype(
        df=df_tgt,
        user_code=user_code
    )

    return {
        "user_info": user_info,
        "field_analysis": field_result,
        "naics_trend": trend,
        "hightech_prompt": hightech_prompt,
        "crossborder_analysis": crossborder_payload,
        "ma_type_analysis": ma_type_payload
    }

def build_ma_prompt(payload):
    return f"""
당신은 기술·산업·M&A 분석을 통해 해당 기술의 M&A 시장 매력도를 산출하는 에이전트입니다.

아래 JSON 데이터는 특정 특허와 해당 산업의
기술적 특성, M&A 동향, 국경 간 거래 구조, 인수 전략 유형을 포함합니다.

[입력 데이터]
{json.dumps(payload, ensure_ascii=False, indent=2)}

[판단 기준 가이드]
- M&A 매력도는 다음 요소를 종합적으로 고려하여 판단하십시오.
  1. 산업의 최근 M&A 성장성 및 회복성
  2. 국경 간 M&A 구조의 개방성 및 확장성
  3. 인수 전략 유형(지분 인수 vs 전체 인수)의 전략적 의미
  4. 기술의 High-Tech 여부 및 기술적 파급력

- "높음":
  성장성 또는 회복성이 명확하고 전략적 인수가 활발한 경우
- "중간":
  긍정적 신호와 부정적 신호가 혼재된 경우
- "낮음":
  산업 확장성 또는 전략적 매력도가 낮은 경우

[출력 형식]
아래 JSON 형식으로만 답변하십시오.

{{
  "ma_attractiveness": "높음 | 중간 | 낮음",
  "analysis_reasoning": [
    "판단 근거 1 (수치 또는 명확한 지표 포함)",
    "판단 근거 2 (수치 또는 명확한 지표 포함)",
    "판단 근거 3 (선택, 최대 3개)"
  ],
  "user_report": "사용자에게 제공할 M&A 매력도 설명 (최대 5줄)"
}}

[주의 사항]
- analysis_reasoning은 최대 3개까지만 작성하십시오.
- 반드시 제공된 수치 및 지표를 근거로 판단하십시오.
- user_report는 비전문가도 이해할 수 있는 표현으로 작성하십시오.
- 출력은 반드시 JSON 형식만 사용하십시오.
"""

def build_rights_prompt(
    row,
    avg_claim_count
):

    return f"""
너는 특허 권리성 평가 전문가이다.
아래 특허공보의 청구항 구성과 실제 청구항 내용을 종합적으로 검토하여 권리성을 평가하라.

[평가 기준 요약]
- 전체 청구항 수, 독립항 수, 종속항 수, 청구항 계열 수는 많을수록 권리 설계에 유리하다.
- 독립항의 길이는 짧을수록 권리 범위가 명확하고 안정적이다.
- 방법청구항 또는 시스템청구항만으로 구성된 경우 권리성은 불리하다.
- 독립항에 수학식이나 수치 범위가 포함될 경우 권리 범위가 제한될 수 있다.
- 전체 청구항 수는 평균값({avg_claim_count})과 비교하여 판단한다.
- 수치적 지표뿐만 아니라, 실제 청구항 문구와 특허 데이터 전반을 함께 고려하여 정성적으로 판단하라.

[특허 데이터]
- 전체 청구항 수: {row['claim_count']}
- 독립항 수: {row['independent_claim_count']}
- 독립항 단어 수: {row['independent_claim_word_count']}
- 종속항 수: {row['dependent_claim_count']}
- 청구항 계열 수: {row['claim_family_count']}
- 독립항 내용:
{row['independent_claim']}

[출력 지침]
1. evaluation
  - evaluation에는 수치나 항목을 나열하지 말고, 권리 설계의 인상과 전반적인 수준을 유저 친화적으로 설명하라.
  - 평가 기준이나 내부 판단 로직이 드러나지 않도록 작성하라.
  - 3~5문장 이내로 작성하라.
  - "~습니다", "~보입니다" 형태의 설명체로 작성하라.

2. reason
  - reason에는 정량·정성 평가가 어떻게 반영되었는지 솔직하고 구체적으로 작성하라.
  - reason의 각 문장은 반드시 "~임.", "~함.", "~존재함."과 같은 서술형으로 끝내라.

[출력 형식]
반드시 JSON 형식으로만 출력하라.

{{
  "rights_score": "높음 | 중간 | 낮음",
  "evaluation": "유저에게 제공되는 권리성 평가 요약",
  "reason": "정량·정성 평가 기준이 어떻게 반영되었는지에 대한 내부 설명"
}}
"""

def build_tech_prompt(row, avg_ipc_count, avg_citation_count):
    return f"""
너는 특허 기술성 평가 전문가이다.
아래 특허공보의 기술성을 평가하라.

[작성 지침]
- evaluation은 일반 사용자에게 보여줄 문장이다.
  · 특허 내용 요약은 하지 마라.
  · 평가 기준이나 수치 비교 방식은 직접 언급하지 마라.
  · 기술 설계의 완성도, 구체성, 신뢰도를 설명하듯 작성하라.
  · "~습니다", "~보입니다" 형태의 설명체로 작성하라.
- reason은 내부 확인용이다.
  · IPC 개수, 인용문헌 수, 청구항 내 정량 정보 등
    정량·정성 평가 기준이 어떻게 반영되었는지 솔직하게 작성하라.
    각 문장은 반드시 "~임.", "~함.", "~존재함."과 같은 서술형으로 끝내라.
- ipc코드는 많을수록, 인용문헌은 적을수록 좋으며 아래 평균값과 비교하라.
  · ipc코드 개수의 평균은 {avg_ipc_count}
  · 인용문헌 개수의 평균은 {avg_citation_count}

[특허 정보]
- IPC 개수: {row['ipc_count']}
- 인용문헌 수: {row['forward_citation_count']}
- 특허청구항:
{row['independent_claim']}

[출력 형식]
반드시 JSON 형식으로만 출력하라.

{{
  "technical_score": "높음 | 중간 | 낮음",
  "evaluation": "최대 5줄 이내의 유저용 기술성 평가",
  "reason": "정량·정성 평가 기준이 어떻게 반영되었는지에 대한 내부용 설명"
}}
"""

def call_llm(prompt, model="gpt-4.1"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional patent analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    raw_text = response.choices[0].message.content.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]

    return json.loads(raw_text)

def evaluate_patent(
    payload,
    row,
    avg_claim_count,
    avg_ipc_count,
    avg_citation_count
):
    ma_prompt = build_ma_prompt(payload)
    ma_result = call_llm(ma_prompt)

    tech_prompt = build_tech_prompt(
        row=row,
        avg_ipc_count=avg_ipc_count,
        avg_citation_count=avg_citation_count
    )
    tech_result = call_llm(tech_prompt)

    rights_prompt = build_rights_prompt(
        row=row,
        avg_claim_count=avg_claim_count
    )
    rights_result = call_llm(rights_prompt)

    final_output = {
        "evaluation": {
            "ma_market_evaluation": ma_result,
            "tech_evaluation": tech_result,
            "rights_evaluation": rights_result
        }
    }

    return final_output

def run_score(
    user_info_list,
    df,
    field_scores,
    hightech_list,
    user_prefer,
    row,
    avg_claim_count,
    avg_ipc_count,
    avg_citation_count,
    user_prefer_nation=None,
    user_prefer_area=None
):

    payload = get_one_prompt(
        user_info_list=user_info_list,
        df=df,
        field_scores=field_scores,
        hightech_list=hightech_list,
        user_prefer=user_prefer,
        user_prefer_nation=user_prefer_nation,
        user_prefer_area=user_prefer_area
    )

    final_result = evaluate_patent(
        payload=payload,
        row=row,
        avg_claim_count=avg_claim_count,
        avg_ipc_count=avg_ipc_count,
        avg_citation_count=avg_citation_count
    )

    return final_result


