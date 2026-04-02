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

    # print("payload>>>>", llm_payload)

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
    user_prefer = 'nation',
    user_prefer_nation=None,
    user_prefer_area=None
):

    user_info = user_info_list[0]   # 현재 구조는 단일 특허

    user_nation = user_info.get("country")
    user_title = user_info.get("title")
    user_abstract = user_info.get("abstract")
    user_field = user_info.get("field")

    if user_nation == "KR":
        user_prefer_nation = "South Korea"
    else:
        user_prefer_nation = "United States"
        
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
    # print(">>>>>> ma 추론 시 활용하는 payload 전문", payload)
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
  "ma_attractiveness_score": 0,
  "metric_evaluations": [
    {{
      "metric_name": "예: 최근 성장률 (recent_growth)",
      "data_basis": "입력 JSON에서 인용한 원시 값·문자열 (가공·재계산 금지)",
      "interpretation": "그 수치를 M&A 매력도 관점에서 해석한 한 줄"
    }}
  ],
  "evaluation_summary": "전체 판단을 4~7문장으로 보강 요약 (비전문가용, 문장체)",
  "analysis_reasoning": [
    "내부용 판단 근거 1 (~임./~함. 종결)",
    "내부용 판단 근거 2",
    "내부용 판단 근거 3 (선택, 최대 3개)"
  ]
}}

[metric_evaluations 작성 규칙 — 검증 가능하도록]
- 각 원소는 반드시 "metric_name", "data_basis", "interpretation" 키를 가진다.
- **data_basis**에는 반드시 [입력 데이터] JSON에 **실제로 존재하는 수치·문자열을 그대로 또는 그대로 인용 가능한 형태로** 적는다.
  · 임의로 반올림·단위만 바꿔 숫자를 바꾸지 말 것. (예: recent_growth가 0.278이면 data_basis에 0.278 또는 동일 값의 퍼센트 표기만 허용)
- **interpretation**에는 그 data_basis가 M&A 매력도(성장·회복·국경간·인수유형·하이테크 등)에 어떤 의미인지 짧게 쓴다.
- **반드시 포함할 항목** (각각 별도 배열 원소로 작성):
  1) field_analysis.trend_metrics의 **recent_growth** — metric_name에 "최근 성장률" 포함
  2) field_analysis.trend_metrics의 **recovery_ratio** — metric_name에 "회복 비율" 또는 "회복 정도" 포함
  3) field_analysis.trend_metrics의 **recent_trend_slope**
  4) field_analysis.trend_metrics의 **final_score** (분야 M&A 동향 종합)
  5) naics_trend.time_series에서 **최근 연도 1~2개**의 A(건수)·B_ratio 등 핵심 수치 인용 1~2행
  6) crossborder_analysis.crossborder_metrics의 **total_deals, cross_border_ratio, direction** 중 최소 2개 수치·값을 data_basis에 명시하는 행 1개 이상
  7) ma_type_analysis.ma_type_metrics가 비어 있지 않으면 **stake·full 인수 건수** 관련 1행
  8) 특허와 hightech_categories 관련 **High-Tech 해당 여부** 1행 (data_basis에 판단 근거로 든 입력 요약)
- 위 항목을 누락하지 말 것. 입력에 해당 블록이 없거나 값이 비어 있으면 data_basis에 "입력 데이터 없음"이라고 적고 interpretation에 그에 따른 한계를 적는다.

[주의 사항]
- ma_attractiveness_score는 0~50 사이의 정수로 반환하십시오.
- "높음"에 해당하면 35~50, "중간"에 해당하면 18~34, "낮음"에 해당하면 0~17 범위를 사용하십시오.
- analysis_reasoning은 최대 3개, 내부 검토용 서술형(~임./~함.)으로 작성하십시오.
- evaluation_summary는 metric_evaluations와 **수치·표 형태의 반복 나열은 피하고**, 시장·국경간·인수유형·하이테크를 **종합한 서사**로 4~7문장 작성하십시오. "~습니다"체, 비전문가도 이해 가능하게.
- **reason 필드는 출력하지 마십시오.** 내부용 줄 형식은 metric_evaluations로만 제공합니다.
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

[출력 지침 — evaluation]
- 사용자에게 보이는 본문이다. 수치·항목 나열은 하지 말 것.
- 권리 설계의 **인상**, **강·약점**, **실무 관점에서의 의미**를 **분석적으로** 풀어쓴다.
- 다음 각도를 가능한 한 골고루 다룬다 (해당 없으면 생략):
  · 청구 구조가 권리 범위·회피 난이도에 주는 시사점
  · 독립항 문구가 구체적인지, 과도하게 좁히지는 않았는지
  · 방법/시스템/장치 등 청구 유형 구성이 균형적인지
- **5~8문장**, "~습니다", "~보입니다" 설명체.
- 평가 기준이나 내부 로직이 직접 드러나지 않게 쓴다.

[metric_evaluations] (시스템이 이를 조합해 내부용 reason 문자열로 쓴다)
- 아래 항목을 각각 별도 원소로 작성한다. data_basis에는 [특허 데이터]의 **실제 수치**를 그대로 넣는다.
  1) 전체 청구항 수 (평균 {avg_claim_count}과 비교 언급은 interpretation에서)
  2) 독립항 수·종속항 수·청구 계열 수
  3) 독립항 단어 수 (권리 범위 명확성과 연결하여 interpretation 작성)
- interpretation: 각 수치가 권리성에 미치는 영향을 한 줄로.

[출력 형식]
반드시 JSON 형식으로만 출력하라.

{{
  "rights_score": 0,
  "metric_evaluations": [
    {{"metric_name": "전체 청구항 수", "data_basis": "...", "interpretation": "..."}},
    {{"metric_name": "독립항·종속항·계열", "data_basis": "...", "interpretation": "..."}},
    {{"metric_name": "독립항 분량(단어 수)", "data_basis": "...", "interpretation": "..."}}
  ],
  "evaluation": "사용자용 권리성 평가 (5~8문장, 분석적·풍부하게)"
}}

[추가 조건]
- rights_score는 0~25 사이의 정수로 반환하라.
- "높음"에 해당하면 18~25, "중간"에 해당하면 9~17, "낮음"에 해당하면 0~8 범위를 사용하라.
- **reason 필드는 출력하지 마라.** 내부용 줄은 metric_evaluations만으로 충분하다.
"""

def build_tech_prompt(row, avg_ipc_count, avg_citation_count):
    return f"""
너는 특허 기술성 평가 전문가이다.
아래 특허공보의 기술성을 평가하라.

[작성 지침 — evaluation (사용자용)]
- 일반 사용자에게 보이는 본문이다.
- 특허 **발명의 요지를 한 줄로 요약하는 데 그치지 말고**, 아래를 **분석적으로** 5~8문장으로 쓴다.
  · 청구항에 드러난 기술 수단·구성의 **구체성**과 **완성도**
  · 동일 분야에서 흔한 구성 대비 **차별 포인트가 읽히는지**(없다면 그 한계도 서술)
  · 선행(인용) 부담이 **크게 느껴지는지 여부**를 정성적으로 (수치 직접 인용은 피함)
  · 기술 설명의 **신뢰도**(논리 전개, 용어 일관성 등)에 대한 인상
- "~습니다", "~보입니다" 설명체.
- **IPC 개수·인용 건수 등 숫자는 evaluation 본문에 직접 적지 마라.** (해당 내용은 metric_evaluations에만)
- 평가 기준이나 채점 로직이 드러나지 않게 쓴다.

[내부 추적용 metric_evaluations]
- IPC·인용·청구 구조에 대한 **검증 가능한 근거**는 반드시 metric_evaluations에만 둔다.
- ipc코드는 많을수록, 인용문헌은 적을수록 유리한 경향이 있으며 산업 평균과 비교해 해석하라.
  · IPC 개수 평균: {avg_ipc_count}
  · 인용문헌 수 평균: {avg_citation_count}

[특허 정보]
- IPC 개수: {row['ipc_count']}
- 인용문헌 수: {row['forward_citation_count']}
- 특허청구항:
{row['independent_claim']}

[metric_evaluations]
- 반드시 아래 각 항목을 별도 원소로 작성한다. data_basis에는 위 [특허 정보]에 나온 **실제 숫자·문자를 그대로** 인용한다.
  1) IPC 코드 개수 (ipc_count와 산업 평균 {avg_ipc_count} 병기)
  2) 인용문헌 수 (forward_citation_count와 산업 평균 {avg_citation_count} 병기)
  3) 독립항 구조·분량 (독립항 수·단어 수 등 위에 제시된 수치를 data_basis에 포함)
- interpretation에는 각 data_basis가 기술성 점수에 어떻게 반영되는지 한 줄로 쓴다.

[출력 형식]
반드시 JSON 형식으로만 출력하라.

{{
  "technical_score": 0,
  "metric_evaluations": [
    {{"metric_name": "IPC 코드 개수", "data_basis": "...", "interpretation": "..."}},
    {{"metric_name": "인용문헌 수", "data_basis": "...", "interpretation": "..."}},
    {{"metric_name": "독립항·청구 구조", "data_basis": "...", "interpretation": "..."}}
  ],
  "evaluation": "사용자용 기술성 평가 (5~8문장, 분석적·풍부하게)"
}}

[추가 조건]
- technical_score는 0~25 사이의 정수로 반환하라.
- "높음"에 해당하면 18~25, "중간"에 해당하면 9~17, "낮음"에 해당하면 0~8 범위를 사용하라.
- **reason 필드는 출력하지 마라.** 내부용 줄은 metric_evaluations만으로 충분하다.
"""

def build_final_score_prompt(
    user_title,
    user_abstract,
    total_score,
    ma_result,
    tech_result,
    rights_result
):
    ma_score = ma_result.get("ma_attractiveness_score")
    tech_score = tech_result.get("technical_score")
    rights_score = rights_result.get("rights_score")
    return f"""
너는 특허 투자/사업화 점수화 전문가이다.
아래 입력을 종합해 **이미 확정된 종합 점수**를 바탕으로 설명만 작성하라.

[특허 정보]
- 제목: {user_title}
- 요약: {user_abstract}

[기존 평가 결과]
- M&A 시장 매력도 평가:
{json.dumps(ma_result, ensure_ascii=False, indent=2)}

- 기술성 평가:
{json.dumps(tech_result, ensure_ascii=False, indent=2)}

- 권리성 평가:
{json.dumps(rights_result, ensure_ascii=False, indent=2)}

[점수 산출 원칙 — 반드시 준수]
- **최종 점수는 이미 아래와 같이 확정되었다. 이 수치를 그대로 전제로 하라.**
  · 종합 점수: **{total_score}점** (시장 매력도 {ma_score} + 기술성 {tech_score} + 권리성 {rights_score})
- 점수를 다시 계산하거나 다른 총점을 제시하지 마라.
- 세부 항목 점수도 위 JSON에 있는 값만 근거로 삼아라.

[작성 지침 — 기술성/권리성 평가와 동일한 역할 분리]
- **evaluation** (사용자용)
  · 기술성 평가의 evaluation 필드와 **같은 문체**로 쓴다.
  · "~습니다", "~보입니다" 형태의 설명체.
  · 특허 내용 요약은 하지 마라.
  · 위 세 영역(시장·기술·권리)이 종합 점수({total_score}점)에 어떻게 어우러졌는지 균형 있게 설명하라.
  · **금지**: "평가를 받았습니다", "~로 평가되었습니다" 등 수동·보고서식 표현.
  · **권장**: 각 영역의 강점·약점을 직접 서술하는 형태.
  · 최대 5줄.
- **reason** (내부 확인용)
  · 시장·기술·권리 각 점수가 종합 해석에 어떻게 반영되었는지 정량·정성 근거를 솔직히 적는다.
  · 각 문장은 반드시 "~임.", "~함.", "~존재함." 등 **서술형 종결**로 끝낸다.

[출력 형식]
반드시 아래 JSON 형식으로만 출력하라.

{{
  "evaluation": "사용자용 종합 설명 (최대 5줄)",
  "reason": "내부용 근거 설명"
}}
"""

def call_llm(prompt, model="gpt-4.1", max_tokens=None):
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional patent analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)

    raw_text = response.choices[0].message.content.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]

    return json.loads(raw_text)


def _attach_metric_evaluation_lines(result: dict) -> None:
    """metric_evaluations → '지표명 : (데이터) -> (해석)' 형식의 표시용 줄."""
    rows = result.get("metric_evaluations")
    if not isinstance(rows, list):
        rows = []
    lines = []
    for m in rows:
        if not isinstance(m, dict):
            continue
        name = str(m.get("metric_name") or m.get("metric") or "지표").strip()
        data_basis = str(m.get("data_basis") or m.get("data") or "").strip()
        interp = str(m.get("interpretation") or "").strip()
        lines.append(f"{name} : ({data_basis}) -> ({interp})")
    result["metric_evaluation_lines"] = lines


def _join_analysis_reasoning_list(val):
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return str(val)


def _set_reason_from_metric_lines(result: dict, fallback: str = "") -> None:
    """최종 reason: '지표 : (데이터) -> (해석)' 줄들. 비어 있으면 fallback."""
    _attach_metric_evaluation_lines(result)
    lines = result.get("metric_evaluation_lines") or []
    result["reason"] = "\n".join(lines) if lines else (fallback or "")


def _normalize_component_score(value, max_score):
    score_map = {
        "높음": max_score,
        "중간": max_score // 2,
        "낮음": 0,
    }

    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value in score_map:
            return score_map[stripped_value]

    try:
        score = int(float(value))
    except (TypeError, ValueError):
        score = 0

    return max(0, min(max_score, score))

def evaluate_patent(
    payload,
    row,
    avg_claim_count,
    avg_ipc_count,
    avg_citation_count
):
    user_info = payload.get("user_info", {})
    user_title = user_info.get("title")
    user_abstract = user_info.get("abstract")

    ma_prompt = build_ma_prompt(payload)
    ma_result = call_llm(ma_prompt, max_tokens=4096)
    ma_result["ma_attractiveness_score"] = _normalize_component_score(
        ma_result.get("ma_attractiveness_score", ma_result.get("ma_attractiveness")),
        50
    )
    if not ma_result.get("evaluation_summary") and ma_result.get("user_report"):
        ma_result["evaluation_summary"] = ma_result["user_report"]
    _set_reason_from_metric_lines(
        ma_result,
        fallback=_join_analysis_reasoning_list(ma_result.get("analysis_reasoning")),
    )

    tech_prompt = build_tech_prompt(
        row=row,
        avg_ipc_count=avg_ipc_count,
        avg_citation_count=avg_citation_count
    )
    tech_result = call_llm(tech_prompt, max_tokens=3072)
    tech_result["technical_score"] = _normalize_component_score(
        tech_result.get("technical_score"),
        25
    )
    _set_reason_from_metric_lines(tech_result)

    rights_prompt = build_rights_prompt(
        row=row,
        avg_claim_count=avg_claim_count
    )
    rights_result = call_llm(rights_prompt, max_tokens=3072)
    rights_result["rights_score"] = _normalize_component_score(
        rights_result.get("rights_score"),
        25
    )
    _set_reason_from_metric_lines(rights_result)

    total_score = (
        ma_result["ma_attractiveness_score"] +
        tech_result["technical_score"] +
        rights_result["rights_score"]
    )

    final_score_prompt = build_final_score_prompt(
        user_title=user_title,
        user_abstract=user_abstract,
        total_score=total_score,
        ma_result=ma_result,
        tech_result=tech_result,
        rights_result=rights_result
    )
    final_score_result = call_llm(final_score_prompt)

    evaluation_text = (
        final_score_result.get("evaluation")
        or final_score_result.get("final_score_comment")
        or ""
    )
    reason_text = final_score_result.get("reason") or ""

    final_output = {
        "evaluation": {
            "ma_market_evaluation": ma_result,
            "tech_evaluation": tech_result,
            "rights_evaluation": rights_result
        },
        "final_result": {
            "final_score": total_score,
            "evaluation": evaluation_text,
            "reason": reason_text,
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

    return final_result, payload


