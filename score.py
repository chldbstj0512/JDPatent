import pandas as pd
import numpy as np
import ast
import json
import re

from dotenv import load_dotenv
from openai import OpenAI 
import os
from openai_logging import openai_chat_options

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_field(field_scores, user_field):
    field_score = field_scores.get(user_field)

    if field_score is None:
        raise ValueError(f"Unknown field: {user_field}")

    metrics_flat = [
        {
            "metric_key": k,
            "value": field_score[k],
            "definition": {
                "recent_growth": "최근 5개년 기준 산출된 분야 M&A 성장률(입력값)",
                "recovery_ratio": "과거 최저점 대비 회복 수준(입력값)",
                "recent_trend_slope": "최근 추세의 연간 기울기(입력값)",
                "final_score": "성장성과 회복성을 가중 결합한 분야 종합 지표(입력값)",
            }.get(k, ""),
        }
        for k in ("recent_growth", "recovery_ratio", "recent_trend_slope", "final_score")
        if k in field_score
    ]

    llm_payload = {
        "field_code": user_field,
        "trend_metrics": field_score,
        "metrics_flat": metrics_flat,
        "field_metric_definition": {
            "recent_growth": "최근 5개년간 M&A 성장률(분야 집계 입력값)",
            "recovery_ratio": "과거 최저점 대비 회복 수준",
            "recent_trend_slope": "최근 추세의 연간 기울기",
            "final_score": "성장성과 회복성을 가중 결합한 종합 점수",
        },
    }

    return llm_payload

def _acquiror_naics_concentration(df_slice: pd.DataFrame, acq_col: str = "Acquiror Primary NAIC Code 2022") -> dict:
    """피인수 동일 NAICS 거래 집합에서 인수자 NAICS 분포의 산업집중도(HHI 등)."""
    if df_slice is None or len(df_slice) == 0:
        return {
            "hhi_acquiror_naics": None,
            "top3_acquiror_share": None,
            "n_distinct_acquiror_naics": 0,
        }
    s = (
        df_slice[acq_col]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    s = s[(s != "") & (s != "-")]
    if len(s) == 0:
        return {
            "hhi_acquiror_naics": None,
            "top3_acquiror_share": None,
            "n_distinct_acquiror_naics": 0,
        }
    counts = s.value_counts()
    sh = counts / counts.sum()
    hhi = float((sh ** 2).sum())
    top3 = float(sh.head(3).sum()) if len(sh) >= 1 else 0.0
    return {
        "hhi_acquiror_naics": round(hhi, 5),
        "top3_acquiror_share": round(top3, 4),
        "n_distinct_acquiror_naics": int(len(counts)),
    }


def get_naics_trend_payload(
    acquisitions_df,
    user_code,
    n_years: int = 5,
    end_year: int | None = None,
    year_col="Year",
    tgt_col="Target Primary NAIC Code 2022",
    acq_col="Acquiror Primary NAIC Code 2022",
):
    """
    수정요구: 3개년 → 5개년 동향. end_year 미지정 시 데이터 최대 연도 사용.
    요약: 전체 건수, 연도별 최저·최고 A, 2년 전 A 등.
    """
    user_code = str(user_code)
    user_prefix = user_code[:4]

    if end_year is None:
        y = pd.to_numeric(acquisitions_df[year_col], errors="coerce").dropna()
        end_year = int(y.max()) if len(y) else 2025
    start_year = int(end_year) - int(n_years) + 1
    years = list(range(start_year, int(end_year) + 1))

    df_tgt = acquisitions_df[
        acquisitions_df[tgt_col].astype(str).eq(user_code)
        & acquisitions_df[year_col].isin(years)
    ].copy()

    df_tgt["is_internal"] = (
        df_tgt[acq_col]
        .astype(str)
        .str[:4]
        .eq(user_prefix)
    )

    result = (
        df_tgt.groupby(year_col)
        .agg(
            A_target_cnt=(tgt_col, "count"),
            B_internal_cnt=("is_internal", "sum"),
        )
        .reset_index()
    )

    result["B_ratio"] = (
        result["B_internal_cnt"] / result["A_target_cnt"].replace(0, np.nan)
    ).fillna(0).round(3)

    result = (
        pd.DataFrame({year_col: years})
        .merge(result, on=year_col, how="left")
        .fillna(0)
    )

    a_series = result["A_target_cnt"].astype(float)
    total_a = int(a_series.sum())
    min_idx = int(a_series.idxmin()) if len(a_series) else None
    max_idx = int(a_series.idxmax()) if len(a_series) else None
    min_year = int(result.loc[min_idx, year_col]) if min_idx is not None and min_idx >= 0 else None
    max_year = int(result.loc[max_idx, year_col]) if max_idx is not None and max_idx >= 0 else None
    two_years_ago = int(end_year) - 2
    row_2ya = result[result[year_col] == two_years_ago]
    a_two_years_ago = int(row_2ya["A_target_cnt"].iloc[0]) if len(row_2ya) else 0

    conc = _acquiror_naics_concentration(df_tgt.drop(columns=["is_internal"], errors="ignore"))

    payload = {
        "naics_code": user_code,
        "window_years": n_years,
        "year_range": {"start": start_year, "end": int(end_year)},
        "naics_trend_metric_definition": {
            "A_target_cnt": "연도별 Target NAICS 기업 M&A 건수",
            "B_internal_cnt": "동종 산업군 인수자의 인수 건수",
            "B_ratio": "동종 산업 내부 인수 비중 (B/A)",
        },
        "time_series": [
            {
                "year": int(row[year_col]),
                "A": int(row["A_target_cnt"]),
                "B": int(row["B_internal_cnt"]),
                "B_ratio": round(float(row["B_ratio"]), 3),
            }
            for _, row in result.iterrows()
        ],
        "trend_summary": {
            "total_A_target_cnt_window": total_a,
            "min_A": int(a_series.min()) if len(a_series) else 0,
            "min_A_year": min_year,
            "max_A": int(a_series.max()) if len(a_series) else 0,
            "max_A_year": max_year,
            "A_target_cnt_two_years_ago": a_two_years_ago,
            "two_years_ago_label_year": two_years_ago,
        },
        "industry_concentration_acquiror_naics": conc,
    }

    return payload

def _crossborder_counts_by_year(df: pd.DataFrame, user_prefer: str, year_col: str = "Year") -> dict:
    """수정요구: M&A cross border 년 단위 건수 (국경/지역 플래그 기준)."""
    if df is None or len(df) == 0:
        return {}
    flag = "Cross_Border_Deal_Flag_Nation" if user_prefer == "nation" else "Cross_Border_Deal_Flag_Area"
    if flag not in df.columns:
        return {}
    m = df[df[flag].fillna(0).astype(int) == 1]
    if len(m) == 0:
        return {}
    g = m.groupby(year_col).size()
    return {str(int(k)): int(v) for k, v in g.items()}


def _filter_hightech_categories_remove_other(categories) -> list:
    """수정요구: High-Tech 'Other(s)' 계열 제거 — 모델이 잘못 매칭하는 것을 줄임."""
    if not isinstance(categories, list):
        return []
    out = []
    for c in categories:
        s = str(c).strip()
        if not s:
            continue
        low = s.lower()
        if low == "others" or low == "other":
            continue
        if re.match(r"^other\b", low):
            continue
        out.append(c)
    return out


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

        "hightech_categories": _filter_hightech_categories_remove_other(hightech_list),

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
        flag_series = df["Cross_Border_Deal_Flag_Nation"].fillna(0).astype(int) == 1
        cross_border_cnt = int(flag_series.sum())
    elif user_prefer == "area":
        flag_series = df["Cross_Border_Deal_Flag_Area"].fillna(0).astype(int) == 1
        cross_border_cnt = int(flag_series.sum())
    else:
        raise ValueError("user_prefer는 'nation' 또는 'area'여야 합니다.")

    cross_border_majority = cross_border_cnt / total_cnt > 0.5

    # 수정요구: inbound/outbound는 국경·지역 간 거래(플래그=1) 부분집합에서만 집계
    if user_prefer == "nation":
        if user_prefer_nation is None:
            raise ValueError("user_prefer_nation이 필요합니다.")

        outbound_cnt = (
            flag_series
            & (df["Acquiror Nation"] == user_prefer_nation)
            & (df["Target Nation"] != user_prefer_nation)
        ).sum()

        inbound_cnt = (
            flag_series
            & (df["Acquiror Nation"] != user_prefer_nation)
            & (df["Target Nation"] == user_prefer_nation)
        ).sum()

    else:  # area
        if user_prefer_area is None:
            raise ValueError("user_prefer_area가 필요합니다.")

        outbound_cnt = (
            flag_series
            & (df["Acquiror Primary Nation Region"] == user_prefer_area)
            & (df["Target Primary Nation Region"] != user_prefer_area)
        ).sum()

        inbound_cnt = (
            flag_series
            & (df["Acquiror Primary Nation Region"] != user_prefer_area)
            & (df["Target Primary Nation Region"] == user_prefer_area)
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
                "Cross_Border_Deal_Flag가 1인 거래만 대상으로, "
                "사용자 선호 국가/지역이 인수자(Acquiror)인 cross-border 거래 건수 "
                "(선호지가 비선호지 기업을 인수 → 유출)"
            ),

            "inbound_cnt": (
                "Cross_Border_Deal_Flag가 1인 거래만 대상으로, "
                "사용자 선호 국가/지역이 피인수자(Target)인 cross-border 거래 건수 "
                "(비선호지가 선호지 기업을 인수 → 유입)"
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
            "prefer_acquiror_ratio": float(crossborder_result["prefer_acquiror_ratio"]),
            "crossborder_deal_count_by_year": crossborder_result.get("crossborder_deal_count_by_year") or {},
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
    stake_total = int(len(stake_df))

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
            "total_deals": "해당 NAICS 코드의 전체 M&A 건수",
            "stake_multi_over_stake_ratio": (
                "Stake 거래 중 동일 타깃 다회(멀티) 비중 = stake_multi_target_cnt / max(1, stake 총건)"
            ),
            "full_over_all_ratio": (
                "전체 거래 대비 Full 인수 비중 = full_acquisition_cnt / total_deals"
            ),
        },

        "ma_type_metrics": {
            "total_deals": total_deals,
            "stake_single_target_cnt": stake_single_target_cnt,
            "stake_multi_target_cnt": stake_multi_target_cnt,
            "full_acquisition_cnt": full_acquisition_cnt,
            "stake_deals_overall": stake_total,
            "stake_multi_over_stake_ratio": round(
                stake_multi_target_cnt / max(1, stake_total), 4
            ),
            "full_over_all_ratio": round(full_acquisition_cnt / max(1, total_deals), 4),
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

    if user_prefer == "nation" and user_prefer_nation is None:
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
    crossborder_result["crossborder_deal_count_by_year"] = _crossborder_counts_by_year(
        df_tgt, user_prefer
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

def _compute_ma_attractiveness_anchor_detail(payload: dict) -> dict:
    """
    M&A 매력도 앵커(정량 근거 0~50): 사용자용 한 줄 + 내부용 근거 줄.
    LLM 4관점 합산과 별도 파이프라인으로 유지한다.
    """
    try:
        fa = payload.get("field_analysis") or {}
        tm = fa.get("trend_metrics") or {}
        fs = float(tm.get("final_score", 0.0))
        part_field = min(22.0, max(0.0, fs * 22.0))

        tr = payload.get("naics_trend") or {}
        ts = tr.get("time_series") or []
        last_br = [float(r.get("B_ratio", 0.0)) for r in ts[-3:] if isinstance(r, dict)]
        mean_br = sum(last_br) / max(1, len(last_br))
        part_naics = min(16.0, max(0.0, mean_br * 16.0))

        cb = (payload.get("crossborder_analysis") or {}).get("crossborder_metrics") or {}
        cbr = float(cb.get("cross_border_ratio", 0.0))
        part_cb = min(12.0, max(0.0, cbr * 12.0))

        mt = (payload.get("ma_type_analysis") or {}).get("ma_type_metrics") or {}
        tot = int(mt.get("total_deals") or 0)
        full = int(mt.get("full_acquisition_cnt") or 0)
        full_ratio = full / max(1, tot)
        part_ma = min(10.0, max(0.0, full_ratio * 10.0))

        score = int(round(part_field + part_naics + part_cb + part_ma))
        score = max(0, min(50, score))

        lines = [
            f"분야 M&A 종합(final_score×22, 캡22) : ({fs:.4f}) -> (기여 {part_field:.2f}점)",
            f"최근 3년 평균 동종 인수비중 B_ratio×16, 캡16 : ({mean_br:.4f}) -> (기여 {part_naics:.2f}점)",
            f"국경간 거래비중×12, 캡12 : ({cbr:.4f}) -> (기여 {part_cb:.2f}점)",
            f"완전인수 비율 full/total×10, 캡10 : (full={full}, total={tot}, 비율={full_ratio:.4f}) -> (기여 {part_ma:.2f}점)",
        ]
        anchor_reason = "\n".join(lines)
        ma_market_anchor_evaluation = (
            f"정량 앵커는 분야 동향·NAICS 동종 비중·국경간 개방성·완전 인수 비율을 반영해 {score}점"
            f"(50점 만점)으로 산출되었습니다. 이 값은 LLM 네 관점 합산 점수와 가중 블렌딩되어 최종 시장 매력도에 반영됩니다."
        )
        return {
            "ma_anchor_score": score,
            "anchor_reason": anchor_reason,
            "ma_market_anchor_evaluation": ma_market_anchor_evaluation,
        }
    except Exception:
        return {
            "ma_anchor_score": 25,
            "anchor_reason": "앵커 산출 중 예외로 기본값 25점 적용됨.",
            "ma_market_anchor_evaluation": "정량 앵커는 기본값으로 처리되었습니다.",
        }


def _compute_ma_attractiveness_anchor(payload: dict) -> int:
    """앵커 점수만 필요할 때."""
    return int(_compute_ma_attractiveness_anchor_detail(payload)["ma_anchor_score"])


def _ma_segment_json_schema(max_segment_score: int) -> str:
    return f"""
반드시 아래 JSON만 출력하십시오.

{{
  "segment_score": 0,
  "metric_evaluations": [
    {{"metric_name": "...", "data_basis": "입력 JSON의 원시 값", "interpretation": "..."}}
  ],
  "evaluation_paragraphs": [
    "사용자용 첫 번째 단락 (2~3문장, ~습니다체, 비전문가용, 이번 관점만)",
    "사용자용 두 번째 단락 (2~3문장)",
    "사용자용 세 번째 단락 (2~3문장)"
  ],
  "analysis_reasoning": ["내부용 ~임./~함. 종결 문장 1~2개"]
}}

- segment_score: 0~{max_segment_score} 정수. **이번 관점만** 근거로 부여한다.
- evaluation_paragraphs: **정확히 3개** 문자열. 각 단락은 서로 다른 측면을 다룬다.
- reason 필드는 출력하지 마십시오.
"""


def _normalize_ma_segment_evaluation_paragraphs(part: dict) -> list[str]:
    """LLM 출력을 길이 3의 단락 리스트로 맞춘다. 구형 evaluation_paragraph도 허용."""
    raw = part.get("evaluation_paragraphs")
    out: list[str] = []
    if isinstance(raw, list):
        for x in raw[:3]:
            s = str(x).strip() if x is not None else ""
            out.append(s)
    while len(out) < 3:
        out.append("")
    legacy = part.get("evaluation_paragraph")
    if isinstance(legacy, str) and legacy.strip():
        if not any(out):
            chunks = [c.strip() for c in legacy.split("\n\n") if c.strip()]
            if len(chunks) >= 3:
                out = chunks[:3]
            elif len(chunks) == 2:
                out = [chunks[0], chunks[1], ""]
            elif len(chunks) == 1:
                out = [chunks[0], "", ""]
            else:
                out = [legacy.strip(), "", ""]
    return out[:3]


def _segment_internal_reason(part: dict) -> str:
    sub = {
        "metric_evaluations": part.get("metric_evaluations")
        if isinstance(part.get("metric_evaluations"), list)
        else []
    }
    _set_reason_from_metric_lines(
        sub,
        fallback=_join_analysis_reasoning_list(part.get("analysis_reasoning")),
    )
    return str(sub.get("reason") or "")


# (segment_key, eval_field_name, reason_field_name, score_field_name)
MA_MARKET_SEGMENT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("field_trend", "ma_market_field_trend_evaluation", "field_trend_reason", "field_trend_score"),
    ("crossborder", "ma_market_crossborder_evaluation", "crossborder_reason", "crossborder_score"),
    ("ma_type", "ma_market_ma_type_evaluation", "ma_type_reason", "ma_type_score"),
    ("hightech", "ma_market_hightech_evaluation", "hightech_reason", "hightech_score"),
)


def build_ma_segment_prompt_field_trend(payload: dict) -> str:
    sub = {
        "user_info": {
            "title": (payload.get("user_info") or {}).get("title"),
            "abstract": (payload.get("user_info") or {}).get("abstract"),
            "field": (payload.get("user_info") or {}).get("field"),
        },
        "field_analysis": payload.get("field_analysis"),
        "naics_trend": payload.get("naics_trend"),
    }
    return f"""
당신은 M&A 시장 분석가이다. 이번 호출은 **(1) 분야 지표 + (2) NAICS 5개년 동향·집중도**만 다룬다.

[입력 JSON — 이 블록만 사용]
{json.dumps(sub, ensure_ascii=False, indent=2)}

[역할]
- field_analysis.trend_metrics와 naics_trend(time_series, trend_summary, industry_concentration_acquiror_naics)를 근거로 성장·회복·동종 인수 비중·집중도를 해석한다.
- 다른 블록(crossborder, ma_type, hightech)은 언급하지 말 것.

{_ma_segment_json_schema(14)}
"""


def build_ma_segment_prompt_crossborder(payload: dict) -> str:
    sub = {
        "user_info": {
            "title": (payload.get("user_info") or {}).get("title"),
            "country": (payload.get("user_info") or {}).get("country"),
        },
        "crossborder_analysis": payload.get("crossborder_analysis"),
    }
    return f"""
당신은 M&A 시장 분석가이다. 이번 호출은 **국경/지역 간 거래 구조(crossborder)**만 다룬다.

[입력 JSON — 이 블록만 사용]
{json.dumps(sub, ensure_ascii=False, indent=2)}

[역할]
- crossborder_metrics 및 crossborder_deal_count_by_year를 근거로 개방성·방향성(in/out)·연도별 패턴을 해석한다.
- 분야·NAICS 시계열·인수유형·하이테크는 언급하지 말 것.

{_ma_segment_json_schema(12)}
"""


def build_ma_segment_prompt_ma_type(payload: dict) -> str:
    sub = {
        "user_info": {
            "title": (payload.get("user_info") or {}).get("title"),
        },
        "ma_type_analysis": payload.get("ma_type_analysis"),
    }
    return f"""
당신은 M&A 시장 분석가이다. 이번 호출은 **인수 유형(지분 vs 전체, multi 지표)**만 다룬다.

[입력 JSON — 이 블록만 사용]
{json.dumps(sub, ensure_ascii=False, indent=2)}

[역할]
- ma_type_metrics(stake_single, stake_multi, full, 비율 지표)를 근거로 전략적 의미를 해석한다.
- 국경간·분야·하이테크·NAICS 시계열은 언급하지 말 것.

{_ma_segment_json_schema(12)}
"""


def build_ma_segment_prompt_hightech(payload: dict) -> str:
    sub = {
        "user_info": {
            "title": (payload.get("user_info") or {}).get("title"),
            "abstract": (payload.get("user_info") or {}).get("abstract"),
        },
        "hightech_prompt": payload.get("hightech_prompt"),
    }
    return f"""
당신은 M&A 시장 분석가이다. 이번 호출은 **High-Tech 적합성**만 다룬다.

[입력 JSON — 이 블록만 사용]
{json.dumps(sub, ensure_ascii=False, indent=2)}

[역할]
- hightech_prompt와 특허 제목·요약을 근거로 하이테크 해당 가능성·산업 파급을 해석한다.
- 수치형 M&A 테이블은 언급하지 말 것.

{_ma_segment_json_schema(12)}
"""


def run_ma_market_evaluation_four_calls(payload: dict) -> dict:
    """
    M&A 시장 매력도: **4관점별 별도 LLM 호출**(관점별 점수·3단락 평가·내부 reason) +
    **1 정량 앵커 파이프라인**을 합산·블렌딩한다.
    """
    segment_builders = [
        ("field_trend", build_ma_segment_prompt_field_trend, 14),
        ("crossborder", build_ma_segment_prompt_crossborder, 12),
        ("ma_type", build_ma_segment_prompt_ma_type, 12),
        ("hightech", build_ma_segment_prompt_hightech, 12),
    ]
    merged: dict = {
        "metric_evaluations": [],
        "analysis_reasoning": [],
        "ma_segment_results": {},
        "ma_market_perspectives": {},
    }
    perspective_blocks: list[str] = []
    llm_sum = 0
    spec_by_key = {s[0]: s for s in MA_MARKET_SEGMENT_SPECS}

    for key, builder, mx in segment_builders:
        prompt = builder(payload)
        part = call_llm(prompt, max_tokens=2800)
        seg = _normalize_component_score(part.get("segment_score"), mx)
        llm_sum += seg
        paras = _normalize_ma_segment_evaluation_paragraphs(part)
        eval_joined = "\n\n".join(p for p in paras if p)
        seg_reason = _segment_internal_reason(part)
        _, eval_field, reason_field, score_field = spec_by_key[key]
        enriched = {
            **part,
            "segment_score_capped": seg,
            "evaluation_paragraphs_normalized": paras,
            reason_field: seg_reason,
            eval_field: eval_joined,
            score_field: seg,
        }
        merged["ma_segment_results"][key] = enriched
        merged["ma_market_perspectives"][key] = {
            score_field: seg,
            eval_field: eval_joined,
            f"{eval_field}_paragraphs": paras,
            reason_field: seg_reason,
        }
        rows = part.get("metric_evaluations")
        if isinstance(rows, list):
            merged["metric_evaluations"].extend(rows)
        ar = part.get("analysis_reasoning")
        if isinstance(ar, list):
            merged["analysis_reasoning"].extend(ar)
        if eval_joined:
            perspective_blocks.append(eval_joined)

    merged["evaluation_summary"] = "\n\n".join(perspective_blocks)
    merged["ma_attractiveness_score_llm_segments_sum"] = min(50, llm_sum)

    anchor_detail = _compute_ma_attractiveness_anchor_detail(payload)
    anchor = int(anchor_detail["ma_anchor_score"])
    merged["ma_market_anchor"] = {
        "ma_anchor_score": anchor,
        "ma_market_anchor_evaluation": anchor_detail.get("ma_market_anchor_evaluation", ""),
        "anchor_reason": anchor_detail.get("anchor_reason", ""),
    }
    merged["ma_anchor_score"] = anchor
    blended = int(round(0.42 * anchor + 0.58 * merged["ma_attractiveness_score_llm_segments_sum"]))
    merged["ma_attractiveness_score"] = max(0, min(50, blended))
    merged["ma_llm_score_component"] = merged["ma_attractiveness_score_llm_segments_sum"]
    merged["ma_score_blending"] = {
        "anchor_weight": 0.42,
        "llm_four_perspectives_sum_weight": 0.58,
        "llm_four_perspectives_sum_cap": 50,
    }

    reason_sections = []
    titles = {
        "field_trend": "【분야·NAICS·동향】",
        "crossborder": "【국경간 거래】",
        "ma_type": "【인수 유형】",
        "hightech": "【하이테크 적합성】",
    }
    for key, _, _, _ in MA_MARKET_SEGMENT_SPECS:
        p = merged["ma_market_perspectives"].get(key) or {}
        r = ""
        for _k, _ef, rf, _sf in MA_MARKET_SEGMENT_SPECS:
            if _k == key:
                r = str(p.get(rf) or "")
                break
        if r.strip():
            reason_sections.append(f"{titles.get(key, key)}\n{r.strip()}")
    arn = str(anchor_detail.get("anchor_reason") or "").strip()
    if arn:
        reason_sections.append(f"【정량 앵커】\n{arn}")
    merged["reason"] = "\n\n".join(reason_sections)
    return merged


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
    kwargs.update(openai_chat_options("score.call_llm", model=model))
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

    ma_result = run_ma_market_evaluation_four_calls(payload)

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

