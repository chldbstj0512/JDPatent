import json
import re

# LLM/내부 파이프라인은 compu·bio·comm·elec·etc 를 쓰고, 최종 사용자 노출만 풀네임으로 통일합니다.
FIELD_CODE_TO_DISPLAY_NAME = {
    "compu": "computer",
    "bio": "biotechnology",
    "comm": "communications",
    "elec": "electronic",
    "etc": "other",
}


def field_code_to_display_name(field_code) -> str:
    if field_code is None:
        return None
    key = str(field_code).strip().lower()
    return FIELD_CODE_TO_DISPLAY_NAME.get(key, str(field_code).strip())


def strip_excel_xml_unicode_escapes(text):
    """
    Excel/Office sharedStrings 등에서 쓰이는 이스케이프 제거.
    예: _x000D_(CR), _x000A_(LF)가 초록/제목 뒤에 그대로 붙는 경우.
    """
    if text is None or not isinstance(text, str):
        return text
    s = re.sub(r"_x[0-9A-Fa-f]{4}_", "", text, flags=re.IGNORECASE)
    s = s.replace("\r", "")
    return s.rstrip()


def build_final_output(
    user_info_list,
    user_score,
    user_similar_company,
    user_pattern
):
    user_info = user_info_list[0]

    # -----------------------------
    # BASIC INFO
    # -----------------------------
    basic_info = {
        "pdf_name": user_info.get("pdf_name"),
        "country": user_info.get("country"),
        "field": field_code_to_display_name(user_info.get("field")),
        "patent": {
            "title": user_info.get("title"),
            "applicant": {
                "name": user_info.get("applicant_name"),
                "number": user_info.get("applicant_number"),
                "date": user_info.get("applicant_date"),
            },
            "grant": {
                "number": user_info.get("grant_number"),
                "date": user_info.get("grant_date"),
            },
            "ipc_code": [ipc["code"] for ipc in user_info.get("ipc_info", [])],
            "abstract": user_info.get("abstract"),
        },
    }

    # -----------------------------
    # NAICS
    # -----------------------------
    naics = {
        "primary": user_info.get("primary_naic_info"),
        "candidates": user_info.get("candidate_naic_info", []),
    }

    # -----------------------------
    # EVALUATION
    # -----------------------------
    evaluation = {
        "ma_market_evaluation": {
            "ma_market_score": user_score["evaluation"]["ma_market_evaluation"]["ma_attractiveness_score"],
            "ma_market_evaluation": (
                user_score["evaluation"]["ma_market_evaluation"].get("evaluation_summary")
                or user_score["evaluation"]["ma_market_evaluation"].get("user_report")
            ),
            "ma_market_reason": user_score["evaluation"]["ma_market_evaluation"].get("reason", ""),
        },
        "tech_evaluation": {
            "tech_score": user_score["evaluation"]["tech_evaluation"]["technical_score"],
            "tech_evaluation": user_score["evaluation"]["tech_evaluation"]["evaluation"],
            "tech_reason": user_score["evaluation"]["tech_evaluation"].get("reason", ""),
        },
        "rights_evaluation": {
            "rights_score": user_score["evaluation"]["rights_evaluation"]["rights_score"],
            "rights_evaluation": user_score["evaluation"]["rights_evaluation"]["evaluation"],
            "rights_reason": user_score["evaluation"]["rights_evaluation"].get("reason", ""),
        },
        "final_result": {
            "final_score": user_score.get("final_result", {}).get("final_score"),
            "evaluation": user_score.get("final_result", {}).get("evaluation"),
            "reason": user_score.get("final_result", {}).get("reason"),
        },
    }

    # -----------------------------
    # IPC DESCRIPTIONS
    # -----------------------------
    ipc_descriptions = [
        {
            "ipc_code": ipc["code"],
            "short_description": ipc["short_description"],
            "long_description": ipc["long_description"],
        }
        for ipc in user_info.get("ipc_info", [])
    ]

    # -----------------------------
    # RECOMMEND COMPANIES (기업 단위)
    # -----------------------------
    def _transform_patent(meta, is_target):
        ipc_raw = meta.get("ipc_code", "") or ""
        ipc_list = [
            strip_excel_xml_unicode_escapes(part.strip())
            for part in ipc_raw.split("|")
            if part.strip()
        ]
        return {
            "title": strip_excel_xml_unicode_escapes(meta.get("invention_name")),
            "ipc": ipc_list,
            "abstract": strip_excel_xml_unicode_escapes(meta.get("abstract")),
            "application_number": strip_excel_xml_unicode_escapes(meta.get("application_number")),
            "application_date": strip_excel_xml_unicode_escapes(meta.get("application_date")),
        }

    def transform_company_list(company_list, is_target=True):
        output = []
        for idx, company in enumerate(company_list, start=1):
            output.append({
                "rank": idx,
                "company_name": strip_excel_xml_unicode_escapes(company.get("company_name")),
                "company_id": company.get("company_id"),
                "avg_similarity": round(company.get("avg_score", 0), 4),
                "matched_patent_count": company.get("matched_patent_count", 0),
                "patents": [
                    {
                        "similarity": round(p.get("score", 0), 4),
                        **_transform_patent(p.get("metadata", {}), is_target),
                    }
                    for p in company.get("patents", [])
                ],
            })
        return output

    recommend_companies = {
        "target_similar_companies": transform_company_list(
            user_similar_company.get("target_similar_companies", []),
            is_target=True,
        ),
        "acquiror_similar_companies": transform_company_list(
            user_similar_company.get("acquiror_similar_companies", []),
            is_target=False,
        ),
    }

    # -----------------------------
    # MA PATTERNS
    # target_* 는 패턴 행마다 중복되므로 상위에 한 번만 둔다.
    # ratio_percent: 해당 피인수 NAIC 거래 전체 중 이 인수자 NAIC 비중(%), 소수 첫째 자리
    # -----------------------------
    pattern_rows = []
    for item in user_pattern.get("relation_analysis", []):
        ev = item.get("evidence") or {}
        ratio_raw = ev.get("acquirer_ratio_percent")
        if ratio_raw is not None:
            try:
                ratio_percent = round(float(ratio_raw), 1)
            except (TypeError, ValueError):
                ratio_percent = None
        else:
            ratio_percent = None

        pattern_rows.append(
            {
                "acquirer_naic": ev.get("acquirer_naic") or item.get("acquirer_naic"),
                "acquirer_naic_title": ev.get("acquirer_naic_title"),
                "ratio_percent": ratio_percent,
                "relation_label": item.get("relation_label"),
                "one_line_description": item.get("one_line_description"),
                "reason": item.get("reason"),
            }
        )

    ma_patterns = {
        "target_naic": user_pattern.get("user_naic"),
        "target_naic_title": user_pattern.get("user_naic_title"),
        "patterns": pattern_rows,
    }

    # -----------------------------
    # FINAL STRUCTURE
    # -----------------------------
    final_output = {
        "basic_info": basic_info,
        "naics": naics,
        "evaluation": evaluation,
        "ipc_descriptions": ipc_descriptions,
        "recommend_companies": recommend_companies,
        "ma_patterns": ma_patterns,
    }

    return final_output
