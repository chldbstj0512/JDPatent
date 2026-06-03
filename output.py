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


def _normalize_applicant_name_for_output(name):
    """
    legacy output.json 형식: 출원인 법인명만 노출.
    US 서지 'Corp, City, ST (US)' 등 주소·국가 접미는 제거한다.
    """
    if not name or not isinstance(name, str):
        return name
    s = name.strip()
    if "," not in s:
        return s
    head, tail = s.split(",", 1)
    tail = tail.strip()
    if re.search(r"\([A-Z]{2}\)\s*$", tail, re.IGNORECASE):
        return head.strip()
    if re.search(r",\s*[A-Z]{2}\s*\(", tail):
        return head.strip()
    return s


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


def _normalize_country_for_output(value):
    if value is None:
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    alias = {
        "KOREA": "KR",
        "SOUTH KOREA": "KR",
        "REPUBLIC OF KOREA": "KR",
        "대한민국": "KR",
        "한국": "KR",
        "USA": "US",
        "U.S.": "US",
        "U.S.A.": "US",
        "UNITED STATES": "US",
    }
    if s in alias:
        return alias[s]
    if re.fullmatch(r"[A-Z]{2}", s):
        return s
    return None


def _extract_patent_country(meta: dict) -> str | None:
    if not isinstance(meta, dict):
        return None
    for key in ("country", "country_code", "patent_country", "publication_country", "nation", "origin_country"):
        v = _normalize_country_for_output(meta.get(key))
        if v:
            return v

    for key in ("publication_number", "patent_number", "application_number", "grant_number", "doc_number"):
        raw = meta.get(key)
        if not raw:
            continue
        raw_s = str(raw).strip()
        m = re.search(r"\b([A-Z]{2})\s*[\d/\-]", raw_s.upper())
        if m:
            v = _normalize_country_for_output(m.group(1))
            if v:
                return v
        compact = re.sub(r"\s+", "", raw_s)
        if key == "application_number":
            if compact.startswith(("10", "20")) and re.fullmatch(r"\d{10,14}", compact):
                return "KR"
            if re.fullmatch(r"\d{2}/\d{3},\d{3}", compact) or re.fullmatch(r"\d{2}/\d{6}", compact):
                return "US"
            # US application/publication number가 숫자만 있는 형태인 경우(예: 15343948, 17184433)
            if re.fullmatch(r"\d{7,9}", compact) and not compact.startswith(("10", "20")):
                return "US"
        if key in ("patent_number", "grant_number", "doc_number"):
            # US grant number가 숫자만으로 저장되는 경우(예: 12115176)
            if re.fullmatch(r"\d{7,9}", compact) and not compact.startswith(("10", "20")):
                return "US"
    return None


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
                "name": _normalize_applicant_name_for_output(user_info.get("applicant_name")),
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
            "ma_anchor_score": user_score["evaluation"]["ma_market_evaluation"].get("ma_anchor_score"),
            "ma_llm_four_perspectives_sum": user_score["evaluation"]["ma_market_evaluation"].get(
                "ma_attractiveness_score_llm_segments_sum"
            ),
            "ma_score_blending": user_score["evaluation"]["ma_market_evaluation"].get("ma_score_blending"),
            "ma_market_perspectives": user_score["evaluation"]["ma_market_evaluation"].get(
                "ma_market_perspectives", {}
            ),
            "ma_market_anchor": user_score["evaluation"]["ma_market_evaluation"].get("ma_market_anchor", {}),
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
            "patent_country": _extract_patent_country(meta),
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
    parse_audit = user_info.get("parse_audit")
    if parse_audit:
        final_output["parse_audit"] = parse_audit

    return final_output