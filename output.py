import json


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
        "field": user_info.get("field"),
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
            "ma_market_score": user_score["evaluation"]["ma_market_evaluation"]["ma_attractiveness"],
            "ma_market_evaluation": user_score["evaluation"]["ma_market_evaluation"]["user_report"],
            "ma_market_reason": " ".join(
                user_score["evaluation"]["ma_market_evaluation"]["analysis_reasoning"]
            ),
        },
        "tech_evaluation": {
            "tech_score": user_score["evaluation"]["tech_evaluation"]["technical_score"],
            "tech_evaluation": user_score["evaluation"]["tech_evaluation"]["evaluation"],
            "tech_reason": user_score["evaluation"]["tech_evaluation"]["reason"],
        },
        "rights_evaluation": {
            "rights_score": user_score["evaluation"]["rights_evaluation"]["rights_score"],
            "rights_evaluation": user_score["evaluation"]["rights_evaluation"]["evaluation"],
            "rights_reason": user_score["evaluation"]["rights_evaluation"]["reason"],
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
    # RECOMMEND COMPANIES
    # -----------------------------
    def transform_similar_list(similar_list, is_target=True):
        output = []
        for idx, item in enumerate(similar_list, start=1):
            meta = item["metadata"]
            ipc_list = meta.get("ipc_code", "").split("|")

            output.append({
                "rank": idx,
                "similarity": round(item["score"], 4),
                "assignee": {
                    "name": meta.get("target_short_name") if is_target else meta.get("acquiror_short_name"),
                    "id": meta.get("target_id") if is_target else meta.get("acquiror_id"),
                },
                "title": meta.get("invention_name"),
                "ipc": ipc_list,
                "abstract": meta.get("abstract"),
                "application_number": meta.get("application_number"),
                "application_date": meta.get("application_date"),
            })
        return output

    recommend_companies = {
        "target_similar_patents": transform_similar_list(
            user_similar_company.get("target_similar_patents", []),
            is_target=True
        ),
        "acquiror_similar_patents": transform_similar_list(
            user_similar_company.get("acquiror_similar_patents", []),
            is_target=False
        ),
    }

    # -----------------------------
    # MA PATTERNS
    # -----------------------------
    ma_patterns = [
        {
            "acquirer_naics": item["acquirer_naic"],
            "one_line_description": item["one_line_description"],
            "relation_label": item["relation_label"],
        }
        for item in user_pattern.get("relation_analysis", [])
    ]

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
