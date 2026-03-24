import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
import os

load_dotenv() 

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI()

def get_acquiror_payload(
    user_naic,
    acquisitions_df,
    naic_df,
    top_k: int = 5
):

    sub_df = acquisitions_df[
        acquisitions_df['Target Primary NAIC Code 2022'].astype(str) == str(user_naic)
    ].copy()

    if sub_df.empty:
        return {
            "user_naic": user_naic,
            "user_naic_title": None,
            "naic_ratio": {},
            "top_naic_codes": [],
            "top_naic_titles": [],
            "acquirer_list": ""
        }

    naic_ratio = (
        sub_df['Acquiror Primary NAIC Code 2022']
            .dropna()
            .astype(str)
            .str.split(',')
            .explode()
            .str.strip()
            .loc[lambda x: (x != '-') & (x != '')]
            .value_counts(normalize=True)
            * 100
    )

    # top_k 기준으로만 비율 근거를 사용/반환한다.
    naic_ratio_topk = naic_ratio.head(top_k)

    user_naic_title_series = naic_df.loc[
        naic_df['naics_code'].astype(str) == str(user_naic),
        'naics_title'
    ]

    user_naic_title = (
        user_naic_title_series.iloc[0]
        if not user_naic_title_series.empty
        else None
    )

    top_naic_codes = naic_ratio_topk.index.astype(str).tolist()

    naic_title_dict = dict(
        zip(
            naic_df['naics_code'].astype(str),
            naic_df['naics_title']
        )
    )

    top_naic_titles = [
        naic_title_dict.get(code)
        for code in top_naic_codes
    ]

    acquirer_list = "\n".join([
        f"- {code}: {title}"
        for code, title in zip(top_naic_codes, top_naic_titles)
    ])

    return {
        "user_naic": str(user_naic),
        "user_naic_title": user_naic_title,
        "naic_ratio": naic_ratio_topk.to_dict(),
        "top_naic_codes": top_naic_codes,
        "top_naic_titles": top_naic_titles,
        "acquirer_list": acquirer_list
    }

def extract_naic_relation(
    user_naic: str,
    user_naic_title: str,
    acquirer_list: str
):
    prompt = f"""
다음은 기업 인수·피인수 관계 데이터이다.

- 피인수자 산업 코드: {user_naic}
- 피인수자 산업 설명: {user_naic_title}

- 인수자 산업 코드 목록:
{acquirer_list}

각 인수자 산업 코드에 대해
피인수 산업({user_naic})과의 관계를 분석하여
아래 JSON 형식으로 반환하라.

출력은 반드시 **JSON 배열**이어야 하며,
각 원소는 하나의 인수자 산업 코드에 대응한다.

JSON 형식:
[
  {{
    "acquirer_naic": "인수자 산업 코드",
    "relation_label": "관계라벨",
    "one_line_description": "한 줄 설명",
    "reason": "이 관계가 도출된 배경 및 이유 설명"
  }}
]

관계라벨 조건:
- 한국어
- 7글자 미만
- 명사 또는 명사구
- 출력은 한 단어 또는 두 단어 이내로 제한.
- 예시 형식: 수직통합, 수평확장, 기술흡수, 금융지원, 다각화

one_line_description 조건:
- 1문장
- 30자 이내

reason 조건:
- 산업 간 가치사슬, 기술, 시장, 고객, 기능 관점 중 하나 이상 활용

주의:
- JSON 이외의 텍스트 출력 금지
- 키 이름 변경 금지
- 형식을 반드시 정확히 지킬 것
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You analyze M&A industry relationships and return strictly valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=2000,
    )

    content = response.choices[0].message.content.strip()

    # JSON 배열 안전 추출
    start = content.find("[")
    end = content.rfind("]") + 1

    if start == -1 or end == -1:
        raise ValueError("No valid JSON array found in model response")

    return json.loads(content[start:end])

def run_pattern(
    user_info_list: list,
    acquisitions_df,
    naic_df,
    top_k: int = 5
):
    if not user_info_list:
        raise ValueError("user_info_list is empty")

    user_info = user_info_list[0]

    primary_info = user_info.get("primary_naic_info")
    if not primary_info:
        raise ValueError("primary_naic_info not found")

    user_naic = primary_info.get("code")
    user_naic_title = primary_info.get("title")

    payload = get_acquiror_payload(
        user_naic=user_naic,
        acquisitions_df=acquisitions_df,
        naic_df=naic_df,
        top_k=top_k
    )

    acquirer_list = payload["acquirer_list"]

    relation_result = extract_naic_relation(
        user_naic=user_naic,
        user_naic_title=user_naic_title,
        acquirer_list=acquirer_list
    )

    # LLM 추론 이전에 사용된 정량 근거를 코드별로 매핑해 둔다.
    ratio_map = {
        str(code): float(ratio)
        for code, ratio in payload.get("naic_ratio", {}).items()
    }
    title_map = {
        str(code): title
        for code, title in zip(
            payload.get("top_naic_codes", []),
            payload.get("top_naic_titles", [])
        )
    }

    enriched_relation_result = []
    for item in relation_result:
        acquirer_naic = str(item.get("acquirer_naic", ""))
        evidence = {
            "target_naic": str(user_naic) if user_naic is not None else None,
            "target_naic_title": user_naic_title,
            "acquirer_naic": acquirer_naic,
            "acquirer_naic_title": title_map.get(acquirer_naic),
            "acquirer_ratio_percent": ratio_map.get(acquirer_naic)
        }
        enriched_item = dict(item)
        enriched_item["evidence"] = evidence
        enriched_relation_result.append(enriched_item)

    return {
        "user_naic": user_naic,
        "user_naic_title": user_naic_title,
        "acquirer_statistics": payload,
        "relation_analysis": enriched_relation_result
    }
