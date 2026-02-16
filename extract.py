import os
import re
import json
import time
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")      # aws
PINECONE_REGION = os.getenv("PINECONE_REGION")    # us-east-1
INDEX_NAME = os.getenv("INDEX_NAME")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)

def embed_patent_text(text: str) -> list:
    
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    embedding = response.data[0].embedding

    assert len(embedding) == 3072
    return embedding

def retrieve_top_naics(
    query_vector: list,
    top_k: int = 15
) -> list:
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    naics_candidates = []

    for match in result["matches"]:
        meta = match["metadata"]
        naics_candidates.append({
            "code": meta["NAICS_code"],      # 6자리
            "title": meta["NAICS_title"],
            "desc": meta["original_description"],
            "score": match["score"]
        })

    return naics_candidates

def build_naics_context_text(naics_candidates: list) -> str:
    naics_text = "\n".join(
        [
            f"- {str(c['code']).split('.')[0]}: {c['title']} — {c['desc']}"
            for c in naics_candidates
        ]
    )

    return naics_text

def extract_patent_metadata(text: str, naics_context: str):
    prompt = f"""
You are a patent classification and information extraction system.

You will be given:
1) Raw OCR text extracted from a patent publication.
2) A list of candidate NAICS industry codes (with descriptions).

The OCR text may contain OCR errors, duplicated lines, broken line breaks,
or reading-order issues. Do NOT attempt to fix or rewrite the text.

Your tasks are:
(A) Extract bibliographic and technical fields STRICTLY according to INID codes.
(B) Select the MOST APPROPRIATE NAICS industry code(s) from the GIVEN CANDIDATES ONLY.

----------------------------------------
NAICS SELECTION RULES (VERY IMPORTANT)
----------------------------------------
- Please note that the provided NAICS codes are already sorted in order of relevance, so take this into consideration.
- You MUST choose NAICS code(s) ONLY from the candidate list provided below.
- Select the code(s) that BEST match the patent's technical field and application.
- Base your decision primarily on:
  - IPC codes (INID 51)
  - Abstract (INID 57)
- If exactly one NAICS code is clearly the best fit, return ONLY one.
- If two or three codes are strongly relevant, you MAY return up to three.
- NEVER invent or infer NAICS codes not present in the candidate list.
- Return NAICS codes as a list of 6-digit strings.
- If returning multiple NAICS codes, order them by relevance,
  with the most relevant code appearing first in the list.
- You MUST return at least ONE NAICS code.

----------------------------------------
IMPORTANT CONSTRAINTS
----------------------------------------
- Do NOT hallucinate missing information.
- Do NOT infer beyond the text.
- Output MUST be valid JSON.
- Output ONLY the JSON object.
- Return all extracted text fields (including title and abstract) in the ORIGINAL LANGUAGE of the patent as indicated by the country code.
- The "field" value MUST be assigned to EXACTLY ONE of the five allowed categories ("compu", "bio", "comm", "elec", "etc"); it MUST NOT be null under any circumstances.

----------------------------------------
OUTPUT FORMAT (JSON ONLY)
----------------------------------------
Return a JSON object with EXACTLY the following fields.
Use null if a field cannot be confidently extracted.

{{
  "country": "US",
  "title": "Example Title",
  "applicant_name": "Example Applicant",
  "applicant_number": "17/123,456",
  "applicant_date": "2021-02-22",
  "grant_number": null,
  "grant_date": null,
  "naics_code": ["325414"],
  "abstract": "Example abstract text...",
  "field": "bio",
  "ipc_info": [{{
    "code": "A61K 39/05",
    "short_description": "백신기술",
    "long_description": "면역 치료용 백신 조성물 관련 기술"
  }}]
}}

----------------------------------------
FIELD EXTRACTION RULES
----------------------------------------

1. country
- Infer from INID (19).
- If missing, infer from document kind or header in INID (12).
- Return ISO 2-letter country code (e.g., "US", "KR").
- Allowed exceptions: "PCT", "EP".
- If uncertain, return null.

2. title
- Extract ONLY from INID (54).
- Use the original text verbatim.

3. applicant_name
- Extract verbatim.
- Use:
  - INID (71) for U.S. documents.
  - INID (73) for Korean documents.

4. applicant_number
- Extract ONLY from INID (21).

5. applicant_date
- Extract ONLY from INID (22).
- Convert to ISO format YYYY-MM-DD.

6. grant_number
- Extract ONLY from INID (11).

7. grant_date
- Extract ONLY from INID (45).

8. abstract
- Extract FULL abstract from INID (57).
- Preserve verbatim text.

9. ipc_code
- Extract ALL IPC codes under INID (51) WITHOUT OMITTING ANY.
- If multiple IPC codes are listed, you MUST return every single one.
- Do NOT summarize or filter.
- Preserve original formatting exactly as written.
- Return them as a list in the same order as they appear in the document.
- The number of returned IPC codes MUST exactly match the number found in INID (51).

10. ipc_long_description
- For EACH IPC code extracted under INID (51),
  provide a clear technical/industry-oriented explanation.
- Do NOT quote legal definitions.
- Explain in practical terms suitable for M&A or industry analysis.
- Return as a dictionary mapping IPC → description.

11. ipc_short_description
- For EACH IPC code,
  provide a VERY SHORT summary (within 7 Korean characters).
- No legal wording.
- Practical and intuitive.
- Return as a dictionary mapping IPC → short label.

12. field
- Assign EXACTLY ONE of:
  "compu", "bio", "comm", "elec", "etc"
- Decide primarily from IPC codes.

----------------------------------------
NAICS CANDIDATES (CHOOSE FROM THIS LIST ONLY)
----------------------------------------
{naics_context}

----------------------------------------
OCR TEXT
----------------------------------------
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You extract patent metadata and select the most appropriate NAICS codes from provided candidates."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=3000,
    )

    content = response.choices[0].message.content.strip()

    # JSON 안전 추출
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found in model response")

    return json.loads(content[start:end])

def run_NAIC_extract(naic_df, user_id, user_ocr):
    results = []

    patent_text = user_ocr

    query_vector = embed_patent_text(patent_text)

    naics_candidates = retrieve_top_naics(query_vector, top_k=15)
    naics_context = build_naics_context_text(naics_candidates)

    patent_meta = extract_patent_metadata(
        text=patent_text,
        naics_context=naics_context
    )

    naic_map = {
        str(code): {
            "title": title,
            "description": desc
        }
        for code, title, desc in zip(
            naic_df["naics_code"].astype(str),
            naic_df["naics_title"],
            naic_df["description"]
        )
    }

    result_item = {
        "pdf_name": user_id,
        **patent_meta
    }

    codes = result_item.get("naics_code", [])

    if not codes and naics_candidates:
        fallback_code = str(naics_candidates[0]["code"])
        codes = [fallback_code]
        result_item["naics_code"] = codes

    primary_code = codes[0] if codes else None
    candidate_codes = codes[1:] if len(codes) > 1 else []

    result_item["primary_naic_info"] = (
        {
            "code": str(primary_code),
            "title": naic_map.get(str(primary_code), {}).get("title"),
            "description": naic_map.get(str(primary_code), {}).get("description")
        }
        if primary_code else None
    )

    result_item["candidate_naic_info"] = [
        {
            "code": str(code),
            "title": naic_map.get(str(code), {}).get("title"),
            "description": naic_map.get(str(code), {}).get("description")
        }
        for code in candidate_codes
    ]

    results.append(result_item)

    return results
