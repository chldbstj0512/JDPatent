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
    # Embedding API에 보내는 입력 길이를 제한해 불필요한 비용/에러를 방지합니다.
    text = text[:8000]

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

You must first determine whether the provided document is an official patent publication.

A document qualifies as a patent publication ONLY if it clearly contains at least one of the following:
- A patent publication number (e.g., US 2023/0123456 A1, EP 1234567 B1, KR 10-2023-0123456, WO 2023/123456)
- An explicit label such as "Patent", "Patent Application", "Patent Publication", "공개특허", "특허공보", "公開特許", etc.
- A structured patent format including sections such as Abstract, Claims, Description, Inventors, Assignee, Filing Date, Publication Date.

If the document does NOT clearly satisfy these conditions,
OR if there is any uncertainty,
OR if the format does not match an official patent publication structure,

you MUST return exactly the following JSON and nothing else:

{{"error": "not_a_patent_document"}}

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

def extract_patent_claims(text_front: str, text_back: str = None):
    """
    - 50페이지 미만: text_front 에 전체 OCR 텍스트 입력, text_back = None
    - 53페이지 이상: text_front = 앞 3페이지 OCR
                     text_back  = 뒤 50페이지 OCR
    """

    if text_back:
        combined_text = f"""
        [FRONT_PART_OCR]
        {text_front}

        [BACK_PART_OCR]
        {text_back}
        """
    else:
        combined_text = text_front

    prompt = f"""
You are a patent claim extraction and structural analysis system.

You will be given:
1) OCR text extracted from a patent publication.
   - If the patent has fewer than 50 pages: full text is provided.
   - If the patent has 53 pages or more: 
     the first 3 pages and the last 50 pages are provided separately.

If patent claims cannot be identified or extracted from the document, 
return only {{"error": "claims_not_found"}} in JSON format without any additional explanation.

The OCR text may contain:
- OCR noise
- line break errors
- duplicated lines
- broken numbering

You must strictly extract structured claim information according to the rules below.

----------------------------------------
YOUR TASKS
----------------------------------------

(A) Extract IPC codes (INID 51)
(B) Count forward citations (INID 56)
(C) Extract full claim section
(D) Count total claims (excluding deleted claims)
(E) Count dependent claims
(F) Extract independent claims and compute:
    - independent claim count
    - independent claim word count
(G) Compute claim family count (technical category count)

----------------------------------------
CRITICAL RULES (VERY IMPORTANT)
----------------------------------------

- Do NOT hallucinate.
- Do NOT infer beyond text.
- Do NOT rewrite claims.
- Preserve original language (Korean or English).
- Output MUST be valid JSON.
- Output ONLY the JSON object.
- If uncertain, return null (NOT empty string).

----------------------------------------
1. IPC EXTRACTION RULE
----------------------------------------

- Extract ALL IPC codes under INID (51).
- Count them.
- Do NOT return the individual IPC code list.
- Return ONLY the total count as ipc_count.

----------------------------------------
2. FORWARD CITATION COUNT RULE
----------------------------------------

Find citation section:
- English: "References Cited"
- Korean: "선행기술조사문헌"

Count ALL listed references.
If none exist, return 0.
forward_citation_count must exactly match visible entries.

If the citation section includes indications such as "(Continued)", 
you MUST also examine subsequent pages and include all additional listed references 
in the total count.

----------------------------------------
3. CLAIM SECTION EXTRACTION RULE
----------------------------------------

Locate claim section start:
- English likely indicators:
  - "What is claimed is:"
  - "The invention claimed is:"
  - "Claims"
- Korean likely indicator:
  - "청구범위"

Extract entire claim section.
Exclude deleted claims.
Deleted claim indicators:
- English: "(canceled)"
- Korean: "삭제"

Extract entire claim section to analyze.
Do NOT return the full claim section in the output.

----------------------------------------
4. TOTAL CLAIM COUNT RULE
----------------------------------------

Count numbered claims in claim section.
Exclude deleted claims.

Cross-check:
- English front text may state: "N claims"
- Korean front text may state: "총 N 항"

If mismatch occurs, prioritize actual visible claims.

You MUST carefully read each claim individually and explicitly determine whether it is independent or dependent based on its full textual content.
Do NOT rely only on pattern matching.
Analyze the legal structure of each claim before classifying it.
Only after identifying all independent claims, determine independent_claim_count.

Return:
claim_count

----------------------------------------
5. DEPENDENT CLAIM RULE
----------------------------------------

Dependent claim indicators:

English:
- "according to claim"
- "of claim"

Korean:
- "제 n항에 있어서"

Count them strictly.

Return:
dependent_claim_count

----------------------------------------
6. INDEPENDENT CLAIM RULE
----------------------------------------

A claim is independent if it does NOT refer to any other claim.

A claim is dependent if it explicitly refers to another claim.

You MUST read all claims up to the total claim_count and review them completely before providing the final answer.

Step 1:
Read EVERY numbered claim in the claim section from 1 to claim_count.

Step 2:
For each claim:
- If it refers to another claim → classify as dependent.
- If it does NOT refer to any other claim → classify as independent.

Step 3:
Extract the FULL TEXT of all independent claims.
Include the original claim number (e.g., "1.", "12.", "제1항").
Do NOT remove numbering.
Do NOT summarize.

Step 4:
Set independent_claim_count equal to the actual number of independent claims extracted above.

Step 5:
Calculate independent_claim_word_count from the extracted independent_claim text.
Use whitespace splitting for both English and Korean.

----------------------------------------
7. CLAIM FAMILY COUNT RULE
----------------------------------------

Claim family count means:
Number of technically distinct independent claim categories
(e.g., apparatus, method, system, composition, etc.)

This may be equal to independent_claim_count but may differ.

Analyze independent claims based on technical entity type.
Return:
claim_family_count

----------------------------------------
OUTPUT FORMAT (JSON ONLY)
----------------------------------------

Return EXACTLY:

{{
  "claim_count": 0,
  "independent_claim_count": 0,
  "independent_claim_word_count": 0,
  "dependent_claim_count": 0,
  "claim_family_count": 0,
  "independent_claim": "full independent claim text",
  "ipc_count": 0,
  "forward_citation_count": 0
}}

----------------------------------------
OCR TEXT
----------------------------------------
{combined_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You extract structured patent claim information with strict rule-based accuracy."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0,
        max_tokens=10000,
    )

    content = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in model response")

    json_str = match.group(0)
    return json.loads(json_str)

def run_NAIC_extract(
    naic_df,
    user_id,
    user_ocr,      # front OCR (기존 NAICS/metadata용)
    back_ocr=None  # claim 추출용
):
    results = []
    claims = None

    # ---------------------------------
    # 1. NAICS / Metadata (기존 구조 유지)
    # ---------------------------------
    patent_text = user_ocr  # front OCR만 사용

    query_vector = embed_patent_text(patent_text)

    naics_candidates = retrieve_top_naics(query_vector, top_k=15)
    naics_context = build_naics_context_text(naics_candidates)

    try:
        patent_meta = extract_patent_metadata(
            text=patent_text,
            naics_context=naics_context
        )
        if isinstance(patent_meta, str):
            patent_meta = json.loads(patent_meta)
            
        if isinstance(patent_meta, dict) and "error" in patent_meta:
            return [], patent_meta

    except Exception:
        return [], {"error": "metadata_extraction_failed"}

    # ---------------------------------
    # 2. NAICS 코드 매핑
    # ---------------------------------
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

    # ---------------------------------
    # 3. NAICS fallback 처리
    # ---------------------------------
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

    # ---------------------------------
    # 4. Claim 추출 (별도 반환)
    # ---------------------------------
    try:
        claims = extract_patent_claims(
            text_front=user_ocr,
            text_back=back_ocr
        )
        if isinstance(claims, str):
            claims = json.loads(claims)
        
        if isinstance(claims, dict) and "error" in claims:
            return results, claims
        
    except Exception:
        return results, {"error": "claim_extraction_failed"}

    return results, claims
