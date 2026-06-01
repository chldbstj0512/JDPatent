"""
파싱 오류 점검용: extract.py 파이프라인 산출과 변리사 정답 CSV(`user_patent.csv` 등)를 **한 행·넓은 CSV**로 나란히 기록.

CSV 열: 메타 정보 + 체크리스트 N개 × ( _pipeline, _ground_truth ) + **계열 설명 파이프라인 전용 1열**(정답 없음).

정답 행: 기본 `logs/user_patent.csv` 의 2번째 데이터 행(iloc=1, 「테스트 공보 2」).
  - `VALIDATION_GROUND_TRUTH_CSV` : 정답 파일 경로
  - `VALIDATION_GROUND_TRUTH_NO` : `no` 열과 정확히 일치하는 행 우선
  - `VALIDATION_GROUND_TRUTH_ILOC` : 0부터 시작하는 행 인덱스 (NO 미지정 시)
  - `VALIDATION_GROUND_TRUTH_DEFAULT_ILOC` : 위 둘이 없을 때 사용할 기본 iloc (기본 1)

NAICS·field·독립항 단어 수·인용 self-check·**검증용 청구항 전문(첫 독립항)** 열은 검증 CSV에 넣지 않는다(파이프라인 JSON에는 `independent_claim` 등이 남을 수 있음).

기록 시각 열 이름은 하위 호환을 위해 `timestamp_utc`를 유지하나, **값은 Asia/Seoul(KST) 현지 시각** `YYYY-MM-DD HH:MM:SS` 문자열이다.
`run_id`·CSV 백업 접미사 시각도 동일(KST 벽시계)이다.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from extract import normalize_ipc_code, _ipc_display

# 한국 표준시(일광절약 없음). 검증 CSV·run_id·백업 파일명에 사용.
KST = timezone(timedelta(hours=9), "KST")

_FINAL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOG_DIR = os.path.join(_FINAL_DIR, "logs")
_DEFAULT_CSV = os.path.join(_DEFAULT_LOG_DIR, "ocr_parse_validation_wide.csv")
_DEFAULT_OCR_DUMP_DIR = os.path.join(_DEFAULT_LOG_DIR, "validation_ocr")


def _ensure_dirs(csv_path: str) -> None:
    """검증 CSV·상위 `logs/` 등 부모 디렉터리 생성."""
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) 체크리스트 — extract.py 의 extract_patent_metadata / extract_patent_claims
#    출력 필드·규칙과 1:1 대응 (파싱 오류 잡기용)
# ---------------------------------------------------------------------------
PARSE_VALIDATION_CHECKLIST: list[dict[str, str]] = [
    {"id": "country", "source": "metadata", "inid": "19/12", "note": "국가코드"},
    {"id": "title", "source": "metadata", "inid": "54", "note": "발명의 명칭"},
    {"id": "applicant_name", "source": "metadata", "inid": "71/73", "note": "출원인·특허권자"},
    {"id": "applicant_number", "source": "metadata", "inid": "21", "note": "출원번호"},
    {"id": "applicant_date", "source": "metadata", "inid": "22", "note": "출원일(원문·ISO 혼재 가능)"},
    {"id": "grant_number", "source": "metadata", "inid": "11/10", "note": "등록·공보 번호"},
    {"id": "grant_date", "source": "metadata", "inid": "45", "note": "등록일"},
    {"id": "abstract", "source": "metadata", "inid": "57", "note": "요약 전문"},
    {"id": "ipc_codes", "source": "metadata", "inid": "51", "note": "IPC 코드 목록(ipc_info)"},
    {"id": "ipc_code_count", "source": "metadata", "inid": "51", "note": "IPC 개수"},
    {"id": "claims_ipc_count", "source": "claims", "inid": "51", "note": "청구추출 ipc_count"},
    {"id": "claims_forward_citation_count", "source": "claims", "inid": "56", "note": "인용문헌 수(결정론 병합)"},
    {"id": "claims_claim_count", "source": "claims", "inid": "청구", "note": "전체 청구항 수"},
    {"id": "claims_independent_claim_count", "source": "claims", "inid": "청구", "note": "독립항 수"},
    {"id": "claims_dependent_claim_count", "source": "claims", "inid": "청구", "note": "종속항 수"},
    {"id": "claims_claim_family_count", "source": "claims", "inid": "청구", "note": "청구 계열 수(타항 인용 그래프 연결요소; 독립항 수와 달라질 수 있음)"},
]

CHECKLIST_IDS = [x["id"] for x in PARSE_VALIDATION_CHECKLIST]


def expected_validation_wide_columns() -> list[str]:
    """`build_validation_row`와 동일한 열 순서(append 시 헤더·열 수 일치 검사용)."""
    cols = [
        "timestamp_utc",
        "run_id",
        "user_id",
        "ocr_front_chars",
        "ocr_back_chars",
        "ocr_front_dump_path",
        "ocr_back_dump_path",
        "ground_truth_csv",
        "ground_truth_row_no",
    ]
    for cid in CHECKLIST_IDS:
        cols.extend(
            [
                f"{cid}_pipeline",
                f"{cid}_ground_truth",
            ]
        )
    cols.append("claims_claim_family_count_reason")
    return cols


_DEFAULT_GROUND_TRUTH_CSV = os.path.join(_FINAL_DIR, "logs", "user_patent.csv")


def _ground_truth_csv_path() -> str:
    return os.getenv("VALIDATION_GROUND_TRUTH_CSV", _DEFAULT_GROUND_TRUTH_CSV)


def _load_ground_truth_row(user_id: str) -> Optional[dict[str, Any]]:
    """변리사 정답 CSV에서 한 행을 고른다. 실패 시 None."""
    path = _ground_truth_csv_path()
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    if df.empty:
        return None

    no_key = os.getenv("VALIDATION_GROUND_TRUTH_NO", "").strip()
    if no_key and "no" in df.columns:
        hit = df[df["no"].astype(str).str.strip() == no_key]
        if len(hit):
            return hit.iloc[0].to_dict()

    iloc_env = os.getenv("VALIDATION_GROUND_TRUTH_ILOC", "").strip()
    if iloc_env != "":
        try:
            i = int(iloc_env)
            if 0 <= i < len(df):
                return df.iloc[i].to_dict()
        except ValueError:
            pass

    if user_id and "no" in df.columns:
        hit = df[df["no"].astype(str).str.strip() == str(user_id).strip()]
        if len(hit):
            return hit.iloc[0].to_dict()

    try:
        default_i = int(os.getenv("VALIDATION_GROUND_TRUTH_DEFAULT_ILOC", "1"))
    except ValueError:
        default_i = 1
    default_i = max(0, min(default_i, len(df) - 1))
    return df.iloc[default_i].to_dict()


def _parse_gt_ipc_all_cell(raw: Any) -> list[str]:
    """user_patent.csv 의 ipc_all 셀 (예: G06F-011/00,[G11C-029/00, ...]) → 정규화 코드 리스트."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    s = s.replace("[", ",").replace("]", ",")
    seen: list[str] = []
    for part in re.split(r"\s*,\s*", s):
        p = part.strip().strip(",").strip()
        if not p:
            continue
        norm = ""
        m = re.match(r"^([A-H]\d{2})([A-Z])-(\d{1,4})/(\d{1,6})$", p, re.IGNORECASE)
        if m:
            main = m.group(3).lstrip("0") or "0"
            norm = normalize_ipc_code(f"{m.group(1).upper()}{m.group(2).upper()}{main}/{m.group(4)}")
        else:
            norm = normalize_ipc_code(p.replace("-", ""))
        if norm and norm not in seen:
            seen.append(norm)
    return seen


def _ground_truth_checklist_values(gt_row: Optional[dict[str, Any]]) -> dict[str, Any]:
    """체크리스트 id → 정답 CSV에서 온 값 (없으면 None)."""
    empty = {cid: None for cid in CHECKLIST_IDS}
    if not gt_row:
        return empty

    def _g(key: str) -> Any:
        v = gt_row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    ipc_norms = _parse_gt_ipc_all_cell(_g("ipc_all"))
    ipc_disp = [_ipc_display(c) for c in ipc_norms]

    def _int_cell(key: str) -> Optional[int]:
        v = _g(key)
        if v is None:
            return None
        try:
            return int(float(str(v).strip()))
        except (TypeError, ValueError):
            return None

    out = dict(empty)
    out["country"] = _g("country_code")
    out["title"] = _g("invention_title")
    out["applicant_name"] = _g("applicant")
    out["applicant_number"] = _g("application_number")
    out["applicant_date"] = _g("application_date")
    out["grant_number"] = _g("registration_number")
    out["grant_date"] = _g("registration_date")
    out["abstract"] = _g("abstract")
    out["ipc_codes"] = ipc_disp
    out["ipc_code_count"] = _int_cell("ipc_count")
    out["claims_ipc_count"] = _int_cell("ipc_count")
    out["claims_forward_citation_count"] = _int_cell("forward_citation_count")
    out["claims_claim_count"] = _int_cell("claim_count")
    out["claims_independent_claim_count"] = _int_cell("independent_claim_count")
    out["claims_dependent_claim_count"] = _int_cell("dependent_claim_count")
    out["claims_claim_family_count"] = _int_cell("claim_family_count")
    return out


def _dump_full_ocr(run_id: str, user_id: str, front_ocr: str, back_ocr: Optional[str], dump_dir: str) -> dict[str, str]:
    os.makedirs(dump_dir, exist_ok=True)
    safe_uid = re.sub(r"[^\w\-.]", "_", str(user_id))[:120]
    front_path = os.path.join(dump_dir, f"{run_id}_{safe_uid}_front.txt")
    with open(front_path, "w", encoding="utf-8") as f:
        f.write(front_ocr or "")
    back_path = ""
    if back_ocr:
        bp = os.path.join(dump_dir, f"{run_id}_{safe_uid}_back.txt")
        with open(bp, "w", encoding="utf-8") as f:
            f.write(back_ocr)
        back_path = bp
    return {"ocr_front_dump_path": front_path, "ocr_back_dump_path": back_path}


def _pipeline_values(user_info: Optional[dict], claims: Any) -> dict[str, Any]:
    ui = user_info or {}
    ipc_info = ui.get("ipc_info") if isinstance(ui.get("ipc_info"), list) else []
    codes = [x.get("code") for x in ipc_info if isinstance(x, dict) and x.get("code")]
    out: dict[str, Any] = {
        "country": ui.get("country"),
        "title": ui.get("title"),
        "applicant_name": ui.get("applicant_name"),
        "applicant_number": ui.get("applicant_number"),
        "applicant_date": ui.get("applicant_date"),
        "grant_number": ui.get("grant_number"),
        "grant_date": ui.get("grant_date"),
        "abstract": ui.get("abstract"),
        "ipc_codes": codes,
        "ipc_code_count": len(codes),
        "claims_ipc_count": None,
        "claims_forward_citation_count": None,
        "claims_claim_count": None,
        "claims_independent_claim_count": None,
        "claims_dependent_claim_count": None,
        "claims_claim_family_count": None,
        "claims_claim_family_count_reason": None,
    }
    if isinstance(claims, dict) and "error" not in claims:
        out["claims_ipc_count"] = claims.get("ipc_count")
        out["claims_forward_citation_count"] = claims.get("forward_citation_count")
        out["claims_claim_count"] = claims.get("claim_count")
        out["claims_independent_claim_count"] = claims.get("independent_claim_count")
        out["claims_dependent_claim_count"] = claims.get("dependent_claim_count")
        out["claims_claim_family_count"] = claims.get("claim_family_count")
        out["claims_claim_family_count_reason"] = claims.get("claim_family_count_reason")
    return out


def _cell(v: Any, max_len: int = 8000) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False)
    else:
        s = str(v)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def build_validation_row(
    user_id: str,
    front_ocr: str,
    back_ocr: Optional[str],
    user_info_list: Optional[list],
    claims: Any,
    run_id: Optional[str] = None,
    ocr_dump_dir: Optional[str] = None,
) -> dict[str, Any]:
    ocr_dump_dir = ocr_dump_dir or os.getenv("VALIDATION_OCR_DUMP_DIR", _DEFAULT_OCR_DUMP_DIR)
    run_id = run_id or datetime.now(KST).strftime("%Y%m%dT%H%M%S_%f")
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    paths = _dump_full_ocr(run_id, user_id, front_ocr, back_ocr, ocr_dump_dir)

    ui = user_info_list[0] if user_info_list else None
    pipe = _pipeline_values(ui, claims)
    gt_row = _load_ground_truth_row(str(user_id))
    gt_vals = _ground_truth_checklist_values(gt_row)

    row: dict[str, Any] = {
        "timestamp_utc": ts,
        "run_id": run_id,
        "user_id": user_id,
        "ocr_front_chars": len(front_ocr or ""),
        "ocr_back_chars": len(back_ocr or ""),
        "ocr_front_dump_path": paths["ocr_front_dump_path"],
        "ocr_back_dump_path": paths["ocr_back_dump_path"],
        "ground_truth_csv": os.path.abspath(_ground_truth_csv_path()),
        "ground_truth_row_no": _cell(gt_row.get("no")) if gt_row else "",
    }

    for cid in CHECKLIST_IDS:
        pv = pipe.get(cid)
        gv = gt_vals.get(cid)
        row[f"{cid}_pipeline"] = _cell(pv, 8000)
        row[f"{cid}_ground_truth"] = _cell(gv, 8000)

    if isinstance(claims, dict) and claims.get("error"):
        row["claims_claim_family_count_reason"] = _cell(f"[claims error] {claims.get('error')}", 12000)
    else:
        row["claims_claim_family_count_reason"] = _cell(pipe.get("claims_claim_family_count_reason"), 12000)

    return row


def log_ocr_parse_validation(
    user_id: str,
    front_ocr: str,
    back_ocr: Optional[str],
    user_info_list: Optional[list],
    claims: Any,
    csv_path: Optional[str] = None,
    ocr_dump_dir: Optional[str] = None,
) -> pd.DataFrame:
    csv_path = csv_path or os.getenv("VALIDATION_LOG_CSV", _DEFAULT_CSV)
    _ensure_dirs(csv_path)
    row = build_validation_row(
        user_id=user_id,
        front_ocr=front_ocr,
        back_ocr=back_ocr,
        user_info_list=user_info_list,
        claims=claims,
        ocr_dump_dir=ocr_dump_dir,
    )
    cols = expected_validation_wide_columns()
    ordered = {c: row.get(c, "") for c in cols}
    df = pd.DataFrame([ordered], columns=cols)

    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    write_header = not file_exists
    if file_exists:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline().rstrip("\n")
        try:
            existing_header = next(csv.reader([first_line]))
        except Exception:
            existing_header = []
        if tuple(existing_header) != tuple(cols):
            bak = f"{csv_path}.bak.{datetime.now(KST).strftime('%Y%m%dT%H%M%S')}"
            os.replace(csv_path, bak)
            write_header = True
            print(
                f"[validation] wide CSV schema mismatch; rotated old file to {bak} "
                f"(old_cols={len(existing_header)} new_cols={len(cols)})",
                flush=True,
            )

    df.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    print(f"[validation] wide row appended -> {csv_path} (checklist: pipeline+ground_truth)", flush=True)
    return df


def log_ocr_parse_validation_error(
    user_id: str,
    front_ocr: str,
    back_ocr: Optional[str],
    error_reason: str,
    csv_path: Optional[str] = None,
    ocr_dump_dir: Optional[str] = None,
) -> pd.DataFrame:
    return log_ocr_parse_validation(
        user_id=user_id,
        front_ocr=front_ocr,
        back_ocr=back_ocr,
        user_info_list=None,
        claims={"error": error_reason},
        csv_path=csv_path,
        ocr_dump_dir=ocr_dump_dir,
    )
