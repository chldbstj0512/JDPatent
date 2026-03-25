import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from main import main as run_pipeline
from util.ocr import split_ocr_text


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


@lru_cache(maxsize=1)
def _load_data_bundle() -> dict[str, Any]:
    """Load heavy static artifacts once per process."""
    acquisitions_df = pd.read_csv(DATA_DIR / "acquisitions_20260129.csv", low_memory=False)
    naic_df = pd.read_csv(DATA_DIR / "NAICS_descripition.csv")

    with (DATA_DIR / "field_scores.json").open("r", encoding="utf-8") as f:
        field_scores = json.load(f)
    with (DATA_DIR / "hightech_list.json").open("r", encoding="utf-8") as f:
        hightech_list = json.load(f)

    return {
        "acquisitions_df": acquisitions_df,
        "naic_df": naic_df,
        "field_scores": field_scores,
        "hightech_list": hightech_list,
    }


def run_legacy_analysis(
    *,
    user_id: str,
    raw_text: str,
    user_prefer: str = "nation",
    user_prefer_nation: str | None = "South Korea",
    user_prefer_area: str | None = None,
) -> dict[str, Any]:
    """Adapt REST payload to legacy pipeline(main.py)."""
    bundle = _load_data_bundle()
    front_ocr, back_ocr = split_ocr_text(raw_text)

    return run_pipeline(
        user_id=user_id,
        front_ocr=front_ocr,
        back_ocr=back_ocr,
        acquisitions_df=bundle["acquisitions_df"],
        naic_df=bundle["naic_df"],
        field_scores=bundle["field_scores"],
        hightech_list=bundle["hightech_list"],
        user_prefer=user_prefer,
        user_prefer_nation=user_prefer_nation,
        user_prefer_area=user_prefer_area,
        avg_claim_count=7,
        avg_ipc_count=4.15,
        avg_citation_count=10,
    )

