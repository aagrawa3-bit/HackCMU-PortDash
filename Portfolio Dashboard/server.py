# server.py
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from logic import compute_metrics_from_csv

router = APIRouter(prefix="/api", tags=["api"])

class MetricsRequest(BaseModel):
    csv: str = Field(..., description="Portfolio CSV text")
    ff3_csv: str = Field(..., description="F-F Research Data Factors (daily) CSV text")
    mom_txt: str = Field(..., description="F-F Momentum factor daily .txt text")
    date_needed: Optional[str] = Field(None, description="YYYY-MM-DD (optional)")

@router.post("/metrics")
def metrics(req: MetricsRequest):
    try:
        data = compute_metrics_from_csv(
            csv_text=req.csv,
            ff3_csv_text=req.ff3_csv,
            mom_txt_text=req.mom_txt,
            date_needed=req.date_needed,
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"processing_error: {e}")
