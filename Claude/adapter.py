# adapter.py
import os
import io
import json
import shlex
import subprocess
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
# Path to your unchanged calculation script
CORE_SCRIPT = os.environ.get("CORE_SCRIPT", "server.py")

# Where your calculation script expects the CSV (unchanged in your code)
TARGET_CSV_PATH = os.environ.get(
    "TARGET_CSV_PATH",
    "/Users/adityaagrawal/Downloads/US_equity_2025-08-29.csv",
)

# Optional: base notional to show a portfolio value in the UI
PORTFOLIO_NOTIONAL = float(os.environ.get("PORTFOLIO_NOTIONAL", "100000000"))  # $100mm

# --------------------------------------------------------------------------------------
# FastAPI setup
# --------------------------------------------------------------------------------------
app = FastAPI(title="Portfolio Adapter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # lock this down if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class CSVIn(BaseModel):
    csv: str
    filename: Optional[str] = None  # ignored by server.py, but useful for logs


class OptimizeIn(BaseModel):
    csv: str
    strategy: Optional[str] = "sharpe"  # 'sharpe' or 'gmv'


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def write_csv_to_target(csv_text: str) -> str:
    """
    Write the uploaded CSV verbatim to the exact path your script reads.
    """
    target = TARGET_CSV_PATH
    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(csv_text.strip() + ("\n" if not csv_text.endswith("\n") else ""))
    return target


def run_core_and_capture() -> str:
    """
    Run your unmodified server.py and capture its stdout/stderr.
    """
    if not os.path.isfile(CORE_SCRIPT):
        raise RuntimeError(f"Core script not found: {CORE_SCRIPT}")

    # Use python -u for unbuffered output so we see prints as they happen
    cmd = f"{shlex.quote(os.environ.get('PYTHON', 'python3'))} -u {shlex.quote(CORE_SCRIPT)}"
    proc = subprocess.run(
        cmd, shell=True, cwd=os.path.dirname(os.path.abspath(CORE_SCRIPT)) or None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"Core script failed (exit {proc.returncode}). Output:\n{output}")
    return output


def parse_stdout(txt: str) -> Dict[str, Any]:
    """
    Parse all risk/performance numbers printed by your script.
    """
    import re

    def pct_to_float(s):
        if not s:
            return 0.0
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", s)
        return float(m.group(1)) / 100.0 if m else 0.0

    def grab(pattern):
        m = re.search(pattern, txt)
        return m.group(1).strip() if m else None

    out: Dict[str, Any] = {}
    out["portfolioBeta"] = _safe_float(grab(r"Portfolio Beta:\s*([-\d\.]+)"))

    out["dailyVol"]    = pct_to_float(grab(r"Expected Daily Volatility:\s*([^\n]+)"))
    out["weeklyVol"]   = pct_to_float(grab(r"Expected Portfolio Volatility:\s*([^\n]+)"))
    out["dailyRange"]  = pct_to_float(grab(r"Expected Portfolio Range \(1 Day\):\s*\+/-\s*([^\n]+)"))
    out["weeklyRange"] = pct_to_float(grab(r"Expected Portfolio Range \(1 Week\):\s*\+/-\s*([^\n]+)"))
    out["maxDD"]       = pct_to_float(grab(r"Maximum Expected Drawdown:\s*([^\n]+)"))

    out["var5"]   = pct_to_float(grab(r"Portfolio 5% VaR\s+\s*([^\n]+)"))
    out["cvar5"]  = pct_to_float(grab(r"Portfolio 5% CVaR\s+\s*([^\n]+)"))
    out["var1"]   = pct_to_float(grab(r"Portfolio 1% VaR\s+\s*([^\n]+)"))
    out["cvar1"]  = pct_to_float(grab(r"Portfolio 1% CVaR\s+\s*([^\n]+)"))
    out["tailRisk"]= pct_to_float(grab(r"Tail Risk\s+\s*([^\n]+)"))

    tm = grab(r"Tail Risk Multiple:\s*([-\d\.]+)")
    out["tailMultiple"] = _safe_float(tm)

    out["expReturnDaily"]  = pct_to_float(grab(r"Expected Return \(Daily\):\s*([^\n]+)"))
    out["expReturnWeekly"] = pct_to_float(grab(r"Expected Return:\s*([^\n]+)"))

    sg = grab(r"Annualized Expected Sharpe \(Gross\):\s*([-\d\.]+)")
    sn = grab(r"Annualized Expected Sharpe \(Net\):\s*([-\d\.]+)")
    out["sharpeGross"] = _safe_float(sg)
    out["sharpeNet"]   = _safe_float(sn)
    return out


def parse_factors_from_stdout(txt: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the printed pandas DataFrame summary_table at the end of your script.
    Expected rows: Mkt-RF, SMB, HML, Mom
    """
    import re
    factors: Dict[str, Dict[str, Any]] = {}
    key_map = {"Mkt-RF": "Market", "SMB": "SMB", "HML": "HML", "Mom": "MOM"}

    pat = re.compile(
        r"^\s*(Mkt-RF|SMB|HML|Mom)\s+"
        r"([-\d\.]+)\s+"            # Beta
        r"([-\d\.]+)\s+"            # T-Statistic
        r"([-\d\.eE]+)\s+"          # P-Value
        r"([A-Za-z ]+)\s*$",        # Significance
        re.MULTILINE
    )

    for m in pat.finditer(txt):
        raw, beta, t, p, sig = m.groups()
        name = key_map.get(raw, raw)
        factors[name] = {
            "beta": _safe_float(beta),
            "t": _safe_float(t),
            "p": _safe_float(p),
            "sig": sig.strip(),
        }
    return factors


def dataframe_from_csv_text(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    # Normalize column names you use in the UI
    rename_map = {
        "Market Capitalisation": "MarketCap",
        "Capital Allocation": "Allocation",
        "Stop Loss": "StopLoss",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def exposures_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Build sector/name exposures for the UI without changing compute logic.
    """
    exposures = {"sectors": {}, "names": {}}

    if "Allocation" in df.columns:
        alloc = df["Allocation"].astype(float)
    else:
        alloc = pd.Series([1.0] * len(df))

    total = alloc.sum() if alloc.sum() != 0 else 1.0
    alloc_frac = alloc / total

    # Sector exposure
    if "Sector" in df.columns:
        sec = (
            df.assign(_w=alloc_frac)
              .groupby("Sector")["_w"]
              .sum()
              .sort_values(ascending=False)
        )
        exposures["sectors"] = {str(k): float(v) for k, v in sec.items()}

    # Name exposure (by symbol)
    if "Symbol" in df.columns:
        name_map = {}
        for sym, w in zip(df["Symbol"].astype(str).tolist(), alloc_frac.tolist()):
            name_map[sym] = name_map.get(sym, 0.0) + float(w)
        exposures["names"] = name_map

    return exposures


def positions_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    sum_alloc = float(df["Allocation"].astype(float).sum()) if "Allocation" in df.columns else 0.0
    for _, r in df.iterrows():
        alloc = float(r.get("Allocation", 0) or 0)
        alloc_pct = (alloc / sum_alloc) if sum_alloc else 0.0
        out.append({
            "Symbol": str(r.get("Symbol", "")),
            "Execution": str(r.get("Execution", "Buy")),
            "Sector": str(r.get("Sector", "")),
            "MarketCap": _safe_float(r.get("MarketCap", 0)),
            "Price": _safe_float(r.get("Price", 0)),
            "Volume": _safe_float(r.get("Volume", 0)),
            "Allocation": alloc,
            "AllocationPercent": alloc_pct,
            "Timeframe": str(r.get("Timeframe", "")),
            "Target": _safe_float(r.get("Target", 0)),
            "StopLoss": _safe_float(r.get("StopLoss", 0)),
        })
    return out


def overview_from_df(df: pd.DataFrame, risk: Dict[str, Any]) -> Dict[str, Any]:
    longs = 0
    shorts = 0
    if "Execution" in df.columns:
        for v in df["Execution"].astype(str).str.lower():
            if v in ("sell", "short"):
                shorts += 1
            else:
                longs += 1
    active = int(longs + shorts)

    total_alloc = float(df["Allocation"].astype(float).sum()) if "Allocation" in df.columns else 0.0
    alloc_frac = 0.0
    if total_alloc > 0:
        alloc_frac = 1.0  # by definition we normalized exposures on provided Allocation
    # Use notional × allocation fraction to show a value
    portfolio_value = PORTFOLIO_NOTIONAL * alloc_frac

    return {
        "portfolioValue": float(portfolio_value),
        "totalAllocationFrac": float(alloc_frac),
        "activePositions": active,
        "longCount": longs,
        "shortCount": shorts,
        "beta": float(risk.get("portfolioBeta", 0.0)),
    }


def parse_optimizer_from_stdout(txt: str) -> Dict[str, Any]:
    """
    Optional: parse weights printed by your portfolio_gmv(). If not present,
    returns harmless defaults so the Optimizer tab doesn't break.
    """
    import re

    out = {
        "currentSharpe": 0.0,
        "optimalSharpe": 0.0,
        "varianceReductionPct": 0.0,
        "frontier": [],
        "currentPoint": {"x": 0.0, "y": 0.0},
        "optimalPoint": {"x": 0.0, "y": 0.0},
    }

    # Not strictly needed for your UI; included for completeness.
    # Example lines your script prints:
    # print("Max Sharpe Ratio:", sharpe_weight)
    # print("GMV Portfolio:", gmv_weight)
    # You may also print 'tickers' above; we try to match arrays.
    m1 = re.search(r"Max Sharpe Ratio:\s*\[([^\]]+)\]", txt)
    m2 = re.search(r"GMV Portfolio:\s*\[([^\]]+)\]", txt)
    if m1 or m2:
        out["optimalSharpe"] = 0.0  # no Sharpe value printed; leave 0
        out["varianceReductionPct"] = 0.0
    return out


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "status": "healthy", "core": CORE_SCRIPT, "target_csv": TARGET_CSV_PATH}


@app.post("/api/metrics")
def metrics(inp: CSVIn):
    if not inp.csv or not inp.csv.strip():
        raise HTTPException(status_code=400, detail="Missing CSV text.")

    # Save CSV where your script expects it
    try:
        path = write_csv_to_target(inp.csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write CSV to target path: {e}")

    # Run the unchanged script and parse its output
    try:
        txt = run_core_and_capture()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation script failed: {e}")

    # Build positions/exposures/overview from uploaded CSV
    try:
        df = dataframe_from_csv_text(inp.csv)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    risk = parse_stdout(txt)
    factors = parse_factors_from_stdout(txt)
    exposures = exposures_from_df(df)
    positions = positions_from_df(df)
    overview = overview_from_df(df, risk)

    return {
        "ok": True,
        "data": {
            "positions": positions,
            "overview": overview,
            "risk": risk,
            "factors": factors,
            "exposures": exposures,
        },
    }


@app.post("/api/optimize")
def optimize(inp: OptimizeIn):
    """
    Just re-run your script and try to parse the optimizer prints.
    No changes to your optimizer code.
    """
    if not inp.csv or not inp.csv.strip():
        raise HTTPException(status_code=400, detail="Missing CSV text.")

    try:
        write_csv_to_target(inp.csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write CSV to target path: {e}")

    try:
        txt = run_core_and_capture()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation script failed: {e}")

    out = parse_optimizer_from_stdout(txt)
    return {"ok": True, "data": out}


# --------------------------------------------------------------------------------------
# Local dev entrypoint (optional)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("adapter:app", host="0.0.0.0", port=8000, reload=True)
