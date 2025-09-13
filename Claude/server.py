import os
import io
import re
import csv
import json
import tempfile
import subprocess
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Configuration ----------
# Point these at your environment. Defaults match your script paths.
LOGIC_PATH = os.environ.get("LOGIC_PATH", os.path.abspath("./logic.py"))
DEST_CSV_PATH = os.environ.get(
    "DEST_CSV_PATH",
    "/Users/adityaagrawal/Downloads/US_equity_2025-08-29.csv"
)

# Factor files: your script references absolute paths.
# Ensure these exist where your server runs:
#   /Users/adityaagrawal/Desktop/F-F_Research_Data_Factors_daily 3.csv
#   /Users/adityaagrawal/Downloads/F-F_Momentum_Factor_daily 2.txt

# ---------- FastAPI ----------
app = FastAPI(title="Portfolio Backend Adapter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # serve local HTML file without CORS issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class MetricsRequest(BaseModel):
    csv: str

class OptimizeRequest(BaseModel):
    csv: str
    strategy: str  # "sharpe" or "gmv" (front-end sends "sharpe" or "gmv")

# ---------- Helpers ----------
def ensure_csv_destination_path(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv_to_expected_path(raw_csv: str) -> None:
    ensure_csv_destination_path(DEST_CSV_PATH)
    with open(DEST_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        f.write(raw_csv.strip() + ("\n" if not raw_csv.endswith("\n") else ""))

def run_logic_script() -> Tuple[str, str, int]:
    """
    Execute your logic.py exactly as-is and capture stdout/stderr/returncode.
    """
    proc = subprocess.Popen(
        ["python3", LOGIC_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate()
    return stdout, stderr, proc.returncode

def parse_percent_str_to_frac(s: str) -> float:
    # "12.34%" -> 0.1234
    s = s.strip().replace("%", "")
    try:
        return float(s) / 100.0
    except:
        return 0.0

def parse_float(s: str) -> float:
    try:
        return float(s)
    except:
        return 0.0

def normalize_allocation(value: float) -> float:
    """
    Interpret 'Capital Allocation' scale robustly:
    - <=1.0 -> already fraction
    - >100  -> assume bps (e.g., 1550 -> 15.50%) => /10000
    - else  -> assume % (e.g., 15.5 -> 15.5%) => /100
    """
    if value <= 1.0:
        return value
    if value > 100.0:
        return value / 10000.0
    return value / 100.0

def parse_positions_and_exposures(csv_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    """
    Returns (positions, sector_exposure_frac_by_sector, name_exposure_frac_by_symbol)
    """
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    rows = list(reader)

    positions: List[Dict[str, Any]] = []
    sector_sum: Dict[str, float] = {}
    name_sum: Dict[str, float] = {}

    for r in rows:
        sym = (r.get("Symbol") or "").strip()
        exe = (r.get("Execution") or "").strip()
        sec = (r.get("Sector") or "").strip()
        mcap = parse_float(r.get("Market Capitalisation") or r.get("Market Capitalization") or "0")
        price = parse_float(r.get("Price") or "0")
        vol = parse_float(r.get("Volume") or "0")
        alloc_raw = parse_float(r.get("Capital Allocation") or "0")
        tf = (r.get("Timeframe") or "").strip()
        tgt = parse_float(r.get("Target") or "0")
        sl  = parse_float(r.get("Stop Loss") or "0")

        alloc_frac = normalize_allocation(alloc_raw)

        positions.append({
            "Symbol": sym,
            "Execution": exe,
            "Sector": sec,
            "MarketCap": mcap,
            "Price": price,
            "Volume": vol,
            "Allocation": alloc_raw,
            "AllocationPercent": alloc_frac,
            "Timeframe": tf,
            "Target": tgt,
            "StopLoss": sl
        })

        if sec:
            sector_sum[sec] = sector_sum.get(sec, 0.0) + alloc_frac
        if sym:
            name_sum[sym] = name_sum.get(sym, 0.0) + alloc_frac

    # Normalize exposures to fractions of total (in case CSV doesn't sum to 1)
    total_alloc = sum(name_sum.values()) or 1.0
    sector_exposure = {k: v / total_alloc for k, v in sector_sum.items()}
    name_exposure   = {k: v / total_alloc for k, v in name_sum.items()}

    return positions, sector_exposure, name_exposure

METRIC_PATTERNS = {
    "portfolioBeta": re.compile(r"Portfolio Beta:\s*([-\d\.]+)"),
    "dailyVol": re.compile(r"Expected Daily Volatility:\s*([-\d\.]+)%"),
    "weeklyVol": re.compile(r"Expected Portfolio Volatility:\s*([-\d\.]+)%"),
    "dailyRange": re.compile(r"Expected Portfolio Range \(1 Day\): \+/-([-\d\.]+)%"),
    "weeklyRange": re.compile(r"Expected Portfolio Range \(1 Week\): \+/-([-\d\.]+)%"),
    "maxDD": re.compile(r"Maximum Expected Drawdown:\s*([-\d\.]+)%"),
    "var5": re.compile(r"Portfolio 5% VaR\s+([-\d\.]+)%"),
    "cvar5": re.compile(r"Portfolio 5% CVaR\s+([-\d\.]+)%"),
    "var1": re.compile(r"Portfolio 1% VaR\s+([-\d\.]+)%"),
    "cvar1": re.compile(r"Portfolio 1% CVaR\s+([-\d\.]+)%"),
    "tailRisk": re.compile(r"Tail Risk\s+([-\d\.]+)%"),
    "tailMultiple": re.compile(r"Tail Risk Multiple:\s*([-\d\.]+)"),
    "expReturnDaily": re.compile(r"Expected Return \(Daily\):\s*([-\d\.]+)%"),
    "expReturnWeekly": re.compile(r"Expected Return:\s*([-\d\.]+)%"),
    "sharpeGross": re.compile(r"Annualized Expected Sharpe \(Gross\):\s*([-\d\.]+)"),
    "sharpeNet": re.compile(r"Annualized Expected Sharpe \(Net\):\s*([-\d\.]+)"),
    # Optimizer prints
    "maxSharpeWeights": re.compile(r"Max Sharpe Ratio:\s*\[([^\]]+)\]"),
    "gmvWeights": re.compile(r"GMV Portfolio:\s*\[([^\]]+)\]"),
    "printedTickers": re.compile(r"^\[?['\"]?[A-Za-z\-\.]+['\"]?(?:,\s*['\"][A-Za-z\-\.]+['\"]?)*\]?$", re.MULTILINE),
}

def parse_metrics_from_stdout(stdout: str) -> Dict[str, Any]:
    def pick(name: str, to_frac=False, default=None):
        m = METRIC_PATTERNS[name].search(stdout)
        if not m:
            return default
        val = m.group(1)
        return parse_percent_str_to_frac(val) if to_frac else parse_float(val)

    metrics = {
        "portfolioBeta": pick("portfolioBeta", to_frac=False, default=0.0),
        "dailyVol": pick("dailyVol", to_frac=True, default=0.0),
        "weeklyVol": pick("weeklyVol", to_frac=True, default=0.0),
        "dailyRange": pick("dailyRange", to_frac=True, default=0.0),
        "weeklyRange": pick("weeklyRange", to_frac=True, default=0.0),
        "maxDD": pick("maxDD", to_frac=True, default=0.0),
        "var5": pick("var5", to_frac=True, default=0.0),
        "cvar5": pick("cvar5", to_frac=True, default=0.0),
        "var1": pick("var1", to_frac=True, default=0.0),
        "cvar1": pick("cvar1", to_frac=True, default=0.0),
        "tailRisk": pick("tailRisk", to_frac=True, default=0.0),
        "tailMultiple": pick("tailMultiple", to_frac=False, default=0.0),
        "expReturnDaily": pick("expReturnDaily", to_frac=True, default=0.0),
        "expReturnWeekly": pick("expReturnWeekly", to_frac=True, default=0.0),
        "sharpeGross": pick("sharpeGross", to_frac=False, default=0.0),
        "sharpeNet": pick("sharpeNet", to_frac=False, default=0.0),
    }
    return metrics

def parse_optimizer_from_stdout(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m1 = METRIC_PATTERNS["maxSharpeWeights"].search(stdout)
    if m1:
        try:
            out["maxSharpeWeights"] = [float(x.strip()) for x in m1.group(1).split(",")]
        except:
            out["maxSharpeWeights"] = []
    m2 = METRIC_PATTERNS["gmvWeights"].search(stdout)
    if m2:
        try:
            out["gmvWeights"] = [float(x.strip()) for x in m2.group(1).split(",")]
        except:
            out["gmvWeights"] = []

    return out

def count_long_short(positions: List[Dict[str, Any]]) -> Tuple[int, int]:
    longs = sum(1 for p in positions if (p["Execution"] or "").strip().lower() in ("buy", "long"))
    shorts = sum(1 for p in positions if (p["Execution"] or "").strip().lower() in ("sell", "short"))
    return longs, shorts

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "message": "Adapter is running", "logic_path": LOGIC_PATH}

@app.post("/api/metrics")
def api_metrics(payload: MetricsRequest):
    # 1) Save CSV to the exact path your script expects
    write_csv_to_expected_path(payload.csv)

    # 2) Run your logic script as-is and capture output
    stdout, stderr, code = run_logic_script()
    if code != 0:
        return {"ok": False, "error": f"Logic script failed (exit={code})", "stderr": stderr}

    # 3) Parse positions + exposures from the same CSV we just saved
    positions, sector_exposure, name_exposure = parse_positions_and_exposures(payload.csv)
    total_alloc_frac = sum(p["AllocationPercent"] for p in positions)
    long_count, short_count = count_long_short(positions)

    # 4) Parse risk/perf metrics from the script's printed output
    risk = parse_metrics_from_stdout(stdout)

    # 5) Compose overview for the UI
    overview = {
        "portfolioValue": 100_000_000 * total_alloc_frac,  # matches front-end expectation
        "totalAllocationFrac": total_alloc_frac,
        "activePositions": len(positions),
        "longCount": long_count,
        "shortCount": short_count,
        "beta": risk.get("portfolioBeta", 0.0),
    }

    # 6) Return in the shape your front-end expects
    return {
        "ok": True,
        "data": {
            "positions": positions,
            "overview": overview,
            "risk": risk,
            # Factor table is printed as a DataFrame; parsing that reliably from stdout is brittle.
            # If you want it, we can add a second adapter that reads a JSON dump written by your script (still without changing its logic).
            # "factors": {...}
            "exposures": {
                "sectors": sector_exposure,
                "names": name_exposure
            }
        },
        "raw": {"stdout": stdout}
    }

@app.post("/api/optimize")
def api_optimize(payload: OptimizeRequest):
    # Save CSV so your logic reads the same file
    write_csv_to_expected_path(payload.csv)

    # Run logic and parse optimizer prints (weights lines)
    stdout, stderr, code = run_logic_script()
    if code != 0:
        return {"ok": False, "error": f"Logic script failed (exit={code})", "stderr": stderr}

    # Extract optimizer outputs
    opt = parse_optimizer_from_stdout(stdout)
    risk = parse_metrics_from_stdout(stdout)  # for currentSharpe fallback

    # Prepare response. Your script prints weights but not frontier/points numbers,
    # so we leave those empty (UI handles gracefully).
    data = {
        "strategy": payload.strategy,
        "currentSharpe": risk.get("sharpeGross", 0.0),
        "optimalSharpe": risk.get("sharpeGross", 0.0),  # same as current (script doesn’t print optimal Sharpe)
        "varianceReductionPct": 0.0,
        "frontier": [],  # not produced by your script
        "currentPoint": None,
        "optimalPoint": None,
        "weights": {
            "maxSharpe": opt.get("maxSharpeWeights", []),
            "gmv": opt.get("gmvWeights", []),
        }
    }

    return {"ok": True, "data": data, "raw": {"stdout": stdout}}
