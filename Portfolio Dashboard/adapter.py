# adapter.py
import io, math, json, traceback
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ---------------- App & CORS ----------------
app = FastAPI(title="Portfolio Adapter", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # file:// and localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "market capitalisation": "Market Capitalisation",
        "market_capitalisation": "Market Capitalisation",
        "marketcap": "Market Capitalisation",
        "market cap": "Market Capitalisation",
        "capital allocation": "Capital Allocation",
        "allocation": "Capital Allocation",
        "stop loss": "Stop Loss",
        "stoploss": "Stop Loss",
        "time frame": "Timeframe",
    }
    renamed = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        renamed[c] = colmap.get(lc, c.strip())
    return df.rename(columns=renamed)

def _parse_csv_text(csv_text: str) -> pd.DataFrame:
    if not csv_text or not csv_text.strip():
        raise ValueError("CSV body is empty.")
    df = pd.read_csv(io.StringIO(csv_text.strip()))
    df = _norm_cols(df)

    required = ["Symbol", "Execution", "Sector", "Capital Allocation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {required}")

    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["Execution"] = df["Execution"].astype(str).str.strip().str.title()

    if "Market Capitalisation" not in df.columns:
        df["Market Capitalisation"] = np.nan
    for c in ["Price", "Volume", "Timeframe", "Target", "Stop Loss"]:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["Market Capitalisation", "Price", "Volume", "Capital Allocation", "Target", "Stop Loss"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _yf_close(tickers: List[str], start=None, end=None, interval="1d") -> pd.DataFrame:
    if not isinstance(tickers, (list, tuple, set)):
        tickers = [tickers]
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers, start=start, end=end, interval=interval, progress=False,
        auto_adjust=False, group_by="ticker", threads=True
    )

    if isinstance(df.columns, pd.MultiIndex):
        # Try field-first then ticker-first
        try:
            close = df["Close"]
            return close
        except KeyError:
            pass
        try:
            close = df.xs("Close", axis=1, level=1)
            close.columns = [str(c) for c in close.columns]
            return close
        except Exception:
            for lvl in range(df.columns.nlevels):
                if "Close" in df.columns.levels[lvl]:
                    close = df.xs("Close", axis=1, level=lvl)
                    close.columns = [str(c) for c in close.columns]
                    return close
            raise RuntimeError("Unable to locate 'Close' in yfinance result (MultiIndex).")
    else:
        if "Close" in df.columns:
            out = df[["Close"]].copy()
            name = tickers[0] if len(tickers) else "TICKER"
            out.columns = [name]
            return out
        if isinstance(df, pd.Series):
            s = df.copy()
            name = tickers[0] if len(tickers) else "TICKER"
            if s.name != "Close":
                raise RuntimeError(f"Unexpected yfinance Series name: {s.name}")
            return s.to_frame(name=name)
        raise RuntimeError("Unexpected yfinance shape (no 'Close').")

def _yf_ohlc(symbol: str, start=None, end=None, interval="1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data from yfinance for {symbol}")
    needed = ["Open", "High", "Low", "Close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns for {symbol}: {missing}")
    return df

def _gk_series(ohlc: pd.DataFrame) -> pd.Series:
    hl = np.log(ohlc["High"] / ohlc["Low"])
    co = np.log(ohlc["Close"] / ohlc["Open"])
    return np.sqrt(0.5 * (hl**2) - (2 * math.log(2) - 1) * (co**2))

def _yang_zhang(ohlc: pd.DataFrame, window: int) -> pd.Series:
    df = ohlc.dropna()
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    log_oc = np.log(df["Close"] / df["Open"])
    log_co = np.log(df["Close"] / df["Open"].shift(1))
    log_oo = np.log(df["Open"] / df["Close"].shift(1))
    rs = 0.5 * (log_ho - log_lo) ** 2
    rs_vol = (log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)).rolling(window).mean()
    close_vol = log_co.rolling(window).var()
    overnight_vol = log_oo.rolling(window).var()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = overnight_vol + k * close_vol + (1 - k) * rs_vol
    return np.sqrt(yz)

def _expected_ranges_for_symbol(symbol: str, date_end: str) -> Dict[str, float]:
    # Daily
    daily = _yf_ohlc(symbol, interval="1d")
    gk_d = _gk_series(daily).ewm(span=10, adjust=False).mean().dropna()
    yz_d = _yang_zhang(daily, 10).dropna()
    daily_avg = float(np.mean([gk_d.iloc[-1], yz_d.iloc[-1]]))

    # Weekly
    weekly = _yf_ohlc(symbol, start="2000-01-01", end=date_end, interval="1wk")
    gk_w = _gk_series(weekly).ewm(span=10, adjust=False).mean().dropna()
    yz_w = _yang_zhang(weekly, 10).dropna()
    weekly_avg = float(np.mean([gk_w.iloc[-1], yz_w.iloc[-1]]))
    return {"daily": daily_avg, "weekly": weekly_avg}

def _exposures(port: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cap = port["Capital Allocation"].astype(float)
    tot = cap.abs().sum()
    if tot == 0:
        return {"sectors": {}, "names": {}}
    return {
        "sectors": (cap.groupby(port["Sector"]).sum().abs() / tot).to_dict(),
        "names":   (cap.groupby(port["Symbol"]).sum().abs() / tot).to_dict()
    }

def compute_metrics(port: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overview": {}, "risk": {}, "positions": [], "exposures": {}}

    # Positions block
    for _, r in port.iterrows():
        out["positions"].append({
            "Symbol": r["Symbol"],
            "Execution": r["Execution"],
            "Sector": r["Sector"],
            "MarketCap": float(r.get("Market Capitalisation", 0) or 0),
            "Price": float(r.get("Price", 0) or 0),
            "Volume": float(r.get("Volume", 0) or 0),
            "Allocation": float(r.get("Capital Allocation", 0) or 0),
            "AllocationPercent": 0.0,
            "Timeframe": str(r.get("Timeframe", "") or ""),
            "Target": float(r.get("Target", 0) or 0),
            "StopLoss": float(r.get("Stop Loss", 0) or 0),
        })
    tot_abs = port["Capital Allocation"].abs().sum()
    for p in out["positions"]:
        p["AllocationPercent"] = (abs(p["Allocation"]) / tot_abs) if tot_abs else 0.0

    # Overview
    long_count = int((port["Execution"].str.lower().isin(["buy", "long"])).sum())
    short_count = int((port["Execution"].str.lower().isin(["sell", "short"])).sum())
    out["overview"] = {
        "portfolioValue": float(tot_abs * 1e5),  # display only
        "totalAllocationFrac": 1.0 if tot_abs > 0 else 0.0,
        "activePositions": int(len(port)),
        "longCount": long_count,
        "shortCount": short_count,
        "beta": None,  # filled below
    }

    # Risk calc per your structure
    symbols = port["Symbol"].tolist()
    date_needed = pd.Timestamp.today().strftime("%Y-%m-%d")

    close_df = _yf_close(symbols, start="2024-01-01", interval="1d")
    ret = close_df.pct_change().dropna(how="all")
    sym_vol = ret.std()  # per symbol

    beta_total = 0.0
    expected_vol = 0.0
    daily_range = 0.0
    weekly_range = 0.0

    for _, row in port.iterrows():
        sym = row["Symbol"]
        position = 1 if str(row["Execution"]).lower() in ["buy", "long"] else -1
        capital_allocation = float(row["Capital Allocation"]) / 2.0

        est_beta = 1.0  # replace with your FMP call if desired
        beta_total += est_beta * position * capital_allocation

        sv = float(sym_vol.get(sym, np.nan))
        if not np.isnan(sv):
            expected_vol += capital_allocation * sv

        ranges = _expected_ranges_for_symbol(sym, date_needed)
        daily_range += capital_allocation * ranges["daily"]
        weekly_range += capital_allocation * ranges["weekly"]

    out["overview"]["beta"] = float(beta_total)

    out["risk"]["portfolioBeta"] = float(beta_total)
    out["risk"]["dailyVol"]   = float(expected_vol) if expected_vol else 0.0
    out["risk"]["weeklyVol"]  = float(expected_vol * np.sqrt(5)) if expected_vol else 0.0
    out["risk"]["dailyRange"] = float(daily_range) if daily_range else 0.0
    out["risk"]["weeklyRange"]= float(weekly_range) if weekly_range else 0.0
    out["risk"]["maxDD"]      = float(-weekly_range) if weekly_range else 0.0

    # Simple closed-form VaR proxies (you can swap in your heavy MC later)
    if expected_vol and np.isfinite(expected_vol):
        mu = 0.0; sigma = expected_vol; days = 5
        z5, z1 = 1.645, 2.326
        var5 = -(mu*days - z5*sigma*np.sqrt(days))
        var1 = -(mu*days - z1*sigma*np.sqrt(days))
        out["risk"].update({
            "var5": -var5, "cvar5": -var5*1.2,
            "var1": -var1, "cvar1": -var1*1.2,
            "tailRisk": -var1*1.6, "tailMultiple": 1.0
        })
    else:
        out["risk"].update({k: 0.0 for k in ["var5","cvar5","var1","cvar1","tailRisk","tailMultiple"]})

    out["risk"].update({
        "expReturnDaily": 0.0,
        "expReturnWeekly": 0.0,
        "sharpeGross": 0.0,
        "sharpeNet": 0.0
    })

    out["exposures"] = _exposures(port)
    return out

# ---------------- Routes ----------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

def _extract_csv_from_request_body(raw: bytes, content_type: str) -> str:
    """
    Accepts JSON {csv: "..."} or raw text/plain body.
    """
    if "application/json" in (content_type or "").lower():
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON: {e}")
        if not isinstance(data, dict) or "csv" not in data:
            raise ValueError("JSON body must be an object with a 'csv' field.")
        if not isinstance(data["csv"], str):
            raise ValueError("'csv' must be a string.")
        return data["csv"]
    else:
        # treat as raw CSV text
        return raw.decode("utf-8")

@app.post("/api/metrics")
async def api_metrics(request: Request):
    try:
        raw = await request.body()
        csv_text = _extract_csv_from_request_body(raw, request.headers.get("content-type", ""))
        port = _parse_csv_text(csv_text)
        data = compute_metrics(port)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

@app.post("/api/optimize")
async def api_optimize(request: Request):
    try:
        raw = await request.body()
        csv_text = _extract_csv_from_request_body(raw, request.headers.get("content-type", ""))
        port = _parse_csv_text(csv_text)
        # Stub outputs (you can wire your optimizer here)
        return {
            "ok": True,
            "data": {
                "currentSharpe": 0.0,
                "optimalSharpe": 0.0,
                "varianceReductionPct": 0.0,
                "frontier": [],
                "currentPoint": {"x": 0.0, "y": 0.0},
                "optimalPoint": {"x": 0.0, "y": 0.0},
                "exposures": _exposures(port),
            }
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
