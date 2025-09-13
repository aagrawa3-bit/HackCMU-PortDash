# logic.py
import io
import json
import math
import warnings
from datetime import datetime, timedelta

import certifi
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf
from urllib.request import urlopen

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

FMP_API_KEY = "HlLnwPHGLYdEUVKc6vEOY4e9SBDO9b7R"

# ------------------------------
# Helpers (IO / indexing only)
# ------------------------------
def _get_jsonparsed_data(url: str):
    resp = urlopen(url, cafile=certifi.where())
    data = resp.read().decode("utf-8")
    return json.loads(data)

def _get_col(df: pd.DataFrame, field: str, symbol: str = None) -> pd.Series:
    """
    Access OHLCV columns whether yf returns single-level or MultiIndex.
    Keeps math identical to user code, only fixes indexing shape.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if symbol is None:
            raise ValueError("symbol required when columns are MultiIndex")
        return df[(field, symbol)]
    else:
        return df[field]

def _download_ohlc(symbol, **kwargs) -> pd.DataFrame:
    df = yf.download(symbol, **kwargs)
    if df is None or len(df) == 0:
        raise ValueError(f"Empty data for {symbol}")
    # normalize columns to single-level for further ops
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ------------------------------
# EXACT math from your functions
# (formulas unchanged; only indexing robustness added)
# ------------------------------
def target_sl_gkyz_daily(symbol: str) -> float:
    new_download = yf.download(symbol)

    def gkVol(symbol_inner):
        volList = []
        # handle single / multi index consistently
        High = _get_col(new_download, "High", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        Low = _get_col(new_download, "Low", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        Close = _get_col(new_download, "Close", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        Open = _get_col(new_download, "Open", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)

        for i in range(len(new_download)):
            logHighLow = math.log(High.iloc[i] / Low.iloc[i])
            logCloseOpen = math.log(Close.iloc[i] / Open.iloc[i])
            gkVol_valRaw = np.sqrt((0.5 * logHighLow ** 2) - ((2 * math.log(2) - 1)) * logCloseOpen ** 2)
            volList.append(gkVol_valRaw)
        return volList

    def meanChange(symbol_inner):
        meanList = []
        Close = _get_col(new_download, "Close", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        for i in range(1, len(new_download)):
            CloseOpen = (Close.iloc[i] / Close.iloc[i - 1]) - 1
            meanList.append(CloseOpen)
        return meanList

    _ = meanChange(symbol)
    new_download["GK Volatility"] = gkVol(symbol)
    new_download["GK Volatility EMA10"] = new_download["GK Volatility"].ewm(span=10, adjust=False).mean()
    new_download = new_download.dropna()

    def yang_zhang_volatility(df: pd.DataFrame, window: int):
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

    df = yf.download(symbol)
    window = 10
    df = df.dropna()
    new_download["YZ Volatility EMA 10"] = yang_zhang_volatility(df, window)
    # original code indexed with [0]; we just take the scalar last value
    gkVolatility = float(new_download["GK Volatility EMA10"].iloc[-1])
    yzVol = float(new_download["YZ Volatility EMA 10"].iloc[-1])
    return float(np.mean(np.array([gkVolatility, yzVol])))

def target_sl_gkyz_weekly_US(symbol: str, dateEnd: str) -> float:
    new_download = yf.download(symbol, start="2000-01-01", end=dateEnd, interval="1wk")

    def gkVol(symbol_inner):
        volList = []
        High = _get_col(new_download, "High", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        Low = _get_col(new_download, "Low", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        Close = _get_col(new_download, "Close", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        Open = _get_col(new_download, "Open", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)

        for i in range(len(new_download)):
            logHighLow = math.log(High.iloc[i] / Low.iloc[i])
            logCloseOpen = math.log(Close.iloc[i] / Open.iloc[i])
            gkVol_valRaw = np.sqrt((0.5 * logHighLow ** 2) - ((2 * math.log(2) - 1)) * logCloseOpen ** 2)
            volList.append(gkVol_valRaw)
        return volList

    def meanChange(symbol_inner):
        meanList = []
        Close = _get_col(new_download, "Close", symbol_inner if isinstance(new_download.columns, pd.MultiIndex) else None)
        for i in range(1, len(new_download)):
            CloseOpen = (Close.iloc[i] / Close.iloc[i - 1]) - 1
            meanList.append(CloseOpen)
        return meanList

    _ = meanChange(symbol)
    new_download["GK Volatility"] = gkVol(symbol)
    new_download["GK Volatility EMA10"] = new_download["GK Volatility"].ewm(span=10, adjust=False).mean()
    new_download = new_download.dropna()

    def yang_zhang_volatility(df, window):
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

    df = yf.download(symbol, start="2000-01-01", end=dateEnd, interval="1wk").dropna()
    window = 10
    new_download["YZ Volatility EMA 10"] = yang_zhang_volatility(df, window)
    gkVolatility = float(new_download["GK Volatility EMA10"].iloc[-1])
    yzVol = float(new_download["YZ Volatility EMA 10"].iloc[-1])
    return float(np.mean(np.array([gkVolatility, yzVol])))

def target_sl_buy_1(symbol: str) -> float:
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from=2005-01-01&apikey={FMP_API_KEY}"
    data_needed = _get_jsonparsed_data(url)
    historical_data = data_needed.get("historical", [])
    stock_data = pd.DataFrame(historical_data)
    try:
        data = pd.DataFrame(stock_data)[::-1].reset_index(drop=True)
        data.columns = data.columns.get_level_values(0)
        data.rename(columns={"close": "Close", "adjClose": "close"}, inplace=True)
        daily_volume = data["volume"].pct_change().dropna()
        dates = data["date"].to_list()

        data["EMA20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["EMA50"] = data["close"].ewm(span=50, adjust=False).mean()
        data["EMA100"] = data["close"].ewm(span=100, adjust=False).mean()
        data["EMA200"] = data["close"].ewm(span=200, adjust=False).mean()

        moving_avg = data["volume"].rolling(window=50).mean()
        data = data.dropna()

        date_array = []
        for i in range(1, len(data)):
            if data["changePercent"].iloc[i - 1] >= 2.0:
                stock_info = data.iloc[i]
                date_1 = dates[i - 1]
                low = stock_info["low"]; high = stock_info["high"]
                close = stock_info["close"]; open_stock = stock_info["open"]
                volume = stock_info["volume"]
                change_volume = daily_volume.iloc[i - 1]
                moving_average_volume = moving_avg.iloc[i - 1]

                m20 = stock_info["EMA20"]; m50 = stock_info["EMA50"]
                m100 = stock_info["EMA100"]; m200 = stock_info["EMA200"]

                if moving_average_volume:
                    velocity_volume = ((volume - moving_average_volume) / moving_average_volume) * 100
                    if (velocity_volume > 25) and change_volume > 0.25:
                        if (close - open_stock > 0) and (((high - close) / high) < 0.0010):
                            # Keep original if/elif chain intact (logic unchanged)
                            if ((abs(m20 - low) / close) < 0.0015) and (close > m20):
                                date_array.append(date_1); continue
                            elif (low < m20) and (close > m20):
                                date_array.append(date_1); continue
                            elif (low < m20) and (close > m20):
                                date_array.append(date_1); continue
                            elif (((close - m20) / close) > 0.0025) and (open_stock < m20):
                                date_array.append(date_1); continue
                            elif ((abs(m50 - low) / close) < 0.0015) and (close > m50):
                                date_array.append(date_1); continue
                            elif (low < m50) and (close > m50):
                                date_array.append(date_1); continue
                            elif (low < m50) and (close > m50):
                                date_array.append(date_1); continue
                            elif (((close - m50) / close) > 0.0025) and (open_stock < m50):
                                date_array.append(date_1); continue
                            elif ((abs(m100 - low) / close) < 0.0015) and (close > m100):
                                date_array.append(date_1); continue
                            elif (low < m100) and (close > m100):
                                date_array.append(date_1); continue
                            elif (low < m100) and (close > m100):
                                date_array.append(date_1); continue
                            elif (((close - m100) / close) > 0.0025) and (open_stock < m100):
                                date_array.append(date_1); continue
                            elif ((abs(m200 - low) / close) < 0.0015) and (close > m200):
                                date_array.append(date_1); continue
                            elif (low < m200) and (close > m200):
                                date_array.append(date_1); continue
                            elif (low < m200) and (close > m200):
                                date_array.append(date_1); continue
                            elif (((close - m200) / close) > 0.0050) and (open_stock < m200):
                                date_array.append(date_1); continue

                if open_stock < close:
                    if ((abs(m20 - low) / close) < 0.0015) and (close > m20):
                        date_array.append(date_1); continue
                    elif (low < m20) and (close > m20):
                        date_array.append(date_1); continue
                    elif (low < m20) and (close > m20):
                        date_array.append(date_1); continue
                    elif (((close - m20) / close) > 0.0025) and (open_stock < m20):
                        date_array.append(date_1); continue
                    elif ((abs(m50 - low) / close) < 0.0015) and (close > m50):
                        date_array.append(date_1); continue
                    elif (low < m50) and (close > m50):
                        date_array.append(date_1); continue
                    elif (low < m50) and (close > m50):
                        date_array.append(date_1); continue
                    elif (((close - m50) / close) > 0.0025) and (open_stock < m50):
                        date_array.append(date_1); continue
                    elif ((abs(m100 - low) / close) < 0.0015) and (close > m100):
                        date_array.append(date_1); continue
                    elif (low < m100) and (close > m100):
                        date_array.append(date_1); continue
                    elif (low < m100) and (close > m100):
                        date_array.append(date_1); continue
                    elif (((close - m100) / close) > 0.0025) and (open_stock < m100):
                        date_array.append(date_1); continue
                    elif ((abs(m200 - low) / close) < 0.0015) and (close > m200):
                        date_array.append(date_1); continue
                    elif (low < m200) and (close > m200):
                        date_array.append(date_1); continue
                    elif (low < m200) and (close > m200):
                        date_array.append(date_1); continue
                    elif (((close - m200) / close) > 0.0025) and (open_stock < m200):
                        date_array.append(date_1); continue

        prices_period, current_prices = [], []
        for date_2 in date_array:
            final_prices = []
            index = dates.index(date_2)
            final_index = index + 5
            if final_index > len(data) - 1:
                continue
            current_price = data.iloc[index]["close"]
            current_prices.append(current_price)
            for i in range(index + 1, final_index + 1):
                final_prices.append(data.iloc[i]["close"])
            prices_period.append(final_prices)

        prices_computing = [max(element) for element in prices_period] if len(prices_period) else []
        returns = []
        for i in range(len(current_prices)):
            returns.append((prices_computing[i] / current_prices[i]) - 1)

        mean_prices = float(np.mean(returns)) if len(returns) else 0.0
    except Exception:
        mean_prices = 0.0
    return float(mean_prices)

def target_sl_sell(symbol: str) -> float:
    from datetime import date as _date
    try:
        data = yf.download(symbol, "2000-01-01", _date.today().strftime("%Y-%m-%d"))
        data.columns = data.columns.get_level_values(0)
        daily_returns = data["Close"].pct_change().dropna()

        dates = data.index.to_list()
        dates_list = [element.strftime("%Y-%m-%d") for element in dates]

        data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
        data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
        data["EMA100"] = data["Close"].ewm(span=100, adjust=False).mean()
        data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()

        data["MA_Vol"] = data["Volume"].rolling(window=50).mean()
        data = data.dropna()

        date_array = []
        for i in range(1, len(data)):
            if daily_returns.iloc[i - 1] <= -0.02:
                stock_info = data.iloc[i]
                date_1 = dates_list[i - 1]
                high = stock_info["High"]; close = stock_info["Close"]; open_stock = stock_info["Open"]
                m20 = stock_info["EMA20"]; m50 = stock_info["EMA50"]; m100 = stock_info["EMA100"]; m200 = stock_info["EMA200"]

                if open_stock > close:
                    if ((abs(m20 - high) / close) < 0.0015) and (close < m20):
                        date_array.append(date_1); continue
                    elif (high > m20) and (close < m20):
                        date_array.append(date_1); continue
                    elif (high > m20) and (close < m20):
                        date_array.append(date_1); continue
                    elif ((abs(close - m20) / close) < 0.0025) and (close < m20) and (open_stock > m20):
                        date_array.append(date_1); continue
                    elif ((abs(m50 - high) / close) < 0.0015) and (close < m50):
                        date_array.append(date_1); continue
                    elif (high > m50) and (close < m50):
                        date_array.append(date_1); continue
                    elif (high > m50) and (close < m50):
                        date_array.append(date_1); continue
                    elif ((abs(close - m50) / close) < 0.0025) and (close < m50) and (open_stock > m50):
                        date_array.append(date_1); continue
                    elif ((abs(m100 - high) / close) < 0.0015) and (close < m100):
                        date_array.append(date_1); continue
                    elif (high > m100) and (close < m100):
                        date_array.append(date_1); continue
                    elif (high > m100) and (close < m100):
                        date_array.append(date_1); continue
                    elif ((abs(close - m100) / close) < 0.0025) and (close < m100) and (open_stock > m100):
                        date_array.append(date_1); continue
                    elif ((abs(m200 - high) / close) < 0.0015) and (close < m200):
                        date_array.append(date_1); continue
                    elif (high > m200) and (close < m200):
                        date_array.append(date_1); continue
                    elif (high > m200) and (close < m200):
                        date_array.append(date_1); continue
                    elif ((abs(close - m200) / close) < 0.0025) and (close < m200) and (open_stock > m200):
                        date_array.append(date_1); continue

        prices_period, current_prices = [], []
        for date_2 in date_array:
            final_prices = []
            index = dates_list.index(date_2)
            final_index = index + 5
            if final_index > len(data) - 1:
                continue
            current_price = data.iloc[index]["Close"]
            current_prices.append(current_price)
            for i in range(index + 1, final_index):
                final_prices.append(data.iloc[i]["Close"])
            prices_period.append(final_prices)

        prices_computing = [max(element) for element in prices_period] if len(prices_period) else []
        returns = []
        for i in range(len(current_prices)):
            returns.append((prices_computing[i] / current_prices[i]) - 1)

        mean_prices = -(float(np.mean(returns))) if len(returns) else 0.0
    except Exception:
        mean_prices = 0.0

    return float(mean_prices)

# ------------------------------
# Core entry used by API
# ------------------------------
def compute_metrics_from_csv(
    csv_text: str,
    ff3_csv_text: str,
    mom_txt_text: str,
    date_needed: str | None = None,
):
    """
    Runs your original script's logic end-to-end using the provided CSV and factor files.
    No math/formula changes; only IO is adapted to API inputs.
    """
    # --- Portfolio CSV ---
    portfolio = pd.read_csv(io.StringIO(csv_text))
    if "Capital Allocation" not in portfolio.columns or "Symbol" not in portfolio.columns:
        raise ValueError("CSV must include at least 'Symbol' and 'Capital Allocation' columns.")
    if "Execution" not in portfolio.columns:
        portfolio["Execution"] = "Buy"
    if "Sector" not in portfolio.columns:
        portfolio["Sector"] = "Unknown"
    if "Timeframe" not in portfolio.columns:
        portfolio["Timeframe"] = ""
    if "Target" not in portfolio.columns:
        portfolio["Target"] = 0.0
    if "Stop Loss" not in portfolio.columns:
        portfolio["Stop Loss"] = 0.0
    # Keep names aligned with frontend
    portfolio = portfolio.rename(columns={"Market Capitalisation": "MarketCap", "Volume": "Volume", "Price": "Price"})

    date_needed = date_needed or datetime.utcnow().strftime("%Y-%m-%d")

    # --- Sector exposures (same as script grouping /2) + close vols ---
    symbols = portfolio["Symbol"].astype(str).tolist()
    close_data = yf.download(symbols, start="2024-01-01")["Close"]
    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=symbols[0])
    single_stock_vol = close_data.pct_change().dropna().std()

    total_beta = 0.0
    expected_vol = 0.0
    expected_range = 0.0
    expected_range_daily = 0.0

    np.random.seed(42)

    # sector exposure exactly as your script (sum/2)
    sector_exposure_series = portfolio.groupby("Sector")["Capital Allocation"].sum() / 2.0
    sector_exposure = sector_exposure_series.to_dict()

    # name (symbol) exposure (we'll normalize by abs-sum to get fractions for charts)
    name_exposure_raw = portfolio.groupby("Symbol")["Capital Allocation"].sum() / 2.0
    total_abs = name_exposure_raw.abs().sum()
    name_exposure = (name_exposure_raw / (total_abs if total_abs != 0 else 1.0)).to_dict()

    # Loop for betas, expected vol & ranges using your exact steps
    for i in range(len(portfolio)):
        sym = portfolio["Symbol"].iloc[i]
        execn = portfolio["Execution"].iloc[i]
        capital_allocation = float(portfolio["Capital Allocation"].iloc[i]) / 2.0

        beta_url = f"https://financialmodelingprep.com/stable/profile?symbol={sym}&apikey={FMP_API_KEY}"
        data = _get_jsonparsed_data(beta_url)
        position = 1 if execn == "Buy" else -1
        try:
            beta_val = float(data[0]["beta"])
        except Exception:
            beta_val = 0.0
        total_beta += beta_val * position * capital_allocation

        # script uses FIRST symbol's vol for all positions (kept exactly)
        first_sym = portfolio["Symbol"].iloc[0]
        expected_vol += capital_allocation * float(single_stock_vol.loc[first_sym])

        expected_range += capital_allocation * float(target_sl_gkyz_weekly_US(sym, date_needed))
        expected_range_daily += capital_allocation * float(target_sl_gkyz_daily(sym))

    # ---------------- VaR / CVaR (unchanged math, vectorized for speed) ----------------
    years = 5
    endDate = date_needed
    startDate = datetime(int(endDate.split("-")[0]), int(endDate.split("-")[1]), int(endDate.split("-")[2])) - timedelta(days=365 * years)

    tickers = symbols
    cap_all = portfolio["Capital Allocation"].astype(float).tolist()
    percentages = [perc for perc in cap_all]          # no /100 in original
    percentages_divided = [x for x in percentages]    # unchanged

    # Build price matrix
    adj_close_df = pd.DataFrame()
    for t in tickers:
        data = yf.download(t, start=startDate, end=endDate, interval="1d")
        if data is not None and len(data) > 0:
            adj_close_df[t] = data["Close"]
    adj_close_df = adj_close_df.dropna(axis=1, how="all")
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Align weights (unchanged)
    weight_series = pd.Series(percentages_divided, index=tickers)
    weight_series = weight_series.loc[log_returns.columns]

    portfolio_expected_return = (log_returns.mean() * weight_series).sum()
    cov_matrix = log_returns.cov().loc[weight_series.index, weight_series.index]
    portfolio_std_dev = float(np.sqrt(weight_series.values.T @ cov_matrix.values @ weight_series.values))

    # Monte Carlo (unchanged method; vectorized)
    days = 5
    simulations = 1_000_000
    z = np.random.normal(0, 1, simulations)
    port_val = 100_000_000.0
    scenarioReturn = port_val * float(portfolio_expected_return) * days + port_val * portfolio_std_dev * z * np.sqrt(days)

    # Portfolio VaR/CVaR
    def _var_cvar(arr: np.ndarray, q: float):
        VaR = np.percentile(arr, 100 * q)
        cVaR = arr[arr <= VaR].mean()
        return VaR, cVaR

    VaR5, cVaR5 = _var_cvar(scenarioReturn, 0.05)
    VaR1, cVaR1 = _var_cvar(scenarioReturn, 0.01)
    VaR003, cVaR003 = _var_cvar(scenarioReturn, 0.0003)

    # ---------------- Benchmark SPY tail ----------------
    years_b = 5
    endDate_b = datetime.now()
    startDate_b = endDate_b - timedelta(days=365 * years_b)

    adj_close_df_b = pd.DataFrame()
    data_b = yf.download("SPY", start=startDate_b, end="2025-05-10", interval="1d")
    if data_b is not None and len(data_b) > 0:
        adj_close_df_b["SPY"] = data_b["Close"]
    data_b = yf.download("SPY", start=startDate_b, end=endDate_b, interval="1d")
    if data_b is not None and len(data_b) > 0:
        adj_close_df_b["SPY"] = data_b["Close"]
    adj_close_df_b = adj_close_df_b.dropna(axis=1, how="all")
    log_returns_b = np.log(adj_close_df_b / adj_close_df_b.shift(1)).dropna()

    weight_series_b = pd.Series([1.0], index=["SPY"])
    weight_series_b = weight_series_b.loc[log_returns_b.columns]

    portfolio_expected_return_b = (log_returns_b.mean() * weight_series_b).sum()
    cov_matrix_b = log_returns_b.cov().loc[weight_series_b.index, weight_series_b.index]
    portfolio_std_dev_b = float(np.sqrt(weight_series_b.values.T @ cov_matrix_b.values @ weight_series_b.values))

    z_b = np.random.normal(0, 1, simulations)
    scenarioReturn_b = port_val * float(portfolio_expected_return_b) * days + port_val * portfolio_std_dev_b * z_b * np.sqrt(days)
    _, cVaR003_b = _var_cvar(scenarioReturn_b, 0.0003)

    tail_risk_multiple = float(cVaR003 / cVaR003_b) if cVaR003_b != 0 else 0.0

    # ---------------- Expected return & Sharpes (unchanged) ----------------
    expected_mean_return = 0.0
    for i in range(len(portfolio)):
        capital_allocation = float(portfolio["Capital Allocation"].iloc[i]) / 2.0
        if portfolio["Execution"].iloc[i] == "Buy":
            expected_mean_return += capital_allocation * target_sl_buy_1(portfolio["Symbol"].iloc[i])
        else:
            expected_mean_return += capital_allocation * target_sl_sell(portfolio["Symbol"].iloc[i])

    # NOTE: script prints daily as expected_mean_return/5
    exp_return_daily = expected_mean_return / 5.0
    sharpe_gross = (expected_mean_return / expected_range) * np.sqrt(48) if expected_range != 0 else 0.0
    sharpe_net = ((expected_mean_return - (0.0025 * len(portfolio))) / expected_range) * np.sqrt(48) if expected_range != 0 else 0.0

    # position option risk list (kept for completeness)
    tail_risk = cVaR003
    position_option_risk = []
    for i in range(len(portfolio)):
        capital_allocation = float(portfolio["Capital Allocation"].iloc[i]) / 2.0
        if portfolio["Execution"].iloc[i] == "Buy":
            position_option_risk.append(capital_allocation * tail_risk)
        else:
            position_option_risk.append(capital_allocation * -tail_risk)

    # ---------------- Factor regression (exact workflow) ----------------
    ff3 = pd.read_csv(io.StringIO(ff3_csv_text)).dropna()
    if "Unnamed: 0" in ff3.columns:
        ff3 = ff3.rename(columns={"Unnamed: 0": "Date"})
    if "Date" not in ff3.columns:
        raise ValueError("F-F 3 Factors CSV must have a 'Date' column (YYYYMMDD).")
    # momentum text (space separated)
    df_m = pd.read_csv(
        io.StringIO(mom_txt_text),
        sep=r"\s+",
        header=None,
        names=["Date", "Mom"],
        engine="python",
    )

    ff4 = pd.merge(
        ff3.assign(Date=pd.to_datetime(ff3["Date"].astype(str), format="%Y%m%d")),
        df_m.assign(Date=pd.to_datetime(df_m["Date"].astype(str), format="%Y%m%d"))[["Date", "Mom"]],
        on="Date",
        how="inner",
    )

    # weights
    if "Execution" in portfolio.columns:
        execution_sign = (
            portfolio["Execution"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"buy": 1, "long": 1, "sell": -1, "short": -1})
            .fillna(1)
        )
        signed_allocation = portfolio["Capital Allocation"].astype(float) * execution_sign
    else:
        signed_allocation = portfolio["Capital Allocation"].astype(float)

    weights_by_ticker = pd.Series(signed_allocation.values, index=portfolio["Symbol"].values)
    denom = weights_by_ticker.abs().sum()
    weights_by_ticker = weights_by_ticker / (denom if denom != 0 else 1.0)
    tickers_list = weights_by_ticker.index.tolist()

    start_date_for_prices = pd.to_datetime(ff4["Date"]).min()
    close_prices = yf.download(tickers_list, start=start_date_for_prices, progress=False)["Close"]
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=tickers_list[0])
    daily_simple_returns = close_prices.pct_change().dropna(how="all")

    aligned_signed_weights = weights_by_ticker.reindex(daily_simple_returns.columns).fillna(0.0)
    if aligned_signed_weights.abs().sum() != 0:
        aligned_signed_weights = aligned_signed_weights / aligned_signed_weights.abs().sum()

    portfolio_daily_return = (daily_simple_returns @ aligned_signed_weights).rename("PortfolioReturn")

    factors = ff4.copy()
    factors["Date"] = pd.to_datetime(factors["Date"])
    factors = factors.set_index("Date").sort_index()
    for factor_col in ["Mkt-RF", "SMB", "HML", "Mom", "RF"]:
        factors[factor_col] = pd.to_numeric(factors[factor_col], errors="coerce") / 100.0

    regression_data = pd.concat([portfolio_daily_return, factors], axis=1).dropna()
    excess_portfolio_return = regression_data["PortfolioReturn"] - regression_data["RF"]
    design_matrix = sm.add_constant(regression_data[["Mkt-RF", "SMB", "HML", "Mom"]], has_constant="add")

    carhart_results = sm.OLS(excess_portfolio_return, design_matrix).fit()

    factor_betas = carhart_results.params.drop("const")
    factor_tstats = (carhart_results.params / carhart_results.bse).drop("const")
    beta_pvalues = carhart_results.pvalues.drop("const")

    significance = ["Significant" if p < 0.05 else "Not Significant" for p in beta_pvalues.values]
    summary_table = pd.DataFrame(
        {"Beta": factor_betas, "T-Statistic": factor_tstats, "P-Value": beta_pvalues, "Significance": significance},
        index=factor_betas.index,
    ).reindex(["Mkt-RF", "SMB", "HML", "Mom"])

    # ---------------- Output for frontend ----------------
    # Positions table
    total_alloc_abs = portfolio["Capital Allocation"].abs().sum()
    positions = []
    for _, row in portfolio.iterrows():
        alloc = float(row.get("Capital Allocation", 0.0))
        positions.append({
            "Symbol": str(row.get("Symbol", "")),
            "Execution": str(row.get("Execution", "Buy")),
            "Sector": str(row.get("Sector", "")),
            "MarketCap": float(row.get("MarketCap", 0.0)) if pd.notna(row.get("MarketCap", np.nan)) else 0.0,
            "Price": float(row.get("Price", 0.0)) if pd.notna(row.get("Price", np.nan)) else 0.0,
            "Volume": float(row.get("Volume", 0.0)) if pd.notna(row.get("Volume", np.nan)) else 0.0,
            "Allocation": alloc,
            "AllocationPercent": (abs(alloc) / total_alloc_abs) if total_alloc_abs != 0 else 0.0,
            "Timeframe": str(row.get("Timeframe", "")),
            "Target": float(row.get("Target", 0.0)) if pd.notna(row.get("Target", np.nan)) else 0.0,
            "StopLoss": float(row.get("Stop Loss", 0.0)) if pd.notna(row.get("Stop Loss", np.nan)) else 0.0,
        })

    long_count = int((portfolio["Execution"].astype(str).str.lower() == "buy").sum())
    short_count = int((portfolio["Execution"].astype(str).str.lower() == "sell").sum())

    data_out = {
        "overview": {
            "portfolioValue": 100_000_000 * (total_alloc_abs / total_alloc_abs if total_alloc_abs else 0),
            "totalAllocationFrac": float(total_alloc_abs / total_alloc_abs) if total_alloc_abs else 0.0,
            "activePositions": int(len(portfolio)),
            "longCount": long_count,
            "shortCount": short_count,
            "beta": float(total_beta),
        },
        "exposures": {
            "sectors": {k: float(v) / total_abs if total_abs != 0 else 0.0 for k, v in sector_exposure.items()},
            "names": {k: float(v) for k, v in name_exposure.items()},
        },
        "risk": {
            "portfolioBeta": float(total_beta),
            "dailyVol": float(expected_vol),
            "weeklyVol": float(expected_vol * np.sqrt(5)),
            "dailyRange": float(expected_range_daily),
            "weeklyRange": float(expected_range),
            "maxDD": float(-expected_range),
            "var5": float(VaR5 / port_val),
            "cvar5": float(cVaR5 / port_val),
            "var1": float(VaR1 / port_val),
            "cvar1": float(cVaR1 / port_val),
            "tailRisk": float(cVaR003 / port_val),
            "tailMultiple": float(tail_risk_multiple),
            "expReturnDaily": float(exp_return_daily),
            "expReturnWeekly": float(expected_mean_return),
            "sharpeGross": float(sharpe_gross),
            "sharpeNet": float(sharpe_net),
            "positionOptionRisk": [float(x) for x in position_option_risk],
        },
        "factors": {
            k: {
                "beta": float(summary_table.loc[k, "Beta"]),
                "t": float(summary_table.loc[k, "T-Statistic"]),
                "p": float(summary_table.loc[k, "P-Value"]),
                "sig": str(summary_table.loc[k, "Significance"]),
            }
            for k in ["Mkt-RF", "SMB", "HML", "Mom"]
            if k in summary_table.index
        },
        "positions": positions,
    }
    return data_out
