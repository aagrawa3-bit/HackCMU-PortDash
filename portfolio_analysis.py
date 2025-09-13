import requests
import yfinance as yf
import math, numpy as np, pandas as pd
from datetime import datetime, timedelta
from urllib.request import urlopen
import certifi
import json
import warnings

def target_sl_gkyz_daily(symbol):
    new_download = yf.download(symbol)

    def gkVol(symbol):
      volList = []
      for i in range(len(new_download)):
        logHighLow = math.log(new_download['High'][symbol].iloc[i]/new_download['Low'][symbol][i])
        logCloseOpen = math.log(new_download['Close'][symbol].iloc[i]/new_download['Open'][symbol][i])
        gkVol_valRaw = np.sqrt((0.5*logHighLow*2) - ((2 * math.log(2) - 1))*logCloseOpen*2)
        volList.append(gkVol_valRaw)
      return volList

    def meanChange(symbol):
      meanList = []
      for i in range(1,len(new_download)):
        CloseOpen = (new_download['Close'][symbol].iloc[i]/new_download['Close'][symbol][i-1]) - 1
        meanList.append(CloseOpen)
      return meanList

    meanList = meanChange(symbol)
    new_download['GK Volatility'] = gkVol(symbol)
    new_download['GK Volatility EMA10'] = new_download['GK Volatility'].ewm(span=10, adjust=False).mean()
    new_download = new_download.dropna()

    def yang_zhang_volatility(df, window):
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_oc = np.log(df['Close'] / df['Open'])
        log_co = np.log(df['Close'] / df['Open'].shift(1))
        log_oo = np.log(df['Open'] / df['Close'].shift(1))

        rs = 0.5 * (log_ho - log_lo) ** 2
        rs_vol = (log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)).rolling(window).mean()
        close_vol = log_co.rolling(window).var()
        overnight_vol = log_oo.rolling(window).var()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = overnight_vol + k * close_vol + (1 - k) * rs_vol

        return np.sqrt(yz)

    df = yf.download(symbol)
    window = 10
    new_download['YZ Volatility EMA 10'] = yang_zhang_volatility(df, window)
    gkVolatility, yzVol = new_download.iloc[-1]['GK Volatility EMA10'][0], new_download.iloc[-1]['YZ Volatility EMA 10'][0]

    return np.mean(np.array([gkVolatility, yzVol]))

def target_sl_gkyz_weekly_US(symbol, dateEnd):
    new_download = yf.download(symbol, start= "2000-01-01", end=dateEnd, interval='1wk')

    def gkVol(symbol):
      volList = []
      for i in range(len(new_download)):
        logHighLow = math.log(new_download['High'][symbol].iloc[i]/new_download['Low'][symbol][i])
        logCloseOpen = math.log(new_download['Close'][symbol].iloc[i]/new_download['Open'][symbol][i])
        gkVol_valRaw = np.sqrt((0.5*logHighLow*2) - ((2 * math.log(2) - 1))*logCloseOpen*2)
        volList.append(gkVol_valRaw)
      return volList

    def meanChange(symbol):
      meanList = []
      for i in range(1,len(new_download)):
        CloseOpen = (new_download['Close'][symbol].iloc[i]/new_download['Close'][symbol][i-1]) - 1
        meanList.append(CloseOpen)
      return meanList

    meanList = meanChange(symbol)
    new_download['GK Volatility'] = gkVol(symbol)
    new_download['GK Volatility EMA10'] = new_download['GK Volatility'].ewm(span=10, adjust=False).mean()
    new_download = new_download.dropna()

    def yang_zhang_volatility(df, window):

        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_oc = np.log(df['Close'] / df['Open'])
        log_co = np.log(df['Close'] / df['Open'].shift(1))
        log_oo = np.log(df['Open'] / df['Close'].shift(1))

        rs = 0.5 * (log_ho - log_lo) ** 2
        rs_vol = (log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)).rolling(window).mean()
        close_vol = log_co.rolling(window).var()
        overnight_vol = log_oo.rolling(window).var()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = overnight_vol + k * close_vol + (1 - k) * rs_vol

        return np.sqrt(yz)

    df = yf.download(symbol, start= "2000-01-01", end=dateEnd, interval='1wk')
    window = 10
    new_download['YZ Volatility EMA 10'] = yang_zhang_volatility(df, window)
    gkVolatility, yzVol = new_download.iloc[-1]['GK Volatility EMA10'][0], new_download.iloc[-1]['YZ Volatility EMA 10'][0]


    return np.mean(np.array([gkVolatility, yzVol]))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

portfolio = pd.read_csv('US_equity_2025-08-29.csv')
date_needed = '2025-08-29'

sectors = []
sector_exposure = []

close_data = yf.download(portfolio['Symbol'].to_list(), start='2024-01-01')['Close']
single_stock_vol = close_data.pct_change().dropna().std()

total_beta = 0
expected_vol = 0
expected_range = 0
expected_range_daily = 0

np.random.seed(42)

for i in range(len(portfolio)):
    beta_url = f"https://financialmodelingprep.com/stable/profile?symbol={portfolio['Symbol'].iloc[i]}&apikey=HlLnwPHGLYdEUVKc6vEOY4e9SBDO9b7R"
    data = get_jsonparsed_data(beta_url)
    position = 1 if portfolio['Execution'].iloc[i] == "Buy" else -1
    capital_allocation = (portfolio['Capital Allocation'].iloc[i])/2
    total_beta += (data[0]['beta'] * position * capital_allocation)

    sector_exposure = portfolio.groupby('Sector')['Capital Allocation'].sum()/2
    expected_vol += capital_allocation * (single_stock_vol.loc[portfolio['Symbol'].iloc[i]])

    expected_range += capital_allocation * target_sl_gkyz_weekly_US(portfolio['Symbol'].iloc[i], date_needed)
    expected_range_daily += capital_allocation * target_sl_gkyz_daily(portfolio['Symbol'].iloc[i])


#print(sector_exposure)
#print(single_stock_vol)

print(f'Portfolio Beta: {total_beta:.2f}')
print(f'Expected Daily Volatility: {expected_vol:.2%}')
print(f'Expected Portfolio Volatility: {expected_vol * np.sqrt(5):.2%}')
print(f'Expected Portfolio Range (1 Day): +/-{expected_range:.2%}')
print(f'Expected Portfolio Range (1 Week): +/-{expected_range:.2%}')
print(f'Maximum Expected Drawdown: {-expected_range:.2%}')

years = 5

endDate = date_needed
startDate = (datetime(int(endDate.split("-")[0]), int(endDate.split("-")[1]), int(endDate.split("-")[2]))
             - timedelta(days=365 * years))

tickers = portfolio['Symbol'].to_list()
cap_all = portfolio['Capital Allocation'].to_list()

percentages = [perc for perc in cap_all]

percentages_divided = [x for x in percentages]

valid_tickers = []

adj_close_df = pd.DataFrame().dropna(axis=1)

for ticker in tickers:
    try:
        data = yf.download(ticker, start=startDate, end=endDate, interval="1d")
        adj_close_df[ticker] = data['Close']
        valid_tickers.append(tickers)
    except:
        data = yf.download(ticker, start=startDate, end=endDate, interval="1d")
        adj_close_df[ticker] = data['Close']
        valid_tickers.append(tickers)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

for ticker in tickers:
    try:
        data = yf.download(ticker, start=startDate, end=endDate, interval="1d")
        adj_close_df[ticker] = data['Close']
    except:
        data = yf.download(ticker, start=startDate, end=endDate, interval="1d")
        adj_close_df[ticker] = data['Close']

adj_close_df = adj_close_df.dropna(axis=1, how='all')

log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

weight_series = pd.Series(percentages_divided, index=tickers)

weight_series = weight_series.loc[log_returns.columns]

portfolio_expected_return = (log_returns.mean() * weight_series).sum()

cov_matrix = log_returns.cov().loc[weight_series.index, weight_series.index]
portfolio_std_dev = np.sqrt(weight_series.values.T @ cov_matrix.values @ weight_series.values)

def random_z_score():
    return np.random.normal(0, 1)

days = 5

def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(
        days)

simulations = 1000000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(100000000, portfolio_std_dev, z_score, days))

confidence_interval = 0.05
VaR = np.percentile(scenarioReturn, 100 * confidence_interval)
scenarioReturn = np.array(scenarioReturn)

cVaR = scenarioReturn[scenarioReturn <= VaR].mean()
print(f'Portfolio 5% VaR {"":<8} {VaR/100000000:.2%}')
print(f'Portfolio 5% CVaR {"":<8} {cVaR/100000000:.2%}')

print(f'Mean Return: {scenarioReturn.mean()}')

confidence_interval = 0.01
VaR = np.percentile(scenarioReturn, 100 * confidence_interval)
cVaR = scenarioReturn[scenarioReturn <= VaR].mean()
print(f'Portfolio 1% VaR {"":<8} {VaR/100000000:.2%}')
print(f'Portfolio 1% CVaR {"":<8} {cVaR/100000000:.2%}')

confidence_interval = 0.0003
VaR = np.percentile(scenarioReturn, 100 * confidence_interval)
cVaR_tail = scenarioReturn[scenarioReturn <= VaR].mean()
print(f'Tail Risk {"":<8} {cVaR_tail/100000000:.2%}')


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

years = 5

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)

tickers = [
    "SPY"
]

percentages = [100]

percentages_divided = [x / 100 for x in percentages]

valid_tickers = []

adj_close_df = pd.DataFrame().dropna(axis=1)

for ticker in tickers:
    data = yf.download(ticker, start = startDate, end = "2025-05-10", interval="1d")
    adj_close_df[ticker] = data['Close']
    valid_tickers.append(tickers)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)

def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

for ticker in tickers:
    data = yf.download(ticker, start=startDate, end=endDate, interval="1d")
    adj_close_df[ticker] = data['Close']

adj_close_df = adj_close_df.dropna(axis=1, how='all')

log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

weight_series = pd.Series(percentages_divided, index=tickers)

weight_series = weight_series.loc[log_returns.columns]

portfolio_expected_return = (log_returns.mean() * weight_series).sum()

cov_matrix = log_returns.cov().loc[weight_series.index, weight_series.index]
portfolio_std_dev = np.sqrt(weight_series.values.T @ cov_matrix.values @ weight_series.values)

def random_z_score():
    return np.random.normal(0, 1)

days = 5

def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days)

simulations = 1000000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(100000000, portfolio_std_dev, z_score, days))

confidence_interval = 0.05
VaR = np.percentile(scenarioReturn, 100 * confidence_interval)
scenarioReturn = np.array(scenarioReturn)
cVaR = scenarioReturn[scenarioReturn <= VaR].mean()

print(f'Benchmark 5% VaR: {"":<8} {VaR/100000000:.2%}')
print(f'Benchmark 5% CVaR: {"":<8} {cVaR/100000000:.2%}')

confidence_interval = 0.01
VaR = np.percentile(scenarioReturn, 100 * confidence_interval)
cVaR = scenarioReturn[scenarioReturn <= VaR].mean()
print(f'Benchmark 1% VaR: {"":<8} {VaR/100000000:.2%}')
print(f'Benchmark 1% CVaR: {"":<8} {cVaR/100000000:.2%}')

confidence_interval = 0.0003
VaR = np.percentile(scenarioReturn, 100 * confidence_interval)
cVaR_tail_benchmark = scenarioReturn[scenarioReturn <= VaR].mean()
print(f'Tail Risk Multiple: {cVaR_tail/cVaR_tail_benchmark:.2f}')


def target_sl_buy_1(symbol):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from=2005-01-01&apikey=HlLnwPHGLYdEUVKc6vEOY4e9SBDO9b7R"
    data_needed = get_jsonparsed_data(url)
    historical_data = data_needed.get("historical", [])
    stock_data = pd.DataFrame(historical_data)
    close_price_current = (stock_data.iloc[0]['close'])
    try:
        data = stock_data
        data = pd.DataFrame(data)[::-1].reset_index(drop=True)
        data.columns = data.columns.get_level_values(0)
        data.rename(columns={"close": "Close", 'adjClose':'close'})
        #daily_returns = data["close"].pct_change().dropna()
        daily_volume = data["volume"].pct_change().dropna()
        change_percent = data['changePercent'].to_list()
        dates = data['date'].to_list()
        dates_list = dates

        data['EMA20'] = data['close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['close'].ewm(span=50, adjust=False).mean()
        data['EMA100'] = data['close'].ewm(span=100, adjust=False).mean()
        data['EMA200'] = data['close'].ewm(span=200, adjust=False).mean()

        moving_avg = data['volume'].rolling(window=50).mean()
        data = data.dropna()

        date_array = []

        for i in range(1, len(data)):
            if data['changePercent'].iloc[i-1] >= 2.0:
                stock_info = data.iloc[i]
                date_1 = dates_list[i-1]
                low = stock_info["low"]
                high = stock_info["high"]
                close = stock_info["close"]
                open_stock = stock_info["open"]
                volume = stock_info["volume"]
                change_volume = daily_volume.iloc[i-1]
                moving_average_volume = moving_avg.iloc[i-1]

                moving_average_20 = stock_info['EMA20']
                moving_average_50 = stock_info['EMA50']
                moving_average_100 = stock_info["EMA100"]
                moving_average_200 = stock_info['EMA200']

                if moving_average_volume:
                    velocity_volume = ((volume - moving_average_volume) / moving_average_volume) * 100
                    if (velocity_volume > 25) and change_volume > 0.25:
                        if (close - open_stock > 0) and (((high - close) / high) < 0.0010):
                            if ((abs(moving_average_20 - low) / close) < 0.0015) and (close > moving_average_20):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_20) and (close > moving_average_20):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_20) and (close > moving_average_20):
                                date_array.append(date_1)
                                continue
                            elif (((close - moving_average_20) / close) > 0.0025) and (open_stock < moving_average_20):
                                date_array.append(date_1)
                                continue

                            elif ((abs(moving_average_50 - low) / close) < 0.0015) and (close > moving_average_50):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_50) and (close > moving_average_50):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_50) and (close > moving_average_50):
                                date_array.append(date_1)
                                continue
                            elif (((close - moving_average_50) / close) > 0.0025) and (open_stock < moving_average_50):
                                date_array.append(date_1)
                                continue

                            elif ((abs(moving_average_100 - low) / close) < 0.0015) and (close > moving_average_100):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_100) and (close > moving_average_100):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_100) and (close > moving_average_100):
                                date_array.append(date_1)
                                continue
                            elif (((close - moving_average_100) / close) > 0.0025) and (open_stock < moving_average_100):
                                date_array.append(date_1)
                                continue

                            elif ((abs(moving_average_200 - low) / close) < 0.0015) and (close > moving_average_200):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_200) and (close > moving_average_200):
                                date_array.append(date_1)
                                continue
                            elif (low < moving_average_200) and (close > moving_average_200):
                                date_array.append(date_1)
                                continue
                            elif (((close - moving_average_200) / close) > 0.0050) and (open_stock < moving_average_200):
                                date_array.append(date_1)
                                continue
                            else:
                                pass

                if open_stock < close:
                    if ((abs(moving_average_20 - low) / close) < 0.0015) and (close > moving_average_20):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_20) and (close > moving_average_20):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_20) and (close > moving_average_20):
                            date_array.append(date_1)
                            continue
                    elif (((close - moving_average_20) / close) > 0.0025) and (open_stock < moving_average_20):
                            date_array.append(date_1)
                            continue

                    elif ((abs(moving_average_50 - low) / close) < 0.0015) and (close > moving_average_50):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_50) and (close > moving_average_50):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_50) and (close > moving_average_50):
                            date_array.append(date_1)
                            continue
                    elif (((close - moving_average_50) / close) > 0.0025) and (open_stock < moving_average_50):
                            date_array.append(date_1)
                            continue

                    elif ((abs(moving_average_100 - low) / close) < 0.0015) and (close > moving_average_100):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_100) and (close > moving_average_100):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_100) and (close > moving_average_100):
                            date_array.append(date_1)
                            continue
                    elif (((close - moving_average_100) / close) > 0.0025) and (open_stock < moving_average_100):
                            date_array.append(date_1)
                            continue

                    elif ((abs(moving_average_200 - low) / close) < 0.0015) and (close > moving_average_200):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_200) and (close > moving_average_200):
                            date_array.append(date_1)
                            continue
                    elif (low < moving_average_200) and (close > moving_average_200):
                            date_array.append(date_1)
                            continue
                    elif (((close - moving_average_200) / close) > 0.0025) and (open_stock < moving_average_200):
                            date_array.append(date_1)
                            continue
                    else:
                        pass

        prices_period = []
        current_prices = []

        for date_2 in date_array:
            final_prices = []
            index = dates_list.index(date_2)
            final_index = index + 5
            if final_index > len(data):
                continue
            current_price = data.iloc[index]['close']
            current_prices.append(current_price)
            for i in range(index + 1, final_index + 1):
                final_prices.append(data.iloc[i]["close"])

            prices_period.append(final_prices)

        prices_computing = []

        for element in prices_period:
            prices_computing.append(max(element))

        returns = []

        for i in range(len(current_prices)):
            returns.append((prices_computing[i]/current_prices[i])-1)

        mean_prices = np.mean(returns)
        standard_dev = np.std(returns)

    except:
        mean_prices = 0


    return mean_prices


def target_sl_sell(symbol):
    from datetime import date
    try:
        data = yf.download(symbol, "2000-01-01", date.today().strftime("%Y-%m-%d"))

        data.columns = data.columns.get_level_values(0)
        close_price = data['Close'].iloc[-1]
        daily_returns = data["Close"].pct_change().dropna()

        dates = data.index.to_list()
        dates_list = [element.strftime("%Y-%m-%d") for element in dates]

        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

        moving_avg = data['Volume'].rolling(window=50).mean()
        data = data.dropna()

        date_array = []

        for i in range(1, len(data)):
            if daily_returns.iloc[i-1] <= -0.02:
                stock_info = data.iloc[i]
                date_1 = dates_list[i-1]
                low = stock_info["Low"]
                high = stock_info["High"]
                close = stock_info["Close"]
                open_stock = stock_info["Open"]

                moving_average_20 = stock_info['EMA20']
                moving_average_50 = stock_info['EMA50']
                moving_average_100 = stock_info["EMA100"]
                moving_average_200 = stock_info['EMA200']

                if open_stock > close:
                    if ((abs(moving_average_20 - high) / close) < 0.0015) and (close < moving_average_20):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_20) and (close < moving_average_20):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_20) and (close < moving_average_20):
                        date_array.append(date_1)
                        continue
                    elif ((abs(close - moving_average_20) / close) < 0.0025) and (close < moving_average_20) and (
                            open_stock > moving_average_20):
                        date_array.append(date_1)
                        continue

                    elif ((abs(moving_average_50 - high) / close) < 0.0015) and (close < moving_average_50):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_50) and (close < moving_average_50):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_50) and (close < moving_average_50):
                        date_array.append(date_1)
                        continue
                    elif ((abs(close - moving_average_50) / close) < 0.0025) and (close < moving_average_50) and (
                            open_stock > moving_average_50):
                        date_array.append(date_1)
                        continue

                    elif ((abs(moving_average_100 - high) / close) < 0.0015) and (close < moving_average_100):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_100) and (close < moving_average_100):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_100) and (close < moving_average_100):
                        date_array.append(date_1)
                        continue
                    elif ((abs(close - moving_average_100) / close) < 0.0025) and (close < moving_average_100) and (
                            open_stock > moving_average_100):
                        date_array.append(date_1)
                        continue

                    elif ((abs(moving_average_200 - high) / close) < 0.0015) and (close < moving_average_200):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_200) and (close < moving_average_200):
                        date_array.append(date_1)
                        continue
                    elif (high > moving_average_200) and (close < moving_average_200):
                        date_array.append(date_1)
                        continue
                    elif ((abs(close - moving_average_200) / close) < 0.0025) and (close < moving_average_200) and (
                            open_stock > moving_average_200):
                        date_array.append(date_1)
                        continue
                    else:
                        pass

        prices_period = []
        current_prices = []
        for date_2 in date_array:
            final_prices = []
            index = dates_list.index(date_2)
            final_index = index + 5
            if final_index > len(data):
                continue
            current_price = data.iloc[index]['Close']
            current_prices.append(current_price)
            for i in range(index + 1, final_index):
                final_prices.append(data.iloc[i]["Close"])

            prices_period.append(final_prices)

        prices_computing = []

        for element in prices_period:
            prices_computing.append(max(element))

        returns = []

        for i in range(len(current_prices)):
            returns.append((prices_computing[i]/current_prices[i])-1)


        mean_prices = -(np.mean(returns))

    except:
        mean_prices = 0

    return mean_prices

expected_mean_return = 0

for i in range(len(portfolio)):
    capital_allocation = (portfolio['Capital Allocation'].iloc[i])/2
    if portfolio['Execution'].iloc[i] == "Buy":
        expected_mean_return += capital_allocation * target_sl_buy_1(portfolio['Symbol'].iloc[i])
    else:
        expected_mean_return += capital_allocation * target_sl_sell(portfolio['Symbol'].iloc[i])

print(f'Expected Return (Daily): {expected_mean_return/5: .2%}')
print(f'Expected Return: {expected_mean_return: .2%}')
print(f'Annualized Expected Sharpe (Gross): {(expected_mean_return/expected_range) * np.sqrt(48):.2f}')
print(f'Annualized Expected Sharpe (Net): {((expected_mean_return - (0.0025*len(portfolio)))/expected_range) * np.sqrt(48):.2f}')

tail_risk = cVaR_tail

position_option_risk = []

for i in range(len(portfolio)):
    capital_allocation = (portfolio['Capital Allocation'].iloc[i])/2
    if portfolio['Execution'].iloc[i] == "Buy":
        position_option_risk.append(capital_allocation * tail_risk)
    else:
        position_option_risk.append(capital_allocation * -tail_risk)

print(position_option_risk)


def portfolio_gmv(portfolio):
    tickers = portfolio['Symbol'].unique().tolist()

    price_data = yf.download(tickers, period="5y", interval="1wk", group_by='ticker', auto_adjust=True)#['Close']
    price_data = price_data.xs('Close', axis=1, level=1)
    valid_tickers = price_data.columns.tolist()

    returns = price_data.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_portfolios = 1000000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(valid_tickers))
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        weights_record.append(weights)

    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])

    sharpe_weight = weights_record[max_sharpe_idx]
    sharpe_weight /= np.sum(sharpe_weight)
    gmv_weight = weights_record[min_vol_idx]

    print(tickers)

    print("Max Sharpe Ratio:", sharpe_weight)
    print("GMV Portfolio:", gmv_weight)

portfolio_gmv(portfolio)

ff3 = pd.read_csv('F-F_Research_Data_Factors_daily.csv').dropna()
ff3 = ff3.rename(columns={'Unnamed: 0': 'Date'})

df = pd.read_csv(
    "F-F_Momentum_Factor_daily.txt",
    sep=r"\s+",
    header=None,
    names=["Date","Mom"]
)

df.to_csv("momentum_factor.csv", index=False)

ff4 = pd.merge(ff3.assign(Date=pd.to_datetime(ff3['Date'].astype(str), format='%Y%m%d')), df.assign(Date=pd.to_datetime(df['Date'].astype(str), format='%Y%m%d'))[['Date','Mom']], on='Date', how='inner')
portfolio = pd.read_csv('US_equity_2025-08-29.csv')
tickers = portfolio['Symbol'].tolist()
weights = (portfolio['Capital Allocation'] / portfolio['Capital Allocation'].sum()).astype(float)

import yfinance as yf, statsmodels.api as sm

if 'Execution' in portfolio.columns:
    execution_sign = (
        portfolio['Execution'].astype(str).str.strip().str.lower()
        .map({'buy': 1, 'long': 1, 'sell': -1, 'short': -1})
        .fillna(1)
    )
    signed_allocation = portfolio['Capital Allocation'].astype(float) * execution_sign
else:
    signed_allocation = portfolio['Capital Allocation'].astype(float)

weights_by_ticker = pd.Series(signed_allocation.values, index=portfolio['Symbol'].values)
weights_by_ticker = weights_by_ticker / weights_by_ticker.abs().sum()
tickers_list = weights_by_ticker.index.tolist()

start_date_for_prices = pd.to_datetime(ff4['Date']).min()
close_prices = yf.download(tickers_list, start=start_date_for_prices, progress=False)['Close']
if isinstance(close_prices, pd.Series):
    close_prices = close_prices.to_frame(name=tickers_list[0])

daily_simple_returns = close_prices.pct_change().dropna(how='all')

aligned_signed_weights = weights_by_ticker.reindex(daily_simple_returns.columns).fillna(0.0)
if aligned_signed_weights.abs().sum() != 0:
    aligned_signed_weights = aligned_signed_weights / aligned_signed_weights.abs().sum()

portfolio_daily_return = (daily_simple_returns @ aligned_signed_weights).rename('PortfolioReturn')

factors = ff4.copy()
factors['Date'] = pd.to_datetime(factors['Date'])
factors = factors.set_index('Date').sort_index()
for factor_col in ['Mkt-RF', 'SMB', 'HML', 'Mom', 'RF']:
    factors[factor_col] = pd.to_numeric(factors[factor_col], errors='coerce') / 100.0

regression_data = pd.concat([portfolio_daily_return, factors], axis=1).dropna()
excess_portfolio_return = regression_data['PortfolioReturn'] - regression_data['RF']
design_matrix = sm.add_constant(regression_data[['Mkt-RF', 'SMB', 'HML', 'Mom']], has_constant='add')

carhart_results = sm.OLS(excess_portfolio_return, design_matrix).fit()

factor_betas = carhart_results.params.drop('const')
factor_tstats = (carhart_results.params / carhart_results.bse).drop('const')

beta_pvalues = carhart_results.pvalues.drop('const')

significance = []

for i in range(len(beta_pvalues)):
    if beta_pvalues[i] < 0.05:
        significance.append('Significant')
    else:
        significance.append('Not Significant')

summary_table = pd.DataFrame({
    'Beta': factor_betas,
    'T-Statistic': factor_tstats,
    'P-Value': beta_pvalues,
    'Significance': significance
}).reindex(['Mkt-RF', 'SMB', 'HML', 'Mom'])

print(summary_table)