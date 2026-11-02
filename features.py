import numpy as np
import pandas as pd
import yfinance as yf
from utils import calculate_log_returns, calculate_zscore

def load_and_merge_data(file_path: str, logger) -> pd.DataFrame:
    """Loads local XAUUSD data and merges with macro data (DXY, US10Y, VIX)."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = df.columns.str.strip('<>')
        df['Datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str)).dt.tz_localize('UTC')
        df = df.set_index('Datetime').drop(columns=['DATE', 'TIME', 'VOL'], errors='ignore').copy()
        df.columns = df.columns.str.lower()
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load local data: {e}")
        raise

    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    
    try:
        macro = yf.download(['DX-Y.NYB', '^TNX', '^VIX'], start=start_date, end=end_date, progress=False)['Close']
        macro.columns = ['dxy', 'us10y', 'vix']
        macro = macro.shift(1).resample('h').ffill()
        if macro.index.tz is None: macro.index = macro.index.tz_localize('UTC')
        else: macro.index = macro.index.tz_convert('UTC')
        df = df.join(macro, how='left').ffill()
        logger.info("Macro data merged successfully.")
    except Exception as e:
        logger.warning(f"Macro data download failed: {e}.")
    return df



# Generates a feature pool and dynamic labels
def create_features(df: pd.DataFrame, fwd_h: int, cost_bps: float, vol_mult: float) -> tuple[pd.DataFrame, pd.Series]:

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["tickvol"]
    r = calculate_log_returns(c, 1)
    tp = (h + l + c) / 3.0
    f = pd.DataFrame(index=df.index)    
    
    """
    # 1. Technical Indicators (5):
    RSI (period=14 with exponential moving average);
    MACD (Histogram calculated by the difference between exponential moving average fast period = 12 & slow period = 26, then substract signal line period =9);
    Bollinger Bands (rolling period = 20, 2 single side standard error).
    """
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    f['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-12)))

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    f['macd_hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f['bb_width'] = (4 * sd20) / (ma20 + 1e-12)
    f['bb_pos'] = (c - (ma20 - 2*sd20)) / (4*sd20 + 1e-12)

    """
    # 2. Momentum & Volatility (14):
    ret_{n}h: n period momentum from 1 to 48 to capture the momentum across multiple timeframes;
    rv_{n}: n period realized volatility from 1 to 48 to capture the price fluctuations
    atr_ratio_6_48: the ratio of short term ATR (period = 6 EWM) and long term ATR (period = 48 EWM). Values > 1 indicate a sudden surge in volatility (expansion), while values < 1 indicate volatility contraction;
    lag1_autocorr_n: 1-period autocorrelation of returns over a rolling n-hour window;
    donchian_pos_n: Calculates the position of the current price relative to the rolling n-period High-Low range
    """
    for n in [1, 4, 8, 12, 24, 48, 72, 120]:
        f[f"ret_{n}h"] = calculate_log_returns(c, n)
    f["lag1_autocorr_6"] = r.rolling(6).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
    f["lag1_autocorr_24"] = r.rolling(24).apply(lambda x: pd.Series(x).autocorr(1), raw=False)

    """    
    # 3. Volatility & Microstructure:
    volume_zscore_n: Z-score of trading volume over a n-hour window;
    price_volume_corr_n: the rolling n-period correlation between returns (r) and the percentage change in volume (v);
    spread_zscore: Z-score of the bid-ask spread.
    """ 
    f["donchian_pos_20"] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min() + 1e-12)
    f["donchian_pos_48"] = (c - c.rolling(48).min()) / (c.rolling(48).max() - c.rolling(48).min() + 1e-12)
    
    for n in [6, 12, 24, 48]:
        f[f"rv_{n}"] = np.sqrt(r.pow(2).rolling(n).sum())
    f["vol_of_rv_24"] = f["rv_24"].rolling(24).std()
    
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_ratio_6_48"] = tr.ewm(span=6, adjust=False).mean() / (tr.ewm(span=48, adjust=False).mean() + 1e-12)
    f["bb_squeeze"] = f["bb_width"].rolling(48).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-12), raw=True)
    
    
    f["volume_zscore_24"] = calculate_zscore(v, 24)
    f["price_volume_corr_12"] = r.rolling(12).corr(v.pct_change())


    # 4. Volume & VWAP (5)
    vwap_h = (tp * v).rolling(1).sum() / v.replace(0, np.nan)
    vwap_d = (tp * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    f["close_vs_vwap_h1"] = c / (vwap_h + 1e-12) - 1
    f["close_vs_vwap_d1"] = c / (vwap_d + 1e-12) - 1
    f["volume_zscore_24"] = calculate_zscore(v, 24)
    
    # 5. Price Action (3)
    f["price_volume_corr_12"] = r.rolling(12).corr(v.pct_change())
    f["body_ratio"] = ((c - o).abs() / (h - l + 1e-12))
    f["upper_shadow"] = (h - np.maximum(o, c)) / (h - l + 1e-12)
    f["lower_shadow"] = (np.minimum(o, c) - l) / (h - l + 1e-12)
    
    
    """
    # 6. Macro Features (7)
    dxy_chg_{n}h, us10y_chg_{n}h, vix_chg_{n}h: n hour log returns for the DXY (US Dollar Index), US10Y (10-Year Treasury Yield), and VIX (Volatility Index);
    nlp_fear_sentiment: assume that if VIX is rising (panic in the stock market) and the Dollar is falling (weak USD), money will flow into Gold as a safe-haven asset.
    """
    if 'dxy' in df.columns:
        for n in [1, 8, 24]:
            f[f"dxy_chg_{n}h"] = calculate_log_returns(df['dxy'], n)
            f[f"us10y_chg_{n}h"] = calculate_log_returns(df['us10y'], n)
            f[f"vix_chg_{n}h"] = calculate_log_returns(df['vix'], n)
        f["dxy_corr_48"] = r.rolling(48).corr(np.log(df['dxy']/df['dxy'].shift(1)))
        f["nlp_fear_sentiment"] = (f["vix_chg_1h"] - f["dxy_chg_1h"]) / 2.0
        f["nlp_sentiment_smooth"] = f["nlp_fear_sentiment"].rolling(4).mean()

    # 5. Event Driven (NFP), is_nfp_hour: when the US Non-Farm Payrolls report is released;
    is_first_friday = (df.index.dayofweek == 4) & (df.index.day <= 7)
    f["is_nfp_hour"] = (is_first_friday & (df.index.hour.isin([12, 13]))).astype(int)
       
    """
    7. Dynamic Volatility Labeling:
    y_raw: The actual future log return after fwd_h hours;
    raw_rv_24:  24-hour rolling standard deviation;
    dynamic_threshold: A threshold that scales with current market volatility, plus transaction costs.
    """
    X = f.shift(1)
    y_raw = np.log(c.shift(-fwd_h) / c)
    cost = cost_bps * 1e-4
    dynamic_threshold = (1.5 * tr.rolling(14).mean() / c) + cost 
    
    # label = 1 if y_raw > dynamic_threshold, otherwise label = 0
    y = pd.Series(0, index=df.index, name="Label")
    y.loc[y_raw > dynamic_threshold] = 1

    # Drop NaNs and infinities, concatenate the feature matrix X and the labels y
    data = X.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    return data.drop(columns=["Label"]), data["Label"]
