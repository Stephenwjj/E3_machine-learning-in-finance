# modeling.py
import os
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import quantstats as qs
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent GUI errors in server environments
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, roc_curve, confusion_matrix, ConfusionMatrixDisplay, 
    PrecisionRecallDisplay, RocCurveDisplay
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Config:
    """Configuration class to store global parameters."""
    FILE_PATH = 'data/XAUUSD_H1_202001020100_202606302300.csv'
    FORWARD_HOURS = 12
    COST_BPS = 4.0
    VOL_MULTIPLIER = 1.0
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    MODEL_PARAMS = {
        "XGBoost": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "tree_method": "hist", "random_state": 2026},
        "LightGBM": {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.03, "random_state": 2026, "verbose": -1},
        "CatBoost": {"iterations": 200, "depth": 4, "learning_rate": 0.03, "verbose": False, "random_seed": 2026}
    }
    MI_TOP_K = 25
    CORR_THRESHOLD = 0.90
    L1_C = 0.1
    ANNUAL_TRADING_PERIODS = 5796 
    RISK_FREE_RATE = 0.02

config = Config()

def setup_logger():
    """Sets up a production-grade logging system."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(__name__)

def calculate_log_returns(series: pd.Series, n: int = 1) -> pd.Series:
    """Calculates n-period log returns."""
    return np.log(series / series.shift(n))

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculates rolling Z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-12)

def max_drawdown(cum_wealth: pd.Series) -> float:
    """Calculates the maximum drawdown from a cumulative wealth series."""
    rolling_max = cum_wealth.cummax()
    drawdown = (cum_wealth / rolling_max) - 1.0
    return drawdown.min()

def load_and_merge_data(file_path: str, logger) -> pd.DataFrame:
    """Loads local OHLC data and merges it with macro indicators (DXY, US10Y, VIX)."""
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
        # Download macro data and shift by 1 to prevent look-ahead bias
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

def create_features(df: pd.DataFrame, fwd_h: int, cost_bps: float, vol_mult: float) -> tuple[pd.DataFrame, pd.Series]:
    """Generates a comprehensive feature pool and dynamically labeled targets."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["tickvol"]
    r = calculate_log_returns(c, 1)
    tp = (h + l + c) / 3.0
    f = pd.DataFrame(index=df.index)    
    
    # 1. Technical Indicators
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

    # 2. Momentum & Volatility
    for n in [1, 4, 8, 12, 24, 48, 72, 120]:
        f[f"ret_{n}h"] = calculate_log_returns(c, n)
    f["lag1_autocorr_6"] = r.rolling(6).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
    f["lag1_autocorr_24"] = r.rolling(24).apply(lambda x: pd.Series(x).autocorr(1), raw=False)

    f["donchian_pos_20"] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min() + 1e-12)
    f["donchian_pos_48"] = (c - c.rolling(48).min()) / (c.rolling(48).max() - c.rolling(48).min() + 1e-12)
    
    for n in [6, 12, 24, 48]:
        f[f"rv_{n}"] = np.sqrt(r.pow(2).rolling(n).sum())
    f["vol_of_rv_24"] = f["rv_24"].rolling(24).std()
    
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_ratio_6_48"] = tr.ewm(span=6, adjust=False).mean() / (tr.ewm(span=48, adjust=False).mean() + 1e-12)
    f["bb_squeeze"] = f["bb_width"].rolling(48).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-12), raw=True)
    
    # 3. Volume & Price Action
    f["volume_zscore_24"] = calculate_zscore(v, 24)
    f["price_volume_corr_12"] = r.rolling(12).corr(v.pct_change())

    vwap_h = (tp * v).rolling(1).sum() / v.replace(0, np.nan)
    vwap_d = (tp * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    f["close_vs_vwap_h1"] = c / (vwap_h + 1e-12) - 1
    f["close_vs_vwap_d1"] = c / (vwap_d + 1e-12) - 1
    
    f["body_ratio"] = ((c - o).abs() / (h - l + 1e-12))
    f["upper_shadow"] = (h - np.maximum(o, c)) / (h - l + 1e-12)
    f["lower_shadow"] = (np.minimum(o, c) - l) / (h - l + 1e-12)
    
    # 4. Macro Features
    if 'dxy' in df.columns:
        for n in [1, 8, 24]:
            f[f"dxy_chg_{n}h"] = calculate_log_returns(df['dxy'], n)
            f[f"us10y_chg_{n}h"] = calculate_log_returns(df['us10y'], n)
            f[f"vix_chg_{n}h"] = calculate_log_returns(df['vix'], n)
        f["dxy_corr_48"] = r.rolling(48).corr(np.log(df['dxy']/df['dxy'].shift(1)))
        f["nlp_fear_sentiment"] = (f["vix_chg_8h"] - f["dxy_chg_8h"]) / 2.0
        f["nlp_sentiment_smooth"] = f["nlp_fear_sentiment"].rolling(4).mean()

    # 5. Event Driven (NFP Hour)
    is_first_friday = (df.index.dayofweek == 4) & (df.index.day <= 7)
    f["is_nfp_hour"] = (is_first_friday & (df.index.hour.isin([12, 13]))).astype(int)
       
    # 6. Dynamic Volatility Labeling (Shift X to avoid look-ahead bias)
    X = f.shift(1)
    y_raw = np.log(c.shift(-fwd_h) / c)
    cost = cost_bps * 1e-4
    dynamic_threshold = (1.5 * tr.rolling(14).mean() / c) + cost 
    
    y = pd.Series(0, index=df.index, name="Label")
    y.loc[y_raw > dynamic_threshold] = 1

    data = X.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    return data.drop(columns=["Label"]), data["Label"]

def feature_selection_funnel(X: pd.DataFrame, y: pd.Series, mi_top_k: int, corr_thresh: float, l1_c: float, logger) -> list:
    """Executes a three-tier feature selection funnel (MI, Correlation, RFECV, L1)."""
    logger.info(f"Starting feature selection. Initial features: {X.shape[1]}")
    
    # 1. Filter Method: Mutual Information
    mi = mutual_info_classif(X, y, random_state=2026)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    keep_mi = mi_s.head(mi_top_k).index.tolist()
    logger.info(f"Filter (MI) retained {len(keep_mi)} features.")
    
    # 2. Filter Method: Correlation Pruning
    X_mi = X[keep_mi]
    corr_matrix = X_mi.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    drop_cols = set()
    for col in upper.columns:
        high_corr_feats = upper.index[upper[col] > corr_thresh].tolist()
        for feat in high_corr_feats:
            if mi_s[col] > mi_s[feat]:
                drop_cols.add(feat)
            else:
                drop_cols.add(col)
    keep_corr = [c for c in keep_mi if c not in drop_cols]
    logger.info(f"Filter (Correlation) retained {len(keep_corr)} features.")

    # 3. Wrapper Method: RFECV with ColumnTransformer and RobustScaler
    outlier_features = ['macd_hist', 'rsi_14']
    outlier_cols = [c for c in keep_corr if c in outlier_features]
    other_cols = [c for c in keep_corr if c not in outlier_features]

    # Map column names to integer indices to prevent RFECV parallel processing errors
    col_idx = {c: i for i, c in enumerate(X.columns)}
    outlier_idx = [col_idx[c] for c in outlier_cols]
    other_idx = [col_idx[c] for c in other_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), outlier_idx),
            ('standard', StandardScaler(), other_idx)
        ],
        remainder='passthrough'
    )

    tscv = TimeSeriesSplit(n_splits=3, gap=1)
    est = Pipeline([
        ("sc", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear"))
    ])
    
    rfecv = RFECV(
        estimator=est, step=1, cv=tscv, scoring="roc_auc",
        min_features_to_select=8, importance_getter='named_steps.clf.coef_', n_jobs=1
    )
    rfecv.fit(X[keep_corr], y)
    keep_wrap = X[keep_corr].columns[rfecv.support_].tolist()
    logger.info(f"Wrapper (RFECV) retained {len(keep_wrap)} features.")
    
    # 4. Embedded Method: L1 Regularization
    X_scaled = StandardScaler().fit_transform(X[keep_wrap])
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=l1_c, class_weight='balanced', max_iter=2000)
    l1_model.fit(X_scaled, y)
    
    final_feats = [c for c, w in zip(keep_wrap, l1_model.coef_[0]) if abs(w) > 1e-6]
    logger.info(f"Embedded (L1) retained {len(final_feats)} final features.")
    logger.info(f"Final selected features: {final_feats}")
    
    return final_feats

def optimize_threshold_roc(y_true, y_probs):
    """Finds the optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    if optimal_threshold == np.inf:
        optimal_threshold = 1.0
    return optimal_threshold

def print_oos_benchmark(models: dict, X_test: pd.DataFrame, y_test: pd.Series, ret_test: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    """Evaluates OOS performance, prints a summary table, and generates evaluation plots."""
    results_list = []
    
    # Setup subplots for Confusion Matrix, ROC, and PR curves
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 5 * len(models)))
    if len(models) == 1: axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        p_te = model.predict_proba(X_test)[:, 1]
        y_pred = (p_te >= threshold).astype(int)
        
        # Calculate Classification Metrics
        auc_roc = roc_auc_score(y_test, p_te)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Calculate Backtest Metrics (with 1-step shift and transaction costs)
        strat_ret = pd.Series(y_pred, index=ret_test.index).shift(1).fillna(0) * ret_test
        turnover = pd.Series(y_pred, index=ret_test.index).diff().abs().fillna(0)
        strat_ret = (strat_ret - turnover * (config.COST_BPS * 1e-4)).dropna()

        if len(strat_ret) == 0:
            cum_ret, mdd, sharpe = 0.0, 0.0, 0.0
        else:
            cum_wealth = (1 + strat_ret).cumprod()
            cum_ret = cum_wealth.iloc[-1] - 1.0
            mdd = max_drawdown(cum_wealth)
            sharpe = np.sqrt(config.ANNUAL_TRADING_PERIODS) * strat_ret.mean() / (strat_ret.std() + 1e-12) if strat_ret.std() > 0 else 0.0
        
        results_list.append({
            "Model": name, "ROC-AUC": round(auc_roc, 4), "Precision": round(precision, 4),
            "Recall": round(recall, 4), "F1-Score": round(f1, 4), "MCC": round(mcc, 4),
            "Sharpe (H1)": round(sharpe, 2), "Max DD (%)": round(mdd * 100, 2), "Cum Ret (%)": round(cum_ret * 100, 2)
        })

        # Plotting: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=axes[i][0], colorbar=False)
        axes[i][0].set_title(f"{name} - Confusion Matrix")

        # Plotting: ROC Curve
        RocCurveDisplay.from_predictions(y_test, p_te, ax=axes[i][1])
        axes[i][1].set_title(f"{name} - ROC Curve")

        # Plotting: PR Curve
        PrecisionRecallDisplay.from_predictions(y_test, p_te, ax=axes[i][2])
        axes[i][2].set_title(f"{name} - PR Curve")

    plt.tight_layout()
    plt.savefig("Model_Evaluation_Curves.png", dpi=150)
    plt.close()
    
    print("\n" + "="*100)
    print(f"Out-of-Sample (OOS) Benchmarking (Threshold: {threshold:.2f})")
    print("="*100)
    print(tabulate(results_list, headers="keys", tablefmt="pipe", stralign="center"))
    return pd.DataFrame(results_list)

def generate_quantstats_reports(models: dict, X_test: pd.DataFrame, ret_test: pd.Series, bench_ret: pd.Series, threshold: float):
    """Generates HTML performance reports using QuantStats."""
    if bench_ret is not None and bench_ret.index.tz is not None:
        bench_ret.index = bench_ret.index.tz_localize(None)

    for name, model in models.items():
        p_te = model.predict_proba(X_test)[:, 1]
        sig_te = (p_te >= threshold).astype(int)

        strat_ret = pd.Series(sig_te, index=ret_test.index).shift(1).fillna(0) * ret_test
        turnover = pd.Series(sig_te, index=ret_test.index).diff().abs().fillna(0)
        strat_ret = (strat_ret - turnover * (config.COST_BPS * 1e-4)).dropna()
        strat_ret.name = f"{name} Strategy"
        
        if strat_ret.index.tz is not None:
            strat_ret.index = strat_ret.index.tz_localize(None)

        html_filename = f"QuantStats_Report_{name}.html"
        qs.reports.html(
            returns=strat_ret, benchmark=bench_ret, rf=config.RISK_FREE_RATE,
            periods_per_year=config.ANNUAL_TRADING_PERIODS, output=html_filename, title=f"XAUUSD H1 - {name}"
        )
        print(f"Saved: {html_filename}")
