import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import config
from utils import setup_logger
from features import load_and_merge_data, create_features
from selection import feature_selection_funnel
from evaluation import print_oos_benchmark, generate_quantstats_reports

def main():
    logger = setup_logger()
    logger.info("Starting Pipeline...")

    # 1. Data Loading & Feature Engineering
    df = load_and_merge_data(config.FILE_PATH, logger)
    X, y = create_features(df, config.FORWARD_HOURS, config.COST_BPS, config.VOL_MULTIPLIER)
    
    # Extract future returns for backtesting (strictly aligned with X)
    ret_fwd = np.log(df['close'].shift(-config.FORWARD_HOURS) / df['close']).loc[X.index]

    # 2. Feature Selection Funnel
    final_feats = feature_selection_funnel(X, y, config.MI_TOP_K, config.CORR_THRESHOLD, config.L1_C, logger)
    X = X[final_feats]
    
    # ==========================================
    # 3. Dataset Splitting (80% Train, 20% Test)
    # ==========================================
    logger.info("Splitting data into 80% Train and 20% Test (no shuffle)...")
    # Split features and target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Split returns to align perfectly with the train/test indices
    r_train = ret_fwd.loc[X_train.index]
    r_test = ret_fwd.loc[X_test.index]

    # ==========================================
    # 4. Timeseries Cross Validation (5-split, gap=1)
    # ==========================================
    logger.info("Applying TimeSeriesSplit (n_splits=5, gap=1) to extract validation set...")
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    
    # Extract the indices from the last fold to use as our validation set
    # This ensures no look-ahead bias and respects the time series structure
    splits = list(tscv.split(X_train))
    train_idx, val_idx = splits[-1]
    
    # Further split the 80% train into sub-train and validation
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    r_val = r_train.iloc[val_idx]

    # For final OOS testing, we strictly use the untouched 20% X_test
    X_te, y_te, r_te = X_test, y_test, r_test

    # Calculate class weights based on the sub-train set
    scale_pos_weight = (len(y_tr) - sum(y_tr)) / (sum(y_tr) + 1e-5)

    # ==========================================
    # 5. Model Initialization & Training
    # ==========================================
    models = {
        "XGBoost": XGBClassifier(**config.MODEL_PARAMS["XGBoost"], scale_pos_weight=scale_pos_weight),
        "LightGBM": LGBMClassifier(**config.MODEL_PARAMS["LightGBM"], scale_pos_weight=scale_pos_weight),
        "CatBoost": CatBoostClassifier(**config.MODEL_PARAMS["CatBoost"], scale_pos_weight=scale_pos_weight)
    }
    
    logger.info("Training Models...")
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        logger.info(f"{name} trained successfully.")

    # ==========================================
    # 6. Threshold Optimization (Using Validation Set)
    # ==========================================
    best_th, best_sharpe = 0.5, -np.inf
    p_val = models["XGBoost"].predict_proba(X_val)[:, 1]
    
    for th in np.linspace(0.5, 0.8, 31):
        sig = (p_val >= th).astype(int)
        if sig.sum() < 20: continue
        
        strat_ret_val = pd.Series(sig, index=r_val.index).shift(1).fillna(0) * r_val
        strat_ret_val -= pd.Series(sig, index=r_val.index).diff().abs().fillna(0) * (config.COST_BPS * 1e-4)
        
        if strat_ret_val.std() > 0:
            sharpe = np.sqrt(config.ANNUAL_TRADING_PERIODS) * strat_ret_val.mean() / strat_ret_val.std()
            if sharpe > best_sharpe:
                best_sharpe, best_th = sharpe, th
                
    logger.info(f"Optimal Threshold found on Validation Set: {best_th:.3f}")

    # ==========================================
    # 7. OOS Evaluation & Report Generation
    # ==========================================
    print_oos_benchmark(models, X_te, y_te, r_te, threshold=best_th)
    
    # Fetch Benchmark (GLD) for QuantStats
    test_start = r_te.index.min().strftime('%Y-%m-%d')
    test_end = r_te.index.max().strftime('%Y-%m-%d')
    bench_data = yf.download('GLD', start=test_start, end=test_end, progress=False)['Close']
    
    bench_ret_aligned = None
    if not bench_data.empty:
        bench_ret = bench_data.resample('h').ffill().pct_change().dropna()
        bench_ret_aligned = bench_ret.reindex(r_te.index).fillna(0)

    generate_quantstats_reports(models, X_te, r_te, bench_ret_aligned, threshold=best_th)
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()