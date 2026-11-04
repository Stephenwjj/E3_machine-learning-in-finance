# main.py
import os
import json
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from modeling import (
    config, setup_logger, load_and_merge_data, create_features,
    feature_selection_funnel, print_oos_benchmark, 
    generate_quantstats_reports, optimize_threshold_roc
)
from tuning import optimize_xgb, optimize_lgbm, optimize_cat

def convert_to_serializable(obj):
    """Recursively converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# Force current working directory to the script's directory to prevent Colab path nesting issues
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create a dynamic output directory for the current run
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"log/run_{run_time}"
os.makedirs(output_dir, exist_ok=True)

def main():
    logger = setup_logger()
    logger.info("Starting Pipeline...")

    # 1. Data Loading & Feature Engineering
    df = load_and_merge_data(config.FILE_PATH, logger)
    X, y = create_features(df, config.FORWARD_HOURS, config.COST_BPS, config.VOL_MULTIPLIER)
    
    # Extract forward returns for backtesting (strictly aligned with X)
    ret_fwd = np.log(df['close'].shift(-config.FORWARD_HOURS) / df['close']).loc[X.index]

    # 2. Feature Selection Funnel
    final_feats = feature_selection_funnel(X, y, config.MI_TOP_K, config.CORR_THRESHOLD, config.L1_C, logger)
    X = X[final_feats]
    
    # 3. Dataset Splitting (80% Train, 20% Test)
    logger.info("Splitting data into 80% Train and 20% Test...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    r_train_full = ret_fwd.loc[X_train_full.index]
    r_test = ret_fwd.loc[X_test.index]

    # 4. Timeseries Cross Validation (5-split, gap=1)
    logger.info("Applying TimeSeriesSplit (n_splits=5, gap=1) to extract validation set...")
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    splits = list(tscv.split(X_train_full))
    sub_train_idx, val_idx = splits[-1]
    
    # Further split the 80% train into sub-train and validation
    X_sub_train, y_sub_train = X_train_full.iloc[sub_train_idx], y_train_full.iloc[sub_train_idx]
    X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
    r_val = r_train_full.iloc[val_idx]

    X_te, y_te, r_te = X_test, y_test, r_test

    # Calculate class weights based on the sub-train set
    scale_pos_weight = (len(y_sub_train) - sum(y_sub_train)) / (sum(y_sub_train) + 1e-5)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_sub_train)

    # 5. Model Initialization & Training
    logger.info("Starting Bayesian Optimization...")
    best_xgb_params = optimize_xgb(X_sub_train, y_sub_train, sample_weights)
    best_lgbm_params = optimize_lgbm(X_sub_train, y_sub_train, sample_weights)
    best_cat_params = optimize_cat(X_sub_train, y_sub_train, sample_weights)

    logger.info("Initializing Models with Best Parameters...")
    models = {
        "XGBoost": XGBClassifier(**best_xgb_params, scale_pos_weight=scale_pos_weight),
        "LightGBM": LGBMClassifier(**best_lgbm_params, scale_pos_weight=scale_pos_weight),
        "CatBoost": CatBoostClassifier(**best_cat_params, scale_pos_weight=scale_pos_weight)
    }
    
    logger.info("Training Models...")
    for name, model in models.items():
        model.fit(X_sub_train, y_sub_train)
        logger.info(f"{name} trained successfully.")

    # 6. Threshold Optimization (Using Validation Set)
    logger.info("Optimizing Threshold to Maximize ROC-AUC...")
    p_val = models["XGBoost"].predict_proba(X_val)[:, 1]
    best_th = optimize_threshold_roc(y_val, p_val)
    logger.info(f"Optimal Threshold found on Validation Set (Max ROC-AUC): {best_th:.3f}")

    # 7. OOS Evaluation & Report Generation
    metrics_df = print_oos_benchmark(models, X_te, y_te, r_te, threshold=best_th)
    
    # Generate Feature Importance Plot (PDF Page 25)
    logger.info("Generating Feature Importance plot...")
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(models["XGBoost"].feature_importances_, index=X_te.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title("Tuned XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig("XGBoost_Feature_Importance.png", dpi=150)
    plt.close()

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

    logger.info(f"Saving results to {output_dir}...")
    
    # Save metrics to CSV
    if isinstance(metrics_df, pd.DataFrame):
        metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        
    # Save best parameters to JSON
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(convert_to_serializable({
            "XGBoost": best_xgb_params,
            "LightGBM": best_lgbm_params,
            "CatBoost": best_cat_params,
            "Optimal_Threshold": best_th
        }), f, indent=4)

    # Move generated HTML reports and PNG plots to the output directory
    for file in os.listdir("."):
        if (file.startswith("QuantStats_Report_") and file.endswith(".html")) or \
           (file in ["Model_Evaluation_Curves.png", "XGBoost_Feature_Importance.png"]):
            os.rename(file, os.path.join(output_dir, file))

if __name__ == "__main__":
    main()
