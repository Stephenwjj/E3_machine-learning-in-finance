# main.py
import os
import json
import datetime
import logging
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Model imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Custom module imports
from modeling import (
    config, setup_logger, load_and_merge_data, create_features,
    feature_selection_funnel, print_oos_benchmark, 
    generate_quantstats_reports, optimize_threshold_roc,
    mcnemar_test, backtest_with_stop_loss, max_drawdown
)
from tuning import optimize_xgb, optimize_lgbm, optimize_cat

warnings.filterwarnings("ignore")

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

def remove_redundant_features(features):
    """
    Remove redundant rv_ and ret_ features to prevent multicollinearity.
    Keeps only one representative feature for each dimension (e.g., rv_48, ret_12h).
    """
    keep = []
    rv_kept = False
    ret_kept = False
    
    sorted_features = sorted(features, key=lambda x: (
        0 if x == 'rv_48' else (1 if x.startswith('rv_') else 2),
        0 if x == 'ret_12h' else (1 if x.startswith('ret_') else 2)
    ))
    
    for f in sorted_features:
        if f.startswith('rv_'):
            if not rv_kept:
                keep.append(f)
                rv_kept = True
        elif f.startswith('ret_'):
            if not ret_kept:
                keep.append(f)
                ret_kept = True
        else:
            keep.append(f)
            
    return [f for f in features if f in keep]

# Set working directory and output folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"log/run_{run_time}"
os.makedirs(output_dir, exist_ok=True)

def main():
    logger = setup_logger()
    logger.info("Starting Pipeline...")

    # ==========================================
    # 0. Data Loading & Initial Feature Engineering
    # ==========================================
    df = load_and_merge_data(config.FILE_PATH, logger)
    X, y = create_features(df, config.FORWARD_HOURS, 3.0, config.VOL_MULTIPLIER)
    ret_fwd = np.log(df['close'].shift(-config.FORWARD_HOURS) / df['close']).loc[X.index]

    final_feats = feature_selection_funnel(X, y, config.MI_TOP_K, config.CORR_THRESHOLD, config.L1_C, logger)
    logger.info(f"Initially Selected Features ({len(final_feats)}): {final_feats}")

    # ==========================================
    # 1. Dataset Splitting with Gap (Prevent Look-ahead Bias)
    # ==========================================
    logger.info("Splitting data into 80% Train and 20% Test with 12-hour Purge Gap...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X[final_feats], y, test_size=0.2, shuffle=False)
    
    gap = config.FORWARD_HOURS
    X_train_full = X_train_full.iloc[:-gap]
    y_train_full = y_train_full.iloc[:-gap]
    
    r_train_full = ret_fwd.loc[X_train_full.index]
    r_test_log = ret_fwd.loc[X_test.index]
    
    # Convert to simple returns for benchmark and backtest
    r_test = np.exp(r_test_log) - 1
    bench_wealth = np.exp(r_test_log.cumsum())
    bench_ret = bench_wealth.pct_change().fillna(0)
    
    if r_test.index.tz is not None:
        r_test = r_test.tz_localize(None)
    if bench_ret.index.tz is not None:
        bench_ret = bench_ret.tz_localize(None)
    if y_test.index.tz is not None:
        y_test = y_test.tz_localize(None)

    # TimeSeriesSplit for Validation
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    splits = list(tscv.split(X_train_full))
    sub_train_idx, val_idx = splits[-1]
    
    X_sub_train, y_sub_train = X_train_full.iloc[sub_train_idx], y_train_full.iloc[sub_train_idx]
    X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
    
    scale_pos_weight = (len(y_sub_train) - sum(y_sub_train)) / (sum(y_sub_train) + 1e-5)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_sub_train)

    # Helper function for scaling
    def scale_features(train_df, val_df, test_df, features):
        # Apply RobustScaler to outlier features, StandardScaler to others
        outlier_feats = [c for c in ['macd_hist', 'rsi_14'] if c in features]
        other_feats = [c for c in features if c not in outlier_feats]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('robust', RobustScaler(), outlier_feats),
                ('standard', StandardScaler(), other_feats)
            ])
            
        tr_scaled = pd.DataFrame(preprocessor.fit_transform(train_df[features]), columns=features, index=train_df.index)
        val_scaled = pd.DataFrame(preprocessor.transform(val_df[features]), columns=features, index=val_df.index)
        te_scaled = pd.DataFrame(preprocessor.transform(test_df[features]), columns=features, index=test_df.index)
        return tr_scaled, val_scaled, te_scaled

    # ==========================================
    # STAGE 1: Baseline Model (Before Tuning)
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: BASELINE MODEL (All Initial Features)")
    logger.info("="*80)
    
    X_sub_tr_s1, X_val_s1, X_te_s1 = scale_features(X_sub_train, X_val, X_test, final_feats)
    
    baseline_model = XGBClassifier(**config.MODEL_PARAMS["XGBoost"], scale_pos_weight=scale_pos_weight)
    baseline_model.fit(X_sub_tr_s1, y_sub_train)
    
    p_val_s1 = baseline_model.predict_proba(X_val_s1)[:, 1]
    best_th_s1 = optimize_threshold_roc(y_val, p_val_s1)
    
    metrics_base = print_oos_benchmark({"XGBoost Baseline": baseline_model}, X_te_s1, y_test, r_test, 
                                       threshold=best_th_s1, bench_ret=bench_ret, 
                                       stage_name="STAGE 1: BASELINE", plot_curves=True)

    # ==========================================
    # STAGE 2: Tuned Models (Before Deduplication)
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: HYPERPARAMETER TUNING (All Initial Features)")
    logger.info("="*80)
    
    best_xgb = optimize_xgb(X_sub_tr_s1, y_sub_train, sample_weights)
    best_lgb = optimize_lgbm(X_sub_tr_s1, y_sub_train, sample_weights)
    best_cat = optimize_cat(X_sub_tr_s1, y_sub_train, sample_weights)

    best_cat.pop('verbose',None)
    best_cat.pop('Logging_level',None)
    best_cat.pop('verbose_eval',None)
    best_cat.pop('silent',None)
    
    models_tuned = {
        "XGBoost": XGBClassifier(**best_xgb, scale_pos_weight=scale_pos_weight),
        "LightGBM": LGBMClassifier(**best_lgb, scale_pos_weight=scale_pos_weight),
        "CatBoost": CatBoostClassifier(**best_cat, scale_pos_weight=scale_pos_weight,silent=True)
    }
    for model in models_tuned.values():
        model.fit(X_sub_tr_s1, y_sub_train)
        
    p_val_s2 = models_tuned["XGBoost"].predict_proba(X_val_s1)[:, 1]
    best_th_s2 = optimize_threshold_roc(y_val, p_val_s2)
    
    metrics_tuned = print_oos_benchmark(models_tuned, X_te_s1, y_test, r_test, 
                                        threshold=best_th_s2, bench_ret=bench_ret, 
                                        stage_name="STAGE 2: TUNED MODELS", plot_curves=True)

    # Feature Importance Plot (Stage 2)
    logger.info("Generating Feature Importance plot...")
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(models_tuned["XGBoost"].feature_importances_, index=X_te_s1.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title("Tuned XGBoost Feature Importance (Gain) - Before Deduplication")
    plt.tight_layout()
    plt.savefig("XGBoost_Feature_Importance_Stage2.png", dpi=150)
    plt.close()

    # ==========================================
    # STAGE 3: Feature Deduplication & Re-evaluation
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: FEATURE DEDUPLICATION (Removing Multicollinearity)")
    logger.info("="*80)
    
    dedup_feats = remove_redundant_features(final_feats)
    dropped_feats = set(final_feats) - set(dedup_feats)
    logger.info(f"Dropped redundant features: {dropped_feats}")
    logger.info(f"Remaining features ({len(dedup_feats)}): {dedup_feats}")
    
    X_sub_tr_s3, X_val_s3, X_te_s3 = scale_features(X_sub_train, X_val, X_test, dedup_feats)
    
    models_dedup = {
        "XGBoost (Dedup)": XGBClassifier(**best_xgb, scale_pos_weight=scale_pos_weight),
        "LightGBM (Dedup)": LGBMClassifier(**best_lgb, scale_pos_weight=scale_pos_weight),
        "CatBoost (Dedup)": CatBoostClassifier(**best_cat, scale_pos_weight=scale_pos_weight, silent=True)
    }
    for model in models_dedup.values():
        model.fit(X_sub_tr_s3, y_sub_train)
        
    p_val_s3 = models_dedup["XGBoost (Dedup)"].predict_proba(X_val_s3)[:, 1]
    best_th_s3 = optimize_threshold_roc(y_val, p_val_s3)
    
    metrics_dedup = print_oos_benchmark(models_dedup, X_te_s3, y_test, r_test, 
                                        threshold=best_th_s3, bench_ret=bench_ret, 
                                        stage_name="STAGE 3: FEATURE DEDUPLICATION", plot_curves=True)

    # Feature Importance Plot (Stage 3)
    plt.figure(figsize=(10, 6))
    feat_importances_dedup = pd.Series(models_dedup["XGBoost (Dedup)"].feature_importances_, index=X_te_s3.columns)
    feat_importances_dedup.nlargest(15).plot(kind='barh')
    plt.title("Tuned XGBoost Feature Importance (Gain) - After Deduplication")
    plt.tight_layout()
    plt.savefig("XGBoost_Feature_Importance_Stage3.png", dpi=150)
    plt.close()

    # ==========================================
    # STAGE 4: Stop-Loss Risk Management
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 4: IMPACT OF STOP-LOSS (2% Threshold) on Deduplicated Models")
    logger.info("="*80)
    
    metrics_sl = print_oos_benchmark(models_dedup, X_te_s3, y_test, r_test, 
                                     threshold=best_th_s3, bench_ret=bench_ret, 
                                     stop_loss_pct=0.02, stage_name="STAGE 4: STOP-LOSS APPLIED", plot_curves=True)

    # ==========================================
    # STAGE 5: Advanced Risk Management (Volatility Targeting + Stop-Loss)
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 5: VOLATILITY TARGETING & STOP-LOSS (Aiming for Max DD < 15%)")
    logger.info("="*80)
    
    # Target annualized volatility (e.g., 10%)
    target_vol = 0.10
    # Calculate rolling 24-hour volatility (annualized)
    rolling_vol = r_test.rolling(24).std() * np.sqrt(config.ANNUAL_TRADING_PERIODS)
    
    stage5_results = []
    
    # Add Benchmark to Stage 5 results
    bench_wealth_s5 = (1 + bench_ret).cumprod()
    bench_cum_ret_s5 = bench_wealth_s5.iloc[-1] - 1.0
    bench_mdd_s5 = max_drawdown(bench_wealth_s5)
    bench_sharpe_s5 = np.sqrt(config.ANNUAL_TRADING_PERIODS) * bench_ret.mean() / (bench_ret.std() + 1e-12) if bench_ret.std() > 0 else 0.0
    stage5_results.append({
        "Model": "Benchmark (Buy & Hold)", "ROC-AUC": np.nan, "Precision": np.nan, "Recall": np.nan, 
        "F1-Score": np.nan, "MCC": np.nan, "Sharpe (H1)": round(bench_sharpe_s5, 2), 
        "Max DD (%)": round(bench_mdd_s5 * 100, 2), "Cum Ret (%)": round(bench_cum_ret_s5 * 100, 2)
    })

    # Plotting setup for Stage 5
    fig, axes = plt.subplots(len(models_dedup), 3, figsize=(15, 5 * len(models_dedup)))
    if len(models_dedup) == 1: axes = [axes]

    for i, (name, model) in enumerate(models_dedup.items()):
        p_te = model.predict_proba(X_te_s3)[:, 1]
        y_pred = (p_te >= best_th_s3).astype(int)
        
        auc_roc = roc_auc_score(y_test, p_te)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Apply Volatility Targeting Position Sizing
        signals = pd.Series(y_pred, index=r_test.index).shift(1).fillna(0)
        position_size = (target_vol / (rolling_vol + 1e-6)).clip(upper=1.0).fillna(0)
        signals_sized = signals * position_size
        
        # Calculate strategy returns with position sizing
        strat_ret = signals_sized * r_test
        
        # Apply 2% Stop-Loss
        strat_ret[strat_ret < -0.02] = -0.02
        
        # Calculate turnover and costs
        turnover = signals_sized.diff().abs().fillna(0)
        strat_ret = (strat_ret - turnover * (config.COST_BPS * 1e-4)).dropna()
        
        # Calculate metrics
        if len(strat_ret) == 0:
            cum_ret, mdd, sharpe = 0.0, 0.0, 0.0
        else:
            cum_wealth = (1 + strat_ret).cumprod()
            cum_ret = cum_wealth.iloc[-1] - 1.0
            mdd = max_drawdown(cum_wealth)
            sharpe = np.sqrt(config.ANNUAL_TRADING_PERIODS) * strat_ret.mean() / (strat_ret.std() + 1e-12) if strat_ret.std() > 0 else 0.0

        stage5_results.append({
            "Model": f"{name} + VolTarget", "ROC-AUC": round(auc_roc, 4), "Precision": round(precision, 4),
            "Recall": round(recall, 4), "F1-Score": round(f1, 4), "MCC": round(mcc, 4),
            "Sharpe (H1)": round(sharpe, 2), "Max DD (%)": round(mdd * 100, 2), "Cum Ret (%)": round(cum_ret * 100, 2)
        })

        # Plot curves for Stage 5
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=axes[i][0], colorbar=False)
        axes[i][0].set_title(f"{name} - Confusion Matrix (VolTarget)")
        RocCurveDisplay.from_predictions(y_test, p_te, ax=axes[i][1])
        axes[i][1].set_title(f"{name} - ROC Curve")
        PrecisionRecallDisplay.from_predictions(y_test, p_te, ax=axes[i][2])
        axes[i][2].set_title(f"{name} - PR Curve")

    plt.tight_layout()
    plt.savefig("Model_Evaluation_Curves_STAGE_5_VOL_TARGET.png", dpi=150)
    plt.close()

    df_s5 = pd.DataFrame(stage5_results)
    print("\n" + "="*115)
    print(f"STAGE 5: VOLATILITY TARGETING & STOP-LOSS")
    print("="*115)
    print(df_s5.to_markdown(index=False))

    # ==========================================
    # 6. McNemar Test & Benchmark Comparison
    # ==========================================
    y_pred_final = (models_dedup["XGBoost (Dedup)"].predict_proba(X_te_s3)[:, 1] >= best_th_s3).astype(int)
    stat, pval = mcnemar_test(y_test.values, y_pred_final, np.ones(len(y_test)))
    logger.info(f"McNemar Test (Final Model vs Always Up): Statistic={stat:.4f}, p-value={pval:.4e}")

    # Generate QuantStats
    generate_quantstats_reports(models_dedup, X_te_s3, r_test_log, bench_ret, threshold=best_th_s3)

    # ==========================================
    # 7. Save Results
    # ==========================================
    logger.info(f"\nSaving all results to {output_dir}...")
    metrics_base.to_csv(os.path.join(output_dir, "1_metrics_baseline.csv"), index=False)
    metrics_tuned.to_csv(os.path.join(output_dir, "2_metrics_tuned.csv"), index=False)
    metrics_dedup.to_csv(os.path.join(output_dir, "3_metrics_dedup.csv"), index=False)
    metrics_sl.to_csv(os.path.join(output_dir, "4_metrics_stoploss.csv"), index=False)
    df_s5.to_csv(os.path.join(output_dir, "5_metrics_vol_target.csv"), index=False)
    
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(convert_to_serializable({
            "XGBoost": best_xgb, 
            "LightGBM": best_lgb, 
            "CatBoost": best_cat, 
            "Threshold": best_th_s3
        }), f, indent=4)

    # Move generated files
    for file in os.listdir("."):
        if (file.startswith("QuantStats_Report_") and file.endswith(".html")) or \
           (file.startswith("Model_Evaluation_Curves_") and file.endswith(".png")) or \
           (file.startswith("XGBoost_Feature_Importance_") and file.endswith(".png")):
            os.rename(file, os.path.join(output_dir, file))
            
    logger.info("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
