import pandas as pd
import numpy as np
import quantstats as qs
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve
from utils import max_drawdown
import config
import warnings
import logging

# ignore the warnings when running the model
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# search for max ROC-AUC trade-off 
def optimize_threshold_roc(y_true, y_probs):
    """
    Finds the threshold that maximizes the distance to the diagonal (Youden's J),
    which corresponds to maximizing the ROC-AUC trade-off.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Youden's J statistic = Sensitivity + Specificity - 1 = TPR - FPR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Handle infinity threshold edge case
    if optimal_threshold == np.inf:
        optimal_threshold = 1.0
        
    return optimal_threshold

# Calculates metrics AUC, F1，Sharpe, MDD, Cumulative Return and prints a formatted table on Out-of-Sample (OOS) data
def print_oos_benchmark(models: dict, X_test: pd.DataFrame, y_test: pd.Series, ret_test: pd.Series, threshold: float = 0.5) -> pd.DataFrame:

    results_list = []

    # 1. Generate predictions: iterate through each trained model
    # Extract the probability of the positive class (Class 1) as predictions
    # Convert probabilities into binary trading signals (0 or 1) based on the given threshold
    for name, model in models.items():
        p_te = model.predict_proba(X_test)[:, 1]
        y_pred = (p_te >= threshold).astype(int)
        
        # 2. Classification Metrics
        auc_roc = roc_auc_score(y_test, p_te)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # 3. Backtest Metrics (with 1-step shift to prevent look-ahead bias)
        # Deduct transaction costs from strategy returns. 
        strat_ret = pd.Series(y_pred, index=ret_test.index).shift(1).fillna(0) * ret_test
        turnover = pd.Series(y_pred, index=ret_test.index).diff().abs().fillna(0)
        strat_ret = (strat_ret - turnover * (config.COST_BPS * 1e-4)).dropna()

        # 4. Calculate Portfolio Statistics, without edge cases where the strategy never trades or has zero variance
        if len(strat_ret) == 0:
            cum_ret, mdd, sharpe = 0.0, 0.0, 0.0
        else:
            cum_wealth = (1 + strat_ret).cumprod()
            cum_ret = cum_wealth.iloc[-1] - 1.0
            mdd = max_drawdown(cum_wealth)
            sharpe = np.sqrt(config.ANNUAL_TRADING_PERIODS) * strat_ret.mean() / (strat_ret.std() + 1e-12) if strat_ret.std() > 0 else 0.0
        
        # 5. Append to results list for tabular display
        results_list.append({
            "Model": name,
            "ROC-AUC": round(auc_roc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "MCC": round(mcc, 4),
            "Sharpe (H1)": round(sharpe, 2),
            "Max DD (%)": round(mdd * 100, 2),
            "Cum Ret (%)": round(cum_ret * 100, 2)
        })
    
    # 6. Print the results    
    print("\n" + "="*100)
    print(f"Out-of-Sample (OOS) Benchmarking (Threshold: {threshold:.2f})")
    print("="*100)
    print(tabulate(results_list, headers="keys", tablefmt="pipe", stralign="center"))
    
    return pd.DataFrame(results_list)


# Generates and saves QuantStats HTML reports for all models
def generate_quantstats_reports(models: dict, X_test: pd.DataFrame, ret_test: pd.Series, bench_ret: pd.Series, threshold: float):

    # Timezone Handling: strip the timezone (tz_localize(None)) to ensure compatibility.
    if bench_ret is not None and bench_ret.index.tz is not None:
        bench_ret.index = bench_ret.index.tz_localize(None)

    # Re-generate predictions and signals for the specific model
    for name, model in models.items():
        p_te = model.predict_proba(X_test)[:, 1]
        sig_te = (p_te >= threshold).astype(int)

        # Calculate strategy returns
        strat_ret = pd.Series(sig_te, index=ret_test.index).shift(1).fillna(0) * ret_test
        turnover = pd.Series(sig_te, index=ret_test.index).diff().abs().fillna(0)
        strat_ret = (strat_ret - turnover * (config.COST_BPS * 1e-4)).dropna()
        strat_ret.name = f"{name} Strategy"
        
         # Strip timezones from the strategy returns as well to match the benchmark
        if strat_ret.index.tz is not None:
            strat_ret.index = strat_ret.index.tz_localize(None)

        # Generate HTML Repor
        html_filename = f"QuantStats_Report_{name}.html"
        qs.reports.html(
            returns=strat_ret,
            benchmark=bench_ret,
            rf=config.RISK_FREE_RATE,
            periods_per_year=config.ANNUAL_TRADING_PERIODS,
            output=html_filename,
            title=f"XAUUSD H1 - {name}"
        )
        print(f"[+] Saved: {html_filename}")