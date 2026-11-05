# E3_REPORT_Machine-Learning-in-Finance

# 1. Introduction

Gold has a unique position in global financial markets, serving as a store of value, an inflation hedge, and a safe-haven asset during periods of market dislocation. The predictability of gold prices with traditional ARMA, VAR, and GARCH models fail to accommodate the non-linear dynamics and structural breaks characteristic of financial time series. While using machine learning methods, particularly tree-based ensemble classifiers, have emerged as powerful alternatives capable of capturing complex, non-linear interactions without explicit functional form specification.

This report develops a comparative evaluation for a binary classification of XGBoost, LightGBM, and CatBoost classifiers for predicting the direction of hourly XAUUSD price movements. We construct a machine learning pipeline encompassing a three-tier feature selection funnel (mutual information filter, correlation pruning, RFECV wrapper, and L1-embedded method) with Bayesian hyperparameter optimization via Optuna, and a leakage-prevent backtesting framework with purged time-series cross-validation. Using approximately five years of hourly gold data (2020-2026, over 38000 records), we demonstrate that while statistical discrimination remains modest (OOS ROC-AUC: 0.56-0.57), the tuned ensemble models achieve economically meaningful risk-adjusted performance, with XGBoost yielding a Sharpe ratio of 3.95 and CatBoost exhibiting the highest OOS ROC-AUC of 0.5683.


We implement a three-tier feature selection funnel combining filter, wrapper, and embedded methods, explicitly designed to respect the temporal structure of financial data. Second, we employ Bayesian hyperparameter optimization via Optuna with purged time-series cross-validation, ensuring that model selection does not introduce look-ahead bias. Third, we conduct both statistical evaluation (ROC-AUC, MCC, F1) and economic backtesting (Sharpe ratio, maximum drawdown), recognizing that statistical significance and economic viability are distinct constructs.

