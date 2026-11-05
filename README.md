# E3_REPORT_JIAJUNWU

# 1. Introduction
Gold has a unique position in global financial markets, serving as a store of value, an inflation hedge, and a safe-haven asset during periods of market dislocation. The predictability of gold prices with traditional ARMA, VAR, and GARCH models fail to accommodate the non-linear dynamics and structural breaks characteristic of financial time series. While using machine learning methods, particularly tree-based ensemble classifiers, have emerged as powerful alternatives capable of capturing complex, non-linear interactions without explicit functional form specification.

This report develops a comparative evaluation for a binary classification of XGBoost, LightGBM, and CatBoost classifiers for predicting the direction of hourly XAUUSD price movements. We construct a machine learning pipeline encompassing a three-tier feature selection funnel (mutual information filter, correlation pruning, RFECV wrapper, and L1-embedded method) with Bayesian hyperparameter optimization via Optuna, and a leakage-prevent backtesting framework with purged time-series cross-validation. Using approximately five years of hourly gold data (2020-2026, over 38000 records), we demonstrate that while statistical discrimination remains modest (OOS ROC-AUC: 0.56-0.57), the tuned ensemble models achieve economically meaningful risk-adjusted performance, with XGBoost yielding a Sharpe ratio of 3.95 and CatBoost exhibiting the highest OOS ROC-AUC of 0.5683.

We implement a three-tier feature selection funnel combining filter, wrapper, and embedded methods, explicitly designed to respect the temporal structure of financial data. Second, we employ Bayesian hyperparameter optimization via Optuna with purged time-series cross-validation, ensuring that model selection does not introduce look-ahead bias. Third, we conduct both statistical evaluation (ROC-AUC, MCC, F1) and economic backtesting (Sharpe ratio, maximum drawdown), recognizing that statistical significance and economic viability are distinct constructs.

# 2. Methodology
## 2.1 Data Collection
The data was sourced from MT5 Demo accounts through Raw Trading Ltd, with time stamps synchronized to Coordinated Universal Time (UTC). The sample period deliberately includes diverse market regimes—the COVID-19 pandemic, the subsequent monetary easing cycle, the 2022 inflation shock, and the 2024-2026 gold bull market—to enhance the robustness of empirical findings. 

The binomial classification $Y \in \{0,1 \}$ represents the direction of the prediction, where 1 represents a meaningful positive market move and 0 flat/downwards moves. The model was gemerated based on the belowing 7 steps. 

| Steps        | Workflow                  | Remarks                                                         |
|:-------------|:--------------------------|:----------------------------------------------------------------|
|Step 1    | Ideation  | Define objective, success metrics     |
|Step 2    | Data Collection           | Gather and integrate data
|Step 3    | Exploratory Data Analysis (Initial) | Broad exploration: stats, distributions, correlations, missing data |
|Step 4    | Data Cleaning           | Handle missing values, outliers, duplicates.            |
|Step 5    | Feature Engineering & Transformation            | Feature creation, scaling, encoding, selection                         |
|        | Subset Validation EDA            | Re-examine chosen features: check distributions, multicollinearity, relationships                      |               
|Step 6   | Modeling                  | Select algorithm(s), train models, tune hyperparameters                           |
|Step 7   | Evaluation                   | Validate using metrics and backtesting       |

## 2.2 Feature Engineering and Selection

Features were selected based on different measure dimensions: technical indicators (RSI, MACD, Bollinger Bands), momentum & autocorrelation features, trend & volatility features, volume & price action features, macro & sentiment features and event-based feature (NFP).

### 2.2.1 Filter Method
Filter methods evaluate the intrinsic information of each feature relative to the target variable without relying on any machine learning algorithm.

Mutual Information (MI) Score:
We capture both linear and non-linear dependencies between each feature $X_j$ and the target $Y \in \{0,1 \}$. For continuous features, MI is estimated using the k-nearest neighbor estimator:
$$ I(X_j; Y) = \sum_{x \in X_j} \sum_{y \in Y} p(x, y) \log \left( \frac{p(x, y)}{p(x)p(y)} \right)
\tag{1}$$
Features are ranked by their MI score, and the top K=25 features are retained.

Correlation Pruning:
To eliminate multicollinearity and redundancy, we compute the absolute Pearson correlation matrix. For any pair of features $(X_i, X_j)$ where:
$$|\rho_{ij}| = \left| \frac{\text{Cov}(X_i, X_j)}{\sigma_{X_i} \sigma_{X_j}} \right| > 0.90
\tag{2}$$

We discard the feature with the lower MI score, ensuring the retention of the most informative representative from highly collinear clusters.

### 2.2.2 Wrapper Method
Wrapper methods utilize a specific learning algorithm (a baseline Logistic Regression) to evaluate subsets of features through cross-validation.
We apply Recursive Feature Elimination with Cross-Validation (RFECV). The objective is to find a feature subset $S \in F$ that maximizes the cross-validated ROC-AUC based on the strategy's backtest returns. The optimization function is:
$$
S^\star = \arg \max_{S \subseteq F, |S|=m} \frac{1}{K} \sum_{k=1}^K \text{AUC} \left( \text{Signal}(f_S^{(-k)}, \text{Th}_{(-k)}), R_k \right)
\tag{3}$$

Where $K$ is the number of folds in a TimeSeriesSplit (with a gap to prevent look-ahead bias), $f^{(-k)}_S$ is the model trained on the k-th training fold using subset $S$. $\text{Th}_{(-k)}$ is the optimal probability threshold optimized on the training fold. Signal(⋅) generates trading signals (0 or 1) based on the predicted probabilities and the threshold, and $R_k$ represents the actual future returns in the k-th validation fold. At each step, the feature $j$ with the smallest absolute regression coefficient
$\left| w_j \right|$ is eliminated.

### 2.2.3 Embedded Method
Embedded methods perform feature selection inherently during the model training process. We apply an L1-regularized (Lasso) Logistic Regression. The objective is to minimize the negative log-likelihood with an L1 penalty term, which forces the coefficients of irrelevant features to zero:
$$\min_{w,b} \left( -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \sigma(b + x_i^\top w) + (1 - y_i) \log(1 - \sigma(b + x_i^\top w)) \right] + \lambda \sum_{j=1}^{p} |w_j| \right)
\tag{4}$$
Where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function, and $\lambda $ is the regularization strength. The final selected features are those where the estimated coefficient satisfies $|\hat{w}_j| > \epsilon$.

The geometric property of the $L_1$ penalty creates a diamond-shaped constraint region. When minimizing the loss function, the contours often intersect the diamond at the axes, forcing the coefficients of redundant or less informative features to shrink exactly to zero. This induces sparsity and serves as an intrinsic feature selection mechanism, unlike $L_2$ regularization (Ridge) which only shrinks coefficients asymptotically toward zero without reaching it. The feature funnel pruning is documented at each tier:

| Tier | Method | Statistical / Structural Criterion | Features In | Features Out | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Initial Pool | Domain knowledge & Stationarity (Log, *d*) | — | 44 | Includes Macro, Technicals |
| 1a | Filter (MI) | Max Mutual Information $I(X_j; Y)$| 44 | 25 | Removes non-linear noise |
| 1b | Filter (Corr) | Drop if $\|\rho_{ij}\| > 0.90$ (keep higher MI) | 25 | 22 | Remove US10Y |
| 2 | Wrapper (RFECV) | Maximize Cross-Validated AUC | 22 | 13 | Eliminates features without marginal predictive power |
| 3 | Embedded (L1) | Non-zero $\ell_1$ coefficient weights | 13 | 12 | Final sparsity via Lasso optimization |

Features were discarded if they lacked non-linear dependence (low MI), were highly collinear (redundant macro drivers), or failed to improve the cross-validated OOS AUC. The L1 penalty ensures that only the most robust drivers survive for the final Gradient Boosting Decision Tree (GBDT) training.

Following the three-tier funnel (specifically after the L1 regularized embedded selection), the following 12 features were selected to be fed into the XGBoost, LightGBM, and CatBoost architectures:

| NO | Feature Name | Category | Description & Financial Rationale |
| :--- | :--- | :--- | :--- |
| 1 | `vix_chg_24h` | Macro / Sentiment | 24-hour log change of VIX. Captures market fear and safe-haven demand for gold. |
| 2 | `dxy_chg_8h` | Macro / Sentiment | 8-hour log change of DXY. Captures short-term USD strength, which is typically inversely correlated to gold. |
| 3 | `dxy_corr_48` | Macro / Sentiment | 48-hour rolling correlation between gold and DXY. Captures regime shifts in their traditional inverse relationship. |
| 4 | `rv_48` | Volatility | 48-hour Realized Volatility. Captures medium-term volatility clustering. |
| 5 | `rv_12` | Volatility | 12-hour Realized Volatility. Captures short-term volatility spikes. |
| 6 | `rsi_14` | Technical | 14-period RSI. A momentum oscillator identifying overbought or oversold conditions. |
| 7 | `vol_of_rv_24` | Volatility | Volatility of 24-hour realized volatility. Measures the stability of the current volatility regime. |
| 8 | `rv_6` | Volatility | 6-hour Realized Volatility. Captures ultra-short-term intraday volatility. |
| 9 | `bb_pos` | Technical | Bollinger Band Position. Normalized price position within the bands (0 to 1), indicating breakouts or mean-reversion. |
| 10 | `ret_8h` | Momentum | 8-hour log return. Represents short-term trend momentum. |
| 11 | `macd_hist` | Technical | MACD Histogram. Captures trend-following momentum acceleration or deceleration. |
| 12 | `ret_4h` | Momentum | 4-hour log return. Represents ultra-short-term momentum. |
