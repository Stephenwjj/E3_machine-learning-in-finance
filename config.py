# config.py

# Data configuration
FILE_PATH = 'XAUUSD_H1_202001020100_202606302300.csv'
FORWARD_HOURS = 12
COST_BPS = 4.0
VOL_MULTIPLIER = 1.0

# Dataset splitting ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# TEST_RATIO is implicitly the remaining 0.2

# Model hyperparameters
MODEL_PARAMS = {
    "XGBoost": {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.03,
        "tree_method": "hist", "random_state": 42
    },
    "LightGBM": {
        "n_estimators": 200, "num_leaves": 15, "learning_rate": 0.03,
        "random_state": 42, "verbose": -1
    },
    "CatBoost": {
        "iterations": 200, "depth": 4, "learning_rate": 0.03,
        "verbose": False, "random_seed": 42
    }
}

# Feature Selection configuration
MI_TOP_K = 25
CORR_THRESHOLD = 0.90
L1_C = 0.1

# Trading configuration
ANNUAL_TRADING_PERIODS = 5796 # H1 periods per year = 23 * 252 = 5796
RISK_FREE_RATE = 0.02