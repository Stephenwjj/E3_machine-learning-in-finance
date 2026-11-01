# utils.py
import logging
import numpy as np
import pandas as pd

def setup_logger():
    """Sets up a production-grade logging system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
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