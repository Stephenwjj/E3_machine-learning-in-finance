"""
Module for loading and merging XAUUSD data with macroeconomic drivers.
"""
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

def load_and_merge_data(file_path: str) -> pd.DataFrame:
    """
    Loads local XAUUSD H1 data and merges with macro indicators (DXY, US10Y, VIX).
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Merged and timezone-localized dataframe.
    """
    logger.info(f"Loading local data from {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = df.columns.str.strip('<>')
        df['Datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str)).dt.tz_localize('UTC')
        df = df.set_index('Datetime').drop(columns=['DATE', 'TIME', 'VOL'], errors='ignore').copy()
        df.columns = df.columns.str.lower()
    except Exception as e:
        logger.error(f"Failed to load local data: {e}")
        raise

    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    
    logger.info(f"Downloading macro data (DXY, US10Y, VIX) from {start_date} to {end_date}")
    try:
        macro = yf.download(['DX-Y.NYB', '^TNX', '^VIX'], start=start_date, end=end_date, progress=False)['Close']
        macro.columns = ['dxy', 'us10y', 'vix']
        # Shift daily data to prevent look-ahead bias, then resample to hourly
        macro = macro.shift(1).resample('h').ffill()
        
        if macro.index.tz is None: 
            macro.index = macro.index.tz_localize('UTC')
        else: 
            macro.index = macro.index.tz_convert('UTC')
            
        df = df.join(macro, how='left')
        df[['dxy', 'us10y', 'vix']] = df[['dxy', 'us10y', 'vix']].ffill()
    except Exception as e:
        logger.warning(f"Failed to download macro data: {e}. Proceeding with technical features only.")
        
    return df