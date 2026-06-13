from tvDatafeed import TvDatafeed, Interval
import pandas as pd 
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"

def get_XAUUSD_data(symbol='GOLD', exchange='TVC', bars_to_pull=4000):
    tv = TvDatafeed()
    df = tv.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=bars_to_pull)
    
    # --- NEW: NORMALIZE DATE ---
    # This strips the 02:00, 16:30, etc. so only the date remains
    df.index = pd.to_datetime(df.index).date 
    
    df.columns = ['Symbol','XAUUSD_Open', 'XAUUSD_High', 'XAUUSD_Low', 'XAUUSD_Close', 'XAUUSD_Volume']

    df = df[["XAUUSD_Close", "XAUUSD_Open", "XAUUSD_High", "XAUUSD_Low"]].dropna()
    return df

def get_DXY_data(symbol='DXY', exchange='TVC', bars_to_pull=4000):
    tv = TvDatafeed()
    df = tv.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','DXY_Open', 'DXY_High', 'DXY_Low', 'DXY_Close', 'DXY_Volume']
    df = df["DXY_Close"].dropna()         
    return df

def get_SPX_data(symbol='SPX', exchange='SP', bars_to_pull=4000):
    tv = TvDatafeed()
    df = tv.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','SPX_Open', 'SPX_High', 'SPX_Low', 'SPX_Close', 'SPX_Volume']
    df = df["SPX_Close"].dropna()

    return df

def get_US10Y_data(symbol='US10Y', exchange='TVC', bars_to_pull=4000):
    tv = TvDatafeed()
    df = tv.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','US10Y_Open', 'US10Y_High', 'US10Y_Low', 'US10Y_Close', 'US10Y_Volume']
    df = df["US10Y_Close"].dropna()
    return df

def get_VIX_data(symbol='VIX', exchange='CBOE', bars_to_pull=4000):
    tv = TvDatafeed()
    df = tv.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=bars_to_pull)
    df.index = pd.to_datetime(df.index).date 
    df.columns = ['Symbol','VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Close', 'VIX_Volume']
    df = df["VIX_Close"].dropna()

    return df

def pull_master():
    df_XAUUSD = get_XAUUSD_data()
    time.sleep(1)
    df_DXY = get_DXY_data()
    time.sleep(1)
    df_SPX = get_SPX_data()
    time.sleep(1)
    df_US10Y = get_US10Y_data()
    time.sleep(1)
    df_VIX = get_VIX_data()

    # --- THE JOIN ---
    # Use join='inner' to only keep days where ALL 5 assets were open (recommended)
    df_master = pd.concat([df_XAUUSD, df_DXY, df_SPX, df_US10Y, df_VIX], axis=1, join='inner')
    
    # Save with the date index
    DATABASE_DIR.mkdir(exist_ok=True)
    df_master.to_csv(DATABASE_DIR / 'database.csv', index=True)
    print("Master dataset saved! Check the file now - the gaps should be gone.")




pull_master()
