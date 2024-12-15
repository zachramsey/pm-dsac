import os
import sys
import platformdirs as ad

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

def get_data(data_dir, symbols):
    '''Get the historical stock data for each symbol'''
    session = _configure_session()                           # Configure the requests session
    nyse = mcal.get_calendar('NYSE')                        # NYSE calendar for valid trading days
    if not os.path.exists(data_dir): os.makedirs(data_dir)  # Get the data directory
    
    eod_data = {}
    existing_symbols = []
    new_symbols = []
    i = 0

    # Check if the data directory contains the stock data
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            existing_symbols.append((yf.Ticker(symbol, session=session), symbol, file_path))
        else:
            new_symbols.append((yf.Ticker(symbol, session=session), symbol, file_path))

    # Fetch the historical stock data for symbols not in the data directory
    for (symbol, file_path) in new_symbols:
        last_trading_date = _get_last_trading_date(nyse)
        df = yf.download(symbol, end=last_trading_date+pd.Timedelta(days=1),
                        repair=True, progress=False, rounding=True, session=session)
        
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.to_csv(file_path)    # Save to csv
            eod_data[symbol] = df   # Add to the dictionary
            i += 1
            print(f'{"Loading Historical Data: ":<25}{i:>5} / {len(symbols):<5} | {i/len(symbols)*100:.2f}%', end='\r')
        else:
            symbols.remove(symbol)

    # Load and update the historical stock data for symbols already in the data directory
    for (symbol, file_path) in existing_symbols:
        df = pd.read_csv(file_path, index_col='Date')   # Load data from csv
        df.index = pd.to_datetime(df.index)             # Ensure index is in datetime format

        # Check if the data is up-to-date
        last_local_date = df.index[-1].date()           # Get the last local date
        last_trading_date = _get_last_trading_date(nyse) # Get the last trading date
        if last_local_date < last_trading_date:
            new_data = yf.download(symbol, start=df.index[-1]+pd.Timedelta(days=1), end=last_trading_date+pd.Timedelta(days=1),
                                repair=True, progress=False, rounding=True, session=session)
            df.index = pd.to_datetime(df.index)
            new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            new_data.to_csv(file_path, mode='a', header=False)  # Append to csv
            df = pd.concat([df, new_data])                      # Add new data to the dataframe

        eod_data[symbol] = df                                   # Add to the dictionary
        i += 1
        print(f'{"Loading Historical Data: ":<25}{i:>5} / {len(symbols):<5} | {i/len(symbols)*100:.2f}%', end='\r')

    # Remove symbols with insufficient data
    symbols_by_length = [(symbol, len(eod_data[symbol])) for symbol in eod_data]
    sorted_symbols = sorted(symbols_by_length, key=lambda x: x[1])
    sufficient_data = [symbol for symbol, length in sorted_symbols if length >= 7560]
    symbols = sufficient_data[:256]
    eod_data = {symbol: eod_data[symbol] for symbol in symbols}

    return eod_data, symbols


def _get_last_trading_date(nyse):
    '''Get the last trading date from the NYSE calendar'''
    curr_dt = pd.Timestamp.now(tz='America/New_York')
    today = curr_dt.date()
    schedule = nyse.schedule(start_date=today-pd.DateOffset(days=7), end_date=today)
    last_trading_date = schedule['market_close'].iloc[-2 if curr_dt.hour < 16 else -1].date()
    return last_trading_date


def _configure_session():
    '''Configure the requests session for rate-limiting and caching'''
    # Add the parent directory to the system path
    _parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _src_dp = _parent_dp
    sys.path.insert(0, _src_dp)

    # Use adjacent cache folder for testing, delete if already exists and older than today
    testing_cache_dirpath = os.path.join(ad.user_cache_dir(), "py-yfinance-testing")
    yf.set_tz_cache_location(testing_cache_dirpath)
    if os.path.isdir(testing_cache_dirpath):
        mtime = pd.Timestamp(os.path.getmtime(testing_cache_dirpath), unit='s', tz='UTC')
        if mtime.date() < pd.Timestamp.now().date():
            import shutil
            shutil.rmtree(testing_cache_dirpath)

    # Setup a session to rate-limit and cache persistently:
    class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
        pass
    history_rate = RequestRate(1, Duration.SECOND*0.25)
    limiter = Limiter(history_rate)
    cache_fp = os.path.join(testing_cache_dirpath, "unittests-cache")
    session = CachedLimiterSession(
        limiter=limiter,
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache(cache_fp, expire_after=pd.Timedelta(hours=1)),
    )
    return session
