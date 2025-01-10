import os
import sys

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

import platformdirs as ad
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

from preprocess.stock_symbols import StockSymbols

class StockData:
    def __init__(self, symbols: StockSymbols, cfg):
        self.symbols = symbols                                      # List of stock symbols
        self.data_dir = cfg["data_dir"]                             # Directory to store data
        if not os.path.exists(self.data_dir):                       # Create the directory if it doesn't exist
            os.makedirs(self.data_dir)

        self.i = 0                                                  # Initialize counter
        self.data = {}                                              # Initialize dictionary to store data

        self.session = self._configure_session()                    # Configure the requests session
        self.nyse = mcal.get_calendar('NYSE')                       # NYSE calendar for valid trading days

        existing_symbols, new_symbols = self._separate_symbols()    # Separate symbols into existing and new
        self._download_new_data(new_symbols)                        # Download data for new symbols
        self._update_existing_data(existing_symbols)                # Update data for existing symbols

    def __len__(self):
        '''Return the number of stock symbols'''
        return len(self.data)

    def __getitem__(self, symbol):
        '''Return the historical data for the given symbol'''
        return self.data[symbol]
    
    def __setitem__(self, symbol, data):
        '''Set the historical data for the given symbol'''
        self.data[symbol] = data
    
    def __iter__(self):
        '''Return an iterator for the historical data'''
        return iter(self.data.items())
    
    def audit(self, incl_symbols):
        '''Update the data to only include the given symbols'''
        # self.symbols.audit(incl_symbols)    # Ensure symbols match the data
        self.data = {symbol: self.data[symbol] for symbol in incl_symbols}

    def get_dates(self, symbol):
        '''Return the dates included in a symbol's data'''
        return self.data[symbol].index
    
    def get_features(self, symbol):
        '''Return the features included in a symbol's data'''
        return self.data[symbol].columns
    
    def shape(self):
        '''Return the shape of the data (number of symbols, number of dates, number of features)'''
        num_symbols = len(self.data)
        num_dates = len(self.data[list(self.data.keys())[0]])
        num_features = len(self.data[list(self.data.keys())[0]].columns)
        return num_symbols, num_dates, num_features

    def _separate_symbols(self):
        '''Separate symbols into existing and new based on data directory'''
        existing_symbols = []
        new_symbols = []
        for symbol in self.symbols:
            file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            if os.path.exists(file_path):
                existing_symbols.append((symbol, file_path))
            else:
                new_symbols.append((symbol, file_path))
        return existing_symbols, new_symbols

    def _download_new_data(self, new_symbols):
        '''Download historical data for new symbols'''
        failed_symbols = []  # Initialize list to track failed symbols
        for (symbol, file_path) in new_symbols:
            last_trading_date = self._get_last_trading_date()
            df = yf.download(symbol, end=last_trading_date + pd.Timedelta(days=1),
                            repair=True, progress=False, rounding=True, session=self.session)
            
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                df = df.droplevel("Ticker", axis=1) if 'Ticker' in df.columns.names else df
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.to_csv(file_path)    # Save to csv
                self.data[symbol] = df   # Add to the dictionary
                self.i += 1
                print(f'{"Fetching Historical Data: ":<25}{self.i:>5} / {len(new_symbols):<5} | {self.i/len(new_symbols)*100:.2f}%', end='\r')
            else:
                failed_symbols.append(symbol)  # Track the failed symbol

        # Remove failed symbols from the list
        self.symbols = [symbol for symbol in self.symbols if symbol not in failed_symbols]

    def _update_existing_data(self, existing_symbols):
        '''Update historical data for existing symbols'''
        for (symbol, file_path) in existing_symbols:
            df = pd.read_csv(file_path, index_col='Date')   # Load data from csv
            df.index = pd.to_datetime(df.index)             # Ensure index is in datetime format

            # Check if the data is up-to-date
            last_local_date = df.index[-1].date()           # Get the last local date
            last_trading_date = self._get_last_trading_date() # Get the last trading date
            if last_local_date < last_trading_date:
                new_data = yf.download(symbol, start=df.index[-1] + pd.Timedelta(days=1),
                                    end=last_trading_date + pd.Timedelta(days=1),
                                    repair=True, progress=False, rounding=True, session=self.session)
                if not new_data.empty:
                    new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                    new_data.to_csv(file_path, mode='a', header=False)  # Append to csv
                    df = pd.concat([df, new_data])                      # Add new data to the dataframe

            self.data[symbol] = df                                   # Add to the dictionary
            self.i += 1
            print(f'{"Loading Historical Data: ":<25}{self.i:>5} / {len(existing_symbols):<5} | {self.i/len(existing_symbols)*100:.2f}%', end='\r')

    def _get_last_trading_date(self):
        '''Get the last trading date from the NYSE calendar'''
        curr_dt = pd.Timestamp.now(tz='America/New_York')
        today = curr_dt.date()
        schedule = self.nyse.schedule(start_date=today - pd.DateOffset(days=7), end_date=today)
        last_trading_date = schedule['market_close'].iloc[-2 if curr_dt.hour < 16 else -1].date()
        return last_trading_date

    def _configure_session(self):
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
        self.session = CachedLimiterSession(
            limiter=limiter,
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache(cache_fp, expire_after=pd.Timedelta(hours=1)),
        )
