import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess.stock_symbols import StockSymbols
from preprocess.stock_data import StockData

# ==================================================================================================
class StockDataset(Dataset):
    def __init__(self, dates, features, targets, window_size):
        self.dates = dates
        self.features = features
        self.targets = targets
        self.window_size = window_size
    
    def __len__(self):
        return len(self.dates) - self.window_size + 1
    
    def __getitem__(self, idx):
        # Get the window of asset features
        f = self.features[:, idx:idx+self.window_size, :].clone()
        
        # Normalize the prices
        f[:, :, :3] /= f[:, -1, 2].reshape(-1, 1, 1).expand(-1, f.size(1), 3)

        # # Normalize the volume
        # f[:, :, 3] /= f[:, -1, 3].reshape(-1, 1).expand(-1, f.size(1))
        
        # Get the target
        t = self.targets[:, idx+self.window_size-1]

        return f, t

# ==================================================================================================
class StockDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

        self.symbols = None         # Symbol data class
        self.data = None            # Stock data class

        self.dates = None           # Common dates among all stock symbols
        self.features = None        # Features for each stock symbol
        self.targets = None         # Targets for each stock symbol

        # Retrieve the data
        if cfg["debug"] and os.path.exists(cfg["pickle_dir"]):
            self.load_data()
        else:
            self.fetch_data()

        self.train_dataloader = None
        self.test_dataloader = None

        self.create_dataloaders()   # Create the data loaders

    def get_train_data(self):
        return self.train_dataloader
    
    def get_test_data(self):
        return self.test_dataloader
            
    def load_data(self):
        '''Load local pickled data'''
        print(f'{"-"*50}\nLoading Pickled Data...', end='\r')
        self.symbols = pd.read_pickle(self.cfg["pickle_dir"]+"symbols.pkl")
        self.dates = pd.read_pickle(self.cfg["pickle_dir"]+"dates.pkl")
        self.features = pd.read_pickle(self.cfg["pickle_dir"]+"features.pkl")
        self.targets = pd.read_pickle(self.cfg["pickle_dir"]+"targets.pkl")
        print(f'Loading Pickled Data... Done!\n{"-"*50}')

    def save_data(self):
        '''Save the data to local pickle files'''
        print(f'{"Saving Pickled Data...":<25}', end='\r')
        pd.to_pickle(self.symbols, self.cfg["pickle_dir"]+"symbols.pkl")
        pd.to_pickle(self.dates, self.cfg["pickle_dir"]+"dates.pkl")
        pd.to_pickle(self.features, self.cfg["pickle_dir"]+"features.pkl")
        pd.to_pickle(self.targets, self.cfg["pickle_dir"]+"targets.pkl")
        print(f'Saving Pickled Data... Done!\n{"-"*50}')

    def fetch_data(self):
        '''Fetch the stock data from the web'''
        print(f'{"-"*50}\nFetching Stock Symbols...', end='\r')
        self.symbols = StockSymbols()
        print(f'Fetching Stock Symbols... Done!\n{"-"*50}')

        print(f'{"Fetching Historical Data: ":<25}', end='\r')
        self.data = StockData(self.symbols, self.cfg)
        print(f'\n{"-"*50}\nFetching Historical Data... Done!\n{"-"*50}')

        print(f'{"Preparing Features: ":<25}', end='\r')
        self._validate_symbols()
        self._extract_features()
        self._clean_data()
        self._initialize_tensors()
        print(f'Done!\n{"-"*120}')

        if self.cfg["debug"] and not os.path.exists(self.cfg["pickle_dir"]):
            os.makedirs(self.cfg["pickle_dir"])
            self.save_data()

    def _validate_symbols(self):
        '''Validate symbols based on minimum volume criteria and data length'''
        # Filter out symbols with insufficient volume
        suff_vol_symbols = [symbol for symbol in self.symbols if self.data[symbol]['Volume'].mean() > self.cfg["min_volume"]]
        self.symbols.audit(suff_vol_symbols)

        # Choose symbols with top data length
        symbols_by_length = [(symbol, len(self.data[symbol])) for symbol in self.symbols]
        sorted_symbols = sorted(symbols_by_length, key=lambda x: x[1])
        suff_len_symbols = [symbol for symbol, _ in sorted_symbols[-(self.cfg["asset_dim"]-1):]]
        self.symbols.audit(suff_len_symbols)

        # Update the data to only include the chosen symbols
        self.data.audit(self.symbols)

    def _extract_features(self):
        '''Extract features from the stock data'''
        for i, symbol in enumerate(self.symbols):
            print(f'{"Extracting Features: ":<25}{(i+1):>5} / {len(self.symbols):<5} | {(i+1)/len(self.symbols)*100:.2f}%', end='\r')
            df = self.data[symbol].copy()
            df.drop(columns=['Open'], inplace=True)
            df.drop(columns=['Volume'], inplace=True)
            df.dropna(inplace=True)
            self.data[symbol] = df

    def _clean_data(self):
        '''Clean the data by removing dates that are not common among all stock symbols'''
        print(f'\nCleaning Up Data...')
        self.dates = set(self.data.get_dates(self.symbols[0]))
        for symbol in self.symbols:
            self.dates = self.dates.intersection(set(self.data.get_dates(symbol)))
        self.dates = sorted(list(self.dates))

        for symbol in self.symbols:
            self.data[symbol] = self.data[symbol].loc[self.dates]

    def _initialize_tensors(self):
        '''Initialize the tensors for the features and targets'''
        self.targets = torch.zeros(len(self.symbols), len(self.dates))
        self.features = torch.zeros(len(self.symbols), len(self.dates), len(self.data.get_features(self.symbols[0])))

        # Fill prototype tensors with data
        for i, (symbol, df) in enumerate(self.data):
            self.targets[i] = torch.from_numpy((df['Close']/df['Close'].shift(1)).values)
            self.features[i] = torch.from_numpy(df.values)

        # Prepend additional asset representing cash with everything set to 1
        self.targets = torch.cat([torch.ones((1, *self.targets.shape[1:])), self.targets], dim=0)
        self.features = torch.cat([torch.ones((1, *self.features.shape[1:])), self.features], dim=0)

        # Append additional feature representing asset weights with everything set to 0
        self.features = torch.cat([self.features, torch.zeros((*self.features.shape[:-1], 1))], dim=-1)

    def create_dataloaders(self):
        '''Create the data loaders for training and testing'''
        start_eval = int(len(self.dates) * self.cfg["train_ratio"])

        # Slice the data into training and testing sets
        train_dates = self.dates[:start_eval]
        train_features = self.features[:, :start_eval]
        train_targets = self.targets[:, :start_eval]

        test_dates = self.dates[start_eval:]
        test_features = self.features[:, start_eval:]
        test_targets = self.targets[:, start_eval:]

        # Create the data loaders
        train_dataset = StockDataset(train_dates, train_features, train_targets, self.cfg["window_size"])
        self.train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_dataset = StockDataset(test_dates, test_features, test_targets, self.cfg["window_size"])
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Save values to the config
        self.cfg["symbols"] = self.symbols.data
        self.cfg["train_len"] = len(train_dataset)
        self.cfg["test_len"] = len(test_dataset)
        self.cfg["asset_dim"] = self.features.size(0)
        self.cfg["feat_dim"] = self.features.size(2)
