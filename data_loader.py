import os
import pandas as pd
# import pandas_ta as ta
import torch
from tensordict import MemoryMappedTensor, TensorDict
from torch.utils.data import Dataset, DataLoader

from utils.fetch_symbols import get_sp500_symbols
from utils.fetch_data import get_data

# ==================================================================================================
class StockDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]                             # Get the window of data
        x['features'][:, :, :3] /= x['features'][-1, :, 2].unsqueeze(-1)    # Normalize prices
        x['features'][:, :, 3] /= x['features'][-1, :, 3].unsqueeze(-1)     # Normalize volume
        return x

# ==================================================================================================
class StockData:
    def __init__(self, cfg):
        self.cfg = cfg

        self.symbols = None
        self.sector = None
        self.industry = None
        self.eod_data = None

        self.dates = None           # Common dates among all stock symbols
        self.features = None        # Features for each stock symbol
        self.targets = None         # Targets for each stock symbol

        # Load the stock data
        if cfg["debug"] and os.path.exists("pickles/dates.pkl"):
            print(f'{"-"*50}\nLoading Pickled Data...', end='\r')
            self.symbols = pd.read_pickle("pickles/symbols.pkl")
            self.dates = pd.read_pickle("pickles/dates.pkl")
            self.features = pd.read_pickle("pickles/features.pkl")
            self.targets = pd.read_pickle("pickles/targets.pkl")
            print(f'Loading Pickled Data... Done!\n{"-"*50}')
        else:
            print(f'{"-"*50}\Fetching Stock Symbols...', end='\r')
            self.symbols, self.sector, self.industry = get_sp500_symbols()
            print(f'{"-"*50}\Fetching Stock Symbols... Done!\n{"-"*50}')

            print(f'{"Fetching Historical Data: ":<25}', end='\r')
            self.eod_data, self.symbols = get_data(self.cfg["data_dir"], self.symbols)
            print(f'\n{"-"*50}\Fetching Historical Data... Done!\n{"-"*50}')

            print(f'\n{"Extracting Features: ":<25}', end='\r')
            self._prepare_data()
            print(f'Done!\n{"-"*120}')

            if cfg["debug"]:
                pd.to_pickle(self.symbols, "pickles/symbols.pkl")
                pd.to_pickle(self.dates, "pickles/dates.pkl")
                pd.to_pickle(self.features, "pickles/features.pkl")
                pd.to_pickle(self.targets, "pickles/targets.pkl")

    def _prepare_data(self):
        # Extract features for each stock symbol
        features = {}
        for i, symbol in enumerate(self.symbols):
            df = self.eod_data[symbol].copy()
            df['DayofWeek'] = df.index.dayofweek + 1
            df['Month'] = df.index.month
            df['Sector'] = self.sector[symbol]
            df['Industry'] = self.industry[symbol]
            df['Target'] = df['Close'] / df['Close'].shift(1)

            df.drop(columns=['Open'], inplace=True)
            df.dropna(inplace=True)
            features[symbol] = df
            print(f'{"Extracting Features: ":<25}{(i+1):>5} / {len(self.symbols):<5} | {(i+1)/len(self.symbols)*100:.2f}%', end='\r')
        print(f'\nPocessing Data...')

        # Get the common dates among the data
        self.dates = set(self.eod_data[self.symbols[0]].index)
        for symbol in self.symbols[1:]:
            self.dates = self.dates.intersection(set(self.eod_data[symbol].index))
        self.dates = sorted(list(self.dates))

        # Initialize prototype tensors
        self.features = torch.zeros(len(self.dates), len(self.symbols), len(features[self.symbols[0]].columns))
        self.targets = torch.zeros(len(self.dates), len(self.symbols))

        # Fill prototype tensors with data
        for symbol, df in self.eod_data.items():
            df = df.loc[self.dates]
            self.features[:, self.symbols.index(symbol), :] = torch.tensor(df[features[symbol].columns].values)
            self.targets[:, self.symbols.index(symbol)] = torch.tensor(df['Targets'].values)
    
    def get_data_loaders(self):
        start_eval = int(len(self.dates) * self.cfg["train_ratio"])

        # Slice the data into training and testing sets
        train_dates = self.dates[:start_eval]
        train_features = self.features[:start_eval]
        train_targets = self.targets[:start_eval]

        test_dates = self.dates[start_eval:]
        test_features = self.features[start_eval:]
        test_targets = self.targets[start_eval:]

        # Create the tensordicts for training and testing
        train_tensordict = TensorDict({
            'features': MemoryMappedTensor(train_features.shape, dtype=torch.float32),
            'targets': MemoryMappedTensor(train_targets.shape, dtype=torch.float32),
        })
        test_tensordict = TensorDict({
            'features': MemoryMappedTensor(test_features.shape, dtype=torch.float32),
            'targets': MemoryMappedTensor(test_targets.shape, dtype=torch.float32),
        })

        # Load the data into the tensordicts
        train_tensordict['features'][:] = train_features
        train_tensordict['targets'][:] = train_targets
        test_tensordict['features'][:] = test_features
        test_tensordict['targets'][:] = test_targets

        # Create the data loaders
        train_data = StockDataset(train_tensordict, self.cfg["window_size"])
        train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
        test_data = StockDataset(test_tensordict, self.cfg["window_size"])
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        # Save values to the config
        self.cfg['symbols'] = self.symbols
        self.cfg["train_len"] = len(train_dates)
        self.cfg["test_len"] = len(test_dates)
        self.cfg["obs_dim"] = train_features.shape[-2]
        self.cfg["feat_dim"] = train_features.shape[-1]
        self.cfg["act_dim"] = train_targets.shape[-1]

        return train_loader, test_loader, self.cfg