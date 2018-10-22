import numpy as np
import pandas as pd

import numbers
from six import string_types
from itertools import cycle

from .feature_engineering import FeatureEngineering


class Simulator(object):
    def __init__(self, data, train_split, train=True, feature_engineering=True, min_trade_period=0, max_trade_period=1):
        if isinstance(data, string_types):  # Assumes that data is path to a file
            df = pd.read_csv(data, usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Stock'])
            df = df[~np.isnan(df['Open'])]
            df['Date'] = df['Date'].apply(func=lambda x: x.split(' ')[0])
            df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
            df = FeatureEngineering(df, feature_engineering).get_df_processed()
        elif isinstance(data, (set, tuple, list, np.ndarray)):  # Assumes that data is devided into chunks
            data = [FeatureEngineering(chunk.copy(), feature_engineering).get_df_processed() for chunk in data]
            df = pd.concat(data)
            df.reset_index(drop=True, inplace=True)

        # Attributes
        self.chunks = data
        self.all_data = df
        # self.date = df['Date']
        self.count = len(data)
        train_end_index = train_split if isinstance(train_split, (numbers.Integral, np.integer)) else int(train_split * self.count)
        self.train_data = self.chunks[:train_end_index]
        self.test_data = self.chunks[train_end_index:]

        self.state_columns = list(set(FeatureEngineering.features)) + ['Open Trade', 'Duration Trade']  # 'Open Trade' and 'Duration Trade' always should be last columns
        print("State description: " + str(self.state_columns))

        self.min_values = self.all_data[self.state_columns].values.min(axis=0)
        self.min_values[-1] = min_trade_period
        self.max_values = self.all_data[self.state_columns].values.max(axis=0)
        self.max_values[-2] = 1
        self.max_values[-1] = max_trade_period

        self._switch_train_test(train=True)

    def _switch_train_test(self, train=True):
        if train:
            self.episodes = cycle(self.train_data)
        else:
            self.episodes = cycle(self.test_data)

    def reset(self):
        self._data = next(self.episodes)
        self.states = self._data[self.state_columns].values
        self.current_index = 1
        self._end = len(self.states) - 1
        return self.states[0]

    def step(self, open_trade, duration_trade):
        if open_trade:
            obs = self.states[self.current_index]
            obs[-1] += duration_trade
            obs[-2] += open_trade
        else:
            obs = self.states[self.current_index]
            
        self.current_index += 1
        done = self.current_index > self._end

        return obs, done
