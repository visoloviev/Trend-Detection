import numpy as np
import pandas as pd
import talib


class FeatureEngineering(object):
    features = []
    def __init__(self, df, feature_engineering=True):
        df['Open Trade'] = np.zeros(df.shape[0])
        df['Duration Trade'] = np.zeros(df.shape[0])

        if feature_engineering:
            df['Original Open'] = df['Open']
            df['Original Close'] = df['Close']
            df['Original High'] = df['High']
            df['Original Low'] = df['Low']
            df['Original Volume'] = df['Volume']
            df['High'] = np.log(df['High'])
            df['Open'] = np.log(df['Open'])
            df['Low'] = np.log(df['Low'])
            df['Close'] = np.log(df['Close'])

        df = self._calculate_return(df)

        df = self._generate_indicators(df)
        df = self._price_engineering(df)
        df = self._volume_engineering(df)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        self.df = df

    def get_df_processed(self):
        return self.df

    def _calculate_return(self, df):
        # Compute Returns based on consecutive closed prices
        df['Return'] = df['Close']
        self.features.append('Return')
        return df

    def _generate_indicators(self, df):
        _high = df.High.values
        _low = df.Low.values
        _close = df.Close.values
        _volume = df.Volume.values

        # Compute the ATR
        df['ATR-10'] = talib.ATR(_high, _low, _close, timeperiod=10)
        self.features.append('ATR-10')
        df['ATR-21'] = talib.ATR(_high, _low, _close, timeperiod=21)
        self.features.append('ATR-21')

        # Compute the EMA
        df['EMA-10'] = talib.EMA(_close, timeperiod=10)
        self.features.append('EMA-10')
        df['EMA-21'] = talib.EMA(_close, timeperiod=21)
        self.features.append('EMA-21')

        # Compute the TEMA
        df['TEMA-10'] = talib.TEMA(_close, timeperiod=10)
        self.features.append('TEMA-10')
        df['TEMA-21'] = talib.TEMA(_close, timeperiod=21)
        self.features.append('TEMA-21')

        # Compute the ADX
        df['ADX-10'] = talib.ADX(_high, _low, _close, timeperiod=10)
        self.features.append('ADX-10')
        df['ADX-21'] = talib.ADX(_high, _low, _close, timeperiod=21)
        self.features.append('ADX-21')

        # Compute the MOM
        df['MOM-10'] = talib.MOM(_close, timeperiod=10)
        self.features.append('MOM-10')
        df['MOM-21'] = talib.MOM(_close, timeperiod=21)
        self.features.append('MOM-21')

        # Compute the AD
        df['AD'] = talib.AD(_high, _low, _close, _volume)
        self.features.append('AD')

        df.dropna(inplace=True)

        return df

    def _price_engineering(self, df):
        # Price Engineering
        # Get opens
        period_list = [1, 2, 3, 4, 5, 10, 21, 63]
        for x in period_list:
            name = '-' + str(x) + 'd_Open'
            self.features.append(name)
            df[name] = df['Open'].shift(x)

        # Get closes
        week_period = range(1, 5 + 1)
        for x in week_period:
            name = '-' + str(x) + 'd_Close'
            self.features.append(name)
            df[name] = df['Close'].shift(x)

        # Get highs
        for x in week_period:
            name = '-' + str(x) + 'd_High'
            self.features.append(name)
            df[name] = df['High'].shift(x)

        large_period_list = [10, 21, 63, 100]
        for x in large_period_list:
            name = str(x) + 'd_High'
            self.features.append(name)
            df[name] = df['High'].shift().rolling(window=x).max()

        # Get lows
        for x in week_period:
            name = '-' + str(x) + 'd_Low'
            self.features.append(name)
            df[name] = df['Low'].shift(x)

        for x in large_period_list:
            name = str(x) + 'd_Low'
            self.features.append(name)
            df[name] = df['High'].shift().rolling(window=x).min()

        return df

    def _volume_engineering(self, df):
        # Volume Engineering
        # Get volumes
        period_list = [1, 2, 3, 4, 5, 10, 21, 63]
        for x in period_list:
            name = '-' + str(x) + 'd_Volume'
            self.features.append(name)
            df[name] = df['Volume'].shift(x)

        large_period_list = [10, 21, 63, 100]
        for x in large_period_list:
            name = str(x) + 'd_Volume'
            self.features.append(name)
            df[name] = df['Volume'].shift().rolling(window=x).mean()

        return df
