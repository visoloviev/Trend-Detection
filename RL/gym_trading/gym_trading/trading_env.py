import itertools

import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt

import pprint

import talib

import gym

from .simulator import Simulator
from .portfolio import Portfolio


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._seed = 42

    def initialise_simulator(self, data, min_trade_period=0, max_trade_period=1000, train_split=0.8, denom=0.01, cost=3, feature_engineering=True):
        self.sim = Simulator(data, train_split=train_split, train=True, feature_engineering=feature_engineering, min_trade_period=min_trade_period, max_trade_period=max_trade_period)
        self.portfolio = Portfolio(min_trade_period=min_trade_period, max_trade_period=max_trade_period, denom=denom, cost=cost)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=self.sim.min_values, high=self.sim.max_values, dtype=np.float32)

    def step(self, action):
        # Return the observation, done, reward from Simulator and Portfolio
        reward, info = self.portfolio.step(action)
        obs, done = self.sim.step(self.portfolio.open_trade, self.portfolio.curr_trade["Trade Duration"])
        return obs, reward, done, info

    def reset(self):
        obs = self.sim.reset()
        self.portfolio.reset(prices=self.sim._data[['Date', 'Open', 'Close']], stock=self.sim._data['Stock'].iloc[0])
        return obs

    def generate_summary_stats(self, train=True, render_matplotlib=False, render_plotly=False, print_details=False):
        print("SUMMARY STATISTICS OF " + ("TRAIN" if train else "TEST"))

        journal = pd.DataFrame(self.portfolio.journal)
        if len(journal) == 0:
            print("No trades were taken")
            return journal
        journal = journal[journal['Trade Duration'] != 0]
        print("Total trades taken: ", journal.shape[0])
        print("Total profit: ", journal['Profit'].sum())
        print("Average profit per trade: ", journal['Profit'].sum() / journal['Profit'].count())
        print("Win ratio: %s %%" % (((journal.loc[journal['Profit'] > 0, 'Profit'].count()) / journal.shape[0]) * 100))

        if render_matplotlib or render_plotly:
            data = self.sim._data
            buys = journal.loc[journal.Type == 'BUY', :]
            sells = journal.loc[journal.Type == 'SELL', :]

        if render_matplotlib:
            # Get a OHLC list with tuples (dates, Open, High, Low, Close)
            ohlc = list(zip(mdates.date2num(data['Date'].dt.to_pydatetime()),
                            data['Open'].tolist(),
                            data['High'].tolist(),
                            data['Low'].tolist(),
                            data['Close'].tolist()))

            fig, ax = plt.subplots(figsize=(40, 10))

        # Plotting functions
            mf.candlestick_ohlc(ax, ohlc, width=0.02, colorup='green', colordown='red')
            ax.plot(buys['Entry Time'], buys['Entry Price'] - 0.001, 'b^', alpha=1.0)
            ax.plot(buys['Exit Time'], buys['Exit Price'] - 0.001, 'bv', alpha=1.0)
            ax.plot(sells['Entry Time'], sells['Entry Price'] + 0.001, 'r^', alpha=1.0)
            ax.plot(sells['Exit Time'], sells['Exit Price'] + 0.001, 'rv', alpha=1.0)

            plt.show()

        if render_plotly:
            from time import gmtime, strftime
            import plotly as py

            # from plotly.offline import init_notebook_mode, iplot
            # init_notebook_mode(connected=True)

            trace_stock = py.graph_objs.Candlestick(x=data.Date,
                                                    open=data.Open,
                                                    high=data.High,
                                                    low=data.Low,
                                                    close=data.Close,
                                                    name=data['Stock'].iloc[0],
                                                    visible='legendonly')

            buy_trace = py.graph_objs.Scattergl(x=buys['Entry Time'],
                                                y=buys['Entry Price'] - 0.001,
                                                mode='markers',
                                                marker=dict(color='blue',
                                                            symbol="triangle-up",
                                                            size=10),
                                                name='BUY',
                                                visible='legendonly',
                                                text=buys['Trade Duration'])

            buy_trace_close = py.graph_objs.Scattergl(x=buys['Exit Time'],
                                                      y=buys['Exit Price'] - 0.001,
                                                      mode='markers',
                                                      marker=dict(color='blue',
                                                                  symbol="triangle-down",
                                                                  size=10),
                                                      name='Close BUY',
                                                      visible='legendonly',
                                                      text=buys['Trade Duration'])

            sell_trace = py.graph_objs.Scattergl(x=sells['Entry Time'],
                                                 y=sells['Entry Price'] + 0.001,
                                                 mode='markers',
                                                 marker=dict(color='purple',
                                                             symbol="triangle-up",
                                                             size=10),
                                                 name='SELL',
                                                 visible='legendonly',
                                                 text=sells['Trade Duration'])

            sell_trace_close = py.graph_objs.Scattergl(x=sells['Exit Time'],
                                                       y=sells['Exit Price'] + 0.001,
                                                       mode='markers',
                                                       marker=dict(color='purple',
                                                                   symbol="triangle-down",
                                                                   size=10),
                                                       name='Close SELL',
                                                       visible='legendonly',
                                                       text=sells['Trade Duration'])

            all_data = py.graph_objs.Data((trace_stock, buy_trace, buy_trace_close, sell_trace, sell_trace_close))
            layout = py.graph_objs.Layout(xaxis=dict(autorange=True,
                                          rangeslider=dict(visible=False)),
                                          autosize=True,
                                          showlegend=True)

            fig = py.graph_objs.Figure(data=all_data, layout=layout)
            # iplot(fig)
            py.offline.plot(fig,
                            show_link=False,
                            filename='plot-{0}-{1}-{2}.html'.format("train" if train else "test", data['Stock'].iloc[0], strftime("%Y-%m-%d %H-%M-%S", gmtime())),
                            auto_open=False)

        if print_details:
            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint([d for d in self.portfolio.journal if d['Trade Duration'] != 0])

        return journal
