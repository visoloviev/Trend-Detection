class Portfolio(object):
    def __init__(self, min_trade_period=0, max_trade_period=1, denom=0.01, cost=3):
        self.min_trade_period = min_trade_period
        self.max_trade_period = max_trade_period
        self.trading_cost = cost
        self.reward_normalizer = 1. / denom
        self.open_trade = False

    def reset(self, prices, stock):
        # Store list of Open price and Close price to manage reward calculation
        self._open = prices['Open'].values
        self._close = prices['Close'].values
        self._index = prices['Date']
        self._stock = stock

        self.total_reward = 0
        self.total_trades = 0
        self.average_profit_per_trade = 0
        self.count_open_trades = 0
        self.journal = []
        self.current_time = 1
        self._reset_trade()
        self.open_trade = False

    def _reset_trade(self):
        self.curr_trade = {'Entry Price': 0, 'Exit Price': 0, 'Entry Time': None, 'Exit Time': None, 'Profit': 0,
                           'Trade Duration': 0, 'Type': None, 'reward': 0, 'Stock': self._stock}

    def close_trade(self, curr_close_price, curr_time):
        reward = 0
        if self.curr_trade['Type'] == 'SELL':
            self.count_open_trades -= 1

            # Update remaining keys in curr_trade dict
            self.curr_trade['Exit Price'] = curr_close_price
            self.curr_trade['Exit Time'] = curr_time
            reward = -1 * (curr_close_price - self.curr_trade['Entry Price']) * self.reward_normalizer - self.trading_cost
            self.curr_trade['Profit'] = reward
            self.curr_trade['reward'] = reward

        if self.curr_trade['Type'] == 'BUY':
            self.count_open_trades -= 1

            # Update remaining  keys in curr_trade dict
            self.curr_trade['Exit Price'] = curr_close_price
            self.curr_trade['Exit Time'] = curr_time
            reward = (curr_close_price - self.curr_trade['Entry Price']) * self.reward_normalizer - self.trading_cost
            self.curr_trade['Profit'] = reward
            self.curr_trade['reward'] = reward

        # Add curr_trade to journal, then reset curr_trade
        self.journal.append(self.curr_trade)

        self._reset_trade()
        self.open_trade = False

        return reward

    def _holding_trade(self, curr_close_price, prev_close_price):
        self.curr_trade['Trade Duration'] += 1
        return 0

    def step(self, action):
        curr_open_price = self._open[self.current_time]
        curr_close_price = self._close[self.current_time]
        curr_time = self._index.iloc[self.current_time]
        prev_close_price = self._close[self.current_time - 1]
        reward = 0

        if action == 3 or self.curr_trade['Trade Duration'] >= self.max_trade_period:
            # Closing trade or trade duration is reached
            if self.curr_trade['Trade Duration'] >= self.min_trade_period:
                reward = self.close_trade(curr_close_price, curr_time)
            else:
                reward = self._holding_trade(curr_close_price, prev_close_price)

        elif action == 1:
            if not self.open_trade:
                # BUYING
                self.curr_trade['Entry Price'] = curr_open_price
                self.curr_trade['Type'] = "BUY"
                self.curr_trade['Entry Time'] = curr_time
                self.curr_trade['Trade Duration'] += 1
                reward = 0  # (curr_close_price - curr_open_price) * self.reward_normalizer - self.trading_cost
                self.total_trades += 1
                self.open_trade = True
                self.count_open_trades += 1
            else:
                reward = self._holding_trade(curr_close_price, prev_close_price)

        elif action == 2:
            if not self.open_trade:
                # SELLING
                self.curr_trade['Entry Price'] = curr_open_price
                self.curr_trade['Type'] = "SELL"
                self.curr_trade['Entry Time'] = curr_time
                self.curr_trade['Trade Duration'] += 1
                reward = 0  # -1 * (curr_close_price - curr_open_price) * self.reward_normalizer - self.trading_cost
                self.total_trades += 1
                self.open_trade = True
                self.count_open_trades += 1
            else:
                reward = self._holding_trade(curr_close_price, prev_close_price)

        elif action == 0:
            # Holding trade
            if self.open_trade:
                reward = self._holding_trade(curr_close_price, prev_close_price)
            else:
                pass

        self.total_reward += reward

        if self.total_trades > 0:
            self.average_profit_per_trade = self.total_reward / self.total_trades

        self.current_time += 1

        info = {'Average reward per trade': self.average_profit_per_trade,
                'Reward for this trade': reward,
                'Total reward': self.total_reward}

        return reward, info
