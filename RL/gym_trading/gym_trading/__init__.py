from gym.envs.registration import register

from .trading_env import TradingEnv

register(
    id='trading-v1',
    entry_point='gym_trading:TradingEnv'
)
