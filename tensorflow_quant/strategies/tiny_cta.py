# Reference:
# https://github.com/firmai/machine-learning-asset-management
# https://www.linkedin.com/pulse/implement-cta-less-than-10-lines-code-thomas-schmelzer/

import numpy as np
import pandas as pd

__all__ = ["tiny_cta_backtest"]


def osc(prices, fast=32, slow=96):
    f, g = 1 - 1 / fast, 1 - 1 / slow
    return (
        prices.ewm(span=2 * fast - 1).mean() - prices.ewm(span=2 * slow - 1).mean()
    ) / np.sqrt(1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g))


def tiny_cta_backtest(df_prices):
    prices = df_prices.ffill().truncate(before=pd.Timestamp("2000-01-01"))

    volatility = np.log(prices).diff().ewm(com=32).std()

    cum_returns = (np.log(prices).diff() / volatility).clip(-4.2, 4.2).cumsum()

    positions = (50000 * np.tanh(osc(cum_returns, fast=16, slow=48)) / volatility).clip(
        -5e7, 5e7
    )

    profit = (prices.pct_change() * positions.shift(periods=1)).sum(axis=1)

    return profit
