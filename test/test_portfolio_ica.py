from quantfinance.data import get_stocks_prices_yahoo
from quantfinance.portfolio_ica import portfolio_ica


def test_portfolio_ica():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG", "DBC", "^VIX"],
        frequency="daily",
        start="2010-07-31",
        end="2020-12-31",
    )

    returns = prices.pct_change().dropna()

    returns = returns.values

    returns_independent, returns_pca, _, _ = portfolio_ica(returns, num_pca=2)

    assert returns_independent.shape[1] == 2
    assert returns_pca.shape[1] == 2


def test_portfolio_ica_no_components():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG", "DBC", "^VIX"],
        frequency="daily",
        start="2010-07-31",
        end="2020-12-31",
    )

    returns = prices.pct_change().dropna()

    returns = returns.values

    returns_independent, returns_pca, _, _ = portfolio_ica(returns, num_pca=None)

    assert returns_independent.shape[1] == 3
    assert returns_pca.shape[1] == 3
