from tensorflow_quant.data import get_stocks_prices_yahoo


def test_get_stocks_prices_yahoo_daily():
    prices = get_stocks_prices_yahoo(
        "AAPL", frequency="daily", start="2020-01-02", end=None
    )

    assert prices.shape[0] == 524


def test_get_stocks_prices_yahoo_weekly():
    prices = get_stocks_prices_yahoo(
        "AAPL", frequency="weekly", start="2020-01-02", end="2020-12-31"
    )

    assert prices.shape[0] == 53


def test_get_stocks_prices_yahoo_monthly():
    prices = get_stocks_prices_yahoo(
        "AAPL", frequency="monthly", start="2020-01-02", end="2020-12-31"
    )

    assert prices.shape[0] == 12


def test_get_stocks_prices_yahoo_quarterly():
    prices = get_stocks_prices_yahoo(
        "AAPL", frequency="quarterly", start="2020-01-02", end="2020-12-31"
    )

    assert prices.shape[0] == 4
