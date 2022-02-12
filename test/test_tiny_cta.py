from tensorflow_quant.data import get_stocks_prices_yahoo
from tensorflow_quant.library import DOW_30_TICKER
from tensorflow_quant.strategies import tiny_cta_backtest


def test_tiny_cta():
    df_prices = get_stocks_prices_yahoo(
        DOW_30_TICKER[:2],
        frequency="daily",
        start="2020-01-01",
        end="2020-12-31",
    )

    df_prices = df_prices.iloc[
        :50,
    ]

    backtest_profit = tiny_cta_backtest(df_prices)

    assert backtest_profit.shape[0] == 50
