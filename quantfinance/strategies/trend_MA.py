from quantfinance.data import get_stocks_prices_yahoo

def get_data():
    gsci = get_stocks_prices_yahoo(
        ['COMT'],
        frequency="daily",
        start="2020-01-01",
        end="2020-12-31",
    )
