import pandas_datareader.data as web
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def get_stocks_prices_yahoo(symbols, frequency="daily", start="2019-01-01", end=None):
    raw_prices = web.DataReader(symbols, "yahoo", start=start, end=end)

    prices = raw_prices["Adj Close"]

    if frequency == "monthly":
        resample_frequency = "M"
    elif frequency == "weekly":
        resample_frequency = "W"
    elif frequency == "quarterly":
        resample_frequency = "Q"
    elif frequency == "daily":
        resample_frequency = None
    else:
        raise NotImplementedError(f"Frequency specified {frequency} is not valid.")

    if resample_frequency is not None:
        prices = prices.resample(resample_frequency).last()

    return prices
