import investpy
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from quantfinance.utils import get_sharpe_ratio


def get_data_investpy(
    from_date=None,
    to_date=None
):
    if from_date is None:
        from_date = '01/01/1990'
    if to_date is None:
        to_date = '01/05/2022'

    # get_etfs_name_from_symbol('IEF')
    # TLT = investpy.get_etf_historical_data(
    #     etf='iShares 20+ Year Treasury Bond',
    #     country='United States',
    #     from_date=from_date,
    #     to_date=to_date
    # )

    TLH = investpy.get_etf_historical_data(
        etf='iShares 20+ Year Treasury Bond',
        country='United States',
        from_date=from_date,
        to_date=to_date
    )

    EWJ = investpy.get_etf_historical_data(
        etf='iShares MSCI Japan',
        country='United States',
        from_date=from_date,
        to_date=to_date,
        stock_exchange=None
    )

    data = pd.concat([
        TLH[['Close']].rename(columns={'Close': 'UST'}),
        EWJ[['Close']].rename(columns={'Close': 'JPEQ'})
    ], axis=1)

    return data


def get_data_yf():
    TLT = yf.Ticker('TLT').history(period='max')
    EWJ = yf.Ticker('EWJ').history(period='max')

    data = pd.concat([
        TLT[['Close']].rename(columns={'Close': 'UST'}),
        EWJ[['Close']].rename(columns={'Close': 'JPEQ'})
    ], axis=1)

    return data


def backtest(
    data,
    rebal_frequency='weekly'
):
    data['JPEQ_SMA_150'] = ta.sma(data['JPEQ'], length=50)
    data['DAILY_SIGNAL'] = data['JPEQ'] > data['JPEQ_SMA_150']
    data['UST_Returns'] = data['UST'].pct_change()

    # data = data.dropna()

    data = pd.concat([
        data,
        data.index.isocalendar()
    ], axis=1)

    # SIGNAL is either 0 or 1, indicating long or no position
    data['SIGNAL'] = np.nan

    if rebal_frequency == 'weekly':
        last_weekday_signal = data[['DAILY_SIGNAL', 'year', 'week']].groupby(['year', 'week']).tail(1)
        last_weekday_signal['LASTWEEK_SIGNAL'] = last_weekday_signal['DAILY_SIGNAL'].shift(1)

        # start on a fresh week
        trading_index = data.index[1:]
        for _time_index in trading_index:
            _year = data.loc[_time_index].year
            _week = data.loc[_time_index].week
            if np.isnan(last_weekday_signal.query('year==@_year & week==@_week')['LASTWEEK_SIGNAL'].values[0]):
                data.loc[_time_index, 'SIGNAL'] = 1
            elif last_weekday_signal.query('year==@_year & week==@_week')['LASTWEEK_SIGNAL'].values[0]:
                data.loc[_time_index, 'SIGNAL'] = 1
            else:
                data.loc[_time_index, 'SIGNAL'] = 0
    elif rebal_frequency == 'daily':
        data['SIGNAL'] = (data['JPEQ'] > data['JPEQ_SMA_150']).shift(1)

    data['Returns'] = data['SIGNAL'] * data['UST_Returns']
    data['WealthRatio'] = 1. + data['Returns']
    data['CumulativeReturns'] = data['WealthRatio'].cumprod().add(-1)

    return data[['Returns', 'WealthRatio', 'CumulativeReturns', 'SIGNAL']]


def main():
    data = get_data_yf()

    # dont use, it is not adjusted with dividends
    # _data = get_data_investpy()

    weekly_rebal_returns = backtest(
        data=data.copy(),
        rebal_frequency='weekly'
    )

    weekly_rebal_returns.columns = weekly_rebal_returns.columns + '_Weekly'

    daily_rebal_returns = backtest(
        data=data.copy(),
        rebal_frequency='daily'
    )

    daily_rebal_returns.columns = daily_rebal_returns.columns + '_Daily'

    all_returns = pd.concat([
        daily_rebal_returns,
        weekly_rebal_returns
    ], axis=1).dropna()

    print('Sharpe Ratio for Weekly Rebal: ' + '%.2f' % (get_sharpe_ratio(all_returns['Returns_Daily'])))
