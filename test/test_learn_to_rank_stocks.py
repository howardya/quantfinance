import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

from tensorflow_quant.data import get_stocks_prices_yahoo
from tensorflow_quant.strategies import (
    convert_returns_to_features,
    learn_to_rank_stocks_CNN,
    learn_to_rank_stocks_NN,
)


def test_convert_returns_to_features():
    tf.random.set_seed(9710)
    returns_data = tf.random.uniform((250, 2))

    training_data, ranking_data = convert_returns_to_features(
        returns_data, lookback_window=30, lookforward_window=30, num_batches=1
    )

    assert training_data.shape[0] == 1
    assert ranking_data.shape[0] == 1


def test_convert_returns_to_features_large_batches():
    tf.random.set_seed(9710)
    returns_data = tf.random.uniform((250, 2))

    training_data, ranking_data = convert_returns_to_features(
        returns_data, lookback_window=30, lookforward_window=30, num_batches=1000
    )

    assert training_data.shape[1] == 2
    assert ranking_data.shape[1] == 2


def test_learn_to_rank_stocks_NN():
    training_data = np.array([[1.0, 0.0], [0.2, 0.0], [1.0, 2.0], [3.0, -0.5]])[
        None, ...
    ]
    training_data = tf.convert_to_tensor(training_data, dtype=float)
    ranking_data = np.array([30, 40, 1, 20])[None, ...]
    ranking_data = tf.convert_to_tensor(ranking_data, dtype=float)

    tf.random.set_seed(9710)
    model, history = learn_to_rank_stocks_NN(
        training_data, ranking_data, epochs=1, loss=tfr.keras.losses.ListMLELoss()
    )

    assert model.predict(training_data).shape[1] == 4


def test_learn_to_rank_stocks_actual_data():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG"],
        frequency="daily",
        start="2002-07-31",
        end="2020-12-31",
    )
    returns_data = prices.pct_change()[1:].dropna()
    returns_data = tf.convert_to_tensor(returns_data, dtype=tf.float32)

    training_data, ranking_data = convert_returns_to_features(
        returns_data, lookforward_window=125, lookback_window=250, num_batches=100
    )

    tf.random.set_seed(9710)
    model, history = learn_to_rank_stocks_NN(
        training_data, ranking_data, epochs=200, loss=tfr.keras.losses.ListMLELoss()
    )

    assert model.predict(training_data).shape[0] == 100


def test_learn_to_rank_stocks_CNN():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG", "DBC", "^VIX"],
        frequency="daily",
        start="2002-07-31",
        end="2020-12-31",
    )
    returns_data = prices.pct_change()[1:].dropna()
    returns_data = tf.convert_to_tensor(returns_data, dtype=tf.float32)

    training_data, ranking_data = convert_returns_to_features(
        returns_data,
        lookforward_window=125,
        lookback_window=250,
        num_batches=100,
        method="full",
    )

    tf.random.set_seed(9710)
    model, history = learn_to_rank_stocks_CNN(
        training_data, ranking_data, epochs=1, loss=tfr.keras.losses.ListMLELoss()
    )

    assert model.predict(training_data).shape[1] == 4
