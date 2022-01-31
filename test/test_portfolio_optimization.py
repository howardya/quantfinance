import tensorflow as tf

from tensorflow_quant.data import get_stocks_prices_yahoo
from tensorflow_quant.portfolio_optimization import (
    max_ex_post_sharpe_ratio,
    sharpe_loss_generator,
)


def test_sharpe_loss_generator():
    tf.random.set_seed(9710)
    loss_fn = sharpe_loss_generator(tf.random.uniform((1, 250, 2)))

    loss = loss_fn(1.0, tf.constant([0.5, 0.5])[None, :])

    assert abs(loss - -78.2962) < 0.1


def test_max_ex_post_sharpe_ratio_without_training_window():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG"],
        frequency="daily",
        start="2002-07-31",
        end="2020-12-31",
    )
    returns_data = prices.pct_change()[1:].dropna()

    returns_data = tf.convert_to_tensor(returns_data, dtype=tf.float32)

    _, predicted_weights = max_ex_post_sharpe_ratio(
        returns_data, num_batches=1, epochs=1
    )

    assert predicted_weights.shape[0] == 1
    assert predicted_weights.shape[1] == 2


def test_max_ex_post_sharpe_ratio_with_large_num_batches():
    tf.random.set_seed(9710)
    returns_data = tf.random.uniform((50, 2))

    _, predicted_weights = max_ex_post_sharpe_ratio(
        returns_data, lookback_window=40, num_batches=40, epochs=1
    )

    assert predicted_weights.shape[0] == 11
    assert predicted_weights.shape[1] == 2


def test_max_ex_post_sharpe_ratio():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG", "DBC", "^VIX"],
        frequency="daily",
        start="2002-07-31",
        end="2020-12-31",
    )
    returns_data = prices.pct_change()[1:].dropna()

    returns_data = tf.convert_to_tensor(returns_data, dtype=tf.float32)

    _, predicted_weights = max_ex_post_sharpe_ratio(
        returns_data, lookback_window=250, num_batches=50, epochs=1
    )

    assert predicted_weights.shape[0] == 50
    assert predicted_weights.shape[1] == 4
