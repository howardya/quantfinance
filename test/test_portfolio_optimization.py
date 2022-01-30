import tensorflow as tf

from tensorflow_quant.data import get_stocks_prices_yahoo
from tensorflow_quant.portfolio_optimization import (
    max_ex_post_sharpe_ratio,
    sharpe_loss_generator,
)


def test_sharpe_loss_generator():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG"],
        frequency="daily",
        start="2020-12-20",
        end="2020-12-31",
    )
    returns_data = prices.pct_change()[1:].dropna()

    returns_data = tf.convert_to_tensor(returns_data, dtype=tf.float32)
    training_window = returns_data.shape[0]

    tf.random.set_seed(9710)
    loss_fn = sharpe_loss_generator(tf.random.uniform((1, training_window, 2)))

    loss = loss_fn(1.0, tf.constant([0.5, 0.5])[None, :])

    assert abs(loss - -6.607573) < 0.1


def test_max_ex_post_sharpe_ratio_without_training_window():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG"],
        frequency="daily",
        start="2002-07-31",
        end="2020-12-31",
    )
    returns_data = prices.pct_change()[1:].dropna()

    returns_data = tf.convert_to_tensor(returns_data, dtype=tf.float32)

    _, predicted_weights = max_ex_post_sharpe_ratio(returns_data, epochs=1)

    assert predicted_weights.shape[0] == 1
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
        returns_data, training_window=250, epochs=1
    )

    assert predicted_weights.shape[0] == 15
    assert predicted_weights.shape[1] == 4
