import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)


Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
LSTM = tf.keras.layers.LSTM


def sharpe_loss_generator(training_data):
    def sharpe_loss(_, y_pred):
        # y_pred is of dimension (num_batches, data_dim)
        # y_pred is the weights

        cumulative_returns_assets = (
            tf.reduce_prod(1.0 + training_data / 100, axis=1) - 1.0
        )
        batch_cumulative_returns = tf.reduce_sum(
            cumulative_returns_assets * y_pred, axis=1
        )

        batch_volatility = (
            tf.math.reduce_std(
                tf.reduce_sum(
                    tf.expand_dims(y_pred, axis=1) * training_data / 100, axis=2
                ),
                axis=1,
            )
            * tf.sqrt(training_data.shape[1] * 1.0)
        )

        batch_sharpe_ratio = batch_cumulative_returns / batch_volatility

        sharpe_loss = tf.reduce_mean(batch_sharpe_ratio)

        # since we want to maximize Sharpe, while gradient descent minimizes the loss,
        #   we can negate Sharpe (the min of a negated function is its max)
        return -sharpe_loss

    return sharpe_loss


def max_ex_post_sharpe_ratio(
    returns_data, lookback_window=None, num_batches=10, epochs=20
):
    data = tf.convert_to_tensor(returns_data, dtype=tf.float32) * 100
    num_sample_original = data.shape[0]
    data_dim = data.shape[1]

    if lookback_window is None:
        lookback_window = num_sample_original

    sample_indices = tf.range(num_sample_original - lookback_window + 1)
    sample_indices = tf.random.shuffle(sample_indices)

    if num_batches <= len(sample_indices):
        training_index = sample_indices[:num_batches]
    else:
        num_batches = len(sample_indices)
        training_index = sample_indices

    training_data_numpy = np.zeros(shape=(num_batches, lookback_window, data_dim))
    for i in np.arange(num_batches):
        training_data_numpy[i, :, :] = data[
            training_index[i] : (training_index[i] + lookback_window), :
        ]

    training_data = tf.convert_to_tensor(training_data_numpy, dtype=tf.float32)

    model = Sequential(
        [
            LSTM(64, input_shape=(lookback_window, data_dim)),
            Flatten(),
            Dense(data_dim, activation="softmax"),
        ]
    )

    model.compile(loss=sharpe_loss_generator(training_data), optimizer="adam")

    model.fit(
        training_data,
        np.zeros((num_batches, data_dim)),
        epochs=epochs,
        shuffle=False,
        batch_size=num_batches,
    )

    return model, model.predict(training_data)


def max_ex_ante_sharpe_ratio(
    returns_data,
    lookback_window=None,
    lookforward_window=None,
    num_batches=10,
    epochs=20,
):
    data = tf.convert_to_tensor(returns_data, dtype=tf.float32) * 100
    num_sample_original = data.shape[0]
    data_dim = data.shape[1]

    if lookback_window is None:
        lookback_window = int(num_sample_original / 2)

    if lookforward_window is None:
        lookforward_window = int(num_sample_original / 2)

    sample_indices = tf.range(
        num_sample_original - (lookback_window + lookforward_window) + 1
    )
    sample_indices = tf.random.shuffle(sample_indices)

    if num_batches <= len(sample_indices):
        training_index = sample_indices[:num_batches]
    else:
        num_batches = len(sample_indices)
        training_index = sample_indices

    training_data_numpy = np.zeros(shape=(num_batches, lookback_window, data_dim))
    training_target_numpy = np.zeros(shape=(num_batches, lookforward_window, data_dim))
    for i in np.arange(num_batches):
        training_data_numpy[i, :, :] = data[
            training_index[i] : (training_index[i] + lookback_window), :
        ]
        training_target_numpy[i, :, :] = data[
            (training_index[i] + lookback_window) : (
                training_index[i] + lookback_window + lookforward_window
            ),
            :,
        ]

    training_data = tf.convert_to_tensor(training_data_numpy, dtype=tf.float32)
    training_target = tf.convert_to_tensor(training_target_numpy, dtype=tf.float32)

    model = Sequential(
        [
            LSTM(64, input_shape=(lookback_window, data_dim)),
            Flatten(),
            Dense(data_dim, activation="softmax"),
        ]
    )

    model.compile(loss=sharpe_loss_generator(training_target), optimizer="adam")

    model.fit(
        training_data,
        np.zeros((num_batches, data_dim)),
        epochs=epochs,
        shuffle=False,
        batch_size=num_batches,
    )

    return model, model.predict(training_data)
