# Ranking stocks
# References:
# https://www.tensorflow.org/recommenders/examples/listwise_ranking
# https://arxiv.org/abs/2012.07149

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

__all__ = ["learn_to_rank_stocks_NN", "learn_to_rank_stocks_CNN"]


def convert_returns_to_features(
    returns_data, lookback_window, lookforward_window, num_batches, method="simple"
):
    num_sample_original = returns_data.shape[0]
    num_stocks = returns_data.shape[1]
    sample_indices = tf.range(num_sample_original - lookback_window + 1)
    sample_indices = tf.random.shuffle(sample_indices)

    if num_batches <= len(sample_indices):
        training_index = sample_indices[:num_batches]
    else:
        num_batches = len(sample_indices)
        training_index = sample_indices

    if method == "simple":
        training_data_numpy = np.zeros(
            shape=(num_batches, num_stocks, 2)
        )  # annual return and half yearly return
        ranking_data_numpy = np.zeros(shape=(num_batches, num_stocks))
        for i in np.arange(num_batches):
            training_data_numpy[i, :, 0] = tf.reduce_sum(
                returns_data[
                    training_index[i] : (training_index[i] + lookback_window), :
                ],
                axis=0,
            )
            training_data_numpy[i, :, 1] = tf.reduce_sum(
                returns_data[
                    (training_index[i] + int(lookback_window / 2)) : (
                        training_index[i] + lookback_window
                    ),
                    :,
                ],
                axis=0,
            )
            ranking_data_numpy[i, :] = tf.reduce_sum(
                returns_data[
                    (training_index[i] + lookback_window) : (
                        training_index[i] + lookback_window + lookforward_window
                    ),
                    :,
                ],
                axis=0,
            )
    elif method == "full":
        training_data_numpy = np.zeros(
            shape=(num_batches, num_stocks, lookback_window)
        )  # annual return and half yearly return
        ranking_data_numpy = np.zeros(shape=(num_batches, num_stocks))
        for i in np.arange(num_batches):
            training_data_numpy[i, :, :] = tf.transpose(
                returns_data[
                    training_index[i] : (training_index[i] + lookback_window), :
                ]
            )

            ranking_data_numpy[i, :] = tf.reduce_sum(
                returns_data[
                    (training_index[i] + lookback_window) : (
                        training_index[i] + lookback_window + lookforward_window
                    ),
                    :,
                ],
                axis=0,
            )

    training_data = tf.convert_to_tensor(training_data_numpy, dtype=tf.float32)
    ranking_data = tf.convert_to_tensor(ranking_data_numpy, dtype=tf.float32)

    return training_data, ranking_data


def learn_to_rank_stocks_NN(
    training_data, ranking_data, loss=tfr.keras.losses.ListMLELoss(), epochs=1
):
    ranking_model = tf.keras.Sequential(
        [
            # Learn multiple dense layers.
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1)),
        ]
    )
    ranking_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(0.1), loss=loss, metrics=[]
    )
    history = ranking_model.fit(training_data, ranking_data, epochs=epochs)

    return ranking_model, history


def learn_to_rank_stocks_CNN(
    training_data, ranking_data, loss=tfr.keras.losses.ListMLELoss(), epochs=1
):
    _training_data = tf.expand_dims(training_data, axis=-1)
    ranking_model = tf.keras.Sequential()
    ranking_model.add(
        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=5,
            input_shape=_training_data.shape[1:],
            activation="relu",
        )
    )
    ranking_model.add(
        tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 3, 1]))
    )
    ranking_model.add(
        tf.keras.layers.AveragePooling2D(pool_size=(20, 1), data_format="channels_last")
    )
    ranking_model.add(
        tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 3, 1, 2]), name="transpose"
        )
    )
    ranking_model.add(
        tf.keras.layers.Reshape(
            (
                ranking_model.get_layer("transpose").output_shape[1],
                ranking_model.get_layer("transpose").output_shape[2]
                * ranking_model.get_layer("transpose").output_shape[3],
            ),
            input_shape=ranking_model.get_layer("transpose").output_shape[1:],
        )
    )
    ranking_model.add(tf.keras.layers.Dense(64, activation="relu"))
    ranking_model.add(tf.keras.layers.Dense(1))
    ranking_model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1)))

    ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5), loss=loss)
    history = ranking_model.fit(_training_data, ranking_data, epochs=epochs)

    return ranking_model, history
