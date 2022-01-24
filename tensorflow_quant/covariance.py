import numpy as np
import sklearn as sk
import tensorflow as tf
import tensorflow_probability as tfp
from arch.univariate import arch_model

tf.config.experimental_run_functions_eagerly(True)

__ALL__ = [
    "get_covariance",
]


def dcc_garch_loss_fn_generator(
    data_std_resid, a_variable, b_variable, ab_scalar_variable
):
    # https://github.com/Topaceminem/DCC-GARCH/blob/master/examples/dcc_garch_modeling.ipynb
    n_samples = data_std_resid.shape[0]
    data_dim = data_std_resid.shape[1]

    data_batch_correlation = tf.expand_dims(data_std_resid, 1) * tf.expand_dims(
        data_std_resid, 2
    )

    # calculate the averate outerproduct (average correlation)
    average_correlation = tf.reduce_mean(data_batch_correlation, axis=0)

    def dcc_garch_loss_fn():
        a_variable_scaled = ab_scalar_variable * a_variable
        b_variable_scaled = ab_scalar_variable * b_variable
        loss = 0

        correlation_previous = average_correlation
        for sample_i in range(1, n_samples):
            correlation_new = (
                (1.0 - a_variable_scaled - b_variable_scaled) * average_correlation
                + a_variable_scaled * data_batch_correlation[sample_i - 1, :, :]
                + b_variable_scaled * correlation_previous
            )

            inv_vol_scalar = (
                1 / tf.sqrt(tf.maximum(tf.abs(correlation_new), 0.01))
            ) * tf.eye(data_dim)

            correlation_new = inv_vol_scalar @ correlation_new @ inv_vol_scalar

            correlation_new_inverse = tf.linalg.inv(correlation_new)

            loss += tf.math.log(tf.linalg.det(correlation_new)) + tf.reduce_sum(
                data_std_resid[sample_i, :][None, :]
                @ correlation_new_inverse
                @ data_std_resid[sample_i, :][:, None]
            )

            correlation_previous = correlation_new

            return loss

    return dcc_garch_loss_fn


def fit_forecast_garch(data):
    data = tf.cast(data, dtype=tf.float64)
    data_dim = data.shape[1]

    univariate_garch_models = []
    data_std_resid = tf.zeros_like(data).numpy()

    forecast_variance = []
    for dim_i in range(data_dim):
        garchModel = arch_model(
            data[:, dim_i].numpy(),
            mean="constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
        )
        garchModelResult = garchModel.fit(disp="off")
        univariate_garch_models.append(garchModelResult)
        forecast_variance.append(
            garchModelResult.forecast(reindex=False).variance.values[0][0]
        )
        data_std_resid[:, dim_i] = (
            garchModelResult.resid / garchModelResult.conditional_volatility
        )

    data_std_resid = tf.convert_to_tensor(data_std_resid, dtype=tf.float32)

    return data_std_resid, univariate_garch_models, forecast_variance


def fit_forecast_dcc_garch(data, **kwargs):
    optimizer = kwargs.get("optimizer", None)

    # arch package require float64
    data_dim = data.shape[1]
    n_samples = data.shape[0]

    data_std_resid, univariate_garch_models, forecast_variance = fit_forecast_garch(
        data
    )

    data_std_resid = tf.convert_to_tensor(data_std_resid, dtype=tf.float32)

    a_variable = tfp.util.TransformedVariable(0.2, bijector=tfp.bijectors.Sigmoid())

    b_variable = tfp.util.TransformedVariable(0.7, bijector=tfp.bijectors.Sigmoid())

    ab_scalar_variable = tfp.util.TransformedVariable(
        0.5, bijector=tfp.bijectors.Sigmoid()
    )

    gcc_garch_loss_fn = dcc_garch_loss_fn_generator(
        data_std_resid, a_variable, b_variable, ab_scalar_variable
    )

    if optimizer is None:
        optimizer = tf.optimizers.Adam(0.1)

    def trace_fn(traceable_quantities):
        return {
            "loss": traceable_quantities.loss,
            "has_converged": traceable_quantities.has_converged,
        }

    _ = tfp.math.minimize(
        loss_fn=gcc_garch_loss_fn,
        num_steps=1000,
        trace_fn=trace_fn,
        trainable_variables=[a_variable, b_variable, ab_scalar_variable],
        convergence_criterion=tfp.optimizer.convergence_criteria.LossNotDecreasing(
            rtol=0.01, min_num_steps=100, name=None
        ),
        return_full_length_trace=False,
        optimizer=optimizer,
    )

    data_batch_correlation = tf.expand_dims(data_std_resid, 1) * tf.expand_dims(
        data_std_resid, 2
    )

    # calculate the averate outerproduct (average correlation)
    average_correlation = tf.reduce_mean(data_batch_correlation, axis=0)

    a_variable_scaled = ab_scalar_variable * a_variable
    b_variable_scaled = ab_scalar_variable * b_variable

    correlation_now = average_correlation
    for sample_i in range(n_samples):
        correlation_new = (
            (1.0 - a_variable_scaled - b_variable_scaled) * average_correlation
            + a_variable_scaled * data_batch_correlation[sample_i, :, :]
            + b_variable_scaled * correlation_now
        )
        inv_vol_scalar = (
            1 / tf.sqrt(tf.maximum(tf.abs(correlation_new), 0.01))
        ) * tf.eye(data_dim)
        correlation_new = inv_vol_scalar @ correlation_new @ inv_vol_scalar
        correlation_now = correlation_new

    vol_diagonal = tf.convert_to_tensor(np.diag(np.sqrt(forecast_variance)))

    covariance = vol_diagonal @ correlation_now @ vol_diagonal

    return covariance


def fit_forecast_ccc_garch(data):
    data_dim = data.shape[1]

    data_std_resid, univariate_garch_models, forecast_variance = fit_forecast_garch(
        data
    )

    data_std_resid = tf.convert_to_tensor(data_std_resid, dtype=tf.float32)

    correlation = tfp.stats.covariance(data_std_resid)
    inv_vol_scalar = (1 / tf.sqrt(tf.maximum(tf.abs(correlation), 0.01))) * tf.eye(
        data_dim
    )
    correlation = inv_vol_scalar @ correlation @ inv_vol_scalar

    vol_diagonal = tf.convert_to_tensor(np.diag(np.sqrt(forecast_variance)))

    covariance = vol_diagonal @ correlation @ vol_diagonal

    return covariance


def get_covariance(data, method="empirical", return_correlation=False, **kwargs):
    if method == "empirical":
        print("a")
        covariance = tfp.stats.covariance(data, **kwargs)
    elif method == "graphical_lasso":
        covariance, _ = sk.covariance.graphical_lasso(
            tfp.stats.covariance(data).numpy(), **kwargs
        )
        covariance = tf.convert_to_tensor(covariance)
    elif method == "ledoit_wolf":
        covariance, _ = sk.covariance.ledoit_wolf(data, **kwargs)
        covariance = tf.convert_to_tensor(covariance)
    elif method == "shrunk_covariance":
        covariance, _ = sk.covariance.shrunk_covariance(
            tfp.stats.covariance(data).numpy(), **kwargs
        )
        covariance = tf.convert_to_tensor(covariance)
    elif method == "oas":
        covariance, _ = sk.covariance.oas(data, **kwargs)
        covariance = tf.convert_to_tensor(covariance)
    elif method == "dcc-garch":
        covariance = fit_forecast_dcc_garch(data, **kwargs)
    elif method == "ccc-garch":
        covariance = fit_forecast_ccc_garch(data)

    diagonal_inv_vol = tf.eye(covariance.shape[0]) * tf.sqrt(
        1 / tf.maximum(tf.abs(covariance), 0.1)
    )

    correlation = diagonal_inv_vol @ covariance @ diagonal_inv_vol

    if return_correlation:
        return covariance, correlation

    return covariance
