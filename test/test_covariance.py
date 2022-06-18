import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from quantfinance.covariance import (
    dcc_garch_loss_fn_generator,
    fit_forecast_ccc_garch,
    fit_forecast_dcc_garch,
    fit_forecast_garch,
    get_covariance,
)

tf.config.run_functions_eagerly(True)


def generate_multivariate_data(
    n_samples=1500, data_dim=4, correlated=False, unit_volatilities=False
):
    tf.random.set_seed(9711)

    if unit_volatilities:
        volatilities_true = tf.ones(data_dim)
    else:
        volatilities_true = tfp.distributions.Uniform(low=2.0, high=10.0).sample(
            data_dim
        )

    if correlated:
        tril_true = tf.squeeze(
            tfp.distributions.CholeskyLKJ(dimension=data_dim, concentration=1.5).sample(
                1
            )
        )
        scale_tril_true = tril_true * tf.expand_dims(volatilities_true, 0)
        # covariance_true = tf.linalg.diag(volatilities_true) @ correlation_true @  tf.linalg.diag(volatilities_true)

        data = tfp.distributions.MultivariateNormalTriL(
            loc=tf.constant(0.0), scale_tril=scale_tril_true
        ).sample(n_samples)
    else:
        data = tfp.distributions.Normal(loc=0.0, scale=volatilities_true).sample(
            n_samples
        )

    covariance_true = tfp.stats.covariance(data)
    correlation_true = tfp.stats.correlation(data)
    eigenvalues_true, eigenvectors_true = tf.linalg.eig(covariance_true)
    eigenvalues_true = tf.math.real(eigenvalues_true)
    eigenvectors_true = tf.math.real(eigenvectors_true)

    return (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    )


def test_dcc_garch_loss_fn_generator():
    tf.random.set_seed(9710)
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1000,
        data_dim=2,
        correlated=False,
        unit_volatilities=False,
    )
    loss_fn = dcc_garch_loss_fn_generator(data, 0.1, 0.9)
    assert abs(loss_fn().numpy() - 49.984715) < 0.01


def test_fit_forecast_garch():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1000,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )
    data_std_resid, univariate_garch_models, forecast_variance = fit_forecast_garch(
        data
    )
    assert abs(forecast_variance[0] - 1.0) < 0.1
    assert abs(forecast_variance[1] - 1.0) < 0.1


def test_fit_forecast_dcc_garch():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=5000,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )
    covariance = fit_forecast_dcc_garch(data)
    assert abs(covariance.numpy()[0, 0] - 1.0) < 0.1
    assert abs(covariance.numpy()[1, 1] - 1.0) < 0.1


def test_fit_forecast_ccc_garch():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=5000,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )
    covariance = fit_forecast_ccc_garch(data)
    assert abs(covariance.numpy()[0, 0] - 1.0) < 0.1
    assert abs(covariance.numpy()[1, 1] - 1.0) < 0.1


def test_covariance_with_one_dimentional_data():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=1,
        correlated=False,
        unit_volatilities=True,
    )

    with pytest.raises(ValueError):
        get_covariance(data, return_correlation=True)


def test_covariance_empirical():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(data, return_correlation=True)

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_covariance_empirical_without_correlation():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance = get_covariance(data, return_correlation=False)

    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_get_covariance_graphical_lasso():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(
        data,
        method="graphical_lasso",
        return_correlation=True,
    )

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_get_covariance_ledoit_wolf():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(
        data,
        method="ledoit_wolf",
        return_correlation=True,
    )

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_get_covariance_shrunk_covariance():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(
        data,
        method="shrunk_covariance",
        return_correlation=True,
    )

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_get_covariance_oas():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(
        data,
        method="oas",
        return_correlation=True,
    )

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_get_covariance_dcc_garch():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(
        data,
        method="dcc-garch",
        return_correlation=True,
    )

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1


def test_get_covariance_ccc_garch():
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1500,
        data_dim=2,
        correlated=False,
        unit_volatilities=True,
    )

    covariance, correlation = get_covariance(
        data,
        method="ccc-garch",
        return_correlation=True,
    )

    assert abs(correlation[0, 0] - 1.0) < 0.1
    assert abs(correlation[1, 1] - 1.0) < 0.1
    assert abs(covariance[0, 0] - 1.0) < 0.1
    assert abs(covariance[1, 1] - 1.0) < 0.1
