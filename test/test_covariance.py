import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_quant.covariance import get_covariance

tf.config.experimental_run_functions_eagerly(True)


def test_empirical_covariance():
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
            data = tfp.distributions.Normal(loc=0.0, scale=volatilities_true).sample(
                n_samples
            )
        else:
            tril_true = tf.squeeze(
                tfp.distributions.CholeskyLKJ(
                    dimension=data_dim, concentration=1.5
                ).sample(1)
            )
            scale_tril_true = tril_true * tf.expand_dims(volatilities_true, 0)
            # covariance_true = tf.linalg.diag(volatilities_true) @ correlation_true @  tf.linalg.diag(volatilities_true)

            data = tfp.distributions.MultivariateNormalTriL(
                loc=tf.constant(0.0), scale_tril=scale_tril_true
            ).sample(n_samples)

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

    data_dim = 4
    n_samples = 1500

    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=n_samples,
        data_dim=data_dim,
        correlated=False,
        unit_volatilities=False,
    )

    covariance, correlation = get_covariance(data, return_correlation=True)

    assert abs(correlation[0, 0] - 1.0) < 0.01
    # assert covariance[0, 0].numpy() == pytest.approx(
    #     tfp.stats.covariance(data)[0, 0].numpy(), 0.01
    # )
