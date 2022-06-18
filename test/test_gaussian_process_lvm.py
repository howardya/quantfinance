from test.test_covariance import generate_multivariate_data

import tensorflow as tf

from quantfinance.gaussian_process_lvm import gaussian_process_lvm


def test_gaussian_process_lvm_MLE_rbf():
    tf.random.set_seed(9710)
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1000,
        data_dim=4,
        correlated=False,
        unit_volatilities=False,
    )

    loss, kernel, trainable_variables = gaussian_process_lvm(
        tf.transpose(data),
        kernel=None,
        kernel_type="rbf",
        latent_dim=2,
        estimation_method="MLE",
        num_steps=1,
    )

    latent_variables = [a for a in trainable_variables if "latent_variables" in a.name][
        0
    ]

    assert loss.shape[0] == 1
    assert latent_variables.shape[1] == 2
    assert kernel.apply(tf.ones(3), tf.ones(3)).numpy().shape == ()


def test_gaussian_process_lvm_MLE_linear():
    tf.random.set_seed(9710)
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1000,
        data_dim=4,
        correlated=False,
        unit_volatilities=False,
    )

    loss, kernel, trainable_variables = gaussian_process_lvm(
        tf.transpose(data),
        kernel=None,
        kernel_type="linear",
        latent_dim=2,
        estimation_method="MLE",
        num_steps=1,
    )

    latent_variables = [a for a in trainable_variables if "latent_variables" in a.name][
        0
    ]

    assert loss.shape[0] == 1
    assert latent_variables.shape[1] == 2
    assert kernel.apply(tf.ones(3), tf.ones(3)).numpy().shape == ()


def test_gaussian_process_lvm_VI():
    tf.random.set_seed(9710)
    (
        data,
        covariance_true,
        correlation_true,
        eigenvalues_true,
        eigenvectors_true,
    ) = generate_multivariate_data(
        n_samples=1000,
        data_dim=4,
        correlated=False,
        unit_volatilities=False,
    )

    loss, kernel, trainable_variables, surrogate_posterior = gaussian_process_lvm(
        tf.transpose(data),
        kernel=None,
        kernel_type="rbf",
        latent_dim=2,
        estimation_method="VI",
        num_steps=1,
    )

    assert len(trainable_variables) == 4
    assert surrogate_posterior.sample().shape[1] == 2
    assert kernel.apply(tf.ones(3), tf.ones(3)).numpy().shape == ()
    assert loss.shape[0] == 1
