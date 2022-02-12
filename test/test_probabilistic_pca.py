from test.test_covariance import generate_multivariate_data

import numpy as np
import tensorflow as tf

from tensorflow_quant.probabilistic_pca import probabilistic_pca


def test_data_dim_4_components_2_without_constraint():
    data_dim = 4
    n_samples = 1500
    n_components = 2

    tf.random.set_seed(9710)
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

    principal_axes, sigma = probabilistic_pca(
        data=data, n_components=n_components, n_iter=200, constrained_fns=None
    )

    eigen_covariance_subspace_fitted = principal_axes @ tf.transpose(principal_axes)

    principal_axes_true = eigenvectors_true[:, -n_components:]
    eigen_covariance_subspace_true = (
        principal_axes_true
        @ (
            np.diag(eigenvalues_true[-n_components:])
            - eigenvalues_true[: (data_dim - n_components)].mean()
            * tf.eye(n_components)
        )
        @ tf.transpose(principal_axes_true)
    )

    abs_sum_diff = tf.reduce_sum(
        tf.abs(eigen_covariance_subspace_fitted - eigen_covariance_subspace_true)
    ) / tf.reduce_sum(tf.abs(eigen_covariance_subspace_true))

    residual_variance_fitted = sigma ** 2
    residual_variance_true = eigenvalues_true[: (data_dim - n_components)].mean()

    assert abs_sum_diff < 0.01
    assert tf.abs(residual_variance_fitted - residual_variance_true) < 0.4


def test_data_dim_4_components_2_without_constraint_none_num_components():
    data_dim = 4
    n_samples = 1500

    tf.random.set_seed(9710)
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

    principal_axes, sigma = probabilistic_pca(
        data=data, n_iter=200, constrained_fns=None
    )

    assert principal_axes.shape[1] == 4
    assert sigma.numpy() > 0


def test_data_dim_4_components_1_without_constraint():
    data_dim = 4
    n_samples = 1500
    n_components = 1

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

    principal_axes, sigma = probabilistic_pca(
        data=data, n_components=n_components, n_iter=200, constrained_fns=None
    )

    eigen_covariance_subspace_fitted = principal_axes @ tf.transpose(principal_axes)

    principal_axes_true = eigenvectors_true[:, -n_components:]
    eigen_covariance_subspace_true = (
        principal_axes_true
        @ (
            np.diag(eigenvalues_true[-n_components:])
            - eigenvalues_true[: (data_dim - n_components)].mean()
            * tf.eye(n_components)
        )
        @ tf.transpose(principal_axes_true)
    )

    abs_sum_diff = tf.reduce_sum(
        tf.abs(eigen_covariance_subspace_fitted - eigen_covariance_subspace_true)
    ) / tf.reduce_sum(tf.abs(eigen_covariance_subspace_true))

    residual_variance_fitted = sigma ** 2
    residual_variance_true = eigenvalues_true[: (data_dim - n_components)].mean()

    assert abs_sum_diff < 0.01
    assert tf.abs(residual_variance_fitted - residual_variance_true) < 0.4


def test_data_dim_4_components_4():
    data_dim = 4
    n_samples = 1500
    n_components = 4

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

    principal_axes, sigma = probabilistic_pca(
        data=data, n_components=n_components, n_iter=100, constrained_fns=None
    )

    eigen_covariance_subspace_fitted = principal_axes @ tf.transpose(
        principal_axes
    ) + sigma ** 2 * tf.eye(n_components)

    principal_axes_true = eigenvectors_true[:, -n_components:]
    eigen_covariance_subspace_true = (
        principal_axes_true
        @ (np.diag(eigenvalues_true[-n_components:]))
        @ tf.transpose(principal_axes_true)
    )

    abs_sum_diff = tf.reduce_sum(
        tf.abs(eigen_covariance_subspace_fitted - eigen_covariance_subspace_true)
    ) / tf.reduce_sum(tf.abs(eigen_covariance_subspace_true))

    assert abs_sum_diff < 0.01


def test_data_dim_4_components_zero_volatility_without_constraint():
    data_dim = 4
    n_samples = 1500
    n_components = 2

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

    tf.random.set_seed(1110)
    principal_axes, sigma = probabilistic_pca(
        data=data, n_components=n_components, zero_volatility=True, n_iter=50
    )

    assert principal_axes.shape[1] == 2
    assert sigma == 0


def test_data_dim_4_components_2_with_constraints():
    data_dim = 4
    n_samples = 1500
    n_components = 2

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

    tf.random.set_seed(9710)
    principal_axes, sigma = probabilistic_pca(
        data=data,
        n_components=n_components,
        constrained_fns=[],
        n_iter=1,
        m_step_num_steps=10,
    )

    assert principal_axes.shape[1] == 2
    assert sigma.numpy() > 0


def test_data_dim_4_components_2_with_constraints_sigma_constraint():
    data_dim = 4
    n_samples = 1500
    n_components = 2

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

    def constrained_fn(principal_axes_variable, sigma_variable):
        return sigma_variable

    tf.random.set_seed(9710)
    principal_axes, sigma = probabilistic_pca(
        data=data,
        n_components=n_components,
        constrained_fns=[constrained_fn],
        n_iter=1,
        m_step_num_steps=10,
    )

    assert principal_axes.shape[1] == 2
    assert sigma.numpy() > 0
