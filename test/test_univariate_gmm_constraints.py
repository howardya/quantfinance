import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_quant.univariate_gmm_constraints import univariate_gmm_constraints


def generate_data(n_mixtures, n_samples=200, mean_spread=3, volatility=1, means=None):
    if means is None:
        means_true = np.linspace(
            -mean_spread * n_mixtures, mean_spread * n_mixtures, n_mixtures
        )
    else:
        means_true = means

    volatilities_true = np.ones(n_mixtures) * volatility
    mixture_probabilities_true = np.random.dirichlet(np.ones(n_mixtures))

    data = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            probs=mixture_probabilities_true
        ),
        components_distribution=tfp.distributions.Normal(
            loc=means_true, scale=volatilities_true
        ),
    ).sample(n_samples)

    return data, means_true, volatilities_true, mixture_probabilities_true


def test_2_mixtures():
    n_mixtures = 2
    mean_spread = 3

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=1000 * n_mixtures, mean_spread=mean_spread
    )
    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data, n_mixtures=n_mixtures, n_iter=1000, initial_means=np.array([0.0, 5])
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    mixture_probabilities_fitted_sorted = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    volatilities_true_sorted = volatilities_true[sort_idx]
    mixture_probabilities_true_sorted = mixture_probabilities_true[sort_idx]

    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
        assert np.abs(volatilities_fitted_sorted[i] - volatilities_true_sorted[i]) < 0.2
        assert (
            np.abs(
                mixture_probabilities_fitted_sorted[i]
                - mixture_probabilities_true_sorted[i]
            )
            < 0.2
        )


def test_3_mixtures():
    n_mixtures = 3
    mean_spread = 3

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=1000 * n_mixtures, mean_spread=mean_spread
    )
    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data,
        n_mixtures=n_mixtures,
        n_iter=1000,
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    mixture_probabilities_fitted_sorted = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    volatilities_true_sorted = volatilities_true[sort_idx]
    mixture_probabilities_true_sorted = mixture_probabilities_true[sort_idx]

    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
        assert np.abs(volatilities_fitted_sorted[i] - volatilities_true_sorted[i]) < 0.2
        assert (
            np.abs(
                mixture_probabilities_fitted_sorted[i]
                - mixture_probabilities_true_sorted[i]
            )
            < 0.2
        )


def test_4_mixtures():
    np.random.seed(9710)
    n_mixtures = 4
    mean_spread = 3

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=1000 * n_mixtures, mean_spread=mean_spread
    )
    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data,
        n_mixtures=n_mixtures,
        n_iter=1000,
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    mixture_probabilities_fitted_sorted = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    volatilities_true_sorted = volatilities_true[sort_idx]
    mixture_probabilities_true_sorted = mixture_probabilities_true[sort_idx]

    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
        assert np.abs(volatilities_fitted_sorted[i] - volatilities_true_sorted[i]) < 0.2
        assert (
            np.abs(
                mixture_probabilities_fitted_sorted[i]
                - mixture_probabilities_true_sorted[i]
            )
            < 0.2
        )


def test_5_mixtures():
    np.random.seed(97101)
    n_mixtures = 5
    mean_spread = 5

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=5000 * n_mixtures, mean_spread=mean_spread
    )
    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data,
        n_mixtures=n_mixtures,
        n_iter=3000,
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    mixture_probabilities_fitted_sorted = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    volatilities_true_sorted = volatilities_true[sort_idx]
    mixture_probabilities_true_sorted = mixture_probabilities_true[sort_idx]

    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
        assert np.abs(volatilities_fitted_sorted[i] - volatilities_true_sorted[i]) < 0.2
        assert (
            np.abs(
                mixture_probabilities_fitted_sorted[i]
                - mixture_probabilities_true_sorted[i]
            )
            < 0.2
        )


def test_2_mixtures_empty_constraints():
    n_mixtures = 2
    mean_spread = 3

    np.random.seed(9710)
    tf.random.set_seed(9710)

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=1000 * n_mixtures, mean_spread=mean_spread
    )

    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data,
        n_mixtures=n_mixtures,
        n_iter=10,
        m_step_num_steps=200,
        constrained_fns=[],
        constraints_relu_multiplier=2000,
        initial_means=np.array([-10.0, 20.0]),
        verbose=True,
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    mixture_probabilities_fitted_sorted = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    volatilities_true_sorted = volatilities_true[sort_idx]
    mixture_probabilities_true_sorted = mixture_probabilities_true[sort_idx]

    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
        assert np.abs(volatilities_fitted_sorted[i] - volatilities_true_sorted[i]) < 0.2
        assert (
            np.abs(
                mixture_probabilities_fitted_sorted[i]
                - mixture_probabilities_true_sorted[i]
            )
            < 0.2
        )


def test_3_mixtures_empty_constraints():
    n_mixtures = 3
    mean_spread = 3

    np.random.seed(9710)
    tf.random.set_seed(9710)

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=2000 * n_mixtures, mean_spread=mean_spread
    )

    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data,
        n_mixtures=n_mixtures,
        n_iter=10,
        m_step_num_steps=400,
        constrained_fns=[],
        constraints_relu_multiplier=2000,
        initial_means=np.array([-10.0, 2.0, 10.0]),
        optimizer=tf.optimizers.Adam(0.01),
        verbose=True,
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    mixture_probabilities_fitted_sorted = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    volatilities_true_sorted = volatilities_true[sort_idx]
    mixture_probabilities_true_sorted = mixture_probabilities_true[sort_idx]

    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
        assert np.abs(volatilities_fitted_sorted[i] - volatilities_true_sorted[i]) < 0.2
        assert (
            np.abs(
                mixture_probabilities_fitted_sorted[i]
                - mixture_probabilities_true_sorted[i]
            )
            < 0.2
        )


def test_2_mixtures_volatilities_constraints():
    n_mixtures = 2
    mean_spread = 3

    np.random.seed(9710)
    tf.random.set_seed(9710)

    data, means_true, volatilities_true, mixture_probabilities_true = generate_data(
        n_mixtures=n_mixtures, n_samples=2000 * n_mixtures, mean_spread=mean_spread
    )

    def volatilities_constrained(
        theta_mixture_probabilities, theta_means, theta_volatilities
    ):
        return 1.5 - theta_volatilities[0]

    (
        means_fitted,
        volatilities_fitted,
        mixture_probabilities_fitted,
    ) = univariate_gmm_constraints(
        data,
        n_mixtures=n_mixtures,
        n_iter=10,
        m_step_num_steps=200,
        constrained_fns=[volatilities_constrained],
        constraints_relu_multiplier=2000,
        initial_means=np.array([-3.0, 3.0]),
        optimizer=tf.optimizers.Adam(0.01),
        verbose=True,
    )

    sort_idx = np.argsort(means_fitted)
    means_fitted_sorted = means_fitted[sort_idx]
    volatilities_fitted_sorted = volatilities_fitted[sort_idx]
    _ = mixture_probabilities_fitted[sort_idx]

    sort_idx = np.argsort(means_true)
    means_true_sorted = means_true[sort_idx]
    _ = volatilities_true[sort_idx]
    _ = mixture_probabilities_true[sort_idx]

    assert np.abs(volatilities_fitted_sorted[0] - 1.5) < 0.2
    for i in range(n_mixtures):
        assert np.abs(means_fitted_sorted[i] - means_true_sorted[i]) < 0.2
