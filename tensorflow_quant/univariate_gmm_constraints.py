import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm

tf.config.run_functions_eagerly(True)
np_config.enable_numpy_behavior()

__ALL__ = [
    "univariate_gmm_constraints",
]


def univariate_gmm_constraints(
    data,
    n_mixtures,
    constrained_fns=None,
    n_iter=100,
    initial_means=None,
    constraints_relu_multiplier=1000,
    m_step_num_steps=300,
    verbose=False,
    optimizer=None,
    **kwargs,
):
    n_samples = data.shape[0]

    # initialize/randomize the initial parameters and latent class probabilities
    if initial_means is None:
        means = tf.linspace(-3 * n_mixtures, 3 * n_mixtures, n_mixtures)
    else:
        means = tf.convert_to_tensor(initial_means)

    volatilities = (
        tf.ones(n_mixtures, dtype=tf.float64)
        * np.std(data)
        * tf.random.uniform(shape=(n_mixtures,))
    )
    unnormalized_mixture_probabilities = tf.random.uniform(
        [n_mixtures], dtype=tf.float64
    )
    mixture_probabilities = unnormalized_mixture_probabilities / tf.math.reduce_sum(
        unnormalized_mixture_probabilities
    )

    for em_iter in tqdm(range(n_iter)):
        # E-Step
        # calculate responsibilities
        unnormalized_responsibilities = (
            tfp.distributions.Normal(loc=means, scale=volatilities).prob(data[:, None])
            * mixture_probabilities
        )

        responsibilities = unnormalized_responsibilities / tf.reduce_sum(
            unnormalized_responsibilities, axis=1, keepdims=True
        )

        mixture_responsibilities = tf.reduce_sum(responsibilities, axis=0)
        mixture_responsibilities = tf.math.maximum(mixture_responsibilities, 1e-6)

        # M-Step (with constraints)
        if constrained_fns is None:
            # if no constraint is specified (other than mixture probabilities must sum to 1), analytical formula exists.
            # https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/probabilistic_machine_learning/em_gmm_more_comments.py
            mixture_probabilities = mixture_responsibilities / n_samples
            means = tf.reduce_sum(responsibilities * data[:, None], axis=0) / (
                mixture_responsibilities + 1e-10
            )
            volatilities = tf.math.sqrt(
                tf.reduce_sum(responsibilities * (data[:, None] - means) ** 2, axis=0)
                / mixture_responsibilities
            )
        else:
            # resort to numerical optimization
            # define lagrangian (objective function, Q + constraints)
            mixture_probabilities_log = tf.math.log(mixture_probabilities)

            def mixture_probabilities_log_transformer(x):
                return tf.divide(
                    x - mixture_probabilities_log.min(),
                    (mixture_probabilities_log.max() - mixture_probabilities_log.min()),
                )

            def mixture_probabilities_log_invert_transformer(x):
                return (
                    tf.multiply(
                        x,
                        mixture_probabilities_log.max()
                        - mixture_probabilities_log.min(),
                    )
                    + mixture_probabilities_log.min()
                )

            def means_transformer(x):
                return tf.divide(x - means.min(), (means.max() - means.min()))

            def means_invert_transformer(x):
                return tf.multiply(x, means.max() - means.min()) + means.min()

            def volatilities_transformer(x):
                return tf.divide(x, volatilities.max())

            def volatilities_invert_transformer(x):
                return tf.multiply(x, volatilities.max())

            theta_mixture_probabilities_log_scaled_variable = tf.Variable(
                mixture_probabilities_log_transformer(mixture_probabilities_log)
            )
            theta_means_scaled_variable = tf.Variable(means_transformer(means))
            theta_volatilities_scaled_variable = tfp.util.TransformedVariable(
                volatilities_transformer(volatilities), bijector=tfp.bijectors.Exp()
            )

            def loss_fn():
                theta_mixture_probabilities = tf.math.softmax(
                    mixture_probabilities_log_invert_transformer(
                        theta_mixture_probabilities_log_scaled_variable
                    )
                )

                theta_means = means_invert_transformer(theta_means_scaled_variable)

                theta_volatilities = volatilities_invert_transformer(
                    theta_volatilities_scaled_variable
                )

                responsibilities_weighted_variance = tf.reduce_sum(
                    responsibilities * (data[:, None] - theta_means) ** 2,
                    axis=0,
                    keepdims=True,
                )

                # Q(theta_old, theta)
                q_theta = (
                    tf.reduce_sum(
                        mixture_responsibilities
                        * tf.math.log(theta_mixture_probabilities)
                    )
                    - tf.reduce_sum(
                        mixture_responsibilities * tf.math.log(theta_volatilities)
                    )
                    - tf.reduce_sum(
                        responsibilities_weighted_variance
                        / 2
                        / theta_volatilities
                        / theta_volatilities
                    )
                )

                # minimize
                q_theta = -q_theta

                for constrained_fn in constrained_fns:
                    q_theta = q_theta + constraints_relu_multiplier * tf.nn.relu(
                        constrained_fn(
                            theta_mixture_probabilities, theta_means, theta_volatilities
                        )
                    )

                return q_theta

            if optimizer is None:
                optimizer = tf.optimizers.Adam(0.1)

            _ = tfp.math.minimize(
                loss_fn=loss_fn,
                num_steps=m_step_num_steps,
                optimizer=optimizer,
                convergence_criterion=tfp.optimizer.convergence_criteria.LossNotDecreasing(
                    rtol=0.01, min_num_steps=100, name=None
                ),
            )

            mixture_probabilities = tf.math.softmax(
                mixture_probabilities_log_invert_transformer(
                    theta_mixture_probabilities_log_scaled_variable
                )
            )
            means = means_invert_transformer(theta_means_scaled_variable)
            volatilities = volatilities_invert_transformer(
                theta_volatilities_scaled_variable
            )

        if verbose:
            print(
                "---------------------------------------------------------------------------"
            )
            print(f"Iteration: {em_iter + 1}")
            print(f"Mixture probabilities: {mixture_probabilities}")
            print(f"Means: {means}")
            print(f"Volatilities: {volatilities}")

    return means, volatilities, mixture_probabilities
