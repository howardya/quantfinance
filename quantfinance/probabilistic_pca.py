import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm

tf.config.run_functions_eagerly(True)
np_config.enable_numpy_behavior()

__ALL__ = ["probabilistic_pca"]


def probabilistic_pca(
    data,
    n_components=None,
    n_iter=100,
    constraints_relu_multiplier=1000,
    constrained_fns=None,
    zero_volatility=False,
    optimizer=None,
    m_step_num_steps=200,
):
    n_data = data.shape[0]
    data_dim = data.shape[1]

    if n_components is None:
        n_components = data_dim  # full PCA

    principal_axes = tfp.distributions.Normal(loc=0.0, scale=5.0).sample(
        (data_dim, n_components)
    )

    if zero_volatility:
        sigma = tf.constant(0.0)
    else:
        sigma = tf.math.reduce_std(data)

    for em_iter in tqdm(range(n_iter)):
        # E-Step, calculate responsibilities
        # Bishop Chapter 12.2.2 EM algorithm for PCA
        # M = W^T*W + sigma^2*I
        M = tf.transpose(principal_axes) @ principal_axes + sigma ** 2 * tf.eye(
            n_components
        )

        M_inverse = tf.linalg.inv(M)

        # Here, I want E_zn to be dimension n_data x n_components
        E_zn = data @ principal_axes @ M_inverse

        # recall that outer product of A (nx1) and B(1xn) can be computed based on tf.multiply(A,B) due to broadcasting
        # Equation (12.55)
        E_zn_znT = tf.expand_dims(sigma ** 2 * M_inverse, axis=0) + tf.multiply(
            E_zn[..., None], tf.expand_dims(E_zn, axis=1)
        )

        # M-Step
        if constrained_fns is None:
            # if no constraint is specified (other than mixture probabilities must sum to 1), analytical formula exists.
            # Bishop, Chapater 12.2.2

            # Equation 12.57
            if zero_volatility:
                omega = (
                    tf.linalg.inv(tf.transpose(principal_axes) @ principal_axes)
                    @ tf.transpose(principal_axes)
                    @ tf.transpose(data)
                )

                principal_axes = (
                    tf.transpose(data)
                    @ tf.transpose(omega)
                    @ tf.linalg.inv(omega @ tf.transpose(omega))
                )

            else:
                principal_axes = tf.matmul(
                    tf.reduce_sum(
                        tf.multiply(data[..., None], tf.expand_dims(E_zn, 1)), axis=0
                    ),
                    tf.linalg.inv(tf.reduce_sum(E_zn_znT, axis=0)),
                )

                principal_axes_T_principal_axes = (
                    tf.transpose(principal_axes) @ principal_axes
                )

                sigma = tf.sqrt(
                    (
                        tf.reduce_sum(data * data)
                        - 2
                        * tf.reduce_sum(
                            tf.matmul(E_zn, tf.transpose(principal_axes)) * data
                        )
                        + tf.reduce_sum(
                            tf.multiply(
                                tf.matmul(
                                    E_zn_znT, principal_axes_T_principal_axes[None, ...]
                                ),
                                tf.eye(n_components)[None, ...],
                            )
                        )
                    )
                    / n_data
                    / data_dim
                )
        else:
            principal_axes_variable = tf.Variable(principal_axes)

            sigma_scalar = sigma.numpy() / 2.0

            sigma_scaled_variable = tfp.util.TransformedVariable(
                0.1, bijector=tfp.bijectors.Softplus()
            )

            def loss_fn():
                # Log likelihood, Equation 12.53 in Bishop
                sigma_variable = sigma_scaled_variable * sigma_scalar

                log_likelihood = (
                    data_dim * n_data * tf.math.log(sigma_variable)
                    + tf.reduce_sum(
                        tf.multiply(E_zn_znT, tf.eye(n_components)[None, ...])
                    )
                    / 2
                    + tf.reduce_sum(data ** 2) / 2 / sigma_variable ** 2
                    - tf.reduce_sum(
                        tf.matmul(E_zn, tf.transpose(principal_axes_variable)) * data
                    )
                    / sigma_variable ** 2
                    + tf.reduce_sum(
                        tf.multiply(
                            E_zn_znT
                            @ tf.matmul(
                                tf.transpose(principal_axes_variable),
                                principal_axes_variable,
                            )[None, ...],
                            tf.eye(n_components)[None, ...],
                        )
                    )
                    / 2
                    / sigma_variable ** 2
                )

                log_likelihood = log_likelihood

                for constrained_fn in constrained_fns:
                    log_likelihood = (
                        log_likelihood
                        + constraints_relu_multiplier
                        / sigma_scalar ** 2
                        * tf.nn.relu(
                            constrained_fn(principal_axes_variable, sigma_variable)
                        )
                    )

                return log_likelihood

            if optimizer is None:
                optimizer = tf.optimizers.RMSprop(0.1)

            _ = tfp.math.minimize(
                loss_fn=loss_fn,
                num_steps=m_step_num_steps,
                optimizer=optimizer,
                convergence_criterion=tfp.optimizer.convergence_criteria.LossNotDecreasing(
                    rtol=0.01, min_num_steps=100, name=None
                ),
            )

            principal_axes = principal_axes_variable.value()
            sigma = sigma_scaled_variable * sigma_scalar

    return principal_axes, sigma
