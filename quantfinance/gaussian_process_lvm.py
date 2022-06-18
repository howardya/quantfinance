# Gaussian Process Latent Variable Model
# https://www.tensorflow.org/probability/examples/Gaussian_Process_Latent_Variable_Model
# https://proceedings.neurips.cc/paper/2003/file/9657c1fffd38824e5ab0472e022e577e-Paper.pdf
# https://arxiv.org/pdf/1806.03294v1.pdf#page13
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = ["gaussian_process_lvm"]

EPS = np.finfo(np.float32).eps


def gaussian_process_lvm(
    data,
    kernel=None,
    kernel_type="rbf",
    latent_dim=2,
    estimation_method="MLE",
    num_steps=100,
):
    num_samples = data.shape[0]
    if kernel is None:
        if kernel_type == "rbf":  # ExponentiatedQuadratic
            amplitude = tfp.util.TransformedVariable(
                EPS + 1.0,
                bijector=tfp.bijectors.Softplus(),
                name="amplitude",
                dtype=tf.float32,
            )
            length_scale = tfp.util.TransformedVariable(
                EPS + 1.0,
                bijector=tfp.bijectors.Softplus(),
                name="length_scale",
                dtype=tf.float32,
            )
            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
                amplitude=amplitude, length_scale=length_scale
            )
        elif kernel_type == "linear":
            bias_variance = tfp.util.TransformedVariable(
                EPS + 1.0,
                bijector=tfp.bijectors.Softplus(),
                name="bias_variance",
                dtype=tf.float32,
            )
            slope_variance = tfp.util.TransformedVariable(
                EPS + 1.0,
                bijector=tfp.bijectors.Softplus(),
                name="slope_variance",
                dtype=tf.float32,
            )
            shift = tf.Variable(
                0.0,
                name="shift",
                dtype=tf.float32,
            )
            kernel = tfp.math.psd_kernels.Linear(
                bias_variance=bias_variance,
                slope_variance=slope_variance,
                shift=shift,
            )
        else:
            raise NotImplementedError(f"kernel_type provided {kernel_type} not valid.")

    trainable_variables = list(kernel.trainable_variables)

    data_noise_variance = tfp.util.TransformedVariable(
        EPS + 1.0,
        bijector=tfp.bijectors.Softplus(),
        name="data_noise_variance",
        dtype=tf.float32,
    )

    trainable_variables += [data_noise_variance]

    if estimation_method == "MLE":
        latent_variables = tf.Variable(
            tfp.distributions.Normal(
                loc=tf.zeros((num_samples, latent_dim)),
                scale=1.0,
                name="latent_variables",
            ).sample(),
            name="latent_variables",
        )
        trainable_variables += [latent_variables]
        gp = tfp.distributions.GaussianProcess(
            kernel=kernel,
            index_points=latent_variables,
            observation_noise_variance=data_noise_variance,
            name="gaussian_process",
        )

        def neg_log_probs():
            return -tf.reduce_sum(gp.log_prob(tf.transpose(data)))

        loss = tfp.math.minimize(
            loss_fn=neg_log_probs,
            num_steps=num_steps,
            optimizer=tf.optimizers.Adam(0.1),
            convergence_criterion=tfp.optimizer.convergence_criteria.LossNotDecreasing(
                rtol=0.01, min_num_steps=100, name=None
            ),
            trainable_variables=trainable_variables,
        )

        return loss, kernel, trainable_variables

    elif estimation_method == "VI":

        def generative_model(num_samples, latent_dim, kernel, data_noise_variance):
            latent_variables = yield tfp.distributions.JointDistributionCoroutine.Root(
                tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=tf.zeros((num_samples, latent_dim)),
                        scale=1.0,
                        name="latent_variables",
                    ),
                    reinterpreted_batch_ndims=2,
                )
            )

            yield tfp.distributions.GaussianProcess(
                kernel=kernel,
                index_points=latent_variables,
                observation_noise_variance=data_noise_variance,
                name="gp",
            )

        model_joint = tfp.distributions.JointDistributionCoroutineAutoBatched(
            lambda: generative_model(
                num_samples, latent_dim, kernel, data_noise_variance
            )
        )

        def model_joint_log_prob_observed_data(latent_variable_x):
            latent_variable_x = tf.expand_dims(latent_variable_x, -3)
            data_T = tf.transpose(data)

            # required if using sample_size > 1 in fit_surrogate_posterior
            if len(latent_variable_x.shape) == 4:
                data_T = data_T[None, ...]

            return tf.reduce_sum(
                model_joint.log_prob(latent_variable_x, data_T), axis=-1
            )

        surrogate_scale = tfp.util.TransformedVariable(
            tf.ones((num_samples, latent_dim)), bijector=tfp.bijectors.Softplus()
        )
        trainable_variables += [surrogate_scale]
        surrogate_posterior = tfp.distributions.Independent(
            tfp.distributions.Normal(
                loc=tf.zeros((num_samples, latent_dim)),
                scale=surrogate_scale,
                name="latent_variables",
            ),
            reinterpreted_batch_ndims=2,
        )

        loss = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=model_joint_log_prob_observed_data,
            surrogate_posterior=surrogate_posterior,
            optimizer=tf.optimizers.Adam(0.1),
            convergence_criterion=tfp.optimizer.convergence_criteria.LossNotDecreasing(
                rtol=0.1, min_num_steps=10
            ),
            num_steps=num_steps,
            sample_size=1,
        )

        return loss, kernel, trainable_variables, surrogate_posterior
