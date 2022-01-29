import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_quant_finance as tff

tf.config.run_functions_eagerly(True)

__ALL__ = ["constrained_minimize", "unconstrained_minimize"]


@tf.function
def unconstrained_minimize(objective_fn, algorithm="lbfgs", **kwargs):
    @functools.wraps(objective_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(objective_fn, x)

    if algorithm == "lbfgs":
        minimize = tfp.optimizer.lbfgs_minimize
    elif algorithm == "bfgs":
        minimize = tfp.optimizer.bfgs_minimize
    elif algorithm == "conjugate_gradient":
        minimize = tff.math.optimizer.conjugate_gradient_minimize
    else:
        raise ValueError(
            f"Algorithm specified {algorithm} is not one of (lbfgs, bfgs, conjugate_gradient)."
        )

    try:
        results = minimize(val_and_grad, **kwargs)
    except Exception as e:
        raise e

    return {
        "converged": results.converged.numpy(),
        "position": results.position,
        "results_object": results,
    }


@tf.function
def constrained_minimize(
    objective_fn,
    constrained_fns=None,
    algorithm="lbfgs",
    initial_position=None,
    relu_scalar=None,
    relu_scalar_random_positions_count=1000,
    **kwargs,
):
    if initial_position is None:
        raise ValueError("initial-position must be provided")

    if relu_scalar is None:
        # find reasonable maximum for lagrangian
        random_positions_shape = [
            relu_scalar_random_positions_count
        ] + initial_position.shape

        random_positions = tf.random.uniform(
            shape=random_positions_shape, minval=-1000, maxval=1000
        )

        random_fn_values = objective_fn(random_positions)

        relu_scalar = np.max(random_fn_values.numpy())

    def relu_objective_fn(x):
        output = objective_fn(x)
        if constrained_fns is not None and len(constrained_fns) > 0:
            for fn in constrained_fns:
                output = tf.math.add(output, tf.nn.relu(fn(x)) * relu_scalar)
        return output

    try:
        results = unconstrained_minimize(
            objective_fn=relu_objective_fn,
            algorithm=algorithm,
            initial_position=initial_position,
            **kwargs,
        )
    except Exception as e:
        raise e

    return results
