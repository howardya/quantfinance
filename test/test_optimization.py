import numpy as np
import pytest
import tensorflow as tf

from tensorflow_quant.optimization import constrained_minimize, unconstrained_minimize

tf.config.run_functions_eagerly(True)


def test_unspecified_algorithm():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    with pytest.raises(ValueError):
        unconstrained_minimize(
            objective_fn=value_fn,
            initial_position=tf.constant([1.0]),
            algorithm="random",
        )


def test_minimization_exception_no_initial_position():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    with pytest.raises(TypeError):
        unconstrained_minimize(
            objective_fn=value_fn,
            algorithm="lbfgs",
        )


def test_single_quadratic_lbfgs():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    results = unconstrained_minimize(
        objective_fn=value_fn, initial_position=tf.constant([1.0]), algorithm="lbfgs"
    )
    assert results["converged"]
    assert abs(results["position"].numpy()[0] - 0.87) < 0.01


def test_single_quadratic_bfgs():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    results = unconstrained_minimize(
        objective_fn=value_fn, initial_position=tf.constant([1.0]), algorithm="bfgs"
    )
    assert results["converged"]
    assert abs(results["position"].numpy()[0] - 0.87) < 0.1


def test_single_quadratic_conjugate_gradient():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    results = unconstrained_minimize(
        objective_fn=value_fn,
        initial_position=tf.constant([1.0]),
        algorithm="conjugate_gradient",
    )
    assert results["converged"]
    assert abs(results["position"].numpy()[0] - 0.87) < 0.1


def test_single_quadratic_initial_position():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    results = unconstrained_minimize(
        objective_fn=value_fn,
        initial_position=tf.constant(np.linspace(-10.0, 10.0, 10)),
        algorithm="lbfgs",
    )
    assert results["converged"]
    optimized_positions = results["position"].numpy()
    for position in optimized_positions:
        assert abs(position - 0.87) < 0.1


def test_multiple_quadratic():
    def value_fn(x):
        return tf.reduce_sum((x + 5.0) ** 2 * (x - 5.0) ** 2)

    results = unconstrained_minimize(
        objective_fn=value_fn,
        initial_position=tf.constant(np.linspace(-10.0, 10.0, 20)),
    )
    assert results["converged"]
    optimized_positions = np.sort(results["position"].numpy())
    assert abs(optimized_positions[0] - -5.0) < 0.1
    assert abs(optimized_positions[-1] - 5.0) < 0.1


# constrained optimization
def test_without_initial_position():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    def constraint_fn_1(x):
        # x >= 0 ( -x<=0)
        return -tf.reduce_sum(x, axis=-1)

    with pytest.raises(ValueError):
        constrained_minimize(
            objective_fn=value_fn,
            constrained_fns=[constraint_fn_1],
            relu_scalar=1000,
            algorithm="lbfgs",
        )


def test_wrong_algorithm():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    def constraint_fn_1(x):
        # x >= 0 ( -x<=0)
        return -tf.reduce_sum(x, axis=-1)

    with pytest.raises(ValueError):
        constrained_minimize(
            objective_fn=value_fn,
            constrained_fns=[constraint_fn_1],
            initial_position=tf.constant([1.0]),
            relu_scalar=1000,
            algorithm="random",
        )


def test_single_quadratic_single_constraint():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    def constraint_fn_1(x):
        # x >= 0 ( -x<=0)
        return -tf.reduce_sum(x, axis=-1)

    results = constrained_minimize(
        objective_fn=value_fn,
        constrained_fns=[constraint_fn_1],
        relu_scalar=1000,
        initial_position=tf.constant([1.0]),
        algorithm="lbfgs",
    )
    assert results["converged"]
    assert abs(results["position"].numpy()[0] - 0.87) < 0.1


def test_single_quadratic_single_constraint_without_relu_scalar():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    def constraint_fn_1(x):
        # x >= 0 ( -x<=0)
        return -tf.reduce_sum(x, axis=-1)

    results = constrained_minimize(
        objective_fn=value_fn,
        constrained_fns=[constraint_fn_1],
        initial_position=tf.constant([1.0]),
        algorithm="lbfgs",
    )
    assert results["converged"]
    assert abs(results["position"].numpy()[0] - 0.87) < 0.1


def test_single_quadratic_multiple_constraints():
    def value_fn(x):
        return tf.reduce_sum((x - 0.87) ** 2)

    def constraint_fn_1(x):
        # x >= 0 ( -x<=0)
        return -tf.reduce_sum(x, axis=-1)

    def constraint_fn_2(x):
        # x - 5 <= 0
        return tf.reduce_sum(x - 5, axis=-1)

    def constraint_fn_3(x):
        # x^2 - 4 <= 0
        return tf.reduce_sum(x ** 2 - 4, axis=-1)

    results = constrained_minimize(
        objective_fn=value_fn,
        constrained_fns=[constraint_fn_1, constraint_fn_2, constraint_fn_3],
        relu_scalar=1000,
        initial_position=tf.constant([1.0]),
        algorithm="lbfgs",
    )
    assert results["converged"]
    assert abs(results["position"].numpy()[0] - 0.87) < 0.1
