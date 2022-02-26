import numpy as np
import scipy
import scipy.optimize


def positive_pca(cov, n_components=None):
    N = cov.shape[0]

    positive_pca = []

    optim_objects = []

    def objective_fn(weights):
        return (
            -weights[None, ...]
            @ cov
            @ weights[..., None]
            / (np.linalg.norm(weights) ** 2)
        )

    np.random.seed(9710)
    i = 0
    has_not_failed = True
    while i < n_components and has_not_failed:
        # >= 0
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            # {"type": "ineq", "fun": lambda w: 1.001 - w.sum()},
        ]

        for j in range(i):
            constraints.append(
                {"type": "eq", "fun": lambda w, j=j: (w * positive_pca[j]).sum()}
            )

        initial_position = np.random.uniform(size=N)
        initial_position = initial_position / initial_position.sum()
        results = scipy.optimize.minimize(
            objective_fn,
            x0=initial_position,
            bounds=[(0, None)] * N,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 500, "disp": False},
        )

        optim_objects.append(results)
        if not results.success:
            print(f"Component {i+1} did not converged")
            has_not_failed = False
        else:
            positive_pca.append(results.x)

        i += 1

    positive_pca = np.array(positive_pca).T

    return positive_pca, optim_objects
