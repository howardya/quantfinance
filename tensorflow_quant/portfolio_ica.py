from sklearn.decomposition import PCA, FastICA


def portfolio_ica(returns, num_pca=None):
    if num_pca is None:
        num_pca = "mle"

    pca = PCA(n_components=num_pca)
    returns_pca = pca.fit_transform(returns)

    ica = FastICA(n_components=returns_pca.shape[1], max_iter=500)
    returns_independent = ica.fit_transform(returns_pca)

    return returns_independent, returns_pca, pca, ica
