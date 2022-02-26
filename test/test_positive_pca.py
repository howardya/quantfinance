from tensorflow_quant.data import get_stocks_prices_yahoo
from tensorflow_quant.positive_pca import positive_pca


def test_positive_pca():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG", "DBC", "^VIX"],
        frequency="daily",
        start="2010-07-31",
        end="2020-12-31",
    )

    returns = prices.pct_change().dropna()

    cov_df = returns.cov()

    cov = cov_df.values

    positive_pca_components, optim_results = positive_pca(cov, 2)

    assert positive_pca_components.shape[0] == 4
    assert positive_pca_components.shape[1] == 2


def test_positive_pca_large_components():
    prices = get_stocks_prices_yahoo(
        ["VTI", "AGG", "DBC", "^VIX"],
        frequency="daily",
        start="2010-07-31",
        end="2020-12-31",
    )

    returns = prices.pct_change().dropna()

    cov_df = returns.cov()

    cov = cov_df.values

    positive_pca_components, optim_results = positive_pca(cov, 5)

    assert positive_pca_components.shape[0] == 4
    assert positive_pca_components.shape[1] == 2
