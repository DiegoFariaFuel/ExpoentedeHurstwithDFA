import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_pca_kmeans(df: pd.DataFrame, hurst_cols, n_clusters: int = 3, random_state: int = 42):
    """Run PCA on Hursts and cluster the transformed data.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the Hurst features.
    hurst_cols : list
        List of column names in ``df`` that hold Hurst values.
    n_clusters : int
        Number of clusters for KMeans.
    random_state : int
        Seed for reproducibility.
    Returns
    -------
    pandas.Series
        Cluster labels indexed like ``df``.
    PCA
        Fitted PCA object (useful for inspection).
    KMeans
        Fitted KMeans object.
    """
    X = df[hurst_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # choose number of PCA components:
    # - if there is only one feature we must request at least 1 component
    # - otherwise allow a float <1 to specify explained variance ratio
    n_features = X.shape[1]
    if n_features <= 1:
        n_components = 1
    else:
        n_components = 0.95  # keep enough components to explain 95% variance
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50, max_iter=500)
    labels = kmeans.fit_predict(X_pca)

    return pd.Series(labels, index=df.index), pca, kmeans


# neural architectures support
# imports moved into functions so that module can be used without torch


def apply_neural(df: pd.DataFrame, feature_cols, model_name: str, window=30, **kw):
    """Train a neural architecture and return generated signal series.

    ``model_name`` is passed to ``get_neural_model``.  ``kw`` are forwarded as
    keyword arguments for the model constructor (e.g. ``epochs``).
    """
    from .transformer_model import get_neural_model, transformer_signal

    X = df[feature_cols].values
    y = df['retorno'].values
    # forward window/other keywords to the model constructor
    model = get_neural_model(model_name, window=window, **kw)
    model.fit(X, y)

    # build predictions for every available prefix length in one pass
    n = len(X)
    if n <= window:
        return pd.Series(np.zeros(n), index=df.index)
    # model.predict can now return all sliding-window forecasts in one call
    all_preds = model.predict(X)
    # ``all_preds`` may be shorter than ``n`` (length depends on model).
    sig = transformer_signal(all_preds)
    pad_len = n - sig.shape[0]
    if pad_len < 0:
        # something went wrong, truncate
        sig = sig[:n]
        pad_len = 0
    prefix = np.zeros(pad_len)
    return pd.Series(np.concatenate([prefix, sig]), index=df.index)

def grid_search_neural(df: pd.DataFrame, feature_cols, model_name: str, param_grid: dict, window=None, **kw):
    """Brute-force grid search over model keyword arguments.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features and the column ``'retorno'``.
    feature_cols : list
        Columns to use as input features.
    model_name : str
        Name accepted by ``get_neural_model`` (e.g. 'transformer').
    param_grid : dict
        Mapping from parameter names to lists of values to try.  These
        parameters will be forwarded to the model constructor along with
        ``window`` and any ``kw`` passed below.
    window : int or None
        Window size for the model (optional - may also be part of
        ``param_grid``).
    **kw : additional keyword arguments
        Fixed kwargs for all models (e.g. ``epochs``).

    Returns
    -------
    best_params : dict
        The parameter combination that produced the lowest mean squared
        error on the training data.
    results : dict
        Mapping from frozen parameter tuples to the corresponding MSE.
    """
    from .transformer_model import get_neural_model
    import itertools
    # prepare training data
    X = df[feature_cols].values
    y = df['retorno'].values

    keys = list(param_grid.keys())
    results = {}
    best_score = float('inf')
    best_params = None
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if window is not None:
            params.setdefault('window', window)
        params.update(kw)
        model = get_neural_model(model_name, **params)
        try:
            model.fit(X, y)
            preds = model.predict(X)
            # compute mse on available preds (could be shorter than X)
            if len(preds) == 0:
                score = float('inf')
            else:
                score = ((preds - y[params.get('window', window) - 1:])**2).mean()
        except Exception:
            score = float('inf')
        results[tuple(sorted(params.items()))] = score
        if score < best_score:
            best_score = score
            best_params = params.copy()
    return best_params, results


def interpret_clusters(labels: pd.Series, df: pd.DataFrame, hurst_cols):
    """Determine which cluster corresponds to long/short/neutral.

    Parameters
    ----------
    labels : pandas.Series
        Cluster labels indexed by date.
    df : pandas.DataFrame
        DataFrame that contains the Hurst columns.
    hurst_cols : list
        Names of the Hurst feature columns.

    Returns
    -------
    tuple
        (buy_cluster, sell_cluster, neutral_cluster)
    """
    mean_h = (
        df.join(labels.rename('cluster'))
          .groupby('cluster')[hurst_cols]
          .mean()
          .mean(axis=1)
    )
    buy = mean_h.idxmax()
    sell = mean_h.idxmin()
    neutral = mean_h.index.difference([buy, sell])[0]
    return buy, sell, neutral


def generate_signals(labels: pd.Series, buy_cluster, sell_cluster, neutral_cluster=None):
    """Create a signal series from cluster labels.

    1 -> buy cluster, -1 -> sell cluster, 0 otherwise.

    ``neutral_cluster`` may be omitted if only long/short logic is needed.
    """
    def _map(c):
        if c == buy_cluster:
            return 1
        if c == sell_cluster:
            return -1
        return 0

    return labels.map(_map)
