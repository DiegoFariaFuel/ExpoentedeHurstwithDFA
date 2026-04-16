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
