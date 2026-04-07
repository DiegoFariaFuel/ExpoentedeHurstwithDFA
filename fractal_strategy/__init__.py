"""Package for fractal strategy utilities.

This project provides reusable functions to download price data, compute the Hurst
exponent via Detrended Fluctuation Analysis (DFA), create features, cluster with
PCA + KMeans, generate trading signals, and backtest strategies.

The package is intentionally lightweight and dependency-aware.
"""

from .data import download_price
from .hurst import hurst_exponent, hurst_series
from .strategy import run_pca_kmeans, interpret_clusters, generate_signals, apply_neural
from .backtest import backtest, performance_metrics

__all__ = [
    "download_price",
    "hurst_exponent",
    "hurst_series",
    "run_pca_kmeans",
    "interpret_clusters",
    "generate_signals",
    "apply_neural",
    "backtest",
    "performance_metrics",
]
