"""
Fractal Trading Strategy: Theoretical Foundations and Implementation

This script implements a comprehensive fractal-based trading strategy using Hurst exponent analysis,
incorporating DFA calculation, PCA decorrelation, K-means clustering, and backtesting without look-ahead bias.

THEORETICAL FOUNDATIONS:
=======================

1. Fractals in Finance:
   - Fractals are geometric patterns that are self-similar across different scales
   - In financial markets, price series often exhibit fractal properties
   - Benoit Mandelbrot introduced fractal geometry to finance in the 1960s
   - Fractal dimension (D) measures complexity: 1 < D < 2 for time series
   - Hurst exponent (H) relates to fractal dimension: D = 2 - H  

2. Hurst Exponent (H):
   - H measures long-term memory and persistence in time series
   - H = 0.5: Random walk (no memory, efficient market hypothesis)
   - H > 0.5: Persistent (trending behavior, momentum)
   - H < 0.5: Anti-persistent (mean-reverting behavior)
   - In finance: H ≈ 0.6-0.7 for most assets (persistent but not too much)

3. Detrended Fluctuation Analysis (DFA):
   - Robust method for estimating H in non-stationary series
   - Removes local trends to focus on long-range correlations
   - More reliable than R/S analysis for financial data
   - Algorithm: cumulative sum → detrending → fluctuation function → scaling

4. PCA for Feature Decorrelation:
   - Hurst features from different windows are correlated
   - PCA transforms correlated features into uncorrelated principal components
   - Reduces dimensionality while preserving variance
   - Improves clustering quality by removing multicollinearity

5. K-means Clustering:
   - Groups similar market regimes based on Hurst patterns
   - Interprets clusters as long/short/neutral signals
   - Automatic regime detection without manual thresholds
   - Robust to outliers when combined with PCA

6. Trading Signals and Backtesting:
   - Cluster with highest mean H → Long (persistent regime)
   - Cluster with lowest mean H → Short (mean-reverting regime)
   - Neutral cluster → No position
   - One-day signal lag prevents look-ahead bias
   - Compare vs Buy & Hold using risk-adjusted metrics

IMPLEMENTATION:
===============
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
from typing import List, Optional, Dict
from numba import jit, prange

# Hurst Exponent via DFA (Optimized with Numba)
@jit(nopython=True, parallel=True, cache=True)
def hurst_exponent(ts, min_scale: int = 4, max_scale: int = None) -> float:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).
    Optimized with Numba for faster execution.
    """
    N = len(ts)
    if N < 20:
        return np.nan

    if max_scale is None:
        max_scale = N // 4

    # Generate scales logarithmically (fewer scales for speed)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 6).astype(np.int32)
    scales = np.unique(scales)
    scales = scales[scales < N // 2]
    if len(scales) < 3:
        return np.nan

    # Step 1: Integrate series
    y = np.cumsum(ts - np.mean(ts))

    F = np.zeros(len(scales))
    for i in range(len(scales)):
        scale = scales[i]
        segments = N // scale
        if segments < 2:
            F[i] = np.nan
            continue

        rms_vals = np.zeros(segments)
        for j in prange(segments):
            seg = y[j*scale:(j+1)*scale]
            if len(seg) < 3:
                rms_vals[j] = np.nan
                continue

            # Linear detrending
            x = np.arange(len(seg), dtype=np.float64)
            # Manual linear regression for detrending (faster than polyfit)
            n_seg = len(seg)
            sum_x = np.sum(x)
            sum_y = np.sum(seg)
            sum_xy = np.sum(x * seg)
            sum_x2 = np.sum(x**2)
            denominator = n_seg * sum_x2 - sum_x**2
            if denominator == 0:
                rms_vals[j] = np.nan
                continue
            slope = (n_seg * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n_seg
            trend = slope * x + intercept
            residuals = seg - trend
            rms_vals[j] = np.sqrt(np.mean(residuals**2))

        # Remove NaN values
        valid_rms = rms_vals[~np.isnan(rms_vals)]
        if len(valid_rms) > 0:
            F[i] = np.mean(valid_rms)
        else:
            F[i] = np.nan

    # Remove NaN F values
    valid_idx = ~np.isnan(F)
    if np.sum(valid_idx) < 3:
        return np.nan

    scales_valid = scales[valid_idx]
    F_valid = F[valid_idx]

    # Step 5: Scaling relation
    log_scales = np.log(scales_valid.astype(np.float64))
    log_F = np.log(F_valid)

    # Manual linear regression for Hurst exponent
    n = len(log_scales)
    sum_x = np.sum(log_scales)
    sum_y = np.sum(log_F)
    sum_xy = np.sum(log_scales * log_F)
    sum_x2 = np.sum(log_scales**2)

    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        return np.nan

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

def compute_hurst_features(returns: pd.Series, windows=None) -> pd.DataFrame:
    """
    Compute Hurst exponents for multiple rolling windows.

    This creates features that capture different time scales of market memory.
    """
    if windows is None:
        windows = [50, 100, 150, 200, 300, 400]

    df = pd.DataFrame(index=returns.index)

    returns_arr = returns.to_numpy(dtype=np.float64)
    hurst_cols = []
    for w in windows:
        col = f'Hurst_{w}'
        hurst_cols.append(col)
        h_vals = np.full(len(returns_arr), np.nan, dtype=np.float64)
        if w < len(returns_arr):
            for i in range(w, len(returns_arr)):
                h_vals[i] = hurst_exponent(returns_arr[i-w:i])
        df[col] = h_vals

    df[hurst_cols] = df[hurst_cols].ffill().bfill().fillna(0.5)
    df['Hurst_std'] = df[hurst_cols].std(axis=1).fillna(0.0)

    # Compute Hurst trend across windows as an additional regime feature.
    log_windows = np.log(np.array(windows, dtype=np.float64))
    x_mean = np.mean(log_windows)
    denom = np.sum((log_windows - x_mean) ** 2)
    hurst_matrix = df[hurst_cols].to_numpy(dtype=np.float64)
    hurst_means = np.mean(hurst_matrix, axis=1)
    hurst_trend = np.empty(len(hurst_matrix), dtype=np.float64)
    for i in range(len(hurst_matrix)):
        hurst_trend[i] = np.sum((log_windows - x_mean) * (hurst_matrix[i] - hurst_means[i])) / denom
    df['Hurst_trend'] = hurst_trend
    return df

def compute_momentum_volatility(returns: pd.Series, momentum_window: int = 10, volatility_window: int = 10) -> pd.DataFrame:
    """
    Compute momentum and volatility features to improve regime classification.
    """
    df = pd.DataFrame(index=returns.index)
    df['momentum'] = returns.rolling(momentum_window, min_periods=1).mean()
    df['volatility'] = returns.rolling(volatility_window, min_periods=1).std().fillna(0)
    return df


def compute_price_trend(prices: pd.Series, fast_window: int = 20, slow_window: int = 50) -> pd.Series:
    """
    Compute a stronger market trend indicator using moving averages and price slope.
    """
    ma_fast = prices.rolling(window=fast_window, min_periods=1).mean()
    ma_slow = prices.rolling(window=slow_window, min_periods=1).mean()
    slope = np.log(prices).diff().rolling(fast_window, min_periods=1).mean()
    trend_signal = pd.Series(0, index=prices.index)
    trend_signal[(ma_fast > ma_slow) & (slope > 0)] = 1
    trend_signal[(ma_fast < ma_slow) & (slope < 0)] = -1
    return trend_signal.fillna(0).astype(int)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram})


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    """
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.fillna(atr.mean())


def compute_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Compute Bollinger Bands.
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return pd.DataFrame({'sma': sma, 'upper': upper, 'lower': lower})


def compute_additional_features(prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """
    Compute additional technical indicators for enhanced feature set.
    """
    df = pd.DataFrame(index=prices.index)
    
    # RSI
    df['rsi'] = compute_rsi(prices)
    
    # MACD
    macd_df = compute_macd(prices)
    df = df.join(macd_df)
    
    # ATR (assuming high/low/close are prices for simplicity, but ideally need OHLC)
    # For simplicity, use prices as proxy
    df['atr'] = compute_atr(prices, prices, prices)
    
    # Bollinger Bands
    bb_df = compute_bollinger_bands(prices)
    df = df.join(bb_df)
    
    # Additional momentum features
    df['roc'] = prices.pct_change(periods=10)  # Rate of Change
    df['williams_r'] = ((prices - prices.rolling(14).max()) / (prices.rolling(14).max() - prices.rolling(14).min())) * -100
    
    # Fill NaN values
    df = df.bfill().ffill().fillna(0)
    
    return df


def smooth_signals(signals: pd.Series, window: int = 3) -> pd.Series:
    """
    Smooth trading signals to reduce whipsaw and noise.
    """
    smoothed = signals.rolling(window, min_periods=1).median()
    return smoothed.round().fillna(0).astype(int)


def require_stable_signals(signals: pd.Series, min_consecutive: int = 2) -> pd.Series:
    """
    Require a signal to persist for a number of consecutive bars before taking a position.
    """
    stable = pd.Series(0, index=signals.index)
    current_signal = 0
    consecutive = 0

    for idx in signals.index:
        s = signals.loc[idx]
        if s != 0 and s == current_signal:
            consecutive += 1
        elif s != 0:
            current_signal = s
            consecutive = 1
        else:
            current_signal = 0
            consecutive = 0

        stable.loc[idx] = s if consecutive >= min_consecutive else 0

    return stable


def filter_signals_by_distance(signals: pd.Series, labels: np.ndarray, pca_df: pd.DataFrame, kmeans, threshold_quantile: float = 0.75) -> pd.Series:
    """
    Filter low-confidence cluster assignments using distance to KMeans centroid.
    """
    distances = kmeans.transform(pca_df.values)
    assigned_distances = distances[np.arange(len(labels)), labels]
    threshold = np.quantile(assigned_distances, threshold_quantile)
    filtered = signals.copy()
    low_confidence = assigned_distances > threshold
    filtered.iloc[low_confidence] = 0
    return filtered


def compute_hurst_rule_signal(feature_df: pd.DataFrame, hurst_cols, hurst_threshold: float = 0.56, momentum_threshold: float = 0.0, volatility_threshold: float = 0.025, hurst_std_threshold: float = 0.08) -> pd.Series:
    """
    Compute a fallback long-only signal using Hurst average, price trend, momentum, volatility, and Hurst stability.
    """
    avg_hurst = feature_df[hurst_cols].mean(axis=1)
    strong_hurst = avg_hurst >= hurst_threshold
    positive_trend = feature_df['price_trend'] == 1
    positive_momentum = feature_df['momentum'] > momentum_threshold
    low_volatility = feature_df['volatility'] < volatility_threshold
    stable_hurst = feature_df['Hurst_std'] < hurst_std_threshold

    rule_signal = (strong_hurst & positive_trend & positive_momentum & low_volatility & stable_hurst).astype(int)
    return pd.Series(rule_signal.values, index=feature_df.index)


def use_fallback_signals(cluster_signals: pd.Series, rule_signals: pd.Series, silhouette: float, min_silhouette: float = 0.12, hard_fallback: float = 0.05) -> pd.Series:
    """
    Use fallback rule-based signals when clustering quality is poor.
    """
    if np.isnan(silhouette) or silhouette < hard_fallback:
        return rule_signals
    if silhouette < min_silhouette:
        combined = ((cluster_signals == 1) | (rule_signals == 1)).astype(int)
        return combined
    return cluster_signals


def smooth_cluster_labels(labels: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Smooth cluster labels to avoid rapid regime switching.
    """
    smoothed = np.copy(labels)
    for i in range(len(labels)):
        start = max(0, i - window + 1)
        window_labels = labels[start:i + 1]
        smoothed[i] = pd.Series(window_labels).mode().iloc[0]
    return smoothed


def apply_pca_decorrelation(df: pd.DataFrame, hurst_cols, variance_threshold=0.95):
    """
    Apply PCA to decorrelate Hurst features.

    PCA transforms correlated Hurst features into uncorrelated principal components,
    improving clustering quality and reducing dimensionality.
    """
    X = df[hurst_cols].dropna().values
    if X.shape[0] == 0:
        return df, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Create new columns for PCA components
    pca_cols = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, index=df.dropna().index, columns=pca_cols)

    return pca_df, pca, scaler

def perform_kmeans_clustering(X_pca, n_clusters=3, random_state=42):
    """
    Perform K-means clustering on PCA-transformed features.

    Clusters represent different market regimes based on Hurst patterns and derived momentum/volatility signals.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5, max_iter=200)
    labels = kmeans.fit_predict(X_pca)

    silhouette = np.nan
    if X_pca.shape[0] > n_clusters:
        silhouette = silhouette_score(X_pca, labels)

    return labels, kmeans, silhouette


def select_best_kmeans(X_pca, cluster_range=(2, 4), random_state=42):
    """
    Select the best KMeans clustering configuration based on silhouette score.
    """
    best_result = None
    best_silhouette = -np.inf
    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        labels, kmeans, silhouette = perform_kmeans_clustering(X_pca, n_clusters, random_state)
        if np.isnan(silhouette):
            silhouette = -np.inf
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_result = (labels, kmeans, silhouette, n_clusters)

    if best_result is None:
        return perform_kmeans_clustering(X_pca, 3, random_state)

    labels, kmeans, silhouette, _ = best_result
    return labels, kmeans, silhouette

def interpret_clusters_auto(labels, pca_df, feature_df, hurst_cols, min_hurst: float = 0.55, min_momentum: float = 0.0, min_cluster_fraction: float = 0.05):
    """
    Automatically interpret clusters using Hurst, momentum, and volatility features.

    - Highest mean H and positive momentum: Long
    - Lowest mean H and negative momentum: Short
    - Middle/uncertain cluster: Neutral
    """
    cluster_df = pd.DataFrame({'cluster': labels}, index=pca_df.index)
    combined = feature_df.loc[pca_df.index, hurst_cols + ['Hurst_trend', 'Hurst_std', 'momentum', 'volatility', 'rsi', 'macd', 'signal', 'histogram', 'atr', 'roc', 'williams_r']].join(cluster_df)

    size_fraction = combined['cluster'].value_counts(normalize=True)
    valid_clusters = size_fraction[size_fraction >= min_cluster_fraction].index
    if len(valid_clusters) < 2:
        return None, None, None

    combined = combined[combined['cluster'].isin(valid_clusters)]
    mean_h_per_cluster = combined.groupby('cluster')[hurst_cols].mean().mean(axis=1)
    mean_hurst_trend_per_cluster = combined.groupby('cluster')['Hurst_trend'].mean()
    mean_hurst_std_per_cluster = combined.groupby('cluster')['Hurst_std'].mean()
    mean_momentum_per_cluster = combined.groupby('cluster')['momentum'].mean()
    mean_volatility_per_cluster = combined.groupby('cluster')['volatility'].mean()
    mean_rsi_per_cluster = combined.groupby('cluster')['rsi'].mean()
    mean_macd_per_cluster = combined.groupby('cluster')['macd'].mean()
    mean_signal_per_cluster = combined.groupby('cluster')['signal'].mean()
    mean_histogram_per_cluster = combined.groupby('cluster')['histogram'].mean()
    mean_roc_per_cluster = combined.groupby('cluster')['roc'].mean()
    mean_williams_r_per_cluster = combined.groupby('cluster')['williams_r'].mean()

    score_long = (
        0.3 * mean_h_per_cluster
        + 0.15 * mean_hurst_trend_per_cluster
        + 0.15 * mean_momentum_per_cluster
        - 0.1 * mean_volatility_per_cluster
        - 0.1 * mean_hurst_std_per_cluster
        + 0.1 * (mean_rsi_per_cluster - 50) / 50  # RSI above 50 is bullish
        + 0.1 * mean_macd_per_cluster
        + 0.05 * mean_histogram_per_cluster
        + 0.05 * mean_roc_per_cluster
    )

    buy_cluster = score_long.idxmax()
    dynamic_hurst_threshold = mean_h_per_cluster.mean() + 0.08 * mean_h_per_cluster.std()
    if mean_h_per_cluster.loc[buy_cluster] < max(min_hurst, dynamic_hurst_threshold) or mean_momentum_per_cluster.loc[buy_cluster] < min_momentum:
        buy_cluster = None

    if buy_cluster is not None and mean_hurst_std_per_cluster.loc[buy_cluster] > 0.1:
        buy_cluster = None

    equalized_short = mean_h_per_cluster.max() - mean_h_per_cluster
    score_short = (
        0.25 * equalized_short
        + 0.25 * (-mean_hurst_trend_per_cluster)
        + 0.15 * (-mean_momentum_per_cluster)
        - 0.1 * mean_volatility_per_cluster
        - 0.1 * (mean_rsi_per_cluster - 50) / 50  # RSI below 50 is bearish
        - 0.1 * mean_macd_per_cluster
        - 0.05 * mean_histogram_per_cluster
        - 0.05 * mean_roc_per_cluster
    )

    sell_cluster = score_short.idxmax()
    if sell_cluster == buy_cluster:
        sell_cluster = score_short.drop(buy_cluster).idxmax() if len(score_short) > 1 else sell_cluster

    neutral_candidates = valid_clusters.difference([buy_cluster, sell_cluster])
    neutral_cluster = neutral_candidates[0] if len(neutral_candidates) > 0 else None

    return buy_cluster, sell_cluster, neutral_cluster

def generate_trading_signals(labels, buy_cluster, sell_cluster, neutral_cluster=None, index=None, trend_filter: pd.Series = None, allow_short: bool = False):
    """
    Generate trading signals from cluster labels with an optional trend filter.

    1: Buy (long position)
    -1: Sell (short position)
    0: Neutral (no position)
    """
    if trend_filter is None:
        trend_filter = pd.Series(1, index=index)
    else:
        trend_filter = trend_filter.reindex(index).fillna(0).astype(int)

    def map_signal(cluster, trend):
        if cluster == buy_cluster and trend == 1:
            return 1
        if allow_short and cluster == sell_cluster and trend == -1:
            return -1
        return 0

    signals = pd.Series([map_signal(c, trend_filter.iloc[i]) for i, c in enumerate(labels)], index=index)
    return signals

def backtest_strategy(signals: pd.Series, returns: pd.Series, initial_capital: float = 10000) -> Dict[str, object]:
    """
    Perform backtest without look-ahead bias.

    Signals are shifted by one day before applying to returns.
    """
    aligned_signals = signals.reindex(returns.index).fillna(0)
    lagged_signals = aligned_signals.shift(1).fillna(0)

    strategy_returns = lagged_signals * returns
    strategy_returns = strategy_returns.fillna(0)

    strategy_portfolio = initial_capital * (1 + strategy_returns).cumprod()
    buyhold_portfolio = initial_capital * (1 + returns).cumprod()

    total_return_strategy = strategy_portfolio.iloc[-1] / initial_capital - 1
    total_return_buyhold = buyhold_portfolio.iloc[-1] / initial_capital - 1

    strategy_std = np.std(strategy_returns.values)
    buyhold_std = np.std(returns.values)
    strategy_mean = np.mean(strategy_returns.values)
    buyhold_mean = np.mean(returns.values)

    strategy_sharpe = (strategy_mean / strategy_std) * np.sqrt(252) if strategy_std > 0 else np.nan
    buyhold_sharpe = (buyhold_mean / buyhold_std) * np.sqrt(252) if buyhold_std > 0 else np.nan

    strategy_dd = (strategy_portfolio / strategy_portfolio.cummax() - 1).min()
    buyhold_dd = (buyhold_portfolio / buyhold_portfolio.cummax() - 1).min()

    return {
        'strategy_returns': strategy_returns,
        'strategy_portfolio': strategy_portfolio,
        'buyhold_portfolio': buyhold_portfolio,
        'total_return_strategy': total_return_strategy,
        'total_return_buyhold': total_return_buyhold,
        'sharpe_strategy': strategy_sharpe,
        'sharpe_buyhold': buyhold_sharpe,
        'max_dd_strategy': strategy_dd,
        'max_dd_buyhold': buyhold_dd
    }

def to_float_scalar(value):
    if isinstance(value, pd.Series):
        if len(value) == 1:
            value = value.iloc[0]
        else:
            value = value.iloc[-1]
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return float(value)

def download_data(tickers, start_date, end_date):
    """
    Download historical price data for multiple tickers.
    """
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Use adjusted close if available
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                prices_series = df[price_col].dropna().squeeze()  # Ensure it's a Series
                data[ticker] = prices_series
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    return data


def process_asset(ticker: str, prices: pd.Series, windows: List[int], n_clusters: int) -> Optional[Dict[str, object]]:
    """
    Compute Hurst features, cluster market regimes, generate signals, and backtest.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty:
        return None

    trend_signal = compute_price_trend(prices).reindex(returns.index).fillna(0).astype(int)
    hurst_df = compute_hurst_features(returns, windows)
    momentum_df = compute_momentum_volatility(returns)
    combined_df = hurst_df.join(momentum_df, how='left')
    additional_df = compute_additional_features(prices.reindex(returns.index), returns)
    combined_df = combined_df.join(additional_df, how='left')

    hurst_cols = [f'Hurst_{w}' for w in windows]
    combined_df['price_trend'] = trend_signal.reindex(combined_df.index).fillna(0).astype(int)
    feature_cols = hurst_cols + ['Hurst_trend', 'Hurst_std', 'momentum', 'volatility', 'price_trend'] + list(additional_df.columns)

    pca_df, pca_model, scaler = apply_pca_decorrelation(combined_df, feature_cols)
    if pca_df is None or pca_df.empty:
        return None

    labels, kmeans_model, silhouette = select_best_kmeans(pca_df.values, cluster_range=(2, 4))
    if not np.isnan(silhouette) and silhouette < 0.05:
        labels = smooth_cluster_labels(labels, window=5)

    buy_cluster, sell_cluster, neutral_cluster = interpret_clusters_auto(labels, pca_df, combined_df, hurst_cols)
    rule_signals = compute_hurst_rule_signal(combined_df.loc[pca_df.index], hurst_cols)

    if buy_cluster is None:
        signals = rule_signals
    else:
        cluster_signals = generate_trading_signals(
            labels,
            buy_cluster,
            sell_cluster,
            neutral_cluster,
            pca_df.index,
            trend_filter=trend_signal,
            allow_short=False,
        )
        cluster_signals = filter_signals_by_distance(cluster_signals, labels, pca_df, kmeans_model, threshold_quantile=0.75)
        cluster_signals = smooth_signals(cluster_signals, window=5)
        cluster_signals = require_stable_signals(cluster_signals, min_consecutive=3)

        signals = use_fallback_signals(cluster_signals, rule_signals, silhouette, min_silhouette=0.12)

    full_signals = pd.Series(0, index=returns.index)
    full_signals.loc[pca_df.index] = signals

    backtest_results = backtest_strategy(full_signals, returns)

    cluster_mean = combined_df[hurst_cols].loc[pca_df.index].groupby(pd.Series(labels, index=pca_df.index)).mean().mean(axis=1)

    return {
        'hurst_df': hurst_df,
        'pca_df': pca_df,
        'labels': labels,
        'signals': full_signals,
        'backtest': backtest_results,
        'buy_cluster': buy_cluster,
        'sell_cluster': sell_cluster,
        'neutral_cluster': neutral_cluster,
        'cluster_mean': cluster_mean
    }


def main():
    """
    Complete implementation of fractal trading strategy.
    """
    print("🚀 IMPLEMENTANDO ESTRATÉGIA DE TRADING BASEADA EM FRACTAIS E HURST")
    print("=" * 70)

    # Configuration
    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'AAPL', 'MSFT']  # Restored original assets
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    windows = [50, 100, 150, 200, 300]  # Optimized windows for speed
    n_clusters = 4

    print(f"📊 Ativos: {tickers}")
    print(f"📅 Período: {start_date} até {end_date}")
    print(f"🔍 Janelas Hurst: {windows}")
    print(f"🎯 Clusters: {n_clusters}")
    print()

    # 1. Download data
    print("1️⃣ BAIXANDO DADOS...")
    price_data = download_data(tickers, start_date, end_date)
    if not price_data:
        print("❌ Erro: Nenhum dado baixado")
        return

    # Process each asset
    results = {}
    for ticker, prices in price_data.items():
        print(f"\n📈 Processando {ticker}...")
        asset_result = process_asset(ticker, prices, windows, n_clusters)
        if asset_result is None:
            print(f"   ⚠️ Falha ao processar {ticker}")
            continue

        buy_cluster = asset_result['buy_cluster']
        sell_cluster = asset_result['sell_cluster']
        neutral_cluster = asset_result['neutral_cluster']
        cluster_mean = asset_result['cluster_mean']
        backtest_results = asset_result['backtest']

        print("   └─ Interpretando clusters automaticamente...")
        if buy_cluster is not None:
            print(f"      • Cluster Long (H={cluster_mean.loc[buy_cluster]:.3f}): {buy_cluster}")
        else:
            print("      • Cluster Long: nenhum cluster qualificado")

        if sell_cluster is not None:
            print(f"      • Cluster Short (H={cluster_mean.loc[sell_cluster]:.3f}): {sell_cluster}")
        else:
            print("      • Cluster Short: nenhum cluster qualificado")

        print(f"      • Cluster Neutral: {neutral_cluster}")

        print("   └─ Gerando sinais de negociação e realizando backtest...")
        signal_counts = asset_result['signals'].value_counts().to_dict()
        print(f"      • Distribuição de sinais: {signal_counts}")

        print("   └─ Comparando com Buy & Hold...")
        print(f"      • Retorno Estratégia: {backtest_results['total_return_strategy']:.2%} | Buy & Hold: {backtest_results['total_return_buyhold']:.2%}")
        print(f"      • Sharpe Estratégia: {backtest_results['sharpe_strategy']:.2f} | Buy & Hold: {backtest_results['sharpe_buyhold']:.2f}")
        print(f"      • Max DD Estratégia: {backtest_results['max_dd_strategy']:.2%} | Buy & Hold: {backtest_results['max_dd_buyhold']:.2%}")

        results[ticker] = asset_result

    # Summary
    print("\n" + "=" * 70)
    print("📊 RESUMO CONSOLIDADO:")
    print("=" * 70)

    for ticker, data in results.items():
        bt = data['backtest']
        tr = to_float_scalar(bt['total_return_strategy'])
        sr = to_float_scalar(bt['sharpe_strategy'])
        dd = to_float_scalar(bt['max_dd_strategy'])
        print(f"{ticker}: Retorno={tr:.1%} | Sharpe={sr:.2f} | DD={dd:.1%}")

    print("\n✅ IMPLEMENTAÇÃO CONCLUÍDA!")
    print("Fundamentos teóricos: Fractais e Hurst explicados no código")
    print("DFA: Implementado para cálculo robusto do expoente H")
    print("PCA: Aplicado para descorrelação das features")
    print("K-means: Realizado com interpretação automática")
    print("Sinais: Gerados com lag para evitar look-ahead bias")
    print("Backtest: Comparado com Buy & Hold")

if __name__ == "__main__":
    main()