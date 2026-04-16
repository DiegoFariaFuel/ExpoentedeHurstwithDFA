"""
ESTRATÉGIA DE TRADING FRACTAL + HURST EXPONENT
Versão Final Corrigida - Abril 2026
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

# ====================== HURST VIA DFA (otimizado com Numba) ======================
@jit(nopython=True, parallel=True, cache=True)
def hurst_exponent(ts, min_scale: int = 4, max_scale: int = None) -> float:
    N = len(ts)
    if N < 20:
        return np.nan
    if max_scale is None:
        max_scale = N // 4

    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 6).astype(np.int32)
    scales = np.unique(scales)
    scales = scales[scales < N // 2]
    if len(scales) < 3:
        return np.nan

    y = np.cumsum(ts - np.mean(ts))
    F = np.zeros(len(scales))

    for i in range(len(scales)):
        scale = scales[i]
        segments = N // scale
        if segments < 2:
            continue

        rms_vals = np.zeros(segments)
        for j in prange(segments):
            seg = y[j*scale:(j+1)*scale]
            if len(seg) < 3:
                continue
            x = np.arange(len(seg), dtype=np.float64)
            n_seg = len(seg)
            sum_x = np.sum(x)
            sum_y = np.sum(seg)
            sum_xy = np.sum(x * seg)
            sum_x2 = np.sum(x**2)
            denominator = n_seg * sum_x2 - sum_x**2
            if denominator == 0:
                continue
            slope = (n_seg * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n_seg
            trend = slope * x + intercept
            rms_vals[j] = np.sqrt(np.mean((seg - trend)**2))

        valid_rms = rms_vals[~np.isnan(rms_vals)]
        if len(valid_rms) > 0:
            F[i] = np.mean(valid_rms)

    valid_idx = ~np.isnan(F) & (F > 0)
    if np.sum(valid_idx) < 3:
        return np.nan

    log_scales = np.log(scales[valid_idx].astype(np.float64))
    log_F = np.log(F[valid_idx])

    n = len(log_scales)
    sum_x = np.sum(log_scales)
    sum_y = np.sum(log_F)
    sum_xy = np.sum(log_scales * log_F)
    sum_x2 = np.sum(log_scales**2)
    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        return np.nan
    return (n * sum_xy - sum_x * sum_y) / denominator


def compute_hurst_features(returns: pd.Series, windows: List[int] = None) -> pd.DataFrame:
    """Calcula Hurst para múltiplas janelas + feature de tendência."""
    if windows is None:
        windows = [100, 200, 300]

    df = pd.DataFrame(index=returns.index)
    returns_arr = returns.to_numpy(dtype=np.float64)
    hurst_cols = [f'Hurst_{w}' for w in windows]

    for w, col in zip(windows, hurst_cols):
        h_vals = np.full(len(returns_arr), np.nan, dtype=np.float64)
        for i in range(w, len(returns_arr)):
            h_vals[i] = hurst_exponent(returns_arr[i-w:i])
        df[col] = h_vals

    df[hurst_cols] = df[hurst_cols].ffill().bfill().fillna(0.5)
    df['Hurst_std'] = df[hurst_cols].std(axis=1).fillna(0.0)

    # Hurst_trend - CORRIGIDO
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
    df = pd.DataFrame(index=returns.index)
    df['momentum'] = returns.rolling(momentum_window, min_periods=1).mean()
    df['volatility'] = returns.rolling(volatility_window, min_periods=1).std().fillna(0)
    return df


def compute_price_trend(prices: pd.Series, fast_window: int = 20, slow_window: int = 50) -> pd.Series:
    """Calcula tendência com proteção contra MultiIndex do yfinance"""
    prices = prices.copy()
    # Remove MultiIndex se existir (comum com yfinance + vários tickers)
    if isinstance(prices.index, pd.MultiIndex):
        prices = prices.droplevel(0)

    ma_fast = prices.rolling(window=fast_window, min_periods=1).mean()
    ma_slow = prices.rolling(window=slow_window, min_periods=1).mean()
    slope = np.log(prices).diff().rolling(fast_window, min_periods=1).mean()

    long_condition = ma_fast > ma_slow
    short_condition = ma_fast < ma_slow
    trend_signal = long_condition.astype(int) - short_condition.astype(int)
    return trend_signal


def smooth_signals(signals: pd.Series, window: int = 3) -> pd.Series:
    smoothed = signals.rolling(window, min_periods=1).median()
    return smoothed.round().fillna(0).astype(int)


def require_stable_signals(signals: pd.Series, min_consecutive: int = 2) -> pd.Series:
    stable = pd.Series(0, index=signals.index, dtype=int)
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
    distances = kmeans.transform(pca_df.values)
    assigned_distances = distances[np.arange(len(labels)), labels]
    threshold = np.quantile(assigned_distances, threshold_quantile)
    filtered = signals.copy()
    filtered.iloc[assigned_distances > threshold] = 0
    return filtered


def compute_hurst_rule_signal(feature_df: pd.DataFrame, hurst_cols, hurst_threshold: float = 0.5,
                              momentum_threshold: float = -0.01, volatility_threshold: float = 0.05,
                              hurst_std_threshold: float = 0.1) -> pd.Series:
    avg_hurst = feature_df[hurst_cols].mean(axis=1)
    strong_hurst = avg_hurst >= hurst_threshold
    positive_trend = feature_df['price_trend'] == 1
    positive_momentum = feature_df['momentum'] > momentum_threshold
    low_volatility = feature_df['volatility'] < volatility_threshold
    stable_hurst = feature_df['Hurst_std'] < hurst_std_threshold

    rule_signal = (strong_hurst & positive_trend).astype(int)
    return pd.Series(rule_signal.values, index=feature_df.index)


def use_fallback_signals(cluster_signals: pd.Series, rule_signals: pd.Series, silhouette: float,
                         min_silhouette: float = 0.12, hard_fallback: float = 0.05) -> pd.Series:
    if np.isnan(silhouette) or silhouette < hard_fallback:
        return rule_signals
    if silhouette < min_silhouette:
        return ((cluster_signals == 1) | (rule_signals == 1)).astype(int)
    return cluster_signals


def smooth_cluster_labels(labels: np.ndarray, window: int = 3) -> np.ndarray:
    smoothed = np.copy(labels)
    for i in range(len(labels)):
        start = max(0, i - window + 1)
        window_labels = labels[start:i + 1]
        smoothed[i] = pd.Series(window_labels).mode().iloc[0]
    return smoothed


def apply_pca_decorrelation(df: pd.DataFrame, hurst_cols, variance_threshold=0.95):
    X = df[hurst_cols].dropna().values
    if X.shape[0] == 0:
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, index=df.dropna().index, columns=pca_cols)
    return pca_df, pca, scaler


def perform_kmeans_clustering(X_pca, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5, max_iter=200)
    labels = kmeans.fit_predict(X_pca)
    silhouette = np.nan
    if X_pca.shape[0] > n_clusters:
        silhouette = silhouette_score(X_pca, labels)
    return labels, kmeans, silhouette


def select_best_kmeans(X_pca, cluster_range=(2, 4), random_state=42):
    best_result = None
    best_silhouette = -np.inf
    for n in range(cluster_range[0], cluster_range[1] + 1):
        labels, kmeans, silhouette = perform_kmeans_clustering(X_pca, n, random_state)
        if np.isnan(silhouette):
            silhouette = -np.inf
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_result = (labels, kmeans, silhouette, n)
    if best_result is None:
        return perform_kmeans_clustering(X_pca, 3, random_state)
    return best_result[0], best_result[1], best_result[2]


def interpret_clusters_auto(labels, pca_df, feature_df, hurst_cols, min_hurst: float = 0.5,
                            min_momentum: float = -0.01, min_cluster_fraction: float = 0.05):
    cluster_df = pd.DataFrame({'cluster': labels}, index=pca_df.index)
    combined = feature_df.loc[pca_df.index, hurst_cols + ['Hurst_trend', 'Hurst_std', 'momentum', 'volatility']].join(cluster_df)

    size_fraction = combined['cluster'].value_counts(normalize=True)
    valid_clusters = size_fraction[size_fraction >= min_cluster_fraction].index
    if len(valid_clusters) < 2:
        return None, None, None

    combined = combined[combined['cluster'].isin(valid_clusters)]
    mean_h = combined.groupby('cluster')[hurst_cols].mean().mean(axis=1)
    mean_trend = combined.groupby('cluster')['Hurst_trend'].mean()
    mean_std = combined.groupby('cluster')['Hurst_std'].mean()
    mean_mom = combined.groupby('cluster')['momentum'].mean()
    mean_vol = combined.groupby('cluster')['volatility'].mean()

    score_long = 0.4*mean_h + 0.2*mean_trend + 0.2*mean_mom - 0.1*mean_vol - 0.1*mean_std
    buy_cluster = score_long.idxmax()

    dynamic_thresh = mean_h.mean() + 0.08 * mean_h.std()
    if mean_h.loc[buy_cluster] < max(min_hurst, dynamic_thresh) or mean_mom.loc[buy_cluster] < min_momentum:
        buy_cluster = None
    if buy_cluster is not None and mean_std.loc[buy_cluster] > 0.1:
        buy_cluster = None

    score_short = 0.35*(mean_h.max() - mean_h) + 0.35*(-mean_trend) + 0.2*(-mean_mom) - 0.1*mean_vol
    sell_cluster = score_short.idxmax()
    if sell_cluster == buy_cluster and len(score_short) > 1:
        sell_cluster = score_short.drop(buy_cluster).idxmax()

    neutral = list(set(valid_clusters) - {buy_cluster, sell_cluster})
    neutral_cluster = neutral[0] if neutral else None

    return buy_cluster, sell_cluster, neutral_cluster


def generate_trading_signals(labels, buy_cluster, sell_cluster, neutral_cluster=None, index=None,
                             trend_filter: pd.Series = None, allow_short: bool = False):
    if trend_filter is None:
        trend_filter = pd.Series(1, index=index)
    else:
        trend_filter = trend_filter.reindex(index).fillna(0).astype(int)

    def map_signal(c, t):
        if isinstance(t, pd.Series):
            t = t.iloc[0] if not t.empty else 0
        if buy_cluster is not None and c == buy_cluster and t == 1:
            return 1
        if allow_short and sell_cluster is not None and c == sell_cluster and t == -1:
            return -1
        return 0

    signals = pd.Series([map_signal(c, trend_filter.loc[index[i]]) for i, c in enumerate(labels)], index=index)
    return signals


def backtest_strategy(signals: pd.Series, returns: pd.Series, initial_capital: float = 10000) -> Dict:
    aligned = signals.reindex(returns.index).fillna(0)
    lagged = aligned.shift(1).fillna(0)
    strat_ret = lagged * returns.fillna(0)

    strat_port = initial_capital * (1 + strat_ret).cumprod()
    bh_port = initial_capital * (1 + returns).cumprod()

    return {
        'total_return_strategy': (strat_port.iloc[-1] / initial_capital - 1).mean(),
        'total_return_buyhold': (bh_port.iloc[-1] / initial_capital - 1).mean(),
        'sharpe_strategy': ((strat_ret.mean() / strat_ret.std() * np.sqrt(252)).mean()) if (strat_ret.std() > 0).any() else np.nan,
        'sharpe_buyhold': ((returns.mean() / returns.std() * np.sqrt(252)).mean()) if (returns.std() > 0).any() else np.nan,
        'max_dd_strategy': ((strat_port / strat_port.cummax() - 1).min()).mean(),
        'max_dd_buyhold': ((bh_port / bh_port.cummax() - 1).min()).mean(),
    }


def download_data(tickers: List[str], start_date: str, end_date: str):
    print("📥 Baixando dados reais do Yahoo Finance...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                series = df[col].dropna()
                # Proteção contra MultiIndex
                if isinstance(series.index, pd.MultiIndex):
                    non_date_levels = [name for name in series.index.names if name != 'Date']
                    if non_date_levels:
                        series = series.droplevel(non_date_levels[0])
                    else:
                        series = series.droplevel(0)
                # Ensure DatetimeIndex
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                data[ticker] = series
                print(f"   ✅ {ticker} carregado ({len(series)} dias)")
        except Exception as e:
            print(f"   ❌ Erro em {ticker}: {e}")
    return data


def process_asset(ticker: str, prices: pd.Series, windows: List[int], n_clusters: int) -> Optional[Dict]:
    if isinstance(prices.index, pd.MultiIndex):
        prices = prices.droplevel(0)

    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < 300:
        return None

    trend_signal = compute_price_trend(prices).reindex(returns.index).fillna(0).astype(int)
    hurst_df = compute_hurst_features(returns, windows)
    mom_df = compute_momentum_volatility(returns)

    combined_df = hurst_df.join(mom_df, how='left')
    hurst_cols = [f'Hurst_{w}' for w in windows]
    combined_df['price_trend'] = trend_signal

    pca_df, _, scaler = apply_pca_decorrelation(combined_df, hurst_cols)
    if pca_df is None or pca_df.empty:
        return None

    labels, kmeans, silhouette = select_best_kmeans(pca_df.values)
    if not np.isnan(silhouette) and silhouette < 0.05:
        labels = smooth_cluster_labels(labels, window=5)

    buy_c, sell_c, neu_c = interpret_clusters_auto(labels, pca_df, combined_df, hurst_cols)
    rule_signals = compute_hurst_rule_signal(combined_df.loc[pca_df.index], hurst_cols)

    if buy_c is None:
        signals = rule_signals
    else:
        cluster_signals = generate_trading_signals(labels, buy_c, sell_c, neu_c, pca_df.index, trend_signal, allow_short=True)
        cluster_signals = filter_signals_by_distance(cluster_signals, labels, pca_df, kmeans)
        cluster_signals = smooth_signals(cluster_signals, window=5)
        cluster_signals = require_stable_signals(cluster_signals, min_consecutive=1)
        signals = use_fallback_signals(cluster_signals, rule_signals, silhouette)

    full_signals = pd.Series(0, index=returns.index)
    full_signals.loc[pca_df.index] = signals

    backtest = backtest_strategy(full_signals, returns)

    return {
        'signals': full_signals,
        'backtest': backtest,
        'buy_cluster': buy_c,
        'sell_cluster': sell_c,
        'neutral_cluster': neu_c
    }


def main():
    print("🚀 ESTRATÉGIA DE TRADING FRACTAL + HURST")
    print("=" * 80)

    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'AAPL', 'MSFT']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    windows = [100, 200, 300]

    print(f"📊 Ativos: {tickers}")
    print(f"📅 Período: {start_date} até {end_date}")
    print(f"🔍 Janelas Hurst: {windows}\n")

    price_data = download_data(tickers, start_date, end_date)
    if not price_data:
        print("❌ Nenhum dado baixado!")
        return

    results = {}
    for ticker, prices in price_data.items():
        print(f"\n📈 Processando {ticker}...")
        result = process_asset(ticker, prices, windows, n_clusters=2)
        if result is None:
            print(f"   ⚠️ Falha ao processar {ticker}")
            continue

        bt = result['backtest']
        print(f"   Retorno Estratégia : {bt['total_return_strategy']:.2%}")
        print(f"   Retorno Buy&Hold   : {bt['total_return_buyhold']:.2%}")
        print(f"   Sharpe Estratégia  : {bt['sharpe_strategy']:.2f}")
        print(f"   Sharpe Buy&Hold    : {bt['sharpe_buyhold']:.2f}")
        print(f"   Max Drawdown Strat : {bt['max_dd_strategy']:.2%}")

        results[ticker] = result

    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL")
    print("=" * 80)
    for ticker, res in results.items():
        bt = res['backtest']
        print(f"{ticker:10} | Ret={bt['total_return_strategy']:.1%} | Sharpe={bt['sharpe_strategy']:.2f} | DD={bt['max_dd_strategy']:.1%}")

    print("\n✅ Estratégia executada com sucesso!")


if __name__ == "__main__":
    main()