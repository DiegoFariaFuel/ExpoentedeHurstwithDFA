import numpy as np
import pandas as pd

def hurst_exponent(ts, min_scale: int = 4, max_scale: int = None) -> float:
    """Compute the Hurst exponent of a time series using DFA (Detrended Fluctuation Analysis).
    Parameters
    ----------
    ts : array-like
        1‑D series of values (e.g. log returns).
    min_scale : int
        Minimum window size for the DFA.
    max_scale : int or None
        Maximum window size. If ``None`` it defaults to ``len(ts) // 4``.
    Returns
    -------
    float
        Estimated Hurst exponent. ``np.nan`` is returned if the series is too
        short or if there are not enough valid scales.
    """
    ts = np.asarray(ts, dtype=float)
    N = ts.shape[0]
    if N < 20:
        return np.nan
 
    if max_scale is None:
        max_scale = N // 4

    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=12).astype(int)
    scales = np.unique(scales)
    scales = scales[scales < N // 2]
    if len(scales) < 4:
        return np.nan

    y = np.cumsum(ts - np.mean(ts))
    F = []
    for scale in scales:
        segments = N // scale
        rms_vals = []
        for i in range(segments):
            seg = y[i*scale:(i+1)*scale]
            if len(seg) < 3:
                continue
            x = np.arange(len(seg))
            coef = np.polyfit(x, seg, 1)
            trend = np.polyval(coef, x)
            rms_vals.append(np.sqrt(np.mean((seg - trend)**2)))
        if rms_vals:
            F.append(np.mean(rms_vals))
    if len(F) < 4:
        return np.nan

    log_scales = np.log(scales[:len(F)])
    log_F = np.log(F)
    return np.polyfit(log_scales, log_F, 1)[0]

def hurst_series(returns: pd.Series, windows=None) -> pd.DataFrame:
    """Add Hurst exponent features for multiple rolling windows.

    Parameters
    ----------
    returns : pandas.Series
        Log returns indexed by date.
    windows : iterable of int
        List of window sizes (number of observations) over which to compute
        the Hurst exponent.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same index as ``returns`` and columns named
        ``'Hurst_{window}'`` for each window. The first ``window`` rows for
        each feature are ``np.nan``.
    """
    if windows is None:
        windows = [50, 100, 150, 200, 300, 400]

    df = pd.DataFrame(index=returns.index)
    df['retorno'] = returns

    for w in windows:
        col = f'Hurst_{w}'
        h_vals = [np.nan] * len(returns)
        if w < len(returns):
            for i in range(w, len(returns)):
                h = hurst_exponent(returns.iloc[i-w:i].values)
                # replace NaN estimates with neutral 0.5 for stability
                h_vals[i] = h if not np.isnan(h) else 0.5
        df[col] = h_vals

    return df
