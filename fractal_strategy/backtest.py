import pandas as pd
import numpy as np

def backtest(df: pd.DataFrame, signal_col: str='sinal', return_col: str='retorno') -> pd.DataFrame:
    """Perform a simple backtest given signals and returns.
    Assumes ``signal`` is applied with one-day lag to avoid look‑ahead bias.
    """
    df = df.copy()
    df['ret_estrategia'] = df[signal_col].shift(1) * df[return_col]
    df['ret_buyhold'] = df[return_col]
    df[['ret_estrategia', 'ret_buyhold']] = df[['ret_estrategia', 'ret_buyhold']].fillna(0)

    df['patrimonio_estrategia'] = (1 + df['ret_estrategia']).cumprod()
    df['patrimonio_buyhold'] = (1 + df['ret_buyhold']).cumprod()
    return df


def performance_metrics(df: pd.DataFrame) -> dict:
    """Compute common performance metrics from backtest output."""
    ret_est = df['patrimonio_estrategia'].iloc[-1] - 1
    ret_bh  = df['patrimonio_buyhold'].iloc[-1] - 1
    std_est = df['ret_estrategia'].std()
    std_bh = df['ret_buyhold'].std()
    sharpe_est = (df['ret_estrategia'].mean() / std_est * np.sqrt(252)) if std_est != 0 else np.nan
    sharpe_bh  = (df['ret_buyhold'].mean() / std_bh * np.sqrt(252)) if std_bh != 0 else np.nan
    dd_est = (df['patrimonio_estrategia'] / df['patrimonio_estrategia'].cummax() - 1).min()
    dd_bh  = (df['patrimonio_buyhold'] / df['patrimonio_buyhold'].cummax() - 1).min()

    from scipy.stats import wilcoxon
    try:
        _, p_val = wilcoxon(df['ret_estrategia'], df['ret_buyhold'], alternative='greater')
    except Exception:
        # if the test cannot be performed (e.g. constant arrays)
        p_val = np.nan

    return {
        'retorno_est': ret_est,
        'retorno_bh': ret_bh,
        'sharpe_est': sharpe_est,
        'sharpe_bh': sharpe_bh,
        'dd_est': dd_est,
        'dd_bh': dd_bh,
        'p_val': p_val,
    }
