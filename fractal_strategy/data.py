import pandas as pd
import yfinance as yf


def download_price(ticker: str, start: str = "2010-01-01", end: str = None) -> pd.Series:
    """Download closing prices for a single ticker.
    Handles the MultiIndex format that yfinance may return when requesting
    multiple tickers or in newer versions where the API returns a MultiIndex
    even for a single ticker.
    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "PETR4.SA" or "^BVSP").
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date (inclusive). If None, yfinance will use today's date.
    Returns
    -------
    pandas.Series
        Series of closing prices indexed by date. Name will be ``'close'``.
    Raises
    ------
    ValueError
        If no data could be downloaded for the ticker. 
    """
    kwargs = {"start": start}
    if end:
        kwargs["end"] = end

    df = yf.download(ticker, progress=False, **kwargs)
    if df.empty:
        raise ValueError(f"Could not download data for ticker {ticker}")

    # note: in 2025+ yfinance sometimes returns a MultiIndex even for a single
    # ticker, so we normalize to a simple Series.
    if isinstance(df.columns, pd.MultiIndex):
        # try to locate Adjusted Close first, then Close, else take second-to-last
        col = next((c for c in df.columns if c[0] in ["Adj Close", "Close"]), None)
        if col is None:
            col = df.columns[-2]
        series = df[col].copy()
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        series = df[col].copy()

    series.name = "close"
    series = series.dropna()
    return series
