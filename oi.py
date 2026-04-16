import argparse
import numpy as np
import pandas as pd
import warnings

from fractal_strategy import (
    download_price, hurst_series,
    run_pca_kmeans, interpret_clusters, generate_signals,
    backtest, performance_metrics
)

warnings.filterwarnings('ignore')

DEFAULT_ASSETS = [
    # === PETRÓLEO E ENERGIA ===
    "PETR4.SA", "PETR3.SA", "PRIO3.SA", "UGPA3.SA", "CSAN3.SA",
    
    # === MINERAÇÃO ===
    "VALE3.SA", "GGBR4.SA", "CSNA3.SA", "USIM5.SA", "GOAU4.SA",
    
    # === BANCOS ===
    "ITUB4.SA", "ITSA4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA", "BPAC11.SA",
    
    # === BEBIDAS E ALIMENTOS ===
    "ABEV3.SA", "JBSS3.SA", "MRFG3.SA", "BEEF3.SA", "CAML3.SA",
    
    # === TECNOLOGIA E TELECOM ===
    "B3SA3.SA", "VIVT3.SA", "TIMS3.SA", "MGLU3.SA", "AMER3.SA",
    
    # === CONSTRUÇÃO E IMÓVEIS ===
    "CYRE3.SA", "EZTC3.SA", "MRVE3.SA", "TEND3.SA", "DIRR3.SA",
    
    # === SAÚDE ===
    "RADL3.SA", "HYPE3.SA", "FLRY3.SA", "QUAL3.SA", "RDOR3.SA",
    
    # === VAREJO ===
    "LREN3.SA", "PETZ3.SA", "VIIA3.SA", "SOMA3.SA", "BHIA3.SA",
    
    # === ÍNDICES BRASILEIROS ===
    "^BVSP", "^IBX50", "^IBOV",
    
    # === AÇÕES INTERNACIONAIS ===
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
    "JPM", "BAC", "WMT", "KO", "PFE", "JNJ", "V", "MA", "HD", "MCD",
    "SPY", "QQQ", "IWM", "VTI", "BND", "GLD"
]

DEFAULT_WINDOWS = [50, 100, 150, 200, 300, 400]

def process_ativos(ativos, janelas, start, end):
    """Run the full pipeline for a list of tickers.
    Parameters
    ----------
    ativos : list of str
        Tickers to process.
    janelas : list of int
        Window sizes for Hurst calculation.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    """
    resultados = []
    print(f"Baixando dados de{len(ativos)} ativos de {start} até {end}...")
    for ativo in ativos:
        print(f"\nProcessando{ativo}...")
        try:
            preco = download_price(ativo, start=start, end=end)
        except ValueError:
            print(f"Dados insuficientes para{ativo}")
            continue

        retornos = np.log(preco / preco.shift(1)).dropna()
        df = hurst_series(retornos, windows=janelas)
        df.dropna(inplace=True)
        if len(df) < max(janelas) * 2:
            print("muitas NaNs,pulando")
            continue

        cols = [c for c in df.columns if c.startswith('Hurst_')]
        labels, pca, kmeans = run_pca_kmeans(df, cols)
        buy, sell, neut = interpret_clusters(labels, df, cols)
        signals = generate_signals(labels, buy, sell, neut)

        df['sinal'] = signals
        df_bt = backtest(df, signal_col='sinal', return_col='retorno')
        mets = performance_metrics(df_bt)

        resultados.append({
            'Ativo': ativo.replace('.SA', '').replace('^', ''),
            'Retorno Estratégia': f"{mets['retorno_est']:+.2%}",
            'Retorno B&H': f"{mets['retorno_bh']:+.2%}",
            'Sharpe Estratégia': round(mets['sharpe_est'], 3),
            'Sharpe B&H': round(mets['sharpe_bh'], 3),
            'Max DD Estratégia': f"{mets['dd_est']:.2%}",
            'Max DD B&H': f"{mets['dd_bh']:.2%}",
            'p-valor Wilcoxon': f"{mets['p_val']:.4f}",
            'Vencedor': 'Estratégia' if mets['p_val'] < 0.05 else 'B&H'
        })

    if not resultados:
        print("Nenhum ativo gerou resultados válidos (todas janelas tinham NaNs ou dados insuficientes).")
        return pd.DataFrame()

    df_resultados = pd.DataFrame(resultados)
    print("\n" + "="*100)
    print("RESULTADOS FINAIS - ESTRATÉGIA HURST + PCA + K-MEANS")
    print("="*100)
    print(df_resultados.to_string(index=False))

    vitorias = (df_resultados['Vencedor'] == 'Estratégia').sum()
    print(f"\nEstratégia venceu em {vitorias}/{len(df_resultados)} ativos (p < 0.05)")
    print(f"Taxa de acerto: {vitorias/len(df_resultados)*100:.1f}%")

    return df_resultados

def parse_assets(text):
    """Convert comma-separated list into ticker symbols.
    Preserves any symbol starting with ``^`` or containing a ``.``.  For other
    strings we append ``.SA`` only if the ticker contains a digit (i.e. looks
    like a Brazilian equity); this avoids incorrectly mangling US tickers such
    as ``SPY`` or ``AAPL``.
    """
    out = []
    for t in text.split(','):
        t = t.strip()
        if not t:
            continue
        if t.startswith('^') or '.' in t:
            out.append(t)
        elif any(ch.isdigit() for ch in t):
            out.append(t + '.SA')
        else:
            out.append(t)
    return out


def main():
    parser = argparse.ArgumentParser(description='Run Hurst-PCA-KMeans strategy on assets')
    parser.add_argument('--assets', type=str, default=','.join(DEFAULT_ASSETS),
                        help='comma-separated tickers (e.g. PETR4.SA,VALE3.SA)')
    parser.add_argument('--start', type=str, default='2015-01-01',
                        help='start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='2025-11-30',
                        help='end date YYYY-MM-DD')
    parser.add_argument('--windows', type=str, default=','.join(str(w) for w in DEFAULT_WINDOWS),
                        help='comma-separated Hurst window sizes')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='filename to save result table (CSV)')
    args = parser.parse_args()

    ativos = parse_assets(args.assets)
    windows = [int(x) for x in args.windows.split(',') if x]
    
    df_res = process_ativos(ativos=ativos, janelas=windows, start=args.start, end=args.end)

    if args.output and not df_res.empty:
        df_res.to_csv(args.output, index=False)
        print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()

