import argparse
import numpy as np
import pandas as pd
import warnings

from fractal_strategy import (
    download_price, hurst_series,
    run_pca_kmeans, interpret_clusters, generate_signals,
    backtest, performance_metrics, apply_neural
)

warnings.filterwarnings('ignore')

DEFAULT_ASSETS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "B3SA3.SA", "GGBR4.SA", "^BVSP", "AAPL", "MSFT", "SPY"
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
    parser.add_argument('--method', choices=['cluster','neural'], default='cluster',
                        help='which strategy to use for signal generation')
    parser.add_argument('--arch', choices=['lstm','gru','transformer','tft','nbeats','nhits','gnn','hybrid','sph'],
                        default='transformer', help='neural architecture to employ')
    parser.add_argument('--arch-window', type=int, default=30,
                        help='window length for neural model training')
    parser.add_argument('--arch-epochs', type=int, default=10,
                        help='training epochs for neural model')
    parser.add_argument('--arch-lr', type=float, default=1e-3,
                        help='learning rate for neural model')
    parser.add_argument('--auto-tune', action='store_true',
                        help='perform automatic grid search for neural parameters')
    parser.add_argument('--param-grid', type=str, default=None,
                        help='JSON string specifying grid for tuning')
    args = parser.parse_args()

    ativos = parse_assets(args.assets)
    windows = [int(x) for x in args.windows.split(',') if x]
    # choose processing method
    if args.method == 'cluster':
        df_res = process_ativos(ativos=ativos, janelas=windows, start=args.start, end=args.end)
    else:
        # neural pipeline: train chosen architecture
        resultados = []
        for ativo in ativos:
            print(f"\nProcessando {ativo} ({args.arch})...")
            try:
                preco = download_price(ativo, start=args.start, end=args.end)
            except ValueError:
                print(f"Dados insuficientes para {ativo}")
                continue
            retornos = np.log(preco / preco.shift(1)).dropna()
            df = hurst_series(retornos, windows=windows)
            df.dropna(inplace=True)
            if len(df) < args.arch_window * 2:
                print("muitas NaNs, pulando")
                continue
            # optionally tune parameters automatically
            extra_kwargs = {}
            if args.auto_tune:
                from fractal_strategy.strategy import grid_search_neural
                import json
                grid = None
                if args.param_grid:
                    try:
                        grid = json.loads(args.param_grid)
                    except Exception:
                        print("warning: failed to parse param grid, using defaults")
                if grid is None:
                    # default grid: try small variations of window/epochs/lr
                    grid = {
                        'window': [args.arch_window, max(1, args.arch_window//2)],
                        'epochs': [args.arch_epochs, max(1, args.arch_epochs//2)],
                        'lr': [args.arch_lr if hasattr(args,'arch_lr') else 1e-3, 1e-2]
                    }
                best, _ = grid_search_neural(df,
                                            feature_cols=[c for c in df.columns if c.startswith('Hurst_')],
                                            model_name=args.arch,
                                            param_grid=grid,
                                            window=args.arch_window,
                                            epochs=args.arch_epochs)
                if best is not None:
                    print(f"tuning chose parameters {best}")
                    # apply chosen params to args and extra_kwargs, avoiding
                    # duplicate keyword names
                    for key, val in list(best.items()):
                        if key in ('window', 'epochs', 'lr'):
                            setattr(args, f'arch_{key}', val)
                            # don't send these in extra_kwargs as they are
                            # explicitly provided below
                        else:
                            extra_kwargs[key] = val
            sig = apply_neural(df,
                                feature_cols=[c for c in df.columns if c.startswith('Hurst_')],
                                model_name=args.arch,
                                window=args.arch_window,
                                epochs=args.arch_epochs,
                                lr=args.arch_lr,
                                **extra_kwargs)
            df['sinal'] = sig
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
        df_res = pd.DataFrame(resultados)
        print("\n" + "="*100)
        print(f"RESULTADOS FINAIS - ESTRATÉGIA {args.arch.upper()}")
        print("="*100)
        if df_res.empty:
            print("Nenhum ativo processado gerou resultados válidos.")
        else:
            print(df_res.to_string(index=False))
            vitorias = (df_res['Vencedor'] == 'Estratégia').sum()
            print(f"\nEstratégia venceu em {vitorias}/{len(df_res)} ativos (p < 0.05)")
            print(f"Taxa de acerto: {vitorias/len(df_res)*100:.1f}%")

    if args.output and not df_res.empty:
        df_res.to_csv(args.output, index=False)
        print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()

