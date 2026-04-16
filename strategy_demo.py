#!/usr/bin/env python3
"""
Demonstração: Usando dados processados com algoritmo Hurst + PCA + KMeans

Este script mostra como integrar os dados processados com o pipeline
de machine learning existente para gerar sinais de trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from fractal_strategy import hurst_series, run_pca_kmeans, interpret_clusters, generate_signals

def load_and_prepare_data(filepath, tickers=None):
    """Carrega e prepara dados para o modelo."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Filtrar ativos específicos se solicitado
    if tickers:
        df = df[df['ticker'].isin(tickers)]

    # Selecionar features para o modelo
    feature_cols = [
        'returns', 'volatility_20', 'RSI', 'MACD', 'BB_width',
        'Hurst_50', 'Hurst_100', 'Hurst_150', 'Hurst_200'
    ]

    # Verificar se todas as features existem
    available_features = [col for col in feature_cols if col in df.columns]
    if len(available_features) < len(feature_cols):
        print(f"⚠️  Algumas features não encontradas: {set(feature_cols) - set(available_features)}")
        feature_cols = available_features

    print(f"📊 Usando {len(feature_cols)} features: {feature_cols}")

    # Remover NaN e preparar dados
    model_data = df.dropna(subset=feature_cols + ['returns'])

    # Agrupar por ticker para análise individual
    grouped_data = {}
    for ticker in model_data['ticker'].unique():
        ticker_data = model_data[model_data['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        grouped_data[ticker] = ticker_data

    return grouped_data, feature_cols

def run_trading_strategy(ticker_data, feature_cols, n_clusters=3):
    """Executa estratégia de trading para um ativo."""
    df = ticker_data.copy()

    # Identificar colunas de Hurst
    hurst_cols = [col for col in feature_cols if col.startswith('Hurst_')]

    if not hurst_cols:
        print(f"⚠️  Nenhuma coluna de Hurst encontrada para {df['ticker'].iloc[0]}")
        return None

    # Verificar se há dados suficientes
    if len(df) < 50:
        print(f"⚠️  Dados insuficientes para {df['ticker'].iloc[0]}")
        return None

    # Executar PCA + KMeans
    try:
        clusters, pca_components, kmeans_model = run_pca_kmeans(df, hurst_cols, n_clusters=n_clusters)

        # Interpretar clusters
        buy_cluster, sell_cluster, neutral_cluster = interpret_clusters(clusters, df, hurst_cols)

        # Gerar sinais de trading
        signals = generate_signals(clusters, buy_cluster, sell_cluster, neutral_cluster)

        # Adicionar resultados ao DataFrame
        df['cluster'] = clusters
        df['signal'] = signals

        return df

    except Exception as e:
        print(f"❌ Erro ao processar {df['ticker'].iloc[0]}: {str(e)}")
        return None

def calculate_strategy_performance(df):
    """Calcula performance da estratégia."""
    if 'signal' not in df.columns:
        return None

    # Estratégia: comprar quando signal=1, vender quando signal=-1
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']

    # Remover NaN
    df_clean = df.dropna(subset=['strategy_returns', 'returns'])

    if len(df_clean) == 0:
        return None

    # Métricas de performance
    buy_hold_return = (1 + df_clean['returns']).prod() - 1
    strategy_return = (1 + df_clean['strategy_returns']).prod() - 1

    buy_hold_vol = df_clean['returns'].std() * np.sqrt(252)
    strategy_vol = df_clean['strategy_returns'].std() * np.sqrt(252)

    # Sharpe ratio (assumindo taxa livre de risco = 0)
    buy_hold_sharpe = buy_hold_return / buy_hold_vol if buy_hold_vol > 0 else 0
    strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0

    return {
        'buy_hold_return': buy_hold_return,
        'strategy_return': strategy_return,
        'buy_hold_volatility': buy_hold_vol,
        'strategy_volatility': strategy_vol,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'total_signals': len(df_clean[df_clean['signal'] != 0]),
        'win_rate': len(df_clean[df_clean['strategy_returns'] > 0]) / len(df_clean) if len(df_clean) > 0 else 0
    }

def main():
    # Caminho dos dados
    data_path = Path('data/processed_data.csv')

    if not data_path.exists():
        print(f"❌ Arquivo não encontrado: {data_path}")
        return

    # Carregar dados
    print("📂 Carregando dados processados...")
    grouped_data, feature_cols = load_and_prepare_data(data_path)

    print(f"✅ Dados carregados para {len(grouped_data)} ativos\n")

    # Executar estratégia para cada ativo
    results = {}
    all_results = []

    print("🚀 Executando estratégia de trading...")
    print("-" * 60)

    for ticker, ticker_data in grouped_data.items():
        print(f"📈 Processando {ticker}...")

        # Executar estratégia
        result_df = run_trading_strategy(ticker_data, feature_cols)

        if result_df is not None:
            # Calcular performance
            performance = calculate_strategy_performance(result_df)

            if performance:
                results[ticker] = performance
                result_df['ticker'] = ticker
                all_results.append(result_df)

                print(f"  Retorno Buy&Hold: {performance['buy_hold_return']:.1%}")
                print(f"  Retorno Estratégia: {performance['strategy_return']:.1%}")
                print(f"  Sharpe Buy&Hold: {performance['buy_hold_sharpe']:.2f}")
                print(f"  Sharpe Estratégia: {performance['strategy_sharpe']:.2f}")
                print(f"  Sinais gerados: {performance['total_signals']}")
                print(f"  Taxa de acerto: {performance['win_rate']:.1%}")
            else:
                print("  ❌ Erro no cálculo de performance")
        else:
            print("  ❌ Erro na execução da estratégia")
        print()

    if not results:
        print("❌ Nenhuma estratégia foi executada com sucesso!")
        return

    # Combinar todos os resultados
    final_df = pd.concat(all_results, ignore_index=True)

    # Salvar resultados
    output_path = Path('data/strategy_results.csv')
    final_df.to_csv(output_path, index=False)

    print("📊 RESUMO GERAL DA ESTRATÉGIA")
    print("=" * 60)

    # Estatísticas agregadas
    summary = pd.DataFrame(results).T

    print("📈 Performance por Ativo:")
    print(summary.round(4))
    print()

    print("📊 Estatísticas Gerais:")
    print(f"  Retorno Médio Estratégia: {summary['strategy_return'].mean():.1%}")
    print(f"  Retorno Médio Buy&Hold: {summary['buy_hold_return'].mean():.1%}")
    print(f"  Sharpe Médio: {summary['strategy_sharpe'].mean():.2f}")
    print(f"  Sharpe Médio Buy&Hold: {summary['buy_hold_sharpe'].mean():.2f}")
    print(f"  Taxa de Acerto Média: {summary['win_rate'].mean():.1%}")

    # Melhor e pior performance
    best_asset = summary['strategy_return'].idxmax()
    worst_asset = summary['strategy_return'].idxmin()

    print("\n🏆 Melhor Performance:")
    print(f"  {best_asset}: {summary.loc[best_asset, 'strategy_return']:.1%}")
    print(f"  Sharpe: {summary.loc[best_asset, 'strategy_sharpe']:.2f}")

    print("\n📉 Pior Performance:")
    print(f"  {worst_asset}: {summary.loc[worst_asset, 'strategy_return']:.1%}")
    print(f"  Sharpe: {summary.loc[worst_asset, 'strategy_sharpe']:.2f}")

    print(f"\n💾 Resultados salvos em: {output_path}")
    print("✅ Demonstração concluída!")

if __name__ == '__main__':
    main()