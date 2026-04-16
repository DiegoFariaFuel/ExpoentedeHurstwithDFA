#!/usr/bin/env python3
"""
Estratégia Simplificada: Foco no Hurst + Risk Management

Versão simplificada para testar o conceito básico antes de adicionar complexidade.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from fractal_strategy import run_pca_kmeans, interpret_clusters, generate_signals

class SimpleHurstStrategy:
    """Estratégia simplificada focada em Hurst + risk management básico."""

    def __init__(self, n_clusters=3, stop_loss_pct=0.05, take_profit_pct=0.10,
                 max_holding_days=10):
        self.n_clusters = n_clusters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days

    def generate_signals(self, df, hurst_cols):
        """Gera sinais baseados apenas em Hurst + PCA + KMeans."""
        try:
            # PCA + KMeans nos dados de Hurst
            clusters, _, _ = run_pca_kmeans(df, hurst_cols, n_clusters=self.n_clusters)

            # Interpretar clusters baseado na média do Hurst
            buy_cluster, sell_cluster, neutral_cluster = interpret_clusters(clusters, df, hurst_cols)

            # Gerar sinais básicos
            signals = generate_signals(clusters, buy_cluster, sell_cluster, neutral_cluster)

            # Aplicar filtros básicos
            signals = self.apply_basic_filters(df, signals)

            # Risk management simples
            signals = self.apply_simple_risk_management(df, signals)

            return signals.values

        except Exception as e:
            print(f"  ❌ Erro: {str(e)}")
            return np.zeros(len(df))

    def apply_basic_filters(self, df, signals):
        """Aplica filtros básicos de entrada."""
        # Filtro de volatilidade - evitar entradas em períodos de alta volatilidade
        volatility_filter = df['volatility_20'] < df['volatility_20'].quantile(0.8)

        # Filtro de tendência - usar apenas em mercados com Hurst > 0.5
        hurst_mean = df[[col for col in df.columns if col.startswith('Hurst_')]].mean(axis=1)
        trend_filter = hurst_mean > 0.5

        # Combinar filtros
        combined_filter = volatility_filter & trend_filter

        # Aplicar filtro
        filtered_signals = signals.where(combined_filter, 0)

        return filtered_signals

    def apply_simple_risk_management(self, df, signals):
        """Aplica risk management básico."""
        signals = signals.copy()

        # Calcular preços de entrada
        entry_prices = df['close'].where(signals != signals.shift(1)).ffill()

        # Stop-loss simples
        stop_losses = entry_prices * (1 - self.stop_loss_pct * signals)

        # Take-profit simples
        take_profits = entry_prices * (1 + self.take_profit_pct * signals)

        # Condições de saída
        stop_hit = (
            ((signals == 1) & (df['close'] <= stop_losses)) |
            ((signals == -1) & (df['close'] >= stop_losses))
        )

        profit_hit = (
            ((signals == 1) & (df['close'] >= take_profits)) |
            ((signals == -1) & (df['close'] <= take_profits))
        )

        # Aplicar saídas
        signals.loc[stop_hit | profit_hit] = 0

        return signals

def backtest_simple_strategy(df, signals):
    """Backtest simplificado."""
    df = df.copy()
    df['signal'] = signals
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']

    # Métricas básicas
    buy_hold_return = (1 + df['returns']).prod() - 1
    strategy_return = (1 + df['strategy_returns'].fillna(0)).prod() - 1

    buy_hold_vol = df['returns'].std() * np.sqrt(252)
    strategy_vol = df['strategy_returns'].fillna(0).std() * np.sqrt(252)

    buy_hold_sharpe = buy_hold_return / buy_hold_vol if buy_hold_vol > 0 else 0
    strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0

    # Drawdown
    cumulative = (1 + df['strategy_returns'].fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Estatísticas de trades
    trades = df[df['strategy_returns'] != 0]
    win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0

    return {
        'buy_hold_return': buy_hold_return,
        'strategy_return': strategy_return,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades)
    }

def plot_simple_performance(df, signals, ticker, save_path=None):
    """Plota performance simplificada."""
    df = df.copy()
    df['signal'] = signals
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Preços e sinais
    axes[0].plot(df['date'], df['close'], linewidth=1, alpha=0.7)

    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    axes[0].scatter(buy_signals['date'], buy_signals['close'],
                   marker='^', color='green', s=50, label='Compra')
    axes[0].scatter(sell_signals['date'], sell_signals['close'],
                   marker='v', color='red', s=50, label='Venda')

    axes[0].set_title(f'Preços e Sinais - {ticker}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Retornos cumulativos
    buy_hold_cum = (1 + df['returns']).cumprod()
    strategy_cum = (1 + df['strategy_returns'].fillna(0)).cumprod()

    axes[1].plot(df['date'], buy_hold_cum, label='Buy & Hold', linewidth=2)
    axes[1].plot(df['date'], strategy_cum, label='Estratégia Hurst', linewidth=2)
    axes[1].set_title('Retornos Cumulativos')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.show()

def main():
    # Carregar dados
    data_path = Path('data/processed_data.csv')

    if not data_path.exists():
        print(f"❌ Arquivo não encontrado: {data_path}")
        return

    print("📂 Carregando dados processados...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Dados carregados: {len(df):,} registros\n")

    # Estratégia simplificada
    strategy = SimpleHurstStrategy(
        n_clusters=3,
        stop_loss_pct=0.08,
        take_profit_pct=0.12,
        max_holding_days=15
    )

    # Features de Hurst
    hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]

    assets = df['ticker'].unique()
    results = {}
    all_results = []

    print("🚀 Executando estratégia simplificada...")
    print("-" * 60)

    for ticker in assets:
        print(f"📈 Processando {ticker}...")

        asset_data = df[df['ticker'] == ticker].copy()
        asset_data = asset_data.sort_values('date').reset_index(drop=True)

        if len(asset_data) < 200:
            print("  ⚠️  Dados insuficientes")
            continue

        # Gerar sinais
        signals = strategy.generate_signals(asset_data, hurst_cols)

        # Backtest
        performance = backtest_simple_strategy(asset_data, signals)

        results[ticker] = performance

        print(".1%")
        print(".1%")
        print(".2f")
        print(".1%")
        print(f"  Trades: {performance['total_trades']}")

        # Adicionar aos resultados
        asset_data['signal'] = signals
        asset_data['strategy_returns'] = asset_data['signal'].shift(1) * asset_data['returns']
        all_results.append(asset_data)

        print()

    if not results:
        print("❌ Nenhuma estratégia foi executada!")
        return

    # Resultados consolidados
    final_df = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(results).T

    print("📊 RESULTADO CONSOLIDADO - ESTRATÉGIA SIMPLIFICADA")
    print("=" * 60)
    print(".1%")
    print(".1%")
    print(".2f")
    print(".2f")
    print(".1%")

    # Melhor performance
    best_asset = summary['strategy_return'].idxmax()
    print("\n🏆 Melhor Performance:")
    print(f"  {best_asset}: {summary.loc[best_asset, 'strategy_return']:.1%}")

    # Salvar resultados
    output_path = Path('data/simple_strategy_results.csv')
    final_df.to_csv(output_path, index=False)

    # Gráfico do melhor ativo
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    best_data = final_df[final_df['ticker'] == best_asset].copy()
    best_data = best_data.sort_values('date')
    plot_simple_performance(best_data, best_data['signal'], best_asset,
                          save_path=plots_dir / f'simple_strategy_{best_asset}.png')

    print(f"\n💾 Resultados salvos em: {output_path}")
    print("✅ Estratégia simplificada concluída!")

if __name__ == '__main__':
    main()