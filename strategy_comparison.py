#!/usr/bin/env python3
"""
Relatório Comparativo das Estratégias de Trading

Compara performance da estratégia original vs melhorada vs simplificada.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_strategy_results(filename):
    """Carrega resultados de uma estratégia."""
    path = Path('data') / filename
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Calcular métricas agregadas por ativo
    results = {}
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()

        if 'strategy_returns' in ticker_data.columns:
            # Calcular métricas
            buy_hold_return = (1 + ticker_data['returns']).prod() - 1
            strategy_return = (1 + ticker_data['strategy_returns'].fillna(0)).prod() - 1

            buy_hold_vol = ticker_data['returns'].std() * np.sqrt(252)
            strategy_vol = ticker_data['strategy_returns'].fillna(0).std() * np.sqrt(252)

            buy_hold_sharpe = buy_hold_return / buy_hold_vol if buy_hold_vol > 0 else 0
            strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0

            # Drawdown
            cumulative = (1 + ticker_data['strategy_returns'].fillna(0)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Trades
            trades = ticker_data[ticker_data['strategy_returns'] != 0]
            win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0

            results[ticker] = {
                'buy_hold_return': buy_hold_return,
                'strategy_return': strategy_return,
                'buy_hold_sharpe': buy_hold_sharpe,
                'strategy_sharpe': strategy_sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades)
            }

    return results

def create_comparison_report():
    """Cria relatório comparativo das estratégias."""

    # Carregar resultados das estratégias
    strategies = {
        'Original': load_strategy_results('strategy_results.csv'),
        'Melhorada': load_strategy_results('improved_strategy_results.csv'),
        'Simplificada': load_strategy_results('simple_strategy_results.csv')
    }

    # Remover estratégias que não carregaram
    strategies = {k: v for k, v in strategies.items() if v is not None}

    if not strategies:
        print("❌ Nenhum resultado de estratégia encontrado!")
        return

    print("📊 RELATÓRIO COMPARATIVO DE ESTRATÉGIAS")
    print("=" * 80)

    # Comparação por ativo
    all_tickers = set()
    for strategy_results in strategies.values():
        all_tickers.update(strategy_results.keys())

    all_tickers = sorted(all_tickers)

    print("\n📈 PERFORMANCE POR ATIVO E ESTRATÉGIA")
    print("-" * 80)

    # Cabeçalho
    header = "Ativo | Estratégia   | Retorno BH | Retorno Strat | Sharpe BH | Sharpe Strat | Drawdown | Win Rate | Trades"
    print(header)
    print("-" * len(header))

    for ticker in all_tickers:
        for strategy_name, strategy_results in strategies.items():
            if ticker in strategy_results:
                r = strategy_results[ticker]
                print("15"
                      "12"
                      "10.1%"
                      "13.1%"
                      "8.2f"
                      "11.2f"
                      "8.1%"
                      "7.1%"
                      "5d")

    # Estatísticas agregadas
    print("\n📊 ESTATÍSTICAS CONSOLIDADAS")
    print("-" * 80)

    summary_data = []
    for strategy_name, strategy_results in strategies.items():
        if strategy_results:
            returns_bh = [r['buy_hold_return'] for r in strategy_results.values()]
            returns_strat = [r['strategy_return'] for r in strategy_results.values()]
            sharpes_bh = [r['buy_hold_sharpe'] for r in strategy_results.values()]
            sharpes_strat = [r['strategy_sharpe'] for r in strategy_results.values()]
            win_rates = [r['win_rate'] for r in strategy_results.values()]

            summary_data.append({
                'strategy': strategy_name,
                'avg_bh_return': np.mean(returns_bh),
                'avg_strat_return': np.mean(returns_strat),
                'avg_bh_sharpe': np.mean(sharpes_bh),
                'avg_strat_sharpe': np.mean(sharpes_strat),
                'avg_win_rate': np.mean(win_rates),
                'n_assets': len(strategy_results)
            })

    summary_df = pd.DataFrame(summary_data)

    print("Estratégia     | Retorno Médio BH | Retorno Médio Strat | Sharpe Médio BH | Sharpe Médio Strat | Win Rate Médio | Ativos")
    print("-" * 110)
    for _, row in summary_df.iterrows():
        print("12"
              "15.1%"
              "17.1%"
              "15.2f"
              "17.2f"
              "13.1%"
              "4d")

    # Análise de melhoria
    print("\n🎯 ANÁLISE DE MELHORIA")
    print("-" * 80)

    if len(summary_df) >= 2:
        # Comparar com estratégia original
        original = summary_df[summary_df['strategy'] == 'Original']
        if len(original) > 0:
            original_return = original['avg_strat_return'].iloc[0]

            for _, row in summary_df.iterrows():
                if row['strategy'] != 'Original':
                    improvement = row['avg_strat_return'] - original_return
                    print(f"{row['strategy']} vs Original:")
                    print(f"   Melhoria no retorno: {improvement:.1%}")
                    print(f"   Sharpe ratio: {row['avg_strat_sharpe']:.2f}")
                    print()

    # Recomendações
    print("\n💡 RECOMENDAÇÕES")
    print("-" * 80)

    if summary_df['avg_strat_return'].max() > 0:
        best_strategy = summary_df.loc[summary_df['avg_strat_return'].idxmax(), 'strategy']
        print(f"✅ Melhor performance: {best_strategy}")
        print("   - Focar nesta estratégia para otimização adicional")
    else:
        print("⚠️  Nenhuma estratégia teve retorno positivo consistente")
        print("   - Revisar lógica de sinais e parâmetros")

    print("🔧 Melhorias sugeridas:")
    print("   - Otimização de parâmetros com validação cruzada")
    print("   - Adição de mais features técnicas")
    print("   - Implementação de position sizing dinâmico")
    print("   - Teste em diferentes períodos de mercado")
    print("   - Análise de sensibilidade aos parâmetros")

    # Criar visualização
    create_comparison_plot(summary_df)

def create_comparison_plot(summary_df):
    """Cria gráfico comparativo das estratégias."""
    if len(summary_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Retornos
    axes[0,0].bar(summary_df['strategy'], summary_df['avg_bh_return'],
                  alpha=0.7, label='Buy & Hold', color='lightblue')
    axes[0,0].bar(summary_df['strategy'], summary_df['avg_strat_return'],
                  alpha=0.7, label='Estratégia', color='darkblue')
    axes[0,0].set_title('Retornos Médios')
    axes[0,0].set_ylabel('Retorno (%)')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)

    # Sharpe ratios
    x = np.arange(len(summary_df))
    width = 0.35
    axes[0,1].bar(x - width/2, summary_df['avg_bh_sharpe'], width,
                  label='Buy & Hold', alpha=0.7, color='lightgreen')
    axes[0,1].bar(x + width/2, summary_df['avg_strat_sharpe'], width,
                  label='Estratégia', alpha=0.7, color='darkgreen')
    axes[0,1].set_title('Sharpe Ratios Médios')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(summary_df['strategy'], rotation=45)
    axes[0,1].legend()

    # Win rates
    axes[1,0].bar(summary_df['strategy'], summary_df['avg_win_rate'],
                  alpha=0.7, color='orange')
    axes[1,0].set_title('Taxa de Acerto Média')
    axes[1,0].set_ylabel('Win Rate (%)')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Número de ativos
    axes[1,1].bar(summary_df['strategy'], summary_df['n_assets'],
                  alpha=0.7, color='purple')
    axes[1,1].set_title('Número de Ativos Testados')
    axes[1,1].set_ylabel('Ativos')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Salvar gráfico
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico comparativo salvo em: plots/strategy_comparison.png")

    plt.show()

def main():
    create_comparison_report()

if __name__ == '__main__':
    main()