#!/usr/bin/env python3
"""
Estratégia Híbrida: Combina simplicidade da estratégia básica com
otimização e risk management da avançada.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from datetime import datetime

class HybridStrategy:
    """Estratégia híbrida que combina o melhor das abordagens."""

    def __init__(self, n_clusters=3, stop_loss_pct=0.05, take_profit_pct=0.10,
                 max_holding_days=10, min_confidence=0.6):
        self.n_clusters = n_clusters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.min_confidence = min_confidence

    def optimize_parameters(self, df, param_grid):
        """Otimiza parâmetros usando walk-forward validation."""
        best_params = None
        best_sharpe = -np.inf

        for params in ParameterGrid(param_grid):
            strategy = HybridStrategy(**params)

            # Walk-forward: treina em 70% dos dados, testa em 30%
            split_idx = int(len(df) * 0.7)
            train_data = df.iloc[:split_idx]
            test_data = df.iloc[split_idx:]

            try:
                test_returns = strategy.backtest(test_data)['strategy_returns']
                sharpe = self.calculate_sharpe(test_returns)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            except:
                continue

        return best_params if best_params else {'n_clusters': 3, 'stop_loss_pct': 0.05,
                                               'take_profit_pct': 0.10, 'max_holding_days': 10}

    def generate_signals(self, df):
        """Gera sinais usando abordagem híbrida simplificada."""
        try:
            # 1. Sinais Hurst (base da estratégia simples)
            hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]
            hurst_signals = np.where(df[hurst_cols].mean(axis=1) > 0.5, 1, -1)

            # 2. Momentum simples (como na estratégia simples)
            momentum_20 = (df['close'] / df['close'].shift(20) - 1).fillna(0)
            momentum_signals = np.where(momentum_20 > 0.02, 1,
                                      np.where(momentum_20 < -0.02, -1, 0))

            # 3. Ensemble simplificado (média ponderada)
            ensemble_signals = 0.7 * hurst_signals + 0.3 * momentum_signals

            # 4. Filtro básico de confiança
            final_signals = np.where(np.abs(ensemble_signals) >= self.min_confidence,
                                   np.sign(ensemble_signals), 0)

            # 5. Aplicar risk management básico
            final_signals = self.apply_basic_risk_management(df, final_signals)

            return final_signals

        except Exception as e:
            print(f"❌ Erro na geração de sinais: {str(e)}")
            return np.zeros(len(df))

    def apply_basic_risk_management(self, df, signals):
        """Aplica risk management básico."""
        signals_series = pd.Series(signals, index=df.index)

        # Stop-loss simples
        entry_prices = df['close'].where(signals_series != signals_series.shift(1)).ffill()
        stop_losses = entry_prices * (1 - self.stop_loss_pct * signals_series)

        # Take-profit simples
        take_profits = entry_prices * (1 + self.take_profit_pct * signals_series)

        # Condições de saída
        stop_hit = (
            ((signals_series == 1) & (df['close'] <= stop_losses)) |
            ((signals_series == -1) & (df['close'] >= stop_losses))
        )

        profit_hit = (
            ((signals_series == 1) & (df['close'] >= take_profits)) |
            ((signals_series == -1) & (df['close'] <= take_profits))
        )

        # Time exit
        position_days = (
            signals_series.groupby((signals_series != signals_series.shift(1)).cumsum())
              .cumcount()
              .where(signals_series != 0, 0)
        )
        time_exit = position_days >= self.max_holding_days

        # Aplicar saídas
        exit_conditions = stop_hit | profit_hit | time_exit
        signals_series.loc[exit_conditions] = 0

        return signals_series.values

    def calculate_position_size(self, df, signals):
        """Calcula tamanho de posição baseado na volatilidade."""
        volatility = df['volatility_20'].fillna(df['volatility_20'].mean())

        # Tamanho base inversamente proporcional à volatilidade
        base_size = 1.0 / (volatility + 0.01)  # Evitar divisão por zero

        # Normalizar para manter tamanho médio consistente
        normalized_size = base_size / base_size.rolling(50).mean().fillna(1)

        # Limitar tamanho máximo
        position_size = np.clip(normalized_size, 0.5, 2.0)

        return position_size

    def backtest(self, df):
        """Executa backtest da estratégia."""
        signals = self.generate_signals(df)
        position_sizes = self.calculate_position_size(df, signals)

        # Calcular retornos da estratégia
        df = df.copy()
        df['signal'] = signals
        df['position_size'] = position_sizes
        df['strategy_returns'] = df['signal'].shift(1) * df['returns'] * df['position_size']

        return df

    def calculate_sharpe(self, returns):
        """Calcula ratio de Sharpe."""
        if len(returns) < 2:
            return 0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0

        # Sharpe ratio anualizado (assumindo 252 dias de trading)
        return (mean_return / std_return) * np.sqrt(252)

def plot_hybrid_performance(data, signals, position_sizes, ticker, save_path=None):
    """Plota performance da estratégia híbrida."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Preço e sinais
    axes[0].plot(data['date'], data['close'], label='Preço', alpha=0.7)
    buy_signals = data[signals == 1]
    sell_signals = data[signals == -1]

    if not buy_signals.empty:
        axes[0].scatter(buy_signals['date'], buy_signals['close'],
                       marker='^', color='green', label='Compra', s=50)
    if not sell_signals.empty:
        axes[0].scatter(sell_signals['date'], sell_signals['close'],
                       marker='v', color='red', label='Venda', s=50)

    axes[0].set_title(f'Estratégia Híbrida - {ticker}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Retornos acumulados
    buy_hold_returns = (1 + data['returns']).cumprod() - 1
    strategy_returns = (1 + data['strategy_returns'].fillna(0)).cumprod() - 1

    axes[1].plot(data['date'], buy_hold_returns, label='Buy & Hold', alpha=0.7)
    axes[1].plot(data['date'], strategy_returns, label='Estratégia Híbrida', linewidth=2)
    axes[1].set_title('Retornos Acumulados')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # Tamanho das posições
    axes[2].plot(data['date'], position_sizes, label='Tamanho da Posição', color='orange')
    axes[2].set_title('Tamanho das Posições (Volatilidade-Based)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.close()

def main():
    print("🚀 Executando estratégia HÍBRIDA...")
    print("-" * 60)

    # Carregar dados
    data_path = Path('data/processed_data.csv')
    if not data_path.exists():
        print("❌ Arquivo de dados processados não encontrado!")
        return

    print("📂 Carregando dados processados...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Dados carregados: {len(df):,} registros")

    # Ativos para testar
    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'AAPL', 'MSFT']

    results = {}
    all_results = []

    param_grid = {
        'n_clusters': [3, 4],
        'stop_loss_pct': [0.03, 0.05, 0.07],
        'take_profit_pct': [0.08, 0.10, 0.12],
        'max_holding_days': [8, 10, 12]
    }

    for ticker in tickers:
        print(f"\n📈 Processando {ticker}...")

        try:
            asset_data = df[df['ticker'] == ticker].copy().reset_index(drop=True)

            if len(asset_data) < 200:
                print(f"  ⚠️ Dados insuficientes para {ticker}")
                continue

            # Otimizar parâmetros
            print("🔍 Otimizando parâmetros...")
            strategy = HybridStrategy()
            best_params = strategy.optimize_parameters(asset_data, param_grid)
            print(f"✅ Parâmetros otimizados: {best_params}")

            # Criar estratégia otimizada
            optimized_strategy = HybridStrategy(**best_params)

            # Executar backtest
            result_df = optimized_strategy.backtest(asset_data)
            result_df['ticker'] = ticker

            # Calcular métricas
            buy_hold_return = (1 + asset_data['returns']).prod() - 1
            strategy_return = (1 + result_df['strategy_returns'].fillna(0)).prod() - 1

            buy_hold_sharpe = optimized_strategy.calculate_sharpe(asset_data['returns'])
            strategy_sharpe = optimized_strategy.calculate_sharpe(result_df['strategy_returns'].fillna(0))

            # Calcular drawdown máximo
            cumulative = (1 + result_df['strategy_returns'].fillna(0)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calcular win rate
            trades = result_df[result_df['signal'] != result_df['signal'].shift(1)]
            trades = trades[trades['signal'] != 0]
            win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0

            performance = {
                'buy_hold_return': buy_hold_return,
                'strategy_return': strategy_return,
                'buy_hold_sharpe': buy_hold_sharpe,
                'strategy_sharpe': strategy_sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'avg_position_size': result_df['position_size'].mean()
            }

            results[ticker] = performance

            print(".1%")
            print(".1%")
            print(".2f")
            print(".1%")
            print(".1%")
            print(f"  Trades: {performance['total_trades']}")
            print(".1%")

            # Adicionar aos resultados
            all_results.append(result_df)

        except Exception as e:
            print(f"  ❌ Erro: {str(e)}")
        print()

    if not results:
        print("❌ Nenhuma estratégia foi executada!")
        return

    # Resultados consolidados
    final_df = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(results).T

    print("🎯 RESULTADO CONSOLIDADO - ESTRATÉGIA HÍBRIDA")
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
    output_path = Path('data/hybrid_strategy_results.csv')
    final_df.to_csv(output_path, index=False)

    # Gráfico do melhor ativo
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    best_data = final_df[final_df['ticker'] == best_asset].copy()
    best_data = best_data.sort_values('date')

    plot_hybrid_performance(
        best_data,
        best_data['signal'],
        best_data['position_size'],
        best_asset,
        save_path=plots_dir / f'hybrid_strategy_{best_asset}.png'
    )

    print(f"\n💾 Resultados salvos em: {output_path}")
    print("🎉 Estratégia Híbrida concluída!")

if __name__ == '__main__':
    main()