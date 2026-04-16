#!/usr/bin/env python3
"""
Estratégia Avançada: Otimizada com Position Sizing Dinâmico

Melhorias focadas:
- Otimização de parâmetros com grid search
- Position sizing dinâmico baseado em volatilidade
- Ensemble melhorado de sinais
- Risk management avançado
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from fractal_strategy import run_pca_kmeans, interpret_clusters, generate_signals

warnings.filterwarnings('ignore')

class AdvancedTradingStrategy:
    """Estratégia avançada com otimização e position sizing dinâmico."""

    def __init__(self, n_clusters=3, stop_loss_pct=0.05, take_profit_pct=0.10,
                 max_holding_days=10, min_confidence=0.6, volatility_target=0.15):
        self.n_clusters = n_clusters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.min_confidence = min_confidence
        self.volatility_target = volatility_target  # Volatilidade alvo para position sizing

    def optimize_parameters(self, df, hurst_cols):
        """Otimiza parâmetros usando validação temporal."""
        print("🔍 Otimizando parâmetros...")

        param_combinations = [
            {'n_clusters': 3, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.10, 'max_holding_days': 10},
            {'n_clusters': 3, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.12, 'max_holding_days': 15},
            {'n_clusters': 4, 'stop_loss_pct': 0.06, 'take_profit_pct': 0.15, 'max_holding_days': 12},
            {'n_clusters': 4, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.08, 'max_holding_days': 20},
            {'n_clusters': 5, 'stop_loss_pct': 0.07, 'take_profit_pct': 0.13, 'max_holding_days': 14},
        ]

        best_score = -np.inf
        best_params = None

        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)

        for params in param_combinations:
            scores = []

            for train_idx, test_idx in tscv.split(df):
                train_data = df.iloc[train_idx]
                test_data = df.iloc[test_idx]

                try:
                    # Configurar parâmetros
                    self.n_clusters = params['n_clusters']
                    self.stop_loss_pct = params['stop_loss_pct']
                    self.take_profit_pct = params['take_profit_pct']
                    self.max_holding_days = params['max_holding_days']

                    # Testar
                    signals = self.generate_signals(train_data, hurst_cols)
                    performance = self.quick_backtest(train_data, signals)

                    scores.append(performance['sharpe'])

                except:
                    continue

            avg_score = np.mean(scores) if scores else -np.inf

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        if best_params:
            self.n_clusters = best_params['n_clusters']
            self.stop_loss_pct = best_params['stop_loss_pct']
            self.take_profit_pct = best_params['take_profit_pct']
            self.max_holding_days = best_params['max_holding_days']

            print(f"✅ Parâmetros otimizados: {best_params}")
            print(".2f")

        return best_params

    def volatility_position_sizing(self, df):
        """Calcula position sizing baseado na volatilidade."""
        # Volatilidade realizada
        realized_vol = df['returns'].rolling(20).std() * np.sqrt(252)

        # Position size = volatilidade alvo / volatilidade atual
        position_sizes = self.volatility_target / realized_vol

        # Limitar entre 0.1 e 2.0
        position_sizes = np.clip(position_sizes, 0.1, 2.0)

        # Suavizar mudanças bruscas
        position_sizes = position_sizes.rolling(5, center=True).mean()

        return position_sizes.fillna(1.0)

    def generate_signals(self, df, hurst_cols):
        """Gera sinais com ensemble melhorado."""
        try:
            # 1. Sinais baseados em Hurst + PCA + KMeans
            clusters, _, _ = run_pca_kmeans(df, hurst_cols, n_clusters=self.n_clusters)
            buy_cluster, sell_cluster, _ = interpret_clusters(clusters, df, hurst_cols)
            hurst_signals = generate_signals(clusters, buy_cluster, sell_cluster).values

            # 2. Sinais de momentum
            momentum_signals = self.calculate_momentum_signals(df)

            # 3. Sinais de reversão
            reversal_signals = self.calculate_reversal_signals(df, hurst_cols)

            # 4. Ensemble ponderado
            hurst_weight = 0.5
            momentum_weight = 0.3
            reversal_weight = 0.2

            ensemble_signals = (
                hurst_weight * hurst_signals +
                momentum_weight * momentum_signals +
                reversal_weight * reversal_signals
            )

            final_signals = np.where(
                np.abs(ensemble_signals) >= self.min_confidence,
                np.sign(ensemble_signals),
                0
            )

            # 5. Aplicar filtros e risk management
            final_signals = self.apply_advanced_filters(df, final_signals)
            final_signals = self.apply_risk_management(df, final_signals)

            return final_signals

        except Exception as e:
            print(f"❌ Erro na geração de sinais: {str(e)}")
            return np.zeros(len(df))

    def calculate_momentum_signals(self, df):
        """Calcula sinais de momentum."""
        # Momentum em múltiplas janelas
        momentum_5 = (df['close'] / df['close'].shift(5) - 1)
        momentum_20 = (df['close'] / df['close'].shift(20) - 1)

        # Sinal combinado
        momentum_signal = (momentum_5 + momentum_20) / 2

        # Normalizar e classificar
        momentum_norm = (momentum_signal - momentum_signal.rolling(50).mean()) / momentum_signal.rolling(50).std()

        # Converter para sinais discretos
        momentum_discrete = np.where(momentum_norm > 1, 1,
                                   np.where(momentum_norm < -1, -1, 0))

        return momentum_discrete

    def calculate_reversal_signals(self, df, hurst_cols):
        """Calcula sinais de reversão."""
        # Hurst médio
        hurst_mean = df[hurst_cols].mean(axis=1)

        # Condições de reversão
        low_hurst = hurst_mean < 0.45
        high_volatility = df['volatility_20'] > df['volatility_20'].quantile(0.7)
        extreme_rsi = (df['RSI'] > 70) | (df['RSI'] < 30)

        # Sinal de reversão
        reversal_condition = low_hurst & high_volatility & extreme_rsi

        # Direção baseada em RSI
        reversal_direction = np.where(df['RSI'] > 70, -1,  # Overbought -> sell
                                    np.where(df['RSI'] < 30, 1, 0))  # Oversold -> buy

        reversal_signals = np.where(reversal_condition, reversal_direction, 0)

        return reversal_signals

    def apply_advanced_filters(self, df, signals):
        """Aplica filtros avançados."""
        # Filtro de volatilidade extrema
        vol_filter = df['volatility_20'] < df['volatility_20'].quantile(0.85)

        # Filtro de tendência
        hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]
        hurst_mean = df[hurst_cols].mean(axis=1)
        trend_filter = hurst_mean > 0.5

        # Filtro de momentum consistente
        momentum_5 = (df['close'] / df['close'].shift(5) - 1).fillna(0)
        momentum_20 = (df['close'] / df['close'].shift(20) - 1).fillna(0)
        momentum_consistent = (momentum_5 * momentum_20) > 0

        # Combinar filtros
        combined_filter = vol_filter & trend_filter & momentum_consistent

        # Aplicar filtro usando numpy
        filtered_signals = np.where(combined_filter, signals, 0)

        return filtered_signals

    def apply_risk_management(self, df, signals):
        """Aplica risk management avançado."""
        # Converter para pandas Series para facilitar operações
        signals_series = pd.Series(signals, index=df.index)

        # Calcular preços de entrada
        entry_prices = df['close'].where(signals_series != signals_series.shift(1)).ffill()

        # Stop-loss dinâmico baseado na volatilidade
        vol_multiplier = df['volatility_20'] / df['volatility_20'].rolling(50).mean().fillna(1)
        dynamic_stop = self.stop_loss_pct * vol_multiplier.clip(0.5, 2.0)

        stop_losses = entry_prices * (1 - dynamic_stop * signals_series)

        # Take-profit dinâmico
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

    def quick_backtest(self, df, signals):
        """Backtest rápido para otimização."""
        df = df.copy()
        df['signal'] = signals
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']

        # Métricas básicas
        strategy_return = (1 + df['strategy_returns'].fillna(0)).prod() - 1
        strategy_vol = df['strategy_returns'].fillna(0).std() * np.sqrt(252)
        strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0

        return {
            'return': strategy_return,
            'volatility': strategy_vol,
            'sharpe': strategy_sharpe
        }

def plot_advanced_performance(df, signals, position_sizes, ticker, save_path=None):
    """Plota performance avançada com position sizing."""
    df = df.copy()
    df['signal'] = signals
    df['position_size'] = position_sizes
    df['strategy_returns'] = (
        df['signal'].shift(1) *
        df['position_size'].shift(1) *
        df['returns']
    )

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Preços e sinais com tamanho das posições
    axes[0].plot(df['date'], df['close'], linewidth=1, alpha=0.7)

    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    # Tamanho dos pontos baseado no position size
    buy_sizes = buy_signals['position_size'] * 100
    sell_sizes = sell_signals['position_size'] * 100

    axes[0].scatter(buy_signals['date'], buy_signals['close'],
                   marker='^', color='green', s=buy_sizes, alpha=0.7, label='Compra')
    axes[0].scatter(sell_signals['date'], sell_signals['close'],
                   marker='v', color='red', s=sell_sizes, alpha=0.7, label='Venda')

    axes[0].set_title(f'Preços e Sinais com Position Sizing - {ticker}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Retornos cumulativos
    buy_hold_cum = (1 + df['returns']).cumprod()
    strategy_cum = (1 + df['strategy_returns'].fillna(0)).cumprod()

    axes[1].plot(df['date'], buy_hold_cum, label='Buy & Hold', linewidth=2)
    axes[1].plot(df['date'], strategy_cum, label='Advanced Strategy', linewidth=2)
    axes[1].set_title('Retornos Cumulativos')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Position sizes ao longo do tempo
    axes[2].plot(df['date'], df['position_size'], color='purple', alpha=0.7)
    axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Posição Normal')
    axes[2].set_title('Tamanho Dinâmico das Posições')
    axes[2].set_ylabel('Position Size')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

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

    # Separar por ativo
    assets = df['ticker'].unique()
    results = {}
    all_results = []

    # Features de Hurst
    hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]

    print("🚀 Executando estratégia AVANÇADA...")
    print("-" * 60)

    for ticker in assets:
        print(f"📈 Processando {ticker}...")

        asset_data = df[df['ticker'] == ticker].copy()
        asset_data = asset_data.sort_values('date').reset_index(drop=True)

        if len(asset_data) < 200:
            print("  ⚠️  Dados insuficientes")
            continue

        try:
            # Estratégia avançada
            strategy = AdvancedTradingStrategy()

            # Otimização de parâmetros
            strategy.optimize_parameters(asset_data, hurst_cols)

            # Gerar sinais
            signals = strategy.generate_signals(asset_data, hurst_cols)

            # Position sizing dinâmico
            position_sizes = strategy.volatility_position_sizing(asset_data)

            # Aplicar position sizing aos retornos
            asset_data['signal'] = signals
            asset_data['position_size'] = position_sizes
            asset_data['strategy_returns'] = (
                asset_data['signal'].shift(1) *
                asset_data['position_size'].shift(1) *
                asset_data['returns']
            )

            # Métricas finais
            buy_hold_return = (1 + asset_data['returns']).prod() - 1
            strategy_return = (1 + asset_data['strategy_returns'].fillna(0)).prod() - 1

            buy_hold_vol = asset_data['returns'].std() * np.sqrt(252)
            strategy_vol = asset_data['strategy_returns'].fillna(0).std() * np.sqrt(252)

            buy_hold_sharpe = buy_hold_return / buy_hold_vol if buy_hold_vol > 0 else 0
            strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0

            # Drawdown
            cumulative = (1 + asset_data['strategy_returns'].fillna(0)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Trades
            trades = asset_data[asset_data['strategy_returns'] != 0]
            win_rate = len(trades[trades['strategy_returns'] > 0]) / len(trades) if len(trades) > 0 else 0

            performance = {
                'buy_hold_return': buy_hold_return,
                'strategy_return': strategy_return,
                'buy_hold_sharpe': buy_hold_sharpe,
                'strategy_sharpe': strategy_sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'avg_position_size': position_sizes.mean()
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
            all_results.append(asset_data)

        except Exception as e:
            print(f"  ❌ Erro: {str(e)}")
        print()

    if not results:
        print("❌ Nenhuma estratégia foi executada!")
        return

    # Resultados consolidados
    final_df = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(results).T

    print("🎯 RESULTADO CONSOLIDADO - ESTRATÉGIA AVANÇADA")
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
    output_path = Path('data/advanced_strategy_results.csv')
    final_df.to_csv(output_path, index=False)

    # Gráfico do melhor ativo
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    best_data = final_df[final_df['ticker'] == best_asset].copy()
    best_data = best_data.sort_values('date')

    plot_advanced_performance(
        best_data,
        best_data['signal'],
        best_data['position_size'],
        best_asset,
        save_path=plots_dir / f'advanced_strategy_{best_asset}.png'
    )

    print(f"\n💾 Resultados salvos em: {output_path}")
    print("🎉 Estratégia Avançada concluída!")

if __name__ == '__main__':
    main()