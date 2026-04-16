#!/usr/bin/env python3
"""
Estratégia Melhorada: Hurst + PCA + KMeans com Otimização

Melhorias implementadas:
- Validação cruzada temporal
- Ensemble de sinais (Hurst + Momentum + Reversal)
- Stop-loss e take-profit dinâmicos
- Otimização de parâmetros
- Filtros de entrada/saída
- Risk management
- Análise de performance robusta
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from fractal_strategy import run_pca_kmeans, interpret_clusters, generate_signals

warnings.filterwarnings('ignore')

class ImprovedTradingStrategy:
    """Estratégia de trading melhorada com múltiplas camadas de análise."""

    def __init__(self, n_clusters=3, stop_loss_pct=0.05, take_profit_pct=0.10,
                 max_holding_days=20, min_confidence=0.6):
        self.n_clusters = n_clusters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.min_confidence = min_confidence

    def calculate_momentum_signals(self, df, windows=[5, 10, 20]):
        """Calcula sinais baseados em momentum."""
        signals = pd.DataFrame(index=df.index)

        for window in windows:
            # Momentum simples
            signals[f'momentum_{window}d'] = (df['close'] / df['close'].shift(window) - 1)

            # ROC (Rate of Change)
            signals[f'roc_{window}d'] = df['close'].pct_change(window)

            # Momentum normalizado
            signals[f'momentum_norm_{window}d'] = (
                signals[f'momentum_{window}d'] - signals[f'momentum_{window}d'].rolling(50).mean()
            ) / signals[f'momentum_{window}d'].rolling(50).std()

        # Sinal combinado de momentum
        momentum_cols = [col for col in signals.columns if 'momentum_norm' in col]
        signals['momentum_signal'] = signals[momentum_cols].mean(axis=1)

        # Classificação: 1 (forte alta), -1 (forte baixa), 0 (neutro)
        signals['momentum_class'] = pd.cut(
            signals['momentum_signal'],
            bins=[-np.inf, -1, 1, np.inf],
            labels=[-1, 0, 1]
        ).astype(int)

        return signals[['momentum_signal', 'momentum_class']]

    def calculate_reversal_signals(self, df, hurst_cols):
        """Calcula sinais baseados em reversão usando Hurst."""
        signals = pd.DataFrame(index=df.index)

        # Média do Hurst (quanto menor, mais provável reversão)
        signals['hurst_mean'] = df[hurst_cols].mean(axis=1)

        # Desvio padrão do Hurst (consistência)
        signals['hurst_std'] = df[hurst_cols].std(axis=1)

        # Sinal de reversão: Hurst baixo + alta volatilidade recente
        signals['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(50).mean()

        # Condições para reversão
        signals['reversal_signal'] = (
            (signals['hurst_mean'] < 0.45) &  # Hurst baixo (reversão provável)
            (signals['volatility_ratio'] > 1.2) &  # Volatilidade elevada
            (df['RSI'] > 70) | (df['RSI'] < 30)  # RSI extremo
        ).astype(int)

        # Direção da reversão baseada em RSI
        signals['reversal_direction'] = np.where(
            df['RSI'] > 70, -1,  # RSI alto -> sinal de venda (reversão baixa)
            np.where(df['RSI'] < 30, 1, 0)  # RSI baixo -> sinal de compra (reversão alta)
        )

        return signals[['reversal_signal', 'reversal_direction', 'hurst_mean']]

    def apply_risk_management(self, df, signals):
        """Aplica stop-loss, take-profit e limites de tempo."""
        df = df.copy()
        df['signal'] = signals

        # Calcular preços de referência para stop-loss/take-profit
        df['entry_price'] = df['close'].where(df['signal'] != df['signal'].shift(1))

        # Forward fill dos preços de entrada
        df['entry_price'] = df['entry_price'].fillna(method='ffill')

        # Calcular dias em posição
        df['position_days'] = (
            df.groupby((df['signal'] != df['signal'].shift(1)).cumsum())
              .cumcount()
              .where(df['signal'] != 0, 0)
        )

        # Stop-loss dinâmico (apenas quando há posição)
        df['stop_loss_price'] = np.where(
            df['signal'] != 0,
            df['entry_price'] * (1 - self.stop_loss_pct * df['signal']),
            np.nan
        )

        # Take-profit dinâmico (apenas quando há posição)
        df['take_profit_price'] = np.where(
            df['signal'] != 0,
            df['entry_price'] * (1 + self.take_profit_pct * df['signal']),
            np.nan
        )

        # Condições de saída
        stop_loss_hit = (
            ((df['signal'] == 1) & (df['close'] <= df['stop_loss_price'])) |
            ((df['signal'] == -1) & (df['close'] >= df['stop_loss_price']))
        )

        take_profit_hit = (
            ((df['signal'] == 1) & (df['close'] >= df['take_profit_price'])) |
            ((df['signal'] == -1) & (df['close'] <= df['take_profit_price']))
        )

        max_holding_hit = df['position_days'] >= self.max_holding_days

        # Aplicar saídas
        exit_conditions = stop_loss_hit | take_profit_hit | max_holding_hit
        df.loc[exit_conditions, 'signal'] = 0

        return df['signal']

    def create_ensemble_signal(self, hurst_signal, momentum_signal, reversal_signal,
                              hurst_weight=0.4, momentum_weight=0.3, reversal_weight=0.3):
        """Cria sinal ensemble combinando múltiplas fontes."""

        # Normalizar sinais para mesma escala
        signals_df = pd.DataFrame({
            'hurst': hurst_signal,
            'momentum': momentum_signal,
            'reversal': reversal_signal
        })

        # Preencher NaN com 0
        signals_df = signals_df.fillna(0)

        # Calcular sinal ponderado
        ensemble_signal = (
            hurst_weight * signals_df['hurst'] +
            momentum_weight * signals_df['momentum'] +
            reversal_weight * signals_df['reversal']
        )

        # Aplicar threshold de confiança
        ensemble_signal = np.where(
            np.abs(ensemble_signal) >= self.min_confidence,
            np.sign(ensemble_signal),
            0
        )

        return ensemble_signal.astype(int)

    def optimize_parameters(self, df, hurst_cols, param_grid=None):
        """Otimiza parâmetros usando validação cruzada temporal."""

        if param_grid is None:
            param_grid = {
                'n_clusters': [3, 4, 5],
                'hurst_weight': [0.3, 0.4, 0.5],
                'momentum_weight': [0.2, 0.3, 0.4],
                'reversal_weight': [0.2, 0.3, 0.4]
            }

        # Time series split para validação
        tscv = TimeSeriesSplit(n_splits=3)

        best_score = -np.inf
        best_params = None

        # Testar combinações de parâmetros
        for n_clusters in param_grid['n_clusters']:
            for h_weight in param_grid['hurst_weight']:
                for m_weight in param_grid['momentum_weight']:
                    for r_weight in param_grid['reversal_weight']:
                        if h_weight + m_weight + r_weight != 1.0:
                            continue

                        scores = []
                        for train_idx, test_idx in tscv.split(df):
                            train_data = df.iloc[train_idx]
                            test_data = df.iloc[test_idx]

                            try:
                                # Treinar no conjunto de treino
                                self.n_clusters = n_clusters
                                signals = self.generate_signals(train_data, hurst_cols)

                                # Testar no conjunto de teste
                                test_signals = self.generate_signals(test_data, hurst_cols)

                                # Calcular retorno
                                test_returns = test_data['returns'] * test_signals.shift(1)
                                cumulative_return = (1 + test_returns.fillna(0)).prod() - 1

                                scores.append(cumulative_return)

                            except:
                                continue

                        avg_score = np.mean(scores) if scores else -np.inf

                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'n_clusters': n_clusters,
                                'hurst_weight': h_weight,
                                'momentum_weight': m_weight,
                                'reversal_weight': r_weight
                            }

        if best_params:
            self.n_clusters = best_params['n_clusters']
            print(f"📊 Parâmetros otimizados: {best_params}")
            print(".2%")

        return best_params

    def generate_signals(self, df, hurst_cols):
        """Gera sinais usando ensemble de estratégias."""

        try:
            # 1. Sinais baseados em Hurst + PCA + KMeans
            clusters, _, _ = run_pca_kmeans(df, hurst_cols, n_clusters=self.n_clusters)
            buy_cluster, sell_cluster, _ = interpret_clusters(clusters, df, hurst_cols)
            hurst_signal = generate_signals(clusters, buy_cluster, sell_cluster).values

            # Verificar se há NaN
            if np.any(np.isnan(hurst_signal)):
                print("  ⚠️  NaN detectado em hurst_signal")
                hurst_signal = np.nan_to_num(hurst_signal, nan=0)

            # 2. Sinais de momentum
            momentum_signals = self.calculate_momentum_signals(df)
            momentum_signal = momentum_signals['momentum_class'].values

            if np.any(np.isnan(momentum_signal)):
                print("  ⚠️  NaN detectado em momentum_signal")
                momentum_signal = np.nan_to_num(momentum_signal, nan=0)

            # 3. Sinais de reversão
            reversal_signals = self.calculate_reversal_signals(df, hurst_cols)
            reversal_signal = reversal_signals['reversal_signal'] * reversal_signals['reversal_direction']
            reversal_signal = reversal_signal.values

            if np.any(np.isnan(reversal_signal)):
                print("  ⚠️  NaN detectado em reversal_signal")
                reversal_signal = np.nan_to_num(reversal_signal, nan=0)

            # 4. Ensemble de sinais
            ensemble_signal = self.create_ensemble_signal(
                hurst_signal, momentum_signal, reversal_signal
            )

            # 5. Aplicar risk management
            final_signal = self.apply_risk_management(df, ensemble_signal)

            return final_signal

        except Exception as e:
            print(f"  ❌ Erro na geração de sinais: {str(e)}")
            # Retornar sinal neutro em caso de erro
            return np.zeros(len(df))

def backtest_strategy(df, signals, transaction_cost=0.001):
    """Executa backtest completo da estratégia."""

    df = df.copy()
    df['signal'] = signals
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']

    # Aplicar custos de transação
    trades = df['signal'] != df['signal'].shift(1)
    df.loc[trades, 'strategy_returns'] -= transaction_cost

    # Calcular métricas
    buy_hold_return = (1 + df['returns']).prod() - 1
    strategy_return = (1 + df['strategy_returns'].fillna(0)).prod() - 1

    buy_hold_vol = df['returns'].std() * np.sqrt(252)
    strategy_vol = df['strategy_returns'].fillna(0).std() * np.sqrt(252)

    # Sharpe ratio
    buy_hold_sharpe = buy_hold_return / buy_hold_vol if buy_hold_vol > 0 else 0
    strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0

    # Drawdown máximo
    cumulative_returns = (1 + df['strategy_returns'].fillna(0)).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_trades = len(df[df['strategy_returns'] > 0])
    total_trades = len(df[df['strategy_returns'] != 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Profit factor
    gross_profit = df[df['strategy_returns'] > 0]['strategy_returns'].sum()
    gross_loss = abs(df[df['strategy_returns'] < 0]['strategy_returns'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        'buy_hold_return': buy_hold_return,
        'strategy_return': strategy_return,
        'buy_hold_volatility': buy_hold_vol,
        'strategy_volatility': strategy_vol,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'avg_trade_return': df['strategy_returns'].mean(),
        'best_trade': df['strategy_returns'].max(),
        'worst_trade': df['strategy_returns'].min()
    }

def plot_strategy_performance(df, signals, save_path=None):
    """Plota performance da estratégia."""

    df = df.copy()
    df['signal'] = signals
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 1. Preços e sinais
    axes[0].plot(df['date'], df['close'], linewidth=1, alpha=0.7, label='Preço')

    # Plotar sinais de compra/venda
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    axes[0].scatter(buy_signals['date'], buy_signals['close'],
                   marker='^', color='green', s=50, label='Compra')
    axes[0].scatter(sell_signals['date'], sell_signals['close'],
                   marker='v', color='red', s=50, label='Venda')

    axes[0].set_title('Preços e Sinais de Trading')
    axes[0].set_ylabel('Preço')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Retornos cumulativos
    buy_hold_cum = (1 + df['returns']).cumprod()
    strategy_cum = (1 + df['strategy_returns'].fillna(0)).cumprod()

    axes[1].plot(df['date'], buy_hold_cum, label='Buy & Hold', linewidth=2)
    axes[1].plot(df['date'], strategy_cum, label='Estratégia', linewidth=2)
    axes[1].set_title('Retornos Cumulativos')
    axes[1].set_ylabel('Retorno Cumulativo')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Drawdown
    running_max = strategy_cum.expanding().max()
    drawdown = (strategy_cum - running_max) / running_max
    axes[2].fill_between(df['date'], drawdown, 0, alpha=0.3, color='red')
    axes[2].set_title('Drawdown da Estratégia')
    axes[2].set_ylabel('Drawdown')
    axes[2].set_xlabel('Data')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.show()

def main():
    # Carregar dados processados
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

    # Estratégia melhorada
    strategy = ImprovedTradingStrategy(
        n_clusters=4,  # Mais clusters para melhor granularidade
        stop_loss_pct=0.08,  # Stop loss mais agressivo
        take_profit_pct=0.15,  # Take profit mais conservador
        max_holding_days=15,  # Limite de tempo reduzido
        min_confidence=0.7  # Threshold de confiança mais alto
    )

    # Features de Hurst
    hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]

    print("🚀 Executando estratégia melhorada...")
    print("-" * 60)

    all_results = []

    for ticker in assets:
        print(f"📈 Processando {ticker}...")

        asset_data = df[df['ticker'] == ticker].copy()
        asset_data = asset_data.sort_values('date').reset_index(drop=True)

        if len(asset_data) < 200:  # Dados insuficientes
            print("  ⚠️  Dados insuficientes")
            continue

        try:
            # Otimizar parâmetros (opcional - demorado)
            # strategy.optimize_parameters(asset_data, hurst_cols)

            # Gerar sinais
            signals = strategy.generate_signals(asset_data, hurst_cols)

            # Backtest
            performance = backtest_strategy(asset_data, signals)

            results[ticker] = performance

            # Estatísticas
            print(".1%")
            print(".1%")
            print(".2f")
            print(".1%")
            print(f"  Drawdown Máx: {performance['max_drawdown']:.1%}")
            print(f"  Trades: {performance['total_trades']}")
            print(".1%")

            # Adicionar ao resultado final
            asset_data['signal'] = signals
            asset_data['strategy_returns'] = asset_data['signal'].shift(1) * asset_data['returns']
            all_results.append(asset_data)

        except Exception as e:
            print(f"  ❌ Erro: {str(e)}")
        print()

    if not results:
        print("❌ Nenhuma estratégia foi executada com sucesso!")
        return

    # Resultados consolidados
    final_df = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(results).T

    print("📊 RESULTADO CONSOLIDADO - ESTRATÉGIA MELHORADA")
    print("=" * 60)
    print(".1%")
    print(".1%")
    print(".2f")
    print(".2f")
    print(".1%")
    print(".1%")
    print(f"  Retorno Médio Buy&Hold: {summary['buy_hold_return'].mean():.1%}")
    print(f"  Retorno Médio Estratégia: {summary['strategy_return'].mean():.1%}")

    # Melhor e pior performance
    best_asset = summary['strategy_return'].idxmax()
    worst_asset = summary['strategy_return'].idxmin()

    print("\n🏆 Melhor Performance:")
    print(f"  {best_asset}: {summary.loc[best_asset, 'strategy_return']:.1%}")
    print(f"  Sharpe: {summary.loc[best_asset, 'strategy_sharpe']:.2f}")

    print("\n📉 Pior Performance:")
    print(f"  {worst_asset}: {summary.loc[worst_asset, 'strategy_return']:.1%}")
    print(f"  Sharpe: {summary.loc[worst_asset, 'strategy_sharpe']:.2f}")

    # Salvar resultados
    output_path = Path('data/improved_strategy_results.csv')
    final_df.to_csv(output_path, index=False)

    # Criar gráficos
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Gráfico do melhor ativo
    best_data = final_df[final_df['ticker'] == best_asset].copy()
    best_data = best_data.sort_values('date')
    plot_strategy_performance(best_data, best_data['signal'],
                            save_path=plots_dir / f'best_strategy_{best_asset}.png')

    print(f"\n💾 Resultados salvos em: {output_path}")
    print("✅ Estratégia melhorada concluída!")

if __name__ == '__main__':
    main()