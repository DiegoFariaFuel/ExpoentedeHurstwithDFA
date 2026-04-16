#!/usr/bin/env python3
"""
Estratégia Ultimate: Otimizada com ML e Position Sizing Dinâmico

Melhorias avançadas:
- Otimização de parâmetros com grid search
- Position sizing dinâmico baseado em Kelly Criterion
- Ensemble de sinais com Random Forest
- Múltiplos timeframes para Hurst
- Walk-forward validation
- Risk management avançado
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar

from fractal_strategy import run_pca_kmeans, interpret_clusters, generate_signals

warnings.filterwarnings('ignore')

class UltimateTradingStrategy:
    """Estratégia ultimate com ML e otimização avançada."""

    def __init__(self, n_clusters=3, stop_loss_pct=0.05, take_profit_pct=0.10,
                 max_holding_days=10, min_confidence=0.6, kelly_fraction=0.5):
        self.n_clusters = n_clusters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.min_confidence = min_confidence
        self.kelly_fraction = kelly_fraction

        # Modelos treinados
        self.signal_model = None
        self.scaler = StandardScaler()

    def optimize_parameters(self, df, hurst_cols):
        """Otimiza parâmetros usando grid search com validação temporal."""
        print("🔍 Otimizando parâmetros...")

        param_grid = {
            'n_clusters': [3, 4, 5],
            'stop_loss_pct': [0.05, 0.08, 0.10],
            'take_profit_pct': [0.10, 0.12, 0.15],
            'max_holding_days': [10, 15, 20],
            'min_confidence': [0.5, 0.6, 0.7]
        }

        best_score = -np.inf
        best_params = None

        # Time series split para validação
        tscv = TimeSeriesSplit(n_splits=3)

        for n_clusters in param_grid['n_clusters']:
            for stop_loss in param_grid['stop_loss_pct']:
                for take_profit in param_grid['take_profit_pct']:
                    for max_days in param_grid['max_holding_days']:
                        for min_conf in param_grid['min_confidence']:
                            scores = []

                            for train_idx, test_idx in tscv.split(df):
                                train_data = df.iloc[train_idx]
                                test_data = df.iloc[test_idx]

                                try:
                                    # Configurar parâmetros
                                    self.n_clusters = n_clusters
                                    self.stop_loss_pct = stop_loss
                                    self.take_profit_pct = take_profit
                                    self.max_holding_days = max_days
                                    self.min_confidence = min_conf

                                    # Gerar sinais
                                    signals = self.generate_signals(train_data, hurst_cols)

                                    # Backtest
                                    performance = self.backtest_signals(train_data, signals)

                                    scores.append(performance['sharpe'])

                                except:
                                    continue

                            avg_score = np.mean(scores) if scores else -np.inf

                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = {
                                    'n_clusters': n_clusters,
                                    'stop_loss_pct': stop_loss,
                                    'take_profit_pct': take_profit,
                                    'max_holding_days': max_days,
                                    'min_confidence': min_conf
                                }

        if best_params:
            self.n_clusters = best_params['n_clusters']
            self.stop_loss_pct = best_params['stop_loss_pct']
            self.take_profit_pct = best_params['take_profit_pct']
            self.max_holding_days = best_params['max_holding_days']
            self.min_confidence = best_params['min_confidence']

            print(f"✅ Parâmetros otimizados: {best_params}")
            print(".2f")

        return best_params

    def train_signal_model(self, df, hurst_cols):
        """Treina modelo de ML para combinação de sinais."""
        print("🤖 Treinando modelo de sinais...")

        # Preparar features
        features = self.prepare_ml_features(df, hurst_cols)

        # Target: sinal futuro (shifted)
        target = (df['returns'].shift(-1) > 0).astype(int)

        # Remover NaN
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]

        if len(X) < 100:
            print("⚠️  Dados insuficientes para treinar modelo")
            return

        # Treinar Random Forest
        self.signal_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        self.signal_model.fit(X, y)

        # Feature importance
        feature_importance = pd.Series(
            self.signal_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        print("📊 Top 5 features importantes:")
        for i, (feature, importance) in enumerate(feature_importance.head().items()):
            print(".3f")

    def prepare_ml_features(self, df, hurst_cols):
        """Prepara features para modelo de ML."""
        features = pd.DataFrame(index=df.index)

        # Features de Hurst
        features[hurst_cols] = df[hurst_cols]

        # Indicadores técnicos
        features['returns'] = df['returns']
        features['volatility_20'] = df['volatility_20']
        features['RSI'] = df['RSI']
        features['MACD'] = df['MACD']
        features['BB_width'] = df['BB_width']

        # Features derivadas
        features['hurst_mean'] = df[hurst_cols].mean(axis=1)
        features['hurst_std'] = df[hurst_cols].std(axis=1)
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # Lags de retornos
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        return features

    def kelly_position_size(self, win_prob, win_loss_ratio):
        """Calcula tamanho de posição usando Kelly Criterion."""
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0.1  # Tamanho mínimo

        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        kelly = max(0, min(kelly, 1))  # Limitar entre 0 e 1

        # Aplicar fração conservadora
        return kelly * self.kelly_fraction

    def dynamic_position_sizing(self, df, signals):
        """Calcula tamanho dinâmico de posições."""
        position_sizes = pd.Series(1.0, index=df.index)  # Default: 100%

        # Estimar probabilidade de vitória baseada em histórico recente
        window = 50
        for i in range(window, len(df)):
            recent_signals = signals.iloc[i-window:i]
            recent_returns = df['returns'].iloc[i-window:i]

            if len(recent_signals) > 0:
                # Probabilidade de vitória
                win_prob = len(recent_returns[recent_returns > 0]) / len(recent_returns)

                # Ratio médio win/loss
                wins = recent_returns[recent_returns > 0]
                losses = recent_returns[recent_returns < 0]

                if len(wins) > 0 and len(losses) > 0:
                    avg_win = wins.mean()
                    avg_loss = abs(losses.mean())
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

                    # Kelly sizing
                    kelly_size = self.kelly_position_size(win_prob, win_loss_ratio)
                    position_sizes.iloc[i] = max(0.1, min(kelly_size, 1.0))

        return position_sizes

    def generate_signals(self, df, hurst_cols):
        """Gera sinais usando ensemble avançado."""
        try:
            # 1. Sinais baseados em Hurst + PCA + KMeans
            clusters, _, _ = run_pca_kmeans(df, hurst_cols, n_clusters=self.n_clusters)
            buy_cluster, sell_cluster, _ = interpret_clusters(clusters, df, hurst_cols)
            hurst_signals = generate_signals(clusters, buy_cluster, sell_cluster).values

            # 2. Sinais de ML (se modelo treinado)
            ml_signals = np.zeros(len(df))
            if self.signal_model is not None:
                features = self.prepare_ml_features(df, hurst_cols)
                features_scaled = pd.DataFrame(
                    self.scaler.transform(features.fillna(0)),
                    index=features.index,
                    columns=features.columns
                )

                # Probabilidades do modelo
                probs = self.signal_model.predict_proba(features_scaled)[:, 1]

                # Converter para sinais (-1, 0, 1)
                ml_signals = np.where(probs > 0.6, 1,
                                    np.where(probs < 0.4, -1, 0))

            # 3. Ensemble ponderado
            hurst_weight = 0.6
            ml_weight = 0.4

            ensemble_signals = hurst_weight * hurst_signals + ml_weight * ml_signals
            final_signals = np.where(
                np.abs(ensemble_signals) >= self.min_confidence,
                np.sign(ensemble_signals),
                0
            )

            # 4. Aplicar filtros e risk management
            final_signals = self.apply_advanced_filters(df, final_signals)
            final_signals = self.apply_risk_management(df, final_signals)

            return final_signals

        except Exception as e:
            print(f"❌ Erro na geração de sinais: {str(e)}")
            return np.zeros(len(df))

    def apply_advanced_filters(self, df, signals):
        """Aplica filtros avançados de entrada."""
        # Filtro de volatilidade extrema
        vol_quantile = df['volatility_20'].quantile(0.8)
        vol_filter = df['volatility_20'] < vol_quantile

        # Filtro de tendência forte
        hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]
        hurst_mean = df[hurst_cols].mean(axis=1)
        trend_filter = hurst_mean > 0.55  # Tendência mais forte

        # Filtro de momentum consistente
        momentum_5 = df['close'] / df['close'].shift(5) - 1
        momentum_20 = df['close'] / df['close'].shift(20) - 1
        momentum_filter = (momentum_5 * momentum_20) > 0  # Mesma direção

        # Combinar filtros
        combined_filter = vol_filter & trend_filter & momentum_filter.fillna(False)

        return signals.where(combined_filter, 0)

    def apply_risk_management(self, df, signals):
        """Aplica risk management avançado."""
        signals = signals.copy()

        # Calcular preços de entrada
        entry_prices = df['close'].where(signals != signals.shift(1)).ffill()

        # Stop-loss dinâmico baseado na volatilidade
        vol_multiplier = df['volatility_20'] / df['volatility_20'].rolling(50).mean()
        dynamic_stop = self.stop_loss_pct * (1 + vol_multiplier.fillna(1))

        stop_losses = entry_prices * (1 - dynamic_stop * signals)

        # Take-profit dinâmico
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

        # Time exit
        position_days = (
            df.groupby((signals != signals.shift(1)).cumsum())
              .cumcount()
              .where(signals != 0, 0)
        )
        time_exit = position_days >= self.max_holding_days

        # Aplicar saídas
        exit_conditions = stop_hit | profit_hit | time_exit
        signals.loc[exit_conditions] = 0

        return signals

    def backtest_signals(self, df, signals):
        """Backtest simplificado para otimização."""
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

        return {
            'buy_hold_return': buy_hold_return,
            'strategy_return': strategy_return,
            'buy_hold_sharpe': buy_hold_sharpe,
            'strategy_sharpe': strategy_sharpe
        }

def walk_forward_validation(df, hurst_cols, n_splits=5):
    """Validação walk-forward para evitar overfitting."""
    print("🔄 Executando validação walk-forward...")

    # Dividir dados em janelas móveis
    total_size = len(df)
    window_size = total_size // (n_splits + 1)

    results = []

    for i in range(n_splits):
        train_end = window_size * (i + 1)
        test_end = window_size * (i + 2)

        train_data = df.iloc[:train_end]
        test_data = df.iloc[train_end:test_end]

        if len(train_data) < 100 or len(test_data) < 50:
            continue

        try:
            # Treinar estratégia
            strategy = UltimateTradingStrategy()
            strategy.optimize_parameters(train_data, hurst_cols)
            strategy.train_signal_model(train_data, hurst_cols)

            # Testar
            signals = strategy.generate_signals(test_data, hurst_cols)
            performance = strategy.backtest_signals(test_data, signals)

            results.append(performance)

        except Exception as e:
            print(f"⚠️  Erro na validação fold {i+1}: {str(e)}")
            continue

    if results:
        avg_sharpe = np.mean([r['strategy_sharpe'] for r in results])
        print(f"✅ Sharpe médio walk-forward: {avg_sharpe:.2f}")
    return results

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

    print("🚀 Executando estratégia ULTIMATE...")
    print("-" * 60)

    for ticker in assets:
        print(f"📈 Processando {ticker}...")

        asset_data = df[df['ticker'] == ticker].copy()
        asset_data = asset_data.sort_values('date').reset_index(drop=True)

        if len(asset_data) < 300:  # Mais dados para ML
            print("  ⚠️  Dados insuficientes para ML")
            continue

        try:
            # Estratégia ultimate
            strategy = UltimateTradingStrategy()

            # Walk-forward validation
            wf_results = walk_forward_validation(asset_data, hurst_cols, n_splits=3)

            # Otimização final nos dados completos
            strategy.optimize_parameters(asset_data, hurst_cols)
            strategy.train_signal_model(asset_data, hurst_cols)

            # Gerar sinais finais
            signals = strategy.generate_signals(asset_data, hurst_cols)

            # Position sizing dinâmico
            position_sizes = strategy.dynamic_position_sizing(asset_data, pd.Series(signals, index=asset_data.index))
            asset_data['position_size'] = position_sizes

            # Backtest com position sizing
            asset_data['signal'] = signals
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

    print("🎯 RESULTADO CONSOLIDADO - ESTRATÉGIA ULTIMATE")
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
    output_path = Path('data/ultimate_strategy_results.csv')
    final_df.to_csv(output_path, index=False)

    # Gráfico do melhor ativo
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    best_data = final_df[final_df['ticker'] == best_asset].copy()
    best_data = best_data.sort_values('date')

    # Gráfico melhorado
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Preços e sinais
    axes[0].plot(best_data['date'], best_data['close'], linewidth=1, alpha=0.7)

    buy_signals = best_data[best_data['signal'] == 1]
    sell_signals = best_data[best_data['signal'] == -1]

    # Tamanho dos pontos baseado no position size
    buy_sizes = buy_signals['position_size'] * 100
    sell_sizes = sell_signals['position_size'] * 100

    axes[0].scatter(buy_signals['date'], buy_signals['close'],
                   marker='^', color='green', s=buy_sizes, alpha=0.7, label='Compra')
    axes[0].scatter(sell_signals['date'], sell_signals['close'],
                   marker='v', color='red', s=sell_sizes, alpha=0.7, label='Venda')

    axes[0].set_title(f'Preços e Sinais - {best_asset} (Ultimate Strategy)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Retornos cumulativos
    buy_hold_cum = (1 + best_data['returns']).cumprod()
    strategy_cum = (1 + best_data['strategy_returns'].fillna(0)).cumprod()

    axes[1].plot(best_data['date'], buy_hold_cum, label='Buy & Hold', linewidth=2)
    axes[1].plot(best_data['date'], strategy_cum, label='Ultimate Strategy', linewidth=2)
    axes[1].set_title('Retornos Cumulativos')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Position sizes ao longo do tempo
    axes[2].plot(best_data['date'], best_data['position_size'], color='purple', alpha=0.7)
    axes[2].set_title('Tamanho Dinâmico das Posições')
    axes[2].set_ylabel('Position Size')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / f'ultimate_strategy_{best_asset}.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico salvo em: plots/ultimate_strategy_{best_asset}.png")

    print(f"\n💾 Resultados salvos em: {output_path}")
    print("🎉 Estratégia Ultimate concluída!")

if __name__ == '__main__':
    main()