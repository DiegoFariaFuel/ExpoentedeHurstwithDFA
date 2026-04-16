#!/usr/bin/env python3
"""
Script de análise exploratória dos dados processados.

Funcionalidades:
- Estatísticas descritivas
- Visualizações dos dados
- Correlação entre features
- Análise de Hurst por ativo
- Distribuição de retornos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

def load_processed_data(filepath):
    """Carrega os dados processados."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def plot_price_series(df, save_path=None):
    """Plota séries de preços por ativo."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    assets = df['ticker'].unique()

    for i, asset in enumerate(assets):
        if i >= 6:  # Máximo 6 gráficos
            break

        asset_data = df[df['ticker'] == asset].copy()
        asset_data = asset_data.sort_values('date')

        axes[i].plot(asset_data['date'], asset_data['close'], linewidth=1.5)
        axes[i].set_title(f'Preço - {asset}')
        axes[i].set_xlabel('Data')
        axes[i].set_ylabel('Preço (R$)' if '.SA' in asset else 'Preço (USD)')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.show()

def plot_hurst_distribution(df, save_path=None):
    """Plota distribuição dos expoentes de Hurst."""
    hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]

    if not hurst_cols:
        print("⚠️  Nenhuma coluna de Hurst encontrada")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(hurst_cols[:4]):  # Máximo 4 gráficos
        hurst_data = df[col].dropna()

        # Histograma
        axes[i].hist(hurst_data, bins=30, alpha=0.7, edgecolor='black')
        axes[i].axvline(hurst_data.mean(), color='red', linestyle='--',
                      label=f'Média: {hurst_data.mean():.3f}')
        axes[i].axvline(0.5, color='green', linestyle='--',
                      label='Random Walk (0.5)')

        axes[i].set_title(f'Distribuição - {col}')
        axes[i].set_xlabel('Expoente de Hurst')
        axes[i].set_ylabel('Frequência')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.show()

def plot_correlation_matrix(df, save_path=None):
    """Plota matriz de correlação das features."""
    # Selecionar features numéricas principais
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features_to_corr = []

    # Priorizar features importantes
    priority_features = ['returns', 'volatility_20', 'RSI', 'MACD', 'BB_width']
    hurst_features = [col for col in numeric_cols if col.startswith('Hurst_')]

    features_to_corr.extend(priority_features)
    features_to_corr.extend(hurst_features[:3])  # Máximo 3 features de Hurst

    # Filtrar apenas colunas que existem
    features_to_corr = [col for col in features_to_corr if col in numeric_cols]

    if len(features_to_corr) < 2:
        print("⚠️  Poucas features numéricas para correlação")
        return

    corr_matrix = df[features_to_corr].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, fmt='.2f', square=True, linewidths=0.5)
    plt.title('Matriz de Correlação - Features Principais')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.show()

def plot_returns_distribution(df, save_path=None):
    """Plota distribuição dos retornos."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    assets = df['ticker'].unique()

    for i, asset in enumerate(assets):
        if i >= 6:  # Máximo 6 gráficos
            break

        asset_data = df[df['ticker'] == asset].copy()
        returns = asset_data['returns'].dropna()

        # Histograma com distribuição normal
        axes[i].hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')

        # Linha de distribuição normal
        mu, std = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/std)**2)
        axes[i].plot(x, y, 'r-', linewidth=2, label='Normal')

        axes[i].axvline(mu, color='red', linestyle='--',
                      label=f'Média: {mu:.4f}')
        axes[i].axvline(mu + 2*std, color='orange', linestyle='--',
                      label=f'+2σ: {(mu + 2*std):.4f}')
        axes[i].axvline(mu - 2*std, color='orange', linestyle='--',
                      label=f'-2σ: {(mu - 2*std):.4f}')

        axes[i].set_title(f'Distribuição de Retornos - {asset}')
        axes[i].set_xlabel('Retorno Diário')
        axes[i].set_ylabel('Densidade')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")

    plt.show()

def print_summary_statistics(df):
    """Imprime estatísticas descritivas."""
    print("📊 ESTATÍSTICAS DESCRITIVAS DOS DADOS PROCESSADOS")
    print("=" * 60)

    print(f"Total de registros: {len(df):,}")
    print(f"Número de ativos: {df['ticker'].nunique()}")
    print(f"Período: {df['date'].min()} até {df['date'].max()}")
    print(f"Features criadas: {len(df.columns) - 2}")  # Excluindo ticker e date
    print()

    # Estatísticas por ativo
    print("📈 ESTATÍSTICAS POR ATIVO:")
    print("-" * 60)

    stats = df.groupby('ticker').agg({
        'close': ['count', 'mean', 'std', 'min', 'max'],
        'returns': ['mean', 'std', lambda x: x.skew(), lambda x: x.kurtosis()],
        'volatility_20': 'mean'
    }).round(4)

    stats.columns = ['Count', 'Preço_Médio', 'Preço_Std', 'Preço_Min', 'Preço_Max',
                    'Retorno_Médio', 'Retorno_Std', 'Skewness', 'Kurtosis', 'Volatilidade']

    print(stats)
    print()

    # Estatísticas de Hurst
    hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]
    if hurst_cols:
        print("🌊 ESTATÍSTICAS DE HURST:")
        print("-" * 60)
        hurst_stats = df.groupby('ticker')[hurst_cols].agg(['mean', 'std']).round(4)
        print(hurst_stats)
        print()

    # Detecção de outliers
    print("🔍 DETECÇÃO DE OUTLIERS:")
    print("-" * 60)
    outlier_stats = df.groupby('ticker').agg({
        'outlier_zscore': 'sum',
        'outlier_iqr': 'sum'
    })
    outlier_stats['total'] = outlier_stats['outlier_zscore'] + outlier_stats['outlier_iqr']
    outlier_stats = outlier_stats.astype(int)
    print(outlier_stats)
    print()

def main():
    # Caminho dos dados
    data_path = Path('data/processed_data.csv')

    if not data_path.exists():
        print(f"❌ Arquivo não encontrado: {data_path}")
        return

    # Carregar dados
    print("📂 Carregando dados processados...")
    df = load_processed_data(data_path)
    print(f"✅ Dados carregados: {len(df):,} registros\n")

    # Estatísticas descritivas
    print_summary_statistics(df)

    # Criar diretório para gráficos
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Gerar visualizações
    print("📊 Gerando visualizações...")

    try:
        plot_price_series(df, save_path=plots_dir / 'price_series.png')
        plot_hurst_distribution(df, save_path=plots_dir / 'hurst_distribution.png')
        plot_correlation_matrix(df, save_path=plots_dir / 'correlation_matrix.png')
        plot_returns_distribution(df, save_path=plots_dir / 'returns_distribution.png')

        print(f"\n📁 Gráficos salvos em: {plots_dir}/")

    except Exception as e:
        print(f"⚠️  Erro ao gerar gráficos: {str(e)}")

    print("\n✅ Análise exploratória concluída!")

if __name__ == '__main__':
    main()