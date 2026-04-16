#!/usr/bin/env python3
"""
Script para baixar e tratar dados de ações para análise com Hurst + PCA + KMeans.

Funcionalidades:
- Download de dados históricos via yfinance
- Tratamento de dados (limpeza, normalização, features)
- Detecção de outliers
- Cálculo de retornos e volatilidade
- Salvamento em formatos CSV/Parquet
"""

import argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime, timedelta

from fractal_strategy import download_price, hurst_series
from fractal_strategy.strategy import run_pca_kmeans, interpret_clusters, generate_signals

warnings.filterwarnings('ignore')

# Lista expandida de ativos para análise
ASSETS_LIST = [
    # Brasil - Petróleo/Energia
    "PETR4.SA", "PETR3.SA", "PRIO3.SA", "UGPA3.SA", "CSAN3.SA",
    # Brasil - Mineração
    "VALE3.SA", "GGBR4.SA", "CSNA3.SA", "USIM5.SA",
    # Brasil - Bancos
    "ITUB4.SA", "ITSA4.SA", "BBDC4.SA", "BBAS3.SA", "BPAC11.SA",
    # Brasil - Bebidas/Alimentos
    "ABEV3.SA", "JBSS3.SA", "MRFG3.SA", "BEEF3.SA",
    # Brasil - Tecnologia
    "B3SA3.SA", "VIVT3.SA", "TIMS3.SA", "MGLU3.SA",
    # Brasil - Saúde
    "RADL3.SA", "HYPE3.SA", "FLRY3.SA", "QUAL3.SA",
    # Brasil - Varejo
    "LREN3.SA", "PETZ3.SA", "VIIA3.SA",
    # EUA - Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    # EUA - Financeiro
    "JPM", "BAC", "WFC",
    # EUA - Consumo
    "WMT", "KO", "HD", "MCD",
    # Índices
    "^BVSP", "SPY", "QQQ"
]

def detect_outliers_zscore(data, threshold=3):
    """Detecta outliers usando Z-score."""
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

def detect_outliers_iqr(data):
    """Detecta outliers usando IQR (Interquartile Range)."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def calculate_technical_indicators(df, window=20):
    """Calcula indicadores técnicos básicos."""
    df = df.copy()

    # Médias móveis
    df['SMA_20'] = df['close'].rolling(window=window).mean()
    df['SMA_50'] = df['close'].rolling(window=int(window*2.5)).mean()
    df['EMA_20'] = df['close'].ewm(span=window).mean()

    # Volatilidade
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
    df['volatility_60'] = df['returns'].rolling(window=window*3).std() * np.sqrt(252)

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=window).mean()
    df['BB_std'] = df['close'].rolling(window=window).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    return df

def normalize_features(df, method='robust'):
    """Normaliza features usando diferentes métodos."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == 'standard':
        # Z-score normalization
        for col in numeric_cols:
            if col != 'close':  # Não normalizar preço absoluto
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()

    elif method == 'robust':
        # Robust normalization (menos sensível a outliers)
        for col in numeric_cols:
            if col != 'close':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                df[f'{col}_norm'] = (df[col] - median) / mad if mad != 0 else 0

    elif method == 'minmax':
        # Min-Max normalization
        for col in numeric_cols:
            if col != 'close':
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

    return df

def add_hurst_features(df, windows=[50, 100, 150, 200]):
    """Adiciona features baseadas no expoente de Hurst."""
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()

    # Calcular Hurst para diferentes janelas
    hurst_df = hurst_series(df['returns'].dropna(), windows=windows)

    # Combinar com dados originais
    df = df.join(hurst_df, how='left')

    # Preencher NaN com valor neutro (0.5)
    hurst_cols = [col for col in df.columns if col.startswith('Hurst_')]
    df[hurst_cols] = df[hurst_cols].fillna(0.5)

    return df

def create_target_variables(df, horizons=[1, 5, 10, 20]):
    """Cria variáveis alvo para predição."""
    df = df.copy()

    for horizon in horizons:
        # Retorno futuro
        df[f'return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1

        # Direção (1=subiu, 0=desceu)
        df[f'direction_{horizon}d'] = (df[f'return_{horizon}d'] > 0).astype(int)

        # Volatilidade futura
        df[f'volatility_{horizon}d'] = df['returns'].rolling(horizon).std().shift(-horizon) * np.sqrt(252)

    return df

def download_and_process_asset(ticker, start_date, end_date, save_raw=False):
    """Baixa e processa dados de um único ativo."""
    print(f"📥 Baixando dados de {ticker}...")

    try:
        # Download dos dados
        price_data = download_price(ticker, start=start_date, end=end_date)

        if len(price_data) < 100:
            print(f"⚠️  Dados insuficientes para {ticker} ({len(price_data)} pontos)")
            return None

        # Criar DataFrame base
        df = pd.DataFrame({'close': price_data})

        # Calcular indicadores técnicos
        df = calculate_technical_indicators(df)

        # Adicionar features de Hurst
        df = add_hurst_features(df)

        # Criar variáveis alvo
        df = create_target_variables(df)

        # Detectar outliers
        df['outlier_zscore'] = detect_outliers_zscore(df['returns'])
        df['outlier_iqr'] = detect_outliers_iqr(df['returns'])

        # Normalizar features
        df = normalize_features(df, method='robust')

        # Adicionar metadados
        df['ticker'] = ticker
        df['date'] = df.index
        df = df.reset_index(drop=True)

        print(f"✅ {ticker}: {len(df)} linhas processadas")

        if save_raw:
            # Salvar dados brutos também
            raw_df = pd.DataFrame({'close': price_data, 'ticker': ticker})
            raw_df.to_csv(f'data/raw_{ticker.replace(".", "_")}.csv', index=True)

        return df

    except Exception as e:
        print(f"❌ Erro ao processar {ticker}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download e tratamento de dados de ações')
    parser.add_argument('--assets', type=str, default=','.join(ASSETS_LIST[:10]),
                        help='Lista de ativos separados por vírgula')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Data inicial (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='Data final (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/processed_data.csv',
                        help='Arquivo de saída')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv',
                        help='Formato do arquivo de saída')
    parser.add_argument('--save-raw', action='store_true',
                        help='Salvar também dados brutos')
    parser.add_argument('--max-assets', type=int, default=10,
                        help='Número máximo de ativos para processar')

    args = parser.parse_args()

    # Criar diretório de saída
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Definir data final
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    # Lista de ativos
    assets = args.assets.split(',')[:args.max_assets]

    print(f"🚀 Iniciando processamento de {len(assets)} ativos")
    print(f"📅 Período: {args.start} até {end_date}")
    print(f"💾 Saída: {args.output}")
    print("-" * 60)

    # Processar cada ativo
    all_data = []
    for ticker in assets:
        ticker = ticker.strip()
        df = download_and_process_asset(ticker, args.start, end_date, args.save_raw)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("❌ Nenhum ativo foi processado com sucesso!")
        return

    # Combinar todos os dados
    final_df = pd.concat(all_data, ignore_index=True)

    # Estatísticas finais
    print("-" * 60)
    print("📊 ESTATÍSTICAS FINAIS:")
    print(f"Total de ativos processados: {len(assets)}")
    print(f"Total de linhas: {len(final_df):,}")
    print(f"Período: {final_df['date'].min()} até {final_df['date'].max()}")
    print(f"Features criadas: {len([col for col in final_df.columns if not col in ['ticker', 'date']])}")

    # Estatísticas por ativo
    stats_by_asset = final_df.groupby('ticker').agg({
        'date': ['min', 'max', 'count'],
        'returns': ['mean', 'std'],
        'volatility_20': 'mean'
    }).round(4)

    print("\n📈 ESTATÍSTICAS POR ATIVO:")
    print(stats_by_asset)

    # Salvar dados processados
    if args.format == 'csv':
        final_df.to_csv(args.output, index=False)
    else:
        final_df.to_parquet(args.output.replace('.csv', '.parquet'), index=False)

    print(f"\n💾 Dados salvos em: {args.output}")
    print("✅ Processamento concluído!")

if __name__ == '__main__':
    main()