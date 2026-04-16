# 📊 Processamento e Análise de Dados de Ações

Este projeto contém scripts para baixar, processar e analisar dados de ações usando algoritmos de análise técnica e expoente de Hurst.

## 🚀 Funcionalidades

### `data_processor.py`
Script principal para download e tratamento de dados:

- **Download**: Baixa dados históricos via yfinance
- **Indicadores Técnicos**: Calcula SMA, EMA, RSI, MACD, Bollinger Bands
- **Features de Hurst**: Expoente de Hurst em múltiplas janelas
- **Detecção de Outliers**: Z-score e IQR methods
- **Normalização**: Robust scaling das features
- **Variáveis Alvo**: Retornos e direções futuras

### `data_analysis.py`
Script de análise exploratória:

- **Estatísticas Descritivas**: Métricas por ativo
- **Visualizações**: Séries de preços, distribuições, correlações
- **Análise de Hurst**: Distribuição dos expoentes
- **Distribuição de Retornos**: Histogramas com ajuste normal

## 📁 Estrutura dos Dados

### Dados Processados (`data/processed_data.csv`)
Cada linha contém:
- `ticker`: Código do ativo
- `date`: Data da observação
- `close`: Preço de fechamento
- `returns`: Retorno diário
- Indicadores técnicos (SMA, EMA, RSI, MACD, etc.)
- Features de Hurst (Hurst_50, Hurst_100, etc.)
- Variáveis alvo (return_1d, direction_1d, etc.)
- Detecção de outliers

## 🛠️ Como Usar

### 1. Processamento de Dados

```bash
# Processar 5 ativos específicos
python data_processor.py --assets PETR4.SA,VALE3.SA,ITUB4.SA,AAPL,MSFT --max-assets 5

# Processar mais ativos com período personalizado
python data_processor.py --assets PETR4.SA,VALE3.SA,AAPL,MSFT,GOOGL --start 2019-01-01 --end 2024-12-31

# Salvar também dados brutos
python data_processor.py --assets PETR4.SA,AAPL --save-raw --output data/meus_dados.csv
```

**Parâmetros:**
- `--assets`: Lista de tickers separados por vírgula
- `--start`: Data inicial (YYYY-MM-DD)
- `--end`: Data final (YYYY-MM-DD)
- `--output`: Arquivo de saída
- `--format`: Formato (csv ou parquet)
- `--save-raw`: Salvar dados brutos também
- `--max-assets`: Número máximo de ativos

### 2. Análise Exploratória

```bash
# Executar análise completa
python data_analysis.py
```

Gera estatísticas e gráficos em `plots/`.

## 📊 Resultados da Análise

### Estatísticas dos Ativos Processados (2020-2026)

| Ativo    | Preço Médio | Volatilidade | Retorno Médio | Outliers |
|----------|-------------|--------------|---------------|----------|
| PETR4.SA | R$ 19.05   | 34.94%      | 0.14%        | 102     |
| VALE3.SA | R$ 53.22   | 31.27%      | 0.09%        | 91      |
| ITUB4.SA | R$ 22.35   | 27.17%      | 0.06%        | 80      |
| AAPL     | US$ 168.83 | 28.45%      | 0.10%        | 104     |
| MSFT     | US$ 319.61 | 26.83%      | 0.07%        | 103     |

### Features Criadas
- **67 features** por observação
- **Indicadores Técnicos**: 15+ métricas
- **Hurst Exponents**: 4 janelas (50, 100, 150, 200 dias)
- **Normalizações**: Features robustas para ML
- **Variáveis Alvo**: Retornos futuros (1, 5, 10, 20 dias)

## 🔬 Interpretação dos Resultados

### Expoente de Hurst
- **H > 0.5**: Comportamento persistente (tendência)
- **H = 0.5**: Random walk (mercado eficiente)
- **H < 0.5**: Comportamento anti-persistente (reversão)

**Resultados:**
- AAPL: H ≈ 0.70 (forte persistência)
- PETR4.SA: H ≈ 0.67 (persistência moderada)
- MSFT: H ≈ 0.64 (persistência moderada)

### Distribuição de Retornos
- Todos os ativos mostram **leptocurtose** (Kurtosis > 3)
- Distribuições com caudas pesadas (outliers frequentes)
- Assimetria variada (PETR4.SA mais negativa)

## 📈 Aplicações

1. **Machine Learning**: Features prontas para modelos preditivos
2. **Análise de Risco**: Volatilidade e detecção de outliers
3. **Trading Strategies**: Sinais baseados em Hurst + indicadores
4. **Research**: Dados limpos para estudos acadêmicos

## 📋 Dependências

```
pandas>=1.5.0
numpy>=1.21.0
yfinance>=0.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## 🎯 Próximos Passos

- [ ] Integração com pipeline de ML
- [ ] Backtesting com estratégias baseadas em Hurst
- [ ] Análise de correlação cruzada entre ativos
- [ ] Implementação de features adicionais (momentum, volume)
- [ ] Validação out-of-sample dos modelos

---

**Nota**: Os dados são baixados em tempo real via Yahoo Finance. Resultados podem variar com condições de mercado.