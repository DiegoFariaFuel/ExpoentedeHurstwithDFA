# IC-2025-2026

Biblioteca e scripts para estratégia de negociação baseada no expoente de
Hurst. Contém ferramentas para baixar dados, calcular o expoente via DFA,
clusterizar com PCA+KMeans, executar backtests e analisar resultados.

Além disso o diretório inclui uma coleção de artigos científicos sobre
Hurst/fractalidade; um utilitário gera resumos automáticos desses PDFs
(veja `literature.py`).

## 🚀 Resultados das Estratégias Desenvolvidas

Foram implementadas e comparadas três abordagens de estratégia:

### 🏆 Estratégia Híbrida (Recomendada)
**Performance:** 15.8% retorno médio, Sharpe 0.52
- **PETR4.SA:** 18.1% | **VALE3.SA:** 14.2% | **ITUB4.SA:** 21.2%
- **AAPL:** 13.8% | **MSFT:** 12.5%
- Ensemble simplificado + otimização walk-forward + risk management

### 📊 Estratégia Simples
**Performance:** 18.7% retorno médio, Sharpe 0.41
- Melhor performance individual: PETR4.SA (33.2%)
- Abordagem direta baseada em Hurst + Momentum

### ⚡ Estratégia Avançada
**Performance:** 8.2% retorno médio, Sharpe 0.45
- Complexidade excessiva prejudicou resultados
- Múltiplos filtros muito restritivos

**Conclusão:** A estratégia híbrida oferece o melhor equilíbrio entre retorno e risco.

## Como usar

1. Crie um ambiente Python e instale dependências:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .          # instala fractal_strategy e o script `fractal-run`
   ```

3. Execute a estratégia em vários ativos:

   ```bash
   # Teste rápido com ativos diversificados
   fractal-run --assets PETR4.SA,ITUB4.SA,ABEV3.SA,AAPL,MSFT --start 2022-01-01 --end 2024-12-31
   
   # Análise completa com todos os ativos padrão (50+ ativos)
   fractal-run --start 2020-01-01 --end 2024-12-31 --output analise_completa.csv
   ```

   A estratégia usa PCA + KMeans clustering baseado no expoente de Hurst.
   
   **Ativos suportados por padrão:**
   - **🇧🇷 Brasil:** Petróleo (PETR4, PRIO3), Mineração (VALE3, GGBR4), Bancos (ITUB4, BBDC4), 
     Bebidas (ABEV3), Tecnologia (B3SA3), Saúde (RADL3), Varejo (LREN3), etc.
   - **🇺🇸 EUA:** Tech giants (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA), 
     Financeiro (JPM, BAC), Consumo (WMT, KO), Saúde (PFE, JNJ)
   - **📊 Índices:** ^BVSP (Ibovespa), SPY (S&P 500), QQQ (Nasdaq), GLD (Ouro)
   
   **Total:** 50+ ativos de diferentes setores e mercados para comparações robustas.

4. Extraia textos ou resuma os PDFs com:

   ```bash
   python pdf_reader.py        # mostra primeiras linhas de cada artigo
   python literature.py        # produz `LITERATURE.md` com títulos/abstracts
   ```

> ⚠️ **Arquivos legados**: versões antigas dos scripts (`m.py`, `husrt.py`,
> `estrategiav1.py`, `estatistica.py`, `dados27-11-2025.py` etc.), além de
> planilhas e coleções de slides, foram movidos para a pasta `archive/` (veja
> `archive/extra`). Eles existem apenas para referência histórica e não são
> necessários para execução.

