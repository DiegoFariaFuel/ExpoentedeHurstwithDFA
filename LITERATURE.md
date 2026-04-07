# Literature summary

## applied
*File: applsci-08-02473.pdf*

**Abstract**

: This paper presents new methods and models for forecasting stock prices and computing
hybrid models, combining analytical and neural approaches. First, technical and fractal analyses are
conducted and selected stock market indices calculated, such as moving averages and oscillators.
Next, on the basis of these indices, an artificial neural network (ANN) provides predictions one day
ahead of the closing prices of the assets. New technical analysis indicators using fractal modeling are
also proposed. Three kinds of hybrid model with different degrees of fractal analysis were considered.
The new hybrid modeling approach was compared to previous ANN-based prediction methods. The
results showed that the hybrid model with fractal analysis outperforms other models and is more
robust over longer periods of time.
Keywords: stock exchange; technical analysis; fractal analysis; fractal moving average; fractal
dimension; artificial neural networks; hybrid models

1. Introduction
Information about the current conditions on the Stock Exchange and future closing prices is
crucial for investors, who wish to maximize their profits and make good investment decisions. Over
the years, different theories and stock models have been developed, some of which contradicted each
other. Until the second half of twentieth century, it was believed that price changes were random
and thus could not be predicted. This was a basic assumption of the random walk hypothesis [1],
which is consistent with the efficient market hypothesis (EMH) formulated by Eugene Fama [2]. This
hypothesis states that share prices fully reflect all the available information, and stocks always trade
at a fair value. It is therefore impossible to outperform the market overall, and higher returns can be
only obtained by chance or by purchasing riskier investments. This hypothesis was not in line with
experience, as evidenced by the profits amassed by professional financial institutions trading on stock
markets. It was also refuted by a large group of academics, who observed and analyzed market prices
over long periods of time and proved that market prices were not random, but exhibited trends [3,4].
Moreover, these trends repeated over time and some followed a sinusoidal shape [5]. These results
supported the view that the market is, to some degree, predictable, and that future prices and trends
can be predicted on the basis of past prices. This opened the way for technical analysis. Thus, the
problem of stock price prediction is closely related to time series analysis and modeling trends.
These days, not only spectral analysis of trade signals [5], but also other more complex methods,
such as joint time–frequency–shape analysis, are available [6].

Appl. Sci. 2018, 8, 2473; doi:10.3390/app8122473

www.mdpi.com/journal/applsci

Appl. Sci. 2018, 8, 2473

2 of 17

The financial market provides a variety of software products supporting investors. Traditional
prediction algorithms, such as autoregressive integrated moving average (ARIMA) and generalized
autoregressive conditional heteroscedasticity (GARCH), are based on financial time series modeling
using the stochastic process theory [7]. A linear model ARIMA [8] has dominated financial time series
forecasting methods for more than half a century. Popular financial instruments offered by stock
exchanges may help investors to make a profit. However, there is also the risk of losing assets [9].
With the increasing complexity of trading systems, soft computing methods, i.e., artificial neural
networks (ANN) [7,10,11] or fuzzy logic [12], are gaining greater popularity. The stock quotes time
series can be considered as representations of nonlinear dynamic systems [11]. Artificial neural
networks, which are nonlinear universal approximators [13], are appropriate tools for such complex
modeling tasks. Models based on ANN were introduced into financial forecasting two decades
ago [14]. In these mainly autoregressive (AR) models, a time series is modeled by an ANN and the
next time series value is predicted on the basis of several past observations. Hamzacebi, Akay, and
Kutay [15] applied ARIMA and ANN for periodic time series modeling and concluded that ANN-based
forecasting is more accurate. The prediction of trading signals has been mostly implemented using
feed-forward neural networks (FNN) of the multilayer perceptron (MLP) type [7,14,16]. However,
new ANN structures and ANN-based methods have also been developed, such as: A probabilistic
neural network [11], a dynamic artificial neural network (DAN2) [17], a functional link artificial neural
network [18], a state space wavelet network [19], or a neural-wavelet analysis [20]. Uses of ANNs
include predicting future exchange rates [11], stock market indices [10,19,21], and share prices [22].
A hybrid approach was proposed by Zhang [23], combining ARIMA and ANN for time series
modeling. This hybrid methodology takes advantage of statistical and neural methods and outperforms
both. Today, hybrid modeling is becoming more and more popular for share price forecasting.
Although stock price fluctuations cannot be determined precisely, combining different models reduces
the risk of failure and provides more accurate results than using them individually. Different kinds
of hybrid models have been investigated. A new combination of ARIMA and ANN was proposed
by Khashei and Bijari [24], which proved to be more accurate than ARIMA, ANN, or the Zhang
model. Hybrid models combining GARCH and exponential generalized autoregressive conditional
heteroscedasticity (EGARCH) with different ANN structures have been investigated by Güresen and
Kayakutlu [25,26].
Despite the fact that ANN-based models outperform linear statistical models and some hybrid
statistical-ANN models, it would be unwise to rely only on them without incorporating risk
analysis [9,14]. Technical analysis (TA) provides a number of tools that can support investment
decisions. Witkowska and Marcinkiewicz [27] proposed a hybrid methodology combining these
analytical models and ANN for future value forecasting of the WIG20 index on the Warsaw Stock
Exchange. Another trend is to combine TA and fundamental analyses with ANN for stock price
forecasting [28,29]. Hybrid models combining TA and ANN have also been used in previous works by
the authors for close values prediction [22,30]. The hybrid analytical–neural approach for stock price
prediction was inspired by the authors’ previous studies in the field of engineering, in which hybrid
modeling yielded better results than “pure” ANN solutions [31,32]. In our early work, hybrid models
combining TA with ANN were compared only with the ANN-based approach [22]. We then reported
our initial research into hybrid models containing fractal analysis (FA) [30]. Hybrid ANN-based
models with technical and fractal analyses were applied in the core module of a decision-making
information system supporting investor decisions on the Warsaw Stock Exchange [33–35]. The system
was designed to analyze stock prices trends and choose the assets that will achieve the highest expected
profit in the next day. The system was tested offline on historical data from 70 companies over a
period of 6 months. The investment decisions were correct 60% of the time, and a profit of 17.2% was
made, while the major Polish growth indices for the same time period were −0.99% for WIG20 and
5.69% for mWIG40 [35]. These results proved the effectiveness of the hybrid models and ANN-based
decision algorithms.

## axioms
*File: axioms-12-00127-v2.pdf*

**Abstract**

: Demand for power sources is gradually shifting from ozone-depleting-substances towards
renewable and sustainable energy resources. The growth prospects of the renewable energy industry
coupled with improved cost efficiency means that renewable energy companies offer potential returns
for traders in stock markets. Nonetheless, there have been no studies investigating technical trading
rules in renewable energy stocks by amalgamating fractal geometry with technical indicators that
focus on different market phases. In this paper, we explore the profitability of technical analysis using
a portfolio of 20 component stocks from the NASDAQ OMX Renewable Energy Generation Index
using fractal dimension together with trend-reinforcing and mean-reverting (contrarian) indicators.
Using daily prices for the period 1 July 2012 to 30 June 2022, we apply several tests to measure trading
performance and risk-return dynamics of each form of technical trading system—both in isolation
and simultaneously. Overall, trend (contrarian) trading system outperforms (underperforms) the
naïve buy-and-hold policy on a risk-adjusted basis, while the outcome is further enhanced (reduced)
by the fractal-reinforced strategy. Simultaneous use of both trend-reinforcing and mean-reverting
indicators strengthened by fractal geometry generates the best risk-return trade-off, significantly
outperforming the benchmark. Our findings suggest that renewable energy stock prices do not fully
capture historical price patterns, allowing traders to earn significant profits from the weak form
market inefficiency.

Mean-Reverting and Hybrid Fractal
Trading Systems. Axioms 2023, 12,
127. https://doi.org/10.3390/

Keywords: fractal geometry; technical trading systems; trend-reinforcing; contrarian; market
efficiency; renewable energy stocks

axioms12020127
Academic Editor: Boualem Djehiche

MSC: 91B28

Received: 13 October 2022
Revised: 20 January 2023
Accepted: 23 January 2023
Published: 28 January 2023

Copyright: © 2023 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license (https://
creativecommons.org/licenses/by/
4.0/).

1. Introduction
In recent years, global warming due to increased concentrations of greenhouse gases
has triggered unusual weather conditions and rising sea levels in different parts of the
world. One of the biggest emitters of greenhouse gases in the world is the United States (US),
which contributed about 12% of the greenhouse gas, second only after China. The main
culprit of the greenhouse gas is carbon dioxide which according to the US Environmental
Protection Agency, accounted for about 80% of all gases contributed by the country. The
gas is mostly emitted from fossil fuel and industrial processes and traps heat on earth. The
unabsorbed gas will remain in the air for thousands of years.
The alarming amount of gas pollution has motivated governments and non-profit
organizations to intervene in the operations of corporations. For example, various measures were introduced to promote renewable energy from solar, wind, hydropower, and

Axioms 2023, 12, 127. https://doi.org/10.3390/axioms12020127

https://www.mdpi.com/journal/axioms

Axioms 2023, 12, 127

2 of 18

other alternative sources with lucrative financial incentives to corporations and users. The
financial factors together with the increased importance and demand for sustainable business practices have seen a growing trend of companies moving towards renewable energy
ventures [1,2]. Likewise, in the capital markets of advanced nations such as the US, China
and the United Kingdom, individual investors and mutual funds have started undertaking
socially responsible investment practices. A recent report published by the Centre for Climate Finance and Investment [3] indicates that the performance of clean energy portfolios
in financial markets for 2016–2020 has significantly surpassed fossil fuel companies in
terms of returns and volatility. It also reported that the strength of the renewable power
portfolios continues even with the advent and the recovery period of the COVID-19 crisis.
The positive performance of renewable power portfolios is also documented in several
academic research studies, such as those of Chang et al. [4] and Chang et al. [5]. These
studies employed technical analysis and found superior returns for renewable energy
stocks against fossil energy markets in the US and Europe.
As one of the investment strategies dominating the stock market, technical analysis has
a long history where ancient societies began using historical prices to forecast future profits.
One of the earliest forms of this strategy was used by the ancient Babylonians in the 7th
century BC for the purpose of predicting the prices of commodities. The 20th century has
seen a proliferation of technical analysis techniques due to technological advancements. In
academia, studies such as those of [6–10] have explored different markets and/or financial
instruments. By comparison, studies devoted to exploring technical rules in the energy
markets, particularly renewable energy, remain scarce.
A limited number of studies have investigated energy markets using different tools
for predictions, such as those of [11–14]. Chen et al. [11] found that combining information
content of oil price ranges and graphical information of time series provides more accurate
predictions of oil price movements than traditional models. Based on 168 energy stocks
included in the NYSE Energy Index, Thomakos and Papailias [14] observed a strong
presence of the momentum effect in the energy sector of which risk-reward characteristics
can be exploited through various momentum trading strategies. Lin et al. [13] tested the
predictive ability of technical trading rules proposed by Sullivan et al. [15] based on daily
returns for thirteen energy market indices. Results show that the profitability of trading
rules exists even after considering non-synchronous trading bias but disappears when
transaction costs are considered, which implies that developed energy markets are efficient.
More recently, Gurrib et al. [12] investigated the performance of charting using Ichimoku
Cloud on the top ten stocks listed in the S&P Composite 1500 Energy Index. Using data
from 2012 to 2019, they found that technical charts can offer speculators positive returns
and outperform the buy-and-hold (B&H) strategy.
The profitability of technical analysis has significant implications for investment practice and efficient market hypothesis (EMH) at the weak form level. Although recent studies
have investigated some forms of trading strategies using fractals to detect persistence (antipersistence) by going long (short) in the financial markets, they are explored in isolation
from technical trading rules. For example, Batten et al. [16] built trading rules by utilizing
the Hurst coefficient for the gold-silver spread and showed that these rules can beat the
B&H strategies and moving average rules across diverse holding periods. Auer [17] later
used return-based ratio, but technical analysis was still ignored. Using a different approach,
Paluch and Jackowska-Strumiłło [18] integrated fractal moving averages into technical
indicators, which were then used as inputs to neural networks to forecast next-day closing
prices. However, the study made no attempt to distinguish between market phases for
utilizing appropriate indicators. In exploring different market states, Mahata and Nurujjaman [19] observed randomness in the short-term, while the long-term investment
horizon displayed a correlation with firm fundamentals in some Indian and US companies.
Nonetheless, the authors did not explore the combination trading rule.
Since some technical indicators perform better in certain conditions but show poor
performance in other conditions, investors can benefit during trending or mean-reverting

## entropy
*File: entropy-23-01018-v2.pdf*

**Abstract**

: The financial market is a complex system, which has become more complicated due to the
sudden impact of the COVID-19 pandemic in 2020. As a result there may be much higher degree of
uncertainty and volatility clustering in stock markets. How does this “black swan” event affect the
fractal behaviors of the stock market? How to improve the forecasting accuracy after that? Here we
study the multifractal behaviors of 5-min time series of CSI300 and S&P500, which represents the two
stock markets of China and United States. Using the Overlapped Sliding Window-based Multifractal
Detrended Fluctuation Analysis (OSW-MF-DFA) method, we found that the two markets always
have multifractal characteristics, and the degree of fractal intensified during the first panic period
of pandemic. Based on the long and short-term memory which are described by fractal test results,
we use the Gated Recurrent Unit (GRU) neural network model to forecast these indices. We found
that during the large volatility clustering period, the prediction accuracy of the time series can be
significantly improved by adding the time-varying Hurst index to the GRU neural network.



Keywords: multifractal; forecasting; OSW-MF-DFA; GRU neural network; stock index time series

Citation: Zhang, S.; Fang, W.
Multifractal Behaviors of Stock
Indices and Their Ability to Improve

1. Introduction

Forecasting in a Volatility Clustering

In 1970, the economist Eugene F. Fama put forward the efficient market hypothesis
(EMH), which became the cornerstone of contemporary financial theory. In this hypothesis,
all information on the market will be quickly reflected in the stock price, so the stock prices
are unpredictable [1]. However, later behavioral finance theory studies of market behavior
have shown the limitations of the EMH. The criticisms focus on the irrationality of investors,
market friction and incomplete arbitrage, which are in violation of the effective market
hypothesis [2–4]. In empirical terms, momentum effects, reversal effects, January effects,
and financial anomalies such as peaks and thick tails of time series, volatility clustering, etc.
were also found in financial time series [5–9]. Hence, economists have actively sought new
theories to explain these market anomalies. In 1994, Peters proposed the fractal market
hypothesis (FMH). This hypothesis modifies the strict assumptions of the efficient market,
pointing out that asset prices obey fractional Brownian motion, the return rate sequence
has long memory, and the market may be in a non-equilibrium state [10]. Therefore, a
certain level of predictability of prices has become a general consensus. After the FMH was
proposed, the field mainly focused on two aspects, the study of the fractal characteristics of
the stock market and building various models to try to predict market trends.
For the first aspect, regarding the method of studying fractal properties, the starting
point is the rescaled range method (R/S) proposed by the British hydrologist Hurst [11].
When studying the relationship between the Nile Reservoir discharge and the water level,
he found that a biased random walk (fractional Brownian motion) can well describe the
long-term dependence of the two, so he proposed calculating the Hurst exponent by the
rescaled range method, which was used for characterizing the self-similarity of time series.
Many scholars have continuously optimized and improved the method. Peng et al., proposed the detrending fluctuation analysis method (DFA) when studying the long-range

Period. Entropy 2021, 23, 1018.
https://doi.org/10.3390/e23081018
Academic Editor: Rafał Rak
Received: 14 July 2021
Accepted: 3 August 2021
Published: 6 August 2021

Publisher’s Note: MDPI stays neutral
with regard to jurisdictional claims in
published maps and institutional affiliations.

Copyright: © 2021 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license (https://
creativecommons.org/licenses/by/
4.0/).

Entropy 2021, 23, 1018. https://doi.org/10.3390/e23081018

https://www.mdpi.com/journal/entropy

Entropy 2021, 23, 1018

2 of 19

power-law correlation characteristics of DNA sequences, which became the mainstream for
measuring the long-range correlation of stationary time series [12]. However, Kantelhardt
et al., pointed out that in most cases, the scaling behavior of time series is very complicated and cannot be explained by a simple scaling index [13]. Therefore, the multifractal
detrending volatility method (MF-DFA) was proposed and the author pointed out that a
multifractal structure may come from the thick-tailed distribution and long-range correlation. Thompson et al., used multi-methods for analyzing the fractal characteristics of
GE stock price series. The results showed that the MF-DFA model is better fitted [14]. A
number of existing studies have shown that multifractals are common in financial markets
in various countries, including stock markets [15–19], bonds [20] and Bitcoin markets [21].
The results above definitely all rejected the efficient market hypothesis. However, some
scholars questioned the MF-DFA method. After comparing DFA, CMA, MF-DFA and
other detrend volatility analysis methods, Bashan pointed out that the MF-DFA may result
in false fluctuations, which may be reflected in the larger calculated generalized Hurst
index [22]. This happens because the intervals divided by the MF-DFA method do not
overlap, so the fitting polynomials of adjacent intervals may be discontinuous. Recognizing this shortcoming, many scholars use overlapping smoothing windows to optimize
the model respectively, which reduces the spurious fluctuations caused by partially overlapping adjacent intervals [23,24]. We adopt this optimization method, which is called
OSW-MF-DFA. Some scholars have studied the multifractal changes of the financial market
under the impact of the pandemic and confirmed the reduction of market efficiency caused
by COVID-19 [25–27]. Okorie studied the contagion effect of the fractal of the stock market,
which proved the existence of multifractals from another aspect [28].
For the second aspect of predicting market trends, although stock markets are affected
by various factors such as macroeconomic development, institutions, supervision, noise
trading etc., researchers still try to construct various prediction models: from parametric
models such as ARMA, ARIMA, and GARCH to machine learning such as BP, recurrent
neural network (RNN), LSTM and GRU with gated structure, the prediction accuracy
of the model has been continuously improved. The short-term memory neural network
(LSTM) was proposed by Hochreiter and Schmidhuber in 1997 [29]. The gated recurrent
networks LSTM and GRU, which have been popular in recent years, have been widely
used to predict the trend of stock prices, and they have actually proved to have achieved
good results by catching the long and short term memory of financial time series [30–32].
Yu et al., used the GARCH model and LSTM neural network to predict the volatility of
China’s three major stock indexes, and the results proved that LSTM with long memory
has better predictive ability [33]. However, with the popularization of LSTM, more and
more studies have found that LSTM models have flaws such as limited explanatory power
and slow convergence speed. Aiming at the shortcomings of LSTM, Cho et al., further
optimized on the basis of LSTM and proposed a GRU neural network [34]. Compared
with LSTM, GRU has only two gate control structures: update gate and reset gate, which
reduces parameters while maintaining predictive performance, and it helps to speed up
convergence [35,36].
In this article we aim to study the fractal properties of the Chinese and American
intraday stock markets under the impact of the COVID-19 pandemic and use them to
forecast by applying to the GRU model. According to the impact of the pandemic on the
financial markets, we divide the time interval into three periods: before, during and after
the first panic period of pandemic. In terms of multifractal research, this article utilizes
the OSW-MF-DFA method optimized by overlapping smoothing windows. We obtain
the generalized Hurst index and multifractal spectrum of the two stock indexes, then
analyze and compare the fractal characteristics of the two markets at different periods.
The time-varying Hurst exponent and its decomposition sequence are calculated by the
DFA method, which are used as the input variables of the subsequent predictions. A
time-varying Hurst sequence is also added to regular input variables such as opening price,

## fractal and fractional
*File: fractalfract-06-00394-v2.pdf*

**Abstract**

: The prediction of the stock price index is a challenge even with advanced deep-learning
technology. As a result, the analysis of volatility, which has been widely studied in traditional finance,
has attracted attention among researchers. This paper presents a new forecasting model that combines
asymmetric fractality and deep-learning algorithms to predict a one-day-ahead absolute return series,
the proxy index of stock price volatility. Asymmetric Hurst exponents are measured to capture the
asymmetric long-range dependence behavior of the S&P500 index, and recurrent neural network
groups are applied. The results show that the asymmetric Hurst exponents have predictive power for
one-day-ahead absolute return and are more effective in volatile market conditions. In addition, we
propose a new two-stage forecasting model that predicts volatility according to the magnitude of
volatility. This new model shows the best forecasting performance regardless of volatility.
Keywords: forecasting; volatility; asymmetry; multifractal; Hurst exponent; deep learning



Citation: Cho, P.; Lee, M. Forecasting
the Volatility of the Stock Index with
Deep Learning Using Asymmetric
Hurst Exponents. Fractal Fract. 2022,
6, 394. https://doi.org/10.3390/
fractalfract6070394
Academic Editor: Carlo Cattani
Received: 14 May 2022
Accepted: 15 July 2022
Published: 16 July 2022
Publisher’s Note: MDPI stays neutral
with regard to jurisdictional claims in
published maps and institutional affiliations.

Copyright: © 2022 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license (https://
creativecommons.org/licenses/by/
4.0/).

1. Introduction
Stock price and volatility forecast problems have been studied for a long time in the
financial field [1–3], and many studies have recently applied deep-learning techniques to
predict the problem [4–6]. Recurrent neural network (RNN) is often used for time-series
forecasting in the field of deep learning, which can analyze sequential data using past
information [7]. RNN has been widely used to predict stock prices or volatility in the
financial field [8–10], and extensively employed in other fields to deal with time-series
data [11–13]. However, no matter how predictable the deep-learning technique is, it is
difficult to accurately predict stock price and volatility. Therefore, many studies have
been conducted to forecast stock volatility [14,15], which is the variance of stock price that
can be predicted relatively more accurately than stock price [16–19], while some studies
demonstrate the predictive performance of deep-learning algorithms along with domain
knowledge in the financial field [20,21].
The generalized Hurst exponent and multifractality are econophysics concepts that
inform the characteristics of the time-series, and measure the complexity of the corresponding time-series. They can be measured through multifractal detrended fluctuation analysis
(MFDFA) methodology [22], which is one of the most used approaches for estimating multifractality, while the long-range dependence of the time-series can be evaluated with the
generalized Hurst exponent. If the long-range dependence of the time-series is different
according to the time period, the time-series is known to include multifractality. Conversely,
if long-range dependence is the same according to the time period, the time-series is said
to be monofractal. If time-series has a multifractality feature, the complexity of the timeseries increases, while various stylized factors appear accordingly. The major features of a
multifractal are acknowledged to be fat-tail probability distribution, long-term correlation
for small and large fluctuations, and volatility clustering. Therefore, if a time-series has a
multifractal characteristic, it is likely to have these features. It is generally considered that it

Fractal Fract. 2022, 6, 394. https://doi.org/10.3390/fractalfract6070394

https://www.mdpi.com/journal/fractalfract

Fractal Fract. 2022, 6, 394

2 of 16

has a long-range dependence if the Hurst exponent of time-series is greater than 0.5, so using
this pattern will help predict stock price or stock volatility. Therefore, there have been many
efforts to interpret stock price movement through multifractality [23–28]. The application of
the pattern is used not only to predict stock price movement, but also to forecast VIX [29],
foreign exchange rates [30], and oil price [31]. The authors in Garcin [30] showed that the
forecast of foreign exchange rates with time-varying Hurst exponents estimation is effective
when the Hurst exponent is over 0.5.
In particular, there are two distinct trends in the stock market—the bull market and
the bear market. The asymmetric multifractal detrended fluctuation analysis (A-MFDFA)
model [32] distinctly measures the asymmetric multifractal scaling behavior of the generalized Hurst exponent according to these two market trends. Therefore, it appropriately
measures the detailed directional generalized Hurst exponent and directional multifractal
scale according to stock market conditions. In other words, the A-MFDFA method is
the asymmetric generalized Hurst exponent measurement methodology considering the
asymmetric nature of stock price movement. In particular, the asymmetric efficiency of the
stock market can be estimated with the A-MFDFA according to stock market direction [33].
If the market is not efficient, it implies that the stock market is not a complete random walk
and is therefore predictable. As a result, the analysis of the asymmetric Hurst exponents
to deal with the predictability has attracted much attention, so that research areas include
asymmetric phenomena according to price trend [34–36]. However, little attention has
been paid to applying asymmetric multifractal elements to deep learning to forecast the
financial market. Therefore, in this study, we predict the stock market with a deep-learning
model using the asymmetric Hurst exponent, one of the characteristics of the stock market.
Then, we identify whether the asymmetric Hurst exponent is a feature that helps predict
the stock market.
Since the price series is non-stationary time-series data, the multifractal concept cannot
be applied immediately [37]. This research, therefore, focuses on the prediction of the
return series, which is stationary data. Since the multifractal concept or the Hurst exponent
is related to the volatility of the return series, it is expected that the Hurst exponent will
help predict the volatility of return series. Many volatilities have already been defined
in financial markets. For example, historical volatility considers past price movements
of the underlying asset, and is also referred to as realized volatility. Implied volatility
is a measure of market expectations regarding the asset’s future volatility. Parkinson’s
volatility, which is also called High Low-Range Volatility, aims to estimate volatility using
the high and low prices of the day. Garman–Klass volatility calculates daily volatility using
more factors (high, low, open, close prices). The VIX Index is based on real-time prices of
options on the S&P 500 Index and is designed to reflect investors’ consensus views of future
(30-day) expected stock market volatility. Each value expresses the volatility of the financial
market well; however, in this paper, the absolute value of returns is presented as a proxy of
volatility. The reason for this is that it is the easiest way to obtain daily volatility when we
only have price series as data. Historical volatility does not focus on only one-day volatility,
whereas daily realized volatility needs high-frequency data to compute one-day volatility.
Moreover, it is anticipated that asymmetric Hurst exponents are better than the overall
Hurst exponent in forecasting the stock market since they have more detailed information
of the market situation. Although the volatility of the US stock market is also affected by the
volatility of other markets [38], this study focuses on the predictive power of multifractal
data of its own US market. In the US stock market, past volatility provides information
for future prediction [39]. Therefore, the main subject of this paper is the forecasting of the
absolute return series of S&P500 index by applying the asymmetric Hurst exponent to deep
learning. First, the asymmetric Hurst exponents are calculated through the close price of
the S&P500 index using the A-MFDFA method. Second, along with various RNN models,
we predict the absolute return of the S&P500 index for the next day through its past returns
and the asymmetric Hurst exponents. Third, we divide the period and investigate the
change in forecasting power according to market conditions. Lastly, the new two-stage

## fractal and fractional
*File: fractalfract-06-00499-v2.pdf*

**Abstract**

: It is necessary to quantitatively describe or illustrate the characteristics of abnormal stock
price fluctuations in order to prevent and control financial risks. This paper studies the fractal structure of China’s stock market by calculating the fractal dimension and scaling behavior on the timeline
of its eight big slumps, the results show that the slumps have multifractal characteristics, which are
correlated with the policy intervention, institutional arrangements, and investors’ rationality. The
empirical findings are a perfect match with the anomalous features of the stock prices. The fractal
dimensions of the eight stock collapses are between 0.84 and 0.98. The fractal dimension distribution
of the slumps is sensitive to market conditions and the active degree of speculative trading. The
more mature market conditions and the more risk-averse investors correspond to the higher fractal
dimension and the fall which is less deep. Therefore, the fractal characteristics could reflect the
evolution characteristics of the stock market and investment philosophy. The parameter set calculated
in this paper could be used as an effective tool to foresee the slumps on the horizon.
Keywords: China’s stock market; stock market slump; multifractality

Citation: Li, Y. Multifractal
Characteristics of China’s Stock
Market and Slump’s Fractal
Prediction. Fractal Fract. 2022, 6, 499.
https://doi.org/10.3390/
fractalfract6090499
Academic Editors: Carlo Cattani and
Leung Lung Chan
Received: 10 June 2022
Accepted: 1 September 2022
Published: 5 September 2022
Publisher’s Note: MDPI stays neutral
with regard to jurisdictional claims in
published maps and institutional affiliations.

Copyright:

© 2022 by the author.

Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license (https://
creativecommons.org/licenses/by/
4.0/).

1. Introduction
An effective way to evaluate and predict financial stability on-time is by studying the
characteristics of the stock market and its operation rules. Many empirical studies have
proved that the stock market is a complex fractal object [1–4], its nonlinear evolution and
multi-scale characteristics can be described quantitatively using the self-similar behavior
analysis method of multi-fractal theory. Many valuable research results have been obtained
on the fluctuations of various types of financial markets [5–9]. Many scholars verified that
the Chinese stock market has nonlinear multifractal characteristics. Chen et al. constructed
an indicator of extremes and predicted the financial extremes from the complex network
perspective based on 12 kinds of worldwide stock indices [10]. Zhuang et al. used the
Multifractal Detrended Fluctuation Analysis (MF-DFA) method and generalized Hurst
exponents to evaluate ten important Chinese sectoral stock indices and revealed that they
have different degrees of multifractality [11]. Du and Ning found that the Shanghai stock
market has weak multifractal features and there are long-range power-law correlations
between the index series [12]. Chen et al. verified the multifractal walk of the Chinese stock
market, and established a stock price prediction model combining the wavelet, genetic
algorithm, and neural network, according to the local scale characteristics and multi-scale
correlation of the multifractal process [13]. Li proposed that the temporal spectrum of the
dominant fractal dimension α0 could be used to characterize stock market fluctuations,
and that the spectrum parameter set (α0 , ∆α, −B) could distinguish the bubble from the
normal fluctuation status well [14]. Li et al. found the degree of the marketization of the
stock market has a significant impact on the multifractal spectrum of the bubbles of the
Shanghai Stock Exchange Composite Index (SSECI) [15].
Although multifractality, as a nonlinear method, has been used by many scholars to
study high-frequency financial time series to investigate the problems and phenomena
which cannot be explained by traditional economic theory, so far, most of the related

Fractal Fract. 2022, 6, 499. https://doi.org/10.3390/fractalfract6090499

https://www.mdpi.com/journal/fractalfract

Fractal Fract. 2022, 6, 499

2 of 20

research remains in the stage of inspecting the multifractal characteristics of financial asset
price volatility. To my knowledge, the dynamic mechanism of the formation of such fractal
characteristics and scaling behavior has not been explored, and the relations between fractal
structure and policy, the stock market institutional arrangement, and investors’ rationality
has not been dug into deeply. In recent years, the research on the stock market fractals
has almost stopped, and the existing research results have not been effectively applied in
the prediction of the stock market crash. The main reason is that the huge fluctuation of
the stock market in China is a policy-induced plunge, and institutional arrangements and
investors’ rationality also affect stock market volatility, and in turn, affect its multifractal
characteristics. Correctly understanding and analyzing the impact of them on stock market
fractal characteristics is of great significance to the slump’s early warning. However, as
mentioned above, there is still a lack of research in this area, which is the focus of this paper.
The organization of the paper is as follows. Section 2 is devoted to identifying slump
episodes and describing the trigger and market conditions of the slumps in China’s stock
market. Section 3 gives the calculation results of the multifractal spectrum of the Chinese
stock market slumps. Section 4 reveals the correlation between multifractals and policies,
market institutional surroundings, and investors’ rationality. Section 5 illustrates the fractal
prediction of the stock market slump. Section 6 summarizes the paper’s conclusions.
2. The Eight Slumps of China’s Stock Market and Their Direct Triggers
The Chinese stock market has experienced eight large ups and downs since its opening.
Since the trading volume and stock market value of the Shanghai Stock Exchange are much
larger than those of the Shenzhen Stock Exchange in the past, the SSECI is used in this
paper to represent the trend of the Chinese stock market.
Although there is no numerically specific definition of a stock market slump, the
term commonly applies to steep double-digit percentage losses in a stock market index
over a period of several days. Here we define a stock market slump as an event when the
SSECI declines relative to the historical maximum for more than 25 percent (the 25% was
selected as the threshold value mainly because it could well distinguish the eight widely
recognized major stock market crashes in China. We have also tried to use other values as
the threshold, e.g. 20%, such that three more shocks are also eligible: 1999 (the SSECI index
from 1756 points on 30 June 31999 down to 1361 on 4 January 2000), 2018 (from 3587 points
on 30 January 2018 to 2440 on 4 January 2019), and 2021 (from 3708 on 21 December 2021
to 3023 on 16 March 2022). However, due to the low steepness of the decline of the three
oscillations, the consistency of the variability of the calculated fractal parameters are not
obvious, especially for ∆α. Therefore, only the results of the eight meltdowns selected
using a 25% threshold value are reported here). The beginning of the slump is the date
when the SSECI falls below this threshold level. The time of the trough is the date when
the SSECI reaches its minimum level during the slump. The time of recovery is the first
date when the SSECI reaches 25 percent of the pre-slump maximum level after the slump is
triggered. To avoid counting the same slump twice, additional triggers occurring within
a slump are considered part of the existing slump, instead of being an indicator of a new
slump. The eight slump episodes are identified here by applying an operational version of
this criterion to the data of the SSECI, and their specific time, the extent of the fall, and the
direct triggers are shown in Table 1.

## fractal and fractional
*File: fractalfract-08-00454.pdf*

**Abstract**

: The Athens Stock Exchange (ASE) is a dynamic financial market with complex interactions
and inherent volatility. Traditional models often fall short in capturing the intricate dependencies and
long memory effects observed in real-world financial data. In this study, we explore the application
of fractional Brownian motion (fBm) to model stock price dynamics within the ASE, specifically
utilizing the Athens General Composite (ATG) index. The ATG is considered a key barometer of the
overall health of the Greek stock market. Investors and analysts monitor the index to gauge investor
sentiment, economic trends, and potential investment opportunities in Greek companies. We find
that the Hurst exponent falls outside the range typically associated with fractal Brownian motion.
This, combined with the established non-normality of increments, disfavors both geometric Brownian
motion and fractal Brownian motion models for the ATG index.
Keywords: geometric Brownian motion; fractional Brownian motion; Athens Stock Exchange

Citation: Leventides, J.; Melas, E.;

1. Introduction

distributed under the terms and

Predicting future behavior in all aspects of social–financial behavior has long become
a field of intense study [1,2]. The financial markets have long been a subject of fascination
for researchers and practitioners alike. Understanding the underlying dynamics of stock
prices and predicting their behavior is crucial for investors, traders, and policymakers.
Traditional models, such as The Black–Scholes–Merton model [3–5] framework, assume
that stock prices follow geometric Brownian motion, where the logarithm of the price
exhibits independent increments. However, this assumption often falls short in capturing
the intricate dependencies and long memory effects observed in real-world financial data.
The fractional Brownian motion (fBm) is a stochastic process that has gained prominence in recent years due to its ability to model correlated and persistent behavior. Unlike
standard Brownian motion, fBm allows for dependent increments, making it a powerful
tool for capturing long-range dependence and volatility clustering. One of the key parameters in fBm is the Hurst exponent H, which characterizes the degree of memory in the
process. In this paper, we delve into the theory behind fractional Brownian motion, exploring its mathematical properties and applications. Specifically, we focus on its relevance to
the Athens Stock Exchange (ASE). By incorporating fBm into the modeling of stock price
dynamics, we aim to enhance our understanding of the ASE’s behavior.
Our objectives include the following:

conditions of the Creative Commons

•

Poulios, C.; Livada, M.; Poulios, N.C.;
Boufounou, P. Application of the
Fractal Brownian Motion to the
Athens Stock Exchange. Fractal Fract.
2024, 8, 454. https://doi.org/
10.3390/fractalfract8080454
Academic Editor: Leung Lung Chan
Received: 27 May 2024
Revised: 29 June 2024
Accepted: 28 July 2024
Published: 31 July 2024

Copyright: © 2024 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article

Attribution (CC BY) license (https://
creativecommons.org/licenses/by/

Theoretical Foundations: We provide a rigorous treatment of fBm, emphasizing its
Gaussian nature and the role of the Hurst exponent. We discuss how fBm differs from
standard Brownian motion and its implications for financial modeling.

4.0/).

Fractal Fract. 2024, 8, 454. https://doi.org/10.3390/fractalfract8080454

https://www.mdpi.com/journal/fractalfract

Fractal Fract. 2024, 8, 454

2 of 10

•

•

•

ATG Index Data: We analyze historical data from the ATG index, which represents
the performance of the top companies listed on the ASE. By applying fBm, we aim to
uncover hidden patterns, long-range dependencies, and volatility clustering within
the index.
Modeling Approach: We construct a geometric fractional Brownian motion model
tailored to the ATG index. Through parameter estimation, we simulate index behavior
and compare it with traditional models.
Empirical Results: Our analysis reveals that both geometric Brownian motion and
fractal Brownian motion models are disfavored for the ATG index.

The challenge is to model efficiently the dynamics of the ATG index in the Athens.
Stock Exchange. The main result of our paper is a negative one. Contrary to expectations,
the ATG index of ASE is not modeled efficiently by fBm. We think that despite the fact
that our result is negative, it does carry important information for readers. fBm has been
used extensively in the literature to model similar dynamics. We mention in passing that
the main result in Fama’s seminal article “Efficient Capital Markets: A Review of Theory
and Empirical Work” [6] is a negative one. Fama’s work supports the idea that market
prices already incorporate all available information, including past stock prices. Therefore,
attempting to predict future prices solely based on historical data (such as technical analysis)
is unlikely to yield consistent profits. Fama’s efficient markets hypothesis (EMH) suggests
that stock prices adjust rapidly to new information, making it difficult for investors to
consistently outperform the market by analyzing historical data. The novelty of our work
is that, as far as we know, the ATG index of the Athens Stock Exchange has not been
modeled before by geometric or fBm. The advantage of our approach lies in the fact that
we demonstrate that fBm, which has been used extensively to capture the dynamics of
similar indices, is not sufficient to describe the dynamics of the ATG index in ASE.
This paper is organized as follows: In Section 2, we highlight the particulars of Black–
Scholes–Merton model. We emphasize the importance of independent increments inherent
in geometric Brownian motion assumed in the Black–Scholes–Merton model. Moreover, we
give the essentials of fBm. In Section 3, we explain the importance of the ATG index on ASE,
and we present historical data of the ATG index. In Section 4, we show that the dynamics
of the ATG index cannot be modeled by geometric Brownian motion. In Section 5, we study
the serial dependence of ATG returns. In Section 6, we show that the dynamics of the ATG
index cannot be modeled by fBm either, and we outline avenues for further research.
2. Black–Scholes–Merton Model
Developed by Fischer Black, Myron Scholes, and Robert Merton [3–5], the Black–
Scholes–Merton model is a cornerstone of financial modeling. It provides a mathematical
framework for pricing options, a type of derivative investment. Central to the model is
the Black–Scholes equation, a differential equation of parabolic type that yields the Black–
Scholes formula. This formula calculates the theoretical price of a European-style option
based on underlying asset characteristics, including the volatility and expected return
(represented by the risk-free interest rate). The model operates within a simplified market
structure composed of at least one risky asset (typically a stock) and one risk-free asset (like
cash or a bond). The assumptions which underpin these assets are as follows:

•
•

•

Riskless rate: A risk-free investment exists, offering a constant rate of return known as
the risk-free interest rate.
Random walk: Stock prices follow a geometric Brownian motion, meaning they change
randomly over time with a constant rate of growth (drift) and volatility. If these factors
are not constant, adjustments to the model are necessary, provided volatility itself
remains stable.
Dividends: The stock being analyzed does not pay dividends.

## HURST EXPONENT AND FINANCIAL MARKET PREDICTABILITY
*File: hurst_exponent_and_financial_market_predictability.pdf*

**Abstract**

The Hurst exponent (H) is a statistical measure used to
classify time series. H=0.5 indicates a random series
while H>0.5 indicates a trend reinforcing series. The
larger the H value is, the stronger trend. In this paper we
investigate the use of the Hurst exponent to classify series
of financial data representing different periods of time.
Experiments with backpropagation Neural Networks
show that series with large Hurst exponent can be
predicted more accurately than those series with H value
close to 0.50. Thus Hurst exponent provides a measure
for predictability.

## Article
*File: mathematics-13-03587-v3.pdf*

**Abstract**

Accurate Value-at-Risk (VaR) forecasting is challenged by the non-stationary, fractal, and
chaotic dynamics of financial markets. Standard deep learning models like LSTMs often rely
on static internal mechanisms that fail to adapt to shifting market complexities. To address
these limitations, we propose a novel architecture: the Dynamic Fractal–Chaotic LSTM
(DFC-LSTM). This model incorporates two synergistic innovations: a multifractal-driven
dynamic forget gate that utilizes the multifractal spectrum width (∆α) to adaptively regulate
memory retention, and a chaotic oscillator-based dynamic activation that replaces the
standard tanh function with the peak response of a Lee Oscillator’s trajectory. We evaluate
the DFC-LSTM for one-day-ahead 95% VaR forecasting on S&P 500 and AAPL stock data,
comparing it against a suite of state-of-the-art benchmarks. The DFC-LSTM consistently
demonstrates superior statistical calibration, passing coverage tests with significantly
higher p-values—particularly on the volatile AAPL dataset, where several benchmarks
fail—while maintaining competitive economic loss scores. These results validate that
embedding the intrinsic dynamical principles of financial markets into neural architectures
leads to more accurate and reliable risk forecasts.
Academic Editor: Manuel Alberto M.
Ferreira
Received: 9 October 2025

Keywords: value-at-risk; LSTM; multifractal analysis; chaos theory; time series forecasting;
deep learning; neural networks; financial mathematics; computational modeling

Revised: 4 November 2025
Accepted: 6 November 2025
Published: 8 November 2025
Citation:

MSC: 91G70; 62M10; 68T07; 37M10

Zeng, Y.; Tang, B.; Zhou, Z.;

Lee, R.S.T. DFC-LSTM: A Novel LSTM
Architecture Integrating Dynamic
Fractal Gating and Chaotic Activation
for Value-at-Risk Forecasting.
Mathematics 2025, 13, 3587. https://
doi.org/10.3390/math13223587
Copyright: © 2025 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license
(https://creativecommons.org/
licenses/by/4.0/).

Mathematics 2025, 13, 3587

1. Introduction
In modern financial risk management, accurate forecasting of asset return volatility is
essential. Volatility serves as a fundamental risk measure that informs investment decisions
and constitutes the primary input for risk assessment frameworks such as Value-at-Risk
(VaR). As a widely adopted metric for market risk, VaR is extensively used by financial
institutions for regulatory capital requirements and internal risk management [1]. Consequently, producing precise and adaptive VaR forecasts is critical for effectively navigating
the complexities and turbulence of financial markets.
Throughout the history of VaR estimation, various methods have been developed, with
the parametric approach being one of the primary paradigms. This method evaluates risk
by analyzing the return distribution of an asset over a look-back period and estimating its

https://doi.org/10.3390/math13223587

Mathematics 2025, 13, 3587

2 of 26

volatility and expected return. Within this framework, accurate modeling of volatility has
become crucial, leading to the development of many sophisticated volatility models, among
which the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) family
is the most prominent [2]. Efforts to improve predictive accuracy have also explored the
use of more complex probability density functions and the incorporation of time-varying
higher-order conditional moments. While GARCH models are highly regarded for their
statistical rigor and ability to capture ‘stylized facts’ such as volatility clustering, their
foundation relies on strict statistical assumptions—for example, specific error distributions.
This structural rigidity often results in suboptimal performance amid the profound nonlinearity and dynamic complexity of financial markets, particularly during periods of
extreme stress [3].
To overcome the limitations of traditional models, both academia and industry have
increasingly turned to deep learning (DL), which offers fewer modeling constraints and
enhanced feature extraction capabilities [4,5]. Given the pervasive long-memory property in
financial time series [6], Recurrent Neural Networks (RNNs), capable of retaining historical
information, are considered more appropriate for processing such data than Feed-forward
Neural Networks (FNNs) [7]. However, conventional RNNs encounter difficulties with
vanishing or exploding gradients during the training of long sequences [8]. The Long ShortTerm Memory (LSTM) network, introduced by Hochreiter & Schmidhuber [9], addresses
these issues through its gating mechanisms, enabling the model to capture features over
extended time horizons. Consequently, LSTMs are theoretically well-suited for volatility
forecasting tasks [5].
Building on this foundation, researchers have pursued various approaches to further
enhance LSTM performance in financial forecasting. A prominent trend involves developing hybrid models that combine the statistical insights of GARCH models with the
sequential learning strengths of LSTMs. Early studies, such as those by Kim & Won [4] and
Hu et al. [10], demonstrated the effectiveness of this approach by feeding GARCH-derived
predictions as external features into LSTM networks. More recent advancements have
integrated these methodologies more deeply; for instance, Zhao et al. [11] proposed reengineering and embedding the mathematical structure of GARCH models directly within
the LSTM cell, which not only improved the model’s interpretability in a financial context
but also yielded superior predictive performance over traditional econometric models.
Simultaneously, another avenue of research enhances model capabilities by incorporating
external information, notably utilizing Natural Language Processing (NLP) techniques to
extract market sentiment from textual data as an additional input [12]. Since these hybrid
and information-enhanced methods aim to improve volatility forecast accuracy, they have
naturally been extended to the more complex task of VaR estimation, with promising
results [3].
Despite notable advances in fusion strategies, current state-of-the-art forecasting
models still encounter two fundamental bottlenecks rooted in the intrinsic properties of
financial data. The first pertains to input feature fidelity: financial time series are inherently
highly non-stationary [13], with their mean and variance evolving dynamically over time.
Deeper still, this non-stationarity often manifests as intricate fractal structures, where the
complex temporal correlations give rise to multifractal characteristics—beyond mere fat-tail
distributions [14]. Multifractal analysis, a powerful tool in nonlinear dynamics, can reveal
the “bursty” and heterogeneous nature of market fluctuations at fine scales [15]. However, a
persistent technical challenge in standard multifractal analysis—the use of non-overlapping
segmentation—can introduce spurious fluctuations that undermine the stability of fractal
measurements [16]. Nonetheless, its core metric, the multifractal spectrum width (∆α),
remains an effective indicator of the degree of inhomogeneity and local complexity in

## sustainability
*File: sustainability-11-01699.pdf*

**Abstract**

: This paper examines the daily return series of four main indices, including Shanghai Stock
Exchange Composite Index (SSE), Shenzhen Stock Exchange Component Index (SZSE), Shanghai
Shenzhen 300 Index (SHSE-SZSE300), and CSI Smallcap 500 index (CSI500) in Chinese stock market
from 2000 to 2018 by multifractal detrended fluctuation analysis (MF-DFA). The series of the daily
return of the indices exhibit significant multifractal properties on the whole time scale and SZSE has
the highest multifractal properties among the four indices, indicating the lowest market efficiency.
The multifractal properties of four indices are due to long-range correlation and fat-tail characteristics
of the non-Gaussian probability density function, and these two factors have different effects on the
multifractality of four indices. This paper aims to compare the multifractility degrees of the four
indices in three sub-samples divided by the 2015 stock market crash and to discuss its effects on
efficiency of the Shanghai and Shenzhen stock market in each sub-sample. Meanwhile, we study the
effect of the 2015 stock market crash on market efficiency from the statistical and fractal perspectives,
which has theoretical and practical significance in the application of Effective Market Hypothesis
(EMH) in China’s stock market, and it thereby affects the healthy and sustainability of the market.
The results also provide important implications for further study on the dynamic mechanism and
efficiency in stock market and they are relevant to portfolio managers and policy makers in a number
of ways to maintain the sustainable development of China’s capital market and economy.
Keywords: multifractality analysis; MF-DFA; market efficiency; stock market crash

1. Introduction
The Shanghai Security Exchange and Shenzhen Security Exchange, two major stock markets in
China, are closely connected, which exhibit certain similar features and face same external shocks.
The relationship between stock market development and economic growth is a significant issue that is
related to the sustainable economic development of an economy and financial markets, manifested
in optimizing the allocation of resources. Researchers have clearly shown that a well-functioning
stock market can effectively employ social funds and then allocate these funds to productive sectors
with high efficiency. Stock markets promote financial markets development by spreading risks, thus
improving resource allocation efficiency, supervising managers and utilizing corporate governance,
and influencing the savings rate, which can reduce the cost of information and transactions, thus
advancing sustainable economic growth. Moreover, for a long time, the random walk theory and
Effective Market Hypothesis (EMH) maintain key theoretical cornerstones of mainstream financial
theory. However, the EMH has been challenged by the real operations of the markets and related
Sustainability 2019, 11, 1699; doi:10.3390/su11061699

www.mdpi.com/journal/sustainability

Sustainability 2019, 11, 1699

2 of 15

studies since the 1980s. The fractal market research has modified the basic hypothesis of the linear
model, random walk, and normal distribution, on which the mainstream financial econometric theory
is based. Furthermore, China’s stock market experienced abnormal fluctuations from June to August
in 2015. In the case of the Shanghai Stock Exchange Composite Index (SSE Composite Index, SSE
index), the index dropped from 5178.19 to 2850.71 in 53 trading days after two rounds of cliff-cut
declines, representing a decline of more than 45%. The stock market crash has caused great shocks,
which has a negative impact on China’s social stability and stable economic operation. Furthermore,
the stock market crash also led to a large-scale drop-stop of individual stocks, the lack of stock market
liquidity, huge losses investors being suffered, and less sustainability of economic growth. Under such
circumstances, this paper studies the impacts of the stock crash on the dynamics and efficiency of
China’s stock market.
When considering the complex nonlinear characteristics in the Chinese stock market [1], this
paper applies the theory and method of multifractal analysis to the study of price fluctuation in China’s
stock market, and analyzes the price fluctuations of Chinese stock market from the nonlinear angle.
The paper examines four main indices of Chinese stock markets, including Shanghai Stock Exchange
Composite Index (SSE Composite Index, SSE), Shenzhen Stock Exchange Component Index (SZSE
Component Index, SZSE), Shanghai Shenzhen 300 Index (SHSE-SZSE300), and CSI Smallcap 500 Index
(CSI500). The indices of SSE, SZSE, and SHSE-SZSE300 are selected to study the characteristics of the
main board, while SSE and SZSE are studied to reflect the fluctuations of stock prices of the Shanghai
Stock Exchange and Shenzhen Stock Exchange, respectively. The SHSE-SZSE300, a cross-market index,
is also studied, because it reflects the overall trend of Shanghai and Shenzhen Markets and the joint
changes in stock prices in the two markets. The CSI500 comprehensively reflects the stock price of a
group of small and medium-sized companies in China’s A-share market, which can be a supplement
to research on the other three indices. Therefore, this research aims to investigate and compare the
multifractality levels and dynamic evolutions of the daily return data of four major indices in China’s
stock market throughout January 2000 to November 2018.
Literatures mainly focus on the multifractality of the mainboard market, but only a few studies
have examined the impact of stock market crash on the stock market and its efficiency. Unlike previous
studies, this paper applies empirical methodology to compare and study the dynamic evolution
of multifractality of the series of daily return of the four main indices between January 2000 and
November 2018. By comparing the multifractal properties of the four-index returns, we estimate the
market efficiencies of the two main stock market in China, i.e., Shanghai stock market and Shenzhen
stock market. Particularly, we explore the impacts of the 2015 stock market crash on the market
dynamics and efficiency by dividing the sample into three sub-samples: before, during, and after
the crash.
The four indices show multifractal characteristics with different degrees that are caused by the
long-range correlation and the fat-tail distribution. The multifractal levels of the return series of the
four indices exhibit different dynamic properties during different periods and the fluctuations of
the multifractality of the indices are different. The 2015 stock crash had different effects on the
dynamics and efficiency in different markets during different periods. The study is organized,
as follows. Section 2 proposes the MF-DFA methodology. Section 3 introduces the data and model
settings. Section 4 presents empirical results. Section 5 further discusses the results and corresponding
explanations and Section 7 concludes this paper.
2. Literature Review
An efficient market has been theoretically proven to be a vital component for effective and efficient
resource allocation in an economy. The issue of market efficiency has attracted considerable attention
in literatures. Researchers have studied the efficiencies of different markets with various methods or
models. Rizvi et al. [2] incorporated econophysics with Efficient Market Hypothesis (EMH) and then
used Multi-fractal detrended fluctuation analysis (MF-DFA) to undertake a comparative analysis of

## SS symmetry
*File: symmetry-12-01157-v2.pdf*

**Abstract**

: The use of multifractal approaches has been growing because of the capacity of these
tools to analyze complex properties and possible nonlinear structures such as those in financial time
series. This paper analyzes the presence of long-range dependence and multifractal parameters
in the stock indices of nine MSCI emerging Asian economies. Multifractal Detrended Fluctuation
Analysis (MFDFA) is used, with prior application of the Seasonal and Trend Decomposition using
the Loess (STL) method for more reliable results, as STL separates different components of the time
series and removes seasonal oscillations. We find a varying degree of multifractality in all the markets
considered, implying that they exhibit long-range correlations, which could be related to verification
of the fractal market hypothesis. The evidence of multifractality reveals symmetry in the variation
trends of the multifractal spectrum parameters of financial time series, which could be useful to
develop portfolio management. Based on the degree of multifractality, the Chinese and South Korean
markets exhibit the least long-range dependence, followed by Pakistan, Indonesia, and Thailand.
On the contrary, the Indian and Malaysian stock markets are found to have the highest level of
dependence. This evidence could be related to possible market inefficiencies, implying the possibility of
institutional investors using active trading strategies in order to make their portfolios more profitable.
Keywords: Asian stock markets; emerging stock markets; long-range dependence; multifractal Analysis

1. Introduction
A financial system is an integral part of an economy, allowing for the exchange of funds
between lenders, borrowers, investors, and government entities, and efficient resource allocation.
Correct understanding of financial markets’ structures is important for the design of appropriate
public policies, investment strategies and portfolios, or taxation and legal frameworks, which are
major ingredients of a well-structured economy. The study of financial markets has long been based on
the Random Walk Hypothesis (RWH) introduced by the authors of [1], which assumes that stock prices
are described by a random walk, this being the basis of a fundamental theory of financial markets,
namely, the Efficient Market Hypothesis (EMH) proposed in [2]. In its weak form, the efficiency
hypothesis presumes that markets are inherently efficient to readily adjust their assets’ prices to any new
information, preventing investors from having abnormal returns. However, market structures do not
behave in this way and different research reveals that stock markets have ubiquitous properties [3–5],
Symmetry 2020, 12, 1157; doi:10.3390/sym12071157

www.mdpi.com/journal/symmetry

Symmetry 2020, 12, 1157

2 of 19

with issues described in the literature as stylized facts like fat tails [6], long-term correlations [7],
volatility clustering [8], fractals/multifractals [9], and chaos [10], with these properties making financial
markets inconsistent with both EMH and RWH [11]. These inconsistencies required more logical
explanation of market movements than the ones described by the EMH. Some questions were answered
by proving that fractal models, incorporating issues like geometric shape, parts of which can be isolated,
reflect more realistic market activity. Based on this, the authors of [12,13] developed the Fractal Market
Hypothesis (FMH). This is less restrictive as it considers, for example, the possibility of investors’
heterogeneous behavior, which according to theory is the guarantee of market stability [12–14].
Although building the FMH dates from the 1990s, the work of Mandelbrot [3–5] is seminal, with
the introduction of fractal geometry to investigate the behavior of cotton prices and finding that
commodity prices do not exhibit a random walk behavior. Fractal geometry was inspirational for
the study of financial markets, as, in general, time series of financial assets’ prices show similar complex
multifractal patterns to those found in some commodities. Based on the fractal theory, some initial
studies used Detrended Fluctuation Analysis (DFA) as a monofractal approach to investigate stock
markets’ long-range dependence [12,14,15]. With an R/S analysis, the authors of [16,17] identified
the existence of monofractality. However, this is a property which could not describe correctly
the behavior of correlation functions [18]. Later, research in this particular area showed that a single
scaling component used in a monofractal analysis to study prices’ complex multifractal structures is not
sufficient, opening the way for the development of more robust techniques like Multifractal Detrended
Fluctuation Analysis (MFDFA) [7,19] and the partition function method [20–22]. By using the fractal
dimension algorithm, the authors of [23] analyzed the performance of the Fama and French [24]
five-factor model including the Hurst exponent as an additional factor. The authors reported that
the significance of the H factor is greater than the momentum factor and similar to others such as
the capitalization and book-to-market factors.
Multifractality is central in the study of complexity and has applications in almost all essential
areas of scientific research from physics to economics, but also in other natural and social sciences such
as chemistry, hydrology, or psychology, or even other research areas like linguistics, the environment
or even music (see, e.g., in [25–32] among others). In particular, MFDFA is a strong tool, capable of
detecting long-term dependence even in the presence of non-stationarity [33]. In the context of financial
markets, it could help to determine the existence of long-range dependencies linking them to the EMH
and reveal the extent of inefficiency [34]. This method has already been applied in several fields,
including finance [5,20,35–37], revealing the existence of multifractality in financial markets [38–42].
We find a wide gap in the literature regarding in-depth analysis of the fractal behavior of emerging
markets in general and Asian markets in particular. In order to fill this gap, this paper is unique
and extends the existing literature in three ways. First, this study provides evidence of long-range
dependence in nine Morgan Stanley Capital International (MSCI) emerging Asian markets, namely,
China, South Korea, India, Pakistan, Malaysia, Indonesia, the Philippines, Thailand, and Taiwan.
Using daily price data up to 02 April 2020, this study examines the behavior of nine stock markets
over the long run. Second, the use of Seasonal and Trend Decomposition using Loess (STL) before
applying MFDFA, in order to decompose the original time series of stock returns, isolating its values
from possible seasonal oscillations [11]. This is an advantage because STL can handle several different
types of seasonality, is robust to the existence of outliers, and can detect seasonal variations over
time. According to the work in [43], the application of STL can identify the dynamics of stock returns
at the same time as assuring reliability by decomposing the time series after removing seasonal
components. Third, the use of a robust Econophysics methodology such as MFDFA, as a generalization
of the original DFA proposed in [19], will allow us to detect the pattern of multifractal behavior in
emerging Asian stock markets. The main results of this paper show that the Chinese and South Korean
stock markets have the lowest levels of multifractality, while the Indian and Malaysian ones show
the highest levels. Moreover, only the Chinese and Indonesian stock markets indicate persistent

