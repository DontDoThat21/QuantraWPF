using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Services;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy that looks for divergence between RSI and price movement
    /// </summary>
    public class RsiDivergenceStrategy : TradingStrategyProfile
    {
        private int _rsiPeriod = 14;
        private int _lookbackPeriods = 5;
        private double _oversoldLevel = 30;
        private double _overboughtLevel = 70;

        public RsiDivergenceStrategy()
        {
            Name = "RSI Divergence";
            Description = "Identifies overbought/oversold conditions and divergence between RSI and price. " +
                          "A buy signal is generated when price makes a lower low but RSI makes a higher low (bullish divergence). " +
                          "A sell signal is generated when price makes a higher high but RSI makes a lower high (bearish divergence).";
            RiskLevel = 0.6;
            MinConfidence = 0.65;
        }

        /// <summary>
        /// Period for RSI calculation
        /// </summary>
        public int RsiPeriod
        {
            get => _rsiPeriod;
            set
            {
                if (value > 0 && _rsiPeriod != value)
                {
                    _rsiPeriod = value;
                    OnPropertyChanged(nameof(RsiPeriod));
                }
            }
        }

        /// <summary>
        /// Number of periods to look back for divergence
        /// </summary>
        public int LookbackPeriods
        {
            get => _lookbackPeriods;
            set
            {
                if (value > 0 && _lookbackPeriods != value)
                {
                    _lookbackPeriods = value;
                    OnPropertyChanged(nameof(LookbackPeriods));
                }
            }
        }

        /// <summary>
        /// RSI level considered oversold
        /// </summary>
        public double OversoldLevel
        {
            get => _oversoldLevel;
            set
            {
                if (value > 0 && value < 50 && _oversoldLevel != value)
                {
                    _oversoldLevel = value;
                    OnPropertyChanged(nameof(OversoldLevel));
                }
            }
        }

        /// <summary>
        /// RSI level considered overbought
        /// </summary>
        public double OverboughtLevel
        {
            get => _overboughtLevel;
            set
            {
                if (value > 50 && value < 100 && _overboughtLevel != value)
                {
                    _overboughtLevel = value;
                    OnPropertyChanged(nameof(OverboughtLevel));
                }
            }
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "RSI", "Price", "Volume" };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < RsiPeriod + LookbackPeriods + 1)
                return null;
            
            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < RsiPeriod + LookbackPeriods || currentIndex >= prices.Count)
                return null;

            List<double> rsiValues = CalculateRSI(prices, RsiPeriod);
            if (rsiValues == null || rsiValues.Count <= currentIndex - RsiPeriod)
                return null;

            // Check for bullish divergence (oversold)
            if (IsOversold(rsiValues, currentIndex) && HasBullishDivergence(prices, rsiValues, currentIndex, LookbackPeriods))
            {
                return "BUY";
            }
            
            // Check for bearish divergence (overbought)
            if (IsOverbought(rsiValues, currentIndex) && HasBearishDivergence(prices, rsiValues, currentIndex, LookbackPeriods))
            {
                return "SELL";
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (indicators == null)
                return false;

            // Check if we have all required indicators
            if (!indicators.TryGetValue("RSI", out double rsi))
                return false;

            // Check RSI conditions
            bool isOverbought = rsi > OverboughtLevel;
            bool isOversold = rsi < OversoldLevel;

            // Check additional confirmatory indicators if available
            bool macdConfirms = false;
            if (indicators.TryGetValue("MACD", out double macd) && 
                indicators.TryGetValue("MACDSignal", out double macdSignal))
            {
                macdConfirms = (isOversold && macd > macdSignal) || (isOverbought && macd < macdSignal);
            }

            // Check stochastic if available
            bool stochConfirms = false;
            if (indicators.TryGetValue("StochK", out double stochK) && 
                indicators.TryGetValue("StochD", out double stochD))
            {
                stochConfirms = (isOversold && stochK > stochD) || (isOverbought && stochK < stochD);
            }

            // Return true if RSI condition is met and at least one confirming indicator is available and confirms
            return (isOverbought || isOversold) && (macdConfirms || stochConfirms || !indicators.ContainsKey("MACD"));
        }

        private bool IsOversold(List<double> rsiValues, int index)
        {
            int rsiIndex = index - RsiPeriod;
            return rsiIndex >= 0 && rsiValues[rsiIndex] < OversoldLevel;
        }

        private bool IsOverbought(List<double> rsiValues, int index)
        {
            int rsiIndex = index - RsiPeriod;
            return rsiIndex >= 0 && rsiValues[rsiIndex] > OverboughtLevel;
        }

        private bool HasBullishDivergence(List<HistoricalPrice> prices, List<double> rsiValues, int currentIndex, int lookback)
        {
            int startIdx = currentIndex - lookback;
            if (startIdx < RsiPeriod)
                return false;

            // Adjust indices for RSI values which start at period
            int rsiCurrentIdx = currentIndex - RsiPeriod;
            int rsiStartIdx = startIdx - RsiPeriod;
            
            // Find price lows in the lookback period
            int minPriceIdx = startIdx;
            for (int i = startIdx + 1; i <= currentIndex; i++)
            {
                if (prices[i].Low < prices[minPriceIdx].Low)
                    minPriceIdx = i;
            }

            // Return false if the minimum is the current point (no divergence possible)
            if (minPriceIdx == currentIndex)
                return false;

            // Find the corresponding RSI value for the price low
            int rsiMinPriceIdx = minPriceIdx - RsiPeriod;
            
            // Check if price made lower low but RSI made higher low (bullish divergence)
            return prices[currentIndex].Low < prices[minPriceIdx].Low &&
                   rsiValues[rsiCurrentIdx] > rsiValues[rsiMinPriceIdx];
        }

        private bool HasBearishDivergence(List<HistoricalPrice> prices, List<double> rsiValues, int currentIndex, int lookback)
        {
            int startIdx = currentIndex - lookback;
            if (startIdx < RsiPeriod)
                return false;

            // Adjust indices for RSI values which start at period
            int rsiCurrentIdx = currentIndex - RsiPeriod;
            int rsiStartIdx = startIdx - RsiPeriod;
            
            // Find price highs in the lookback period
            int maxPriceIdx = startIdx;
            for (int i = startIdx + 1; i <= currentIndex; i++)
            {
                if (prices[i].High > prices[maxPriceIdx].High)
                    maxPriceIdx = i;
            }

            // Return false if the maximum is the current point (no divergence possible)
            if (maxPriceIdx == currentIndex)
                return false;

            // Find the corresponding RSI value for the price high
            int rsiMaxPriceIdx = maxPriceIdx - RsiPeriod;
            
            // Check if price made higher high but RSI made lower high (bearish divergence)
            return prices[currentIndex].High > prices[maxPriceIdx].High &&
                   rsiValues[rsiCurrentIdx] < rsiValues[rsiMaxPriceIdx];
        }

        private List<double> CalculateRSI(List<HistoricalPrice> prices, int period)
        {
            if (prices.Count <= period)
                return null;
                
            var rsiValues = new List<double>();
            for (int i = 0; i < period; i++)
                rsiValues.Add(50); // Fill initial values

            var gains = new List<double>();
            var losses = new List<double>();

            // Calculate initial gains and losses
            for (int i = 1; i <= period; i++)
            {
                var change = prices[i].Close - prices[i - 1].Close;
                gains.Add(Math.Max(0, change));
                losses.Add(Math.Max(0, -change));
            }

            // Calculate initial average gain and loss
            var avgGain = gains.Average();
            var avgLoss = losses.Average();

            // Calculate first RSI
            var rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            var rsi = 100 - (100 / (1 + rs));
            rsiValues.Add(rsi);

            // Calculate remaining RSI values
            for (int i = period + 1; i < prices.Count; i++)
            {
                var change = prices[i].Close - prices[i - 1].Close;
                var gain = Math.Max(0, change);
                var loss = Math.Max(0, -change);

                // Use Wilder's smoothing method
                avgGain = ((avgGain * (period - 1)) + gain) / period;
                avgLoss = ((avgLoss * (period - 1)) + loss) / period;

                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiValues.Add(rsi);
            }

            return rsiValues;
        }
    }
}