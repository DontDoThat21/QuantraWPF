using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Services;

namespace Quantra.Models
{
    public class EmaSmaCrossStrategy : StrategyProfile
    {
        // EMA will be more responsive to recent price changes
        private const int EmaPeriod = 10;  // 2 trading weeks
        // SMA will provide more stable baseline
        private const int SmaPeriod = 20;  // 4 trading weeks
        private const double CrossConfirmationThreshold = 0.001; // 0.1% threshold for cross confirmation
        
        public EmaSmaCrossStrategy()
        {
            Name = "EMA/SMA Cross";
            Description = "Strategy based on EMA/SMA crossovers with confirmation threshold. " +
                         $"Uses {EmaPeriod}-period EMA and {SmaPeriod}-period SMA.";
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "EMA", "SMA" };

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            // Validate that we have required indicators
            return indicators.ContainsKey("EMA") && indicators.ContainsKey("SMA");
        }

        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            var currentIndex = index ?? historical.Count - 1;
            
            if (currentIndex < Math.Max(EmaPeriod, SmaPeriod))
                return null;

            var prices = historical.Take(currentIndex + 1).Select(p => p.Close).ToList();
            
            var emaValues = CalculateEMA(prices, EmaPeriod);
            var smaValues = CalculateSMA(prices, SmaPeriod);

            if (emaValues.Count < 2 || smaValues.Count < 2)
                return null;

            // Get current and previous values
            var currentEma = emaValues[emaValues.Count - 1];
            var previousEma = emaValues[emaValues.Count - 2];
            var currentSma = smaValues[smaValues.Count - 1];
            var previousSma = smaValues[smaValues.Count - 2];

            // Calculate percentage difference for confirmation
            var currentDiff = (currentEma - currentSma) / currentSma;
            var previousDiff = (previousEma - previousSma) / previousSma;

            // Detect crossovers with confirmation threshold
            if (previousDiff <= 0 && currentDiff > CrossConfirmationThreshold)
            {
                // Bullish crossover - EMA crosses above SMA
                return "BUY";
            }
            else if (previousDiff >= 0 && currentDiff < -CrossConfirmationThreshold)
            {
                // Bearish crossover - EMA crosses below SMA
                return "SELL";
            }
            // Add EXIT signals when moving averages start converging
            else if (Math.Abs(currentDiff) < CrossConfirmationThreshold && Math.Abs(previousDiff) >= CrossConfirmationThreshold)
            {
                return "EXIT";
            }

            return null;
        }

        public override double GetStopLossPercentage()
        {
            // Conservative stop loss of 2%
            return 0.02;
        }

        public override double GetTakeProfitPercentage()
        {
            // Target profit of 3%
            return 0.03;
        }

        private List<double> CalculateEMA(List<double> prices, int period)
        {
            var ema = new List<double>();
            if (prices.Count < period)
                return ema;

            // First value is SMA
            var sma = prices.Take(period).Average();
            ema.Add(sma);

            // Calculate multiplier
            double multiplier = 2.0 / (period + 1);

            // Calculate subsequent EMAs
            for (int i = period; i < prices.Count; i++)
            {
                var currentEma = (prices[i] - ema[ema.Count - 1]) * multiplier + ema[ema.Count - 1];
                ema.Add(currentEma);
            }

            return ema;
        }

        private List<double> CalculateSMA(List<double> prices, int period)
        {
            var sma = new List<double>();
            if (prices.Count < period)
                return sma;

            // Calculate initial SMA
            double sum = prices.Take(period).Sum();
            sma.Add(sum / period);

            // Calculate subsequent SMAs using sliding window
            for (int i = period; i < prices.Count; i++)
            {
                sum = sum - prices[i - period] + prices[i];
                sma.Add(sum / period);
            }

            return sma;
        }

        private double CalculateWeeklyMomentum(List<double> prices)
        {
            if (prices.Count < 5) // Need at least a week of data
                return 0;

            var weeklyPrices = prices.Skip(prices.Count - 5).Take(5).ToList();
            var firstPrice = weeklyPrices.First();
            var lastPrice = weeklyPrices.Last();

            return (lastPrice - firstPrice) / firstPrice;
        }
    }
}