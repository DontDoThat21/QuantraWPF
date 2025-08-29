using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Services;

namespace Quantra.Models
{
    public class FibonacciRetracementStrategy : StrategyProfile
    {
        // Fibonacci levels (percentages)
        private static readonly double[] FibLevels = { 0.236, 0.382, 0.5, 0.618, 0.786 };
        private const int WeeklyLookback = 5; // Number of days to look back for weekly high/low
        private const int TrendLookback = 20; // Days to determine trend direction
        private const double PriceBuffer = 0.003; // 0.3% buffer for level tests

        private const double FibLevel38 = 0.382;
        private const double FibLevel50 = 0.5;
        private const double FibLevel61 = 0.618;
        
        public FibonacciRetracementStrategy()
        {
            Name = "Fibonacci Retracement";
            Description = "Uses Fibonacci retracement levels to generate trading signals";
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "PRICE", "HIGH", "LOW", "SMA20" };

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (!indicators.TryGetValue("PRICE", out double price) ||
                !indicators.TryGetValue("HIGH", out double high) ||
                !indicators.TryGetValue("LOW", out double low) ||
                !indicators.TryGetValue("SMA20", out double sma))
                return false;

            return true; // Basic validation passes if we have all required indicators
        }

        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            var actualIndex = index ?? historical.Count - 1;
            if (actualIndex < WeeklyLookback) return null;

            // Get the relevant portion of historical data
            var prices = historical.Take(actualIndex + 1).ToList();
            
            // Get the most recent price for signal generation
            double currentPrice = prices.Last().Close;
            
            // Determine trend direction from historical prices
            TrendDirection trend = DetermineTrend(prices);
            
            // If trend is undefined, no signal
            if (trend == TrendDirection.Undefined)
                return null;
                
            // Find swing high and low using the lookback period
            var recentPrices = prices.Skip(Math.Max(0, prices.Count - WeeklyLookback)).ToList();
            double high = recentPrices.Max(p => p.High);
            double low = recentPrices.Min(p => p.Low);
            
            // Calculate Fibonacci retracement levels based on the high, low and trend
            var fibLevels = CalculateFibonacciLevels(high, low, trend);
            
            // Generate signal based on current price and Fibonacci levels
            return GenerateSignalAtLevels(currentPrice, fibLevels, trend);
        }

        public override double GetStopLossPercentage()
        {
            // Stop loss at the next Fibonacci level difference (0.618 - 0.5 = 0.118)
            return 0.0382; // 3.82% based on the distance between Fib levels
        }

        public override double GetTakeProfitPercentage()
        {
            // Take profit at approximately twice the stop loss (risk:reward ratio of 1:2)
            return 0.0764; // 7.64% based on twice the stop loss
        }

        private enum TrendDirection
        {
            Uptrend,
            Downtrend,
            Undefined
        }

        private TrendDirection DetermineTrend(List<HistoricalPrice> prices)
        {
            if (prices.Count < TrendLookback)
                return TrendDirection.Undefined;
                
            // Get recent prices for trend analysis
            var trendPrices = prices.Skip(Math.Max(0, prices.Count - TrendLookback)).ToList();
            
            // Use linear regression to determine trend
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            int n = trendPrices.Count;

            for (int i = 0; i < n; i++)
            {
                double x = i;
                double y = trendPrices[i].Close;
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }

            // Calculate slope of the trend line
            double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            
            if (Math.Abs(slope) < 0.0001) // Very flat trend
                return TrendDirection.Undefined;
                
            return slope > 0 ? TrendDirection.Uptrend : TrendDirection.Downtrend;
        }

        private double DetermineTrend(List<double> prices, int period = 20)
        {
            if (prices.Count < period) return 0;

            var recentPrices = prices.TakeLast(period).ToList();
            var firstAvg = recentPrices.Take(period / 2).Average();
            var secondAvg = recentPrices.Skip(period / 2).Take(period / 2).Average();

            return secondAvg - firstAvg; // Positive = uptrend, Negative = downtrend
        }

        private class FibLevel
        {
            public double Level { get; set; }
            public double Price { get; set; }
        }

        private List<FibLevel> CalculateFibonacciLevels(double high, double low, TrendDirection trend)
        {
            var levels = new List<FibLevel>();
            double range = high - low;

            foreach (var fib in FibLevels)
            {
                if (trend == TrendDirection.Uptrend)
                {
                    // In uptrend, measure retracements from low to high
                    levels.Add(new FibLevel
                    {
                        Level = fib,
                        Price = high - (range * fib)
                    });
                }
                else
                {
                    // In downtrend, measure retracements from high to low
                    levels.Add(new FibLevel
                    {
                        Level = fib,
                        Price = low + (range * fib)
                    });
                }
            }

            return levels;
        }

        private string GenerateSignalAtLevels(double currentPrice, List<FibLevel> fibLevels, TrendDirection trend)
        {
            // Check if price is near any Fibonacci level
            foreach (var level in fibLevels)
            {
                bool isNearLevel = Math.Abs(currentPrice - level.Price) <= (level.Price * PriceBuffer);
                
                if (!isNearLevel)
                    continue;

                // Generate signals based on trend and level
                if (trend == TrendDirection.Uptrend)
                {
                    // In uptrend, look for buying opportunities at retracement levels
                    if (level.Level >= 0.382 && level.Level <= 0.618) // Key retracement zone
                    {
                        return "BUY";
                    }
                }
                else if (trend == TrendDirection.Downtrend)
                {
                    // In downtrend, look for selling opportunities at retracement levels
                    if (level.Level >= 0.382 && level.Level <= 0.618) // Key retracement zone
                    {
                        return "SELL";
                    }
                }

                // Exit signals when price reaches extreme levels
                if ((trend == TrendDirection.Uptrend && level.Level <= 0.236) ||
                    (trend == TrendDirection.Downtrend && level.Level >= 0.786))
                {
                    return "EXIT";
                }
            }

            return null;
        }
    }
}