using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Models
{
    public class VwapAnchoredStrategy : StrategyProfile
    {
        private const int WeeklyPeriod = 5; // Trading days in a week
        private const int MonthlyPeriod = 20; // Approximate trading days in a month

        public VwapAnchoredStrategy()
        {
            Name = "VWAP Anchored";
            Description = "Uses Volume Weighted Average Price (VWAP) anchored to weekly and monthly timeframes to generate signals. Buy signals are generated when price is above both VWAPs, sell signals when price is below both VWAPs.";
        }

        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            if (historical == null || !historical.Any())
                return null;

            var currentIndex = index ?? historical.Count - 1;
            if (currentIndex < WeeklyPeriod)
                return null;

            var prices = historical.Take(currentIndex + 1).ToList();
            var currentPrice = prices[prices.Count - 1].Close;

            // Calculate Weekly and Monthly VWAP
            var weeklyVwap = CalculateVwap(prices.Skip(prices.Count - WeeklyPeriod).ToList());
            var monthlyVwap = currentIndex >= MonthlyPeriod ? 
                CalculateVwap(prices.Skip(prices.Count - MonthlyPeriod).ToList()) : 
                weeklyVwap;

            // No signal if we can't calculate both VWAPs
            if (weeklyVwap == null || monthlyVwap == null)
                return null;

            // Generate signals based on price position relative to both VWAPs
            if (currentPrice > weeklyVwap.Value && currentPrice > monthlyVwap.Value)
            {
                return "BUY";
            }
            else if (currentPrice < weeklyVwap.Value && currentPrice < monthlyVwap.Value)
            {
                return "SELL";
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            // VWAP strategy requires volume data to be valid
            return indicators != null && 
                   indicators.ContainsKey("Volume") && 
                   indicators["Volume"] > 0;
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "Volume" };

        public override double GetStopLossPercentage()
        {
            // Default stop loss at 2% (can be adjusted based on VWAP distances)
            return 0.02 * (1 + RiskLevel);
        }

        public override double GetTakeProfitPercentage()
        {
            // Take profit at 2:1 risk-reward ratio
            return GetStopLossPercentage() * 2;
        }

        private double? CalculateVwap(List<HistoricalPrice> prices)
        {
            if (!prices.Any())
                return null;

            double cumulativeTPV = 0; // Total Price * Volume
            double cumulativeVolume = 0;

            foreach (var price in prices)
            {
                // Typical price = (High + Low + Close) / 3
                var typicalPrice = (price.High + price.Low + price.Close) / 3;
                cumulativeTPV += typicalPrice * price.Volume;
                cumulativeVolume += price.Volume;
            }

            if (cumulativeVolume == 0)
                return null;

            return cumulativeTPV / cumulativeVolume;
        }
    }
}