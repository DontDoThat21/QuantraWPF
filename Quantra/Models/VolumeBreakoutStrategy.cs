using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Services;

namespace Quantra.Models
{
    public class VolumeBreakoutStrategy : StrategyProfile
    {
        private const int VolumeLookbackPeriod = 20; // For average volume calculation
        private const int PriceLookbackPeriod = 20;  // For price range calculation
        private const double VolumeSurgeThreshold = 2.0; // Volume must be 2x average
        private const double BreakoutThreshold = 0.02; // 2% price movement for breakout
        
        public VolumeBreakoutStrategy()
        {
            Name = "Volume Breakout";
            Description = "Identifies breakouts based on volume spikes and price movement";
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "VOLUME", "PRICE", "VWAP" };

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (!indicators.TryGetValue("VOLUME", out double volume) ||
                !indicators.TryGetValue("PRICE", out double price) ||
                !indicators.TryGetValue("VWAP", out double vwap))
                return false;

            return true; // Basic validation passes if we have all required indicators
        }

        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            if (historical == null || !historical.Any())
                return null;

            var actualIndex = index ?? historical.Count - 1;
            if (actualIndex < VolumeLookbackPeriod + 1) 
                return null;

            var prices = historical.Take(actualIndex + 1).ToList();
            var currentBar = prices[actualIndex];
            var volume = currentBar.Volume;
            var close = currentBar.Close;
            var open = currentBar.Open;

            // Calculate average volume
            var avgVolume = CalculateAverageVolume(prices);
            
            // Check for volume breakout
            var volumeRatio = volume / avgVolume;
            if (volumeRatio < VolumeSurgeThreshold) 
                return null;

            // Calculate price change
            var priceChange = (close - open) / open;

            // Generate signal based on price direction with volume confirmation
            if (Math.Abs(priceChange) >= BreakoutThreshold)
            {
                return priceChange > 0 ? "BUY" : "SELL";
            }

            return null;
        }

        public override double GetStopLossPercentage()
        {
            // Default to the breakout threshold
            return BreakoutThreshold;
        }

        public override double GetTakeProfitPercentage()
        {
            // 2:1 risk-reward ratio
            return BreakoutThreshold * 2;
        }

        private double CalculateAverageVolume(List<HistoricalPrice> prices)
        {
            return prices
                .Skip(prices.Count - VolumeLookbackPeriod - 1)  // Skip current candle
                .Take(VolumeLookbackPeriod)
                .Average(p => p.Volume);
        }

        private (double upper, double lower) CalculatePriceChannels(List<HistoricalPrice> prices)
        {
            var lookbackPrices = prices
                .Skip(prices.Count - PriceLookbackPeriod - 1)  // Exclude current candle
                .Take(PriceLookbackPeriod)
                .ToList();

            double upper = lookbackPrices.Max(p => p.High);
            double lower = lookbackPrices.Min(p => p.Low);

            return (upper, lower);
        }

        private bool IsUpwardBreakout(HistoricalPrice current, HistoricalPrice previous, double upperChannel)
        {
            // Price breaks above the upper channel
            bool breaksResistance = current.Close > upperChannel;
            
            // Confirmation: Strong closing above previous high
            bool strongClose = current.Close > previous.High && 
                             (current.Close - current.Open) / current.Open > BreakoutThreshold;

            return breaksResistance && strongClose;
        }

        private bool IsDownwardBreakout(HistoricalPrice current, HistoricalPrice previous, double lowerChannel)
        {
            // Price breaks below the lower channel
            bool breaksSupport = current.Close < lowerChannel;
            
            // Confirmation: Strong closing below previous low
            bool strongClose = current.Close < previous.Low &&
                             (current.Open - current.Close) / current.Open > BreakoutThreshold;

            return breaksSupport && strongClose;
        }

        private double CalculateWeeklyTrend(List<HistoricalPrice> prices)
        {
            // Use last 5 candles (representing a week) for trend
            var weekPrices = prices.Skip(prices.Count - 5).Take(5).ToList();
            
            if (weekPrices.Count < 5)
                return 0;

            // Calculate weekly trend using linear regression slope
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            int n = weekPrices.Count;

            for (int i = 0; i < n; i++)
            {
                double x = i;
                double y = weekPrices[i].Close;
                
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }

            // Calculate slope
            double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            return slope;
        }
    }
}