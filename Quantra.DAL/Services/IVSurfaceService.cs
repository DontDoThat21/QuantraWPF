using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for building and analyzing implied volatility surfaces
    /// </summary>
    public class IVSurfaceService
    {
        private readonly LoggingService _loggingService;

        public IVSurfaceService(LoggingService loggingService)
        {
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Builds a 3D implied volatility surface from options chain data
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="optionsChain">Complete options chain</param>
        /// <returns>IV surface data structure</returns>
        public async Task<IVSurfaceData> BuildIVSurfaceAsync(string symbol, List<OptionData> optionsChain)
        {
            if (optionsChain == null || optionsChain.Count == 0)
            {
                _loggingService.Log("Warning", $"No options chain data provided for {symbol}");
                return null;
            }

            return await Task.Run(() =>
            {
                var surface = new IVSurfaceData
                {
                    Symbol = symbol,
                    GeneratedAt = DateTime.Now,
                    DataPoints = new List<IVDataPoint>()
                };

                // Group by expiration and strike
                var grouped = optionsChain
                    .Where(o => o.ImpliedVolatility > 0)
                    .GroupBy(o => o.ExpirationDate)
                    .OrderBy(g => g.Key)
                    .ToList();

                foreach (var expirationGroup in grouped)
                {
                    var expiration = expirationGroup.Key;
                    var daysToExpiration = (expiration - DateTime.Now).TotalDays;

                    // Create IV points for each strike
                    var strikes = expirationGroup
                        .Select(o => new
                        {
                            Strike = o.StrikePrice,
                            IV = o.ImpliedVolatility,
                            Type = o.OptionType
                        })
                        .OrderBy(x => x.Strike)
                        .ToList();

                    foreach (var strike in strikes)
                    {
                        surface.DataPoints.Add(new IVDataPoint
                        {
                            Strike = strike.Strike,
                            DaysToExpiration = daysToExpiration,
                            Expiration = expiration,
                            ImpliedVolatility = strike.IV,
                            OptionType = strike.Type
                        });
                    }
                }

                _loggingService.Log("Info", $"Built IV surface for {symbol} with {surface.DataPoints.Count} points");
                return surface;
            });
        }

        /// <summary>
        /// Interpolates IV for a missing strike/expiration combination
        /// Uses bilinear interpolation
        /// </summary>
        /// <param name="strike">Strike price</param>
        /// <param name="expiration">Expiration date</param>
        /// <param name="chain">Options chain data</param>
        /// <returns>Interpolated IV value</returns>
        public async Task<double> GetInterpolatedIVAsync(
            double strike, 
            DateTime expiration, 
            List<OptionData> chain)
        {
            return await Task.Run(() =>
            {
                var daysToExp = (expiration - DateTime.Now).TotalDays;

                // Find nearby strikes and expirations
                var nearbyOptions = chain
                    .Where(o => o.ImpliedVolatility > 0)
                    .OrderBy(o => Math.Abs(o.StrikePrice - strike) + 
                                  Math.Abs((o.ExpirationDate - expiration).TotalDays))
                    .Take(4)
                    .ToList();

                if (nearbyOptions.Count == 0)
                    return 0;

                // Simple weighted average based on distance
                double totalWeight = 0;
                double weightedIV = 0;

                foreach (var opt in nearbyOptions)
                {
                    var strikeDiff = Math.Abs(opt.StrikePrice - strike);
                    var daysDiff = Math.Abs((opt.ExpirationDate - expiration).TotalDays);
                    var distance = Math.Sqrt(strikeDiff * strikeDiff + daysDiff * daysDiff);
                    
                    if (distance < 0.01) // Very close match
                        return opt.ImpliedVolatility;

                    var weight = 1.0 / (distance + 0.1); // Add small constant to avoid division by zero
                    weightedIV += opt.ImpliedVolatility * weight;
                    totalWeight += weight;
                }

                return totalWeight > 0 ? weightedIV / totalWeight : 0;
            });
        }

        /// <summary>
        /// Analyzes IV skew patterns (volatility smile)
        /// </summary>
        /// <param name="optionsChain">Options chain data</param>
        /// <returns>IV skew metrics</returns>
        public async Task<IVSkewMetrics> AnalyzeIVSkewAsync(List<OptionData> optionsChain)
        {
            return await Task.Run(() =>
            {
                if (optionsChain == null || optionsChain.Count == 0)
                    return null;

                var metrics = new IVSkewMetrics();

                // Group by expiration, analyze the nearest one
                var nearestExpiration = optionsChain
                    .Where(o => o.ExpirationDate > DateTime.Now)
                    .OrderBy(o => o.ExpirationDate)
                    .FirstOrDefault()?.ExpirationDate;

                if (!nearestExpiration.HasValue)
                    return metrics;

                var nearTermOptions = optionsChain
                    .Where(o => o.ExpirationDate == nearestExpiration.Value && o.ImpliedVolatility > 0)
                    .ToList();

                if (nearTermOptions.Count == 0)
                    return metrics;

                // Calculate ATM IV (at-the-money)
                var atmStrike = nearTermOptions
                    .OrderBy(o => Math.Abs(o.StrikePrice - o.LastPrice))
                    .FirstOrDefault()?.StrikePrice ?? 0;

                metrics.ATMVolatility = nearTermOptions
                    .Where(o => Math.Abs(o.StrikePrice - atmStrike) < 0.01)
                    .Average(o => o.ImpliedVolatility);

                // Calculate skew (put IV - call IV at OTM strikes)
                var calls = nearTermOptions.Where(o => o.OptionType == "CALL").ToList();
                var puts = nearTermOptions.Where(o => o.OptionType == "PUT").ToList();

                if (calls.Any() && puts.Any())
                {
                    // OTM call IV (above ATM)
                    var otmCalls = calls.Where(o => o.StrikePrice > atmStrike).ToList();
                    var otmPuts = puts.Where(o => o.StrikePrice < atmStrike).ToList();

                    if (otmCalls.Any() && otmPuts.Any())
                    {
                        var avgCallIV = otmCalls.Average(o => o.ImpliedVolatility);
                        var avgPutIV = otmPuts.Average(o => o.ImpliedVolatility);
                        
                        metrics.Skew = avgPutIV - avgCallIV;
                        metrics.SkewDirection = metrics.Skew > 0 ? "Put Skew" : "Call Skew";
                    }
                }

                // Calculate term structure (near vs far dated)
                var termStructure = optionsChain
                    .GroupBy(o => o.ExpirationDate)
                    .Where(g => g.Any(o => o.ImpliedVolatility > 0))
                    .OrderBy(g => g.Key)
                    .Select(g => new
                    {
                        Expiration = g.Key,
                        AvgIV = g.Average(o => o.ImpliedVolatility)
                    })
                    .ToList();

                if (termStructure.Count >= 2)
                {
                    metrics.TermStructure = termStructure.Last().AvgIV - termStructure.First().AvgIV;
                    metrics.TermStructureShape = metrics.TermStructure > 0 ? "Upward Sloping" : "Inverted";
                }

                _loggingService.Log("Info", $"Analyzed IV skew: ATM={metrics.ATMVolatility:F2}, Skew={metrics.Skew:F4}");
                return metrics;
            });
        }

        /// <summary>
        /// Compares current IV to historical levels
        /// </summary>
        /// <param name="currentIV">Current implied volatility</param>
        /// <param name="historicalIVs">List of historical IV values</param>
        /// <returns>IV percentile and comparison metrics</returns>
        public IVHistoricalComparison CompareToHistoricalIV(double currentIV, List<double> historicalIVs)
        {
            if (historicalIVs == null || historicalIVs.Count == 0)
                return null;

            var sorted = historicalIVs.OrderBy(iv => iv).ToList();
            var rank = sorted.Count(iv => iv <= currentIV);
            var percentile = (double)rank / sorted.Count * 100;

            return new IVHistoricalComparison
            {
                CurrentIV = currentIV,
                IVPercentile = percentile,
                HistoricalMean = historicalIVs.Average(),
                HistoricalStdDev = CalculateStdDev(historicalIVs),
                IsHighIV = percentile > 75,
                IsLowIV = percentile < 25
            };
        }

        private double CalculateStdDev(List<double> values)
        {
            if (values.Count < 2)
                return 0;

            var mean = values.Average();
            var squaredDiffs = values.Select(v => Math.Pow(v - mean, 2));
            return Math.Sqrt(squaredDiffs.Average());
        }
    }

    /// <summary>
    /// Represents a 3D implied volatility surface
    /// </summary>
    public class IVSurfaceData
    {
        public string Symbol { get; set; }
        public DateTime GeneratedAt { get; set; }
        public List<IVDataPoint> DataPoints { get; set; }
    }

    /// <summary>
    /// Individual point on the IV surface
    /// </summary>
    public class IVDataPoint
    {
        public double Strike { get; set; }
        public double DaysToExpiration { get; set; }
        public DateTime Expiration { get; set; }
        public double ImpliedVolatility { get; set; }
        public string OptionType { get; set; }
    }

    /// <summary>
    /// IV skew analysis metrics
    /// </summary>
    public class IVSkewMetrics
    {
        public double ATMVolatility { get; set; }
        public double Skew { get; set; }
        public string SkewDirection { get; set; }
        public double TermStructure { get; set; }
        public string TermStructureShape { get; set; }
    }

    /// <summary>
    /// Historical IV comparison metrics
    /// </summary>
    public class IVHistoricalComparison
    {
        public double CurrentIV { get; set; }
        public double IVPercentile { get; set; }
        public double HistoricalMean { get; set; }
        public double HistoricalStdDev { get; set; }
        public bool IsHighIV { get; set; }
        public bool IsLowIV { get; set; }
    }
}
