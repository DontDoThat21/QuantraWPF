using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Models;

namespace Quantra.Utilities
{
    /// <summary>
    /// Utility class for analyzing and identifying support and resistance levels
    /// using various techniques including price action, pivot points, Fibonacci retracements,
    /// and volume-based analysis.
    /// </summary>
    public class PriceLevelAnalyzer
    {
        /// <summary>
        /// Represents a detected price level (support or resistance)
        /// </summary>
        public class PriceLevel
        {
            /// <summary>
            /// The price level value
            /// </summary>
            public double Price { get; set; }

            /// <summary>
            /// Number of times price has touched this level
            /// </summary>
            public int TouchCount { get; set; }

            /// <summary>
            /// Whether this level acts as support
            /// </summary>
            public bool IsSupport { get; set; }

            /// <summary>
            /// Whether this level acts as resistance
            /// </summary>
            public bool IsResistance { get; set; }

            /// <summary>
            /// Date of the last price touch
            /// </summary>
            public DateTime LastTouch { get; set; }

            /// <summary>
            /// Strength of the level (0-1, with 1 being strongest)
            /// </summary>
            public double Strength { get; set; }

            /// <summary>
            /// Method used to detect this level
            /// </summary>
            public LevelDetectionMethod DetectionMethod { get; set; }

            /// <summary>
            /// Additional details about this level (e.g., "Fibonacci 0.618", "Daily Pivot S1")
            /// </summary>
            public string Description { get; set; }
        }

        /// <summary>
        /// Method used to detect a price level
        /// </summary>
        public enum LevelDetectionMethod
        {
            PriceAction,     // Swing highs/lows
            PivotPoint,      // Standard pivot points
            Fibonacci,       // Fibonacci retracement/extension
            VolumeProfile    // Volume-based support/resistance
        }

        // Common settings
        private int _lookbackPeriods = 100;
        private int _minTouchesToConfirm = 2;
        private double _levelTolerance = 0.5; // % of price

        /// <summary>
        /// Constructor with default settings
        /// </summary>
        public PriceLevelAnalyzer()
        {
        }

        /// <summary>
        /// Constructor with custom settings
        /// </summary>
        public PriceLevelAnalyzer(int lookbackPeriods, int minTouchesToConfirm, double levelTolerance)
        {
            _lookbackPeriods = lookbackPeriods;
            _minTouchesToConfirm = minTouchesToConfirm;
            _levelTolerance = levelTolerance;
        }

        /// <summary>
        /// Detect all support and resistance levels using multiple methods
        /// </summary>
        /// <param name="prices">Historical price data</param>
        /// <param name="currentIndex">Index for current analysis point</param>
        /// <returns>List of detected price levels</returns>
        public List<PriceLevel> DetectAllLevels(List<HistoricalPrice> prices, int currentIndex)
        {
            List<PriceLevel> allLevels = new List<PriceLevel>();

            // Detect levels using different methods
            allLevels.AddRange(DetectPriceActionLevels(prices, currentIndex));
            allLevels.AddRange(CalculatePivotPoints(prices, currentIndex));
            allLevels.AddRange(CalculateFibonacciLevels(prices, currentIndex));
            allLevels.AddRange(DetectVolumeLevels(prices, currentIndex));

            // Group similar levels to avoid duplicates
            allLevels = GroupSimilarLevels(prices, allLevels);

            // Sort by strength and importance
            return allLevels.OrderByDescending(l => l.Strength)
                           .Take(10) // Limit to most significant levels
                           .ToList();
        }

        /// <summary>
        /// Detect support and resistance levels based on price action (swing highs and lows)
        /// </summary>
        public List<PriceLevel> DetectPriceActionLevels(List<HistoricalPrice> prices, int currentIndex)
        {
            // Calculate the start index for analysis
            int startIndex = Math.Max(0, currentIndex - _lookbackPeriods);

            // Extract price swings (local highs and lows)
            var swingHighs = new List<int>();
            var swingLows = new List<int>();

            // Simple swing detection - look for local peaks and valleys
            // A point is a swing high if it's higher than N points on either side
            const int swingWindow = 3;

            for (int i = startIndex + swingWindow; i < currentIndex - swingWindow; i++)
            {
                bool isSwingHigh = true;
                bool isSwingLow = true;

                for (int j = i - swingWindow; j <= i + swingWindow; j++)
                {
                    if (j == i) continue;

                    if (prices[j].High >= prices[i].High)
                        isSwingHigh = false;

                    if (prices[j].Low <= prices[i].Low)
                        isSwingLow = false;
                }

                if (isSwingHigh) swingHighs.Add(i);
                if (isSwingLow) swingLows.Add(i);
            }

            // Group similar price levels (within tolerance)
            var levels = new List<PriceLevel>();

            // Process swing highs to find resistance levels
            GroupPriceLevels(prices, swingHighs, levels, false, true, LevelDetectionMethod.PriceAction);

            // Process swing lows to find support levels
            GroupPriceLevels(prices, swingLows, levels, true, false, LevelDetectionMethod.PriceAction);

            // Add descriptions
            foreach (var level in levels)
            {
                level.Description = level.IsSupport ? "Swing Low Support" : "Swing High Resistance";
            }

            return levels;
        }

        /// <summary>
        /// Calculate pivot points (standard, Woodie's, and Camarilla)
        /// </summary>
        public List<PriceLevel> CalculatePivotPoints(List<HistoricalPrice> prices, int currentIndex)
        {
            var pivotLevels = new List<PriceLevel>();

            // Need at least one complete session of data
            if (prices == null || currentIndex < 1 || currentIndex >= prices.Count)
                return pivotLevels;

            // Get previous session data
            HistoricalPrice previousSession = prices[currentIndex - 1];

            double high = previousSession.High;
            double low = previousSession.Low;
            double close = previousSession.Close;
            double open = previousSession.Open;

            // Calculate standard pivot points
            double pp = (high + low + close) / 3.0; // Pivot Point

            // Support levels
            double s1 = (2.0 * pp) - high;
            double s2 = pp - (high - low);
            double s3 = low - 2.0 * (high - pp);

            // Resistance levels
            double r1 = (2.0 * pp) - low;
            double r2 = pp + (high - low);
            double r3 = high + 2.0 * (pp - low);

            // Add standard pivot points to the list
            pivotLevels.Add(new PriceLevel
            {
                Price = pp,
                IsSupport = true,
                IsResistance = true,
                TouchCount = 0,
                Strength = 0.8,
                LastTouch = DateTime.Now,
                DetectionMethod = LevelDetectionMethod.PivotPoint,
                Description = "Pivot Point (PP)"
            });

            // Add support levels
            pivotLevels.Add(CreatePivotLevel(s1, true, false, 0.7, "Support 1 (S1)"));
            pivotLevels.Add(CreatePivotLevel(s2, true, false, 0.6, "Support 2 (S2)"));
            pivotLevels.Add(CreatePivotLevel(s3, true, false, 0.5, "Support 3 (S3)"));

            // Add resistance levels
            pivotLevels.Add(CreatePivotLevel(r1, false, true, 0.7, "Resistance 1 (R1)"));
            pivotLevels.Add(CreatePivotLevel(r2, false, true, 0.6, "Resistance 2 (R2)"));
            pivotLevels.Add(CreatePivotLevel(r3, false, true, 0.5, "Resistance 3 (R3)"));

            // Calculate Woodie's pivot points
            double woodiesPP = (high + low + 2 * open) / 4.0;
            double woodiesR1 = (2 * woodiesPP) - low;
            double woodiesR2 = woodiesPP + high - low;
            double woodiesS1 = (2 * woodiesPP) - high;
            double woodiesS2 = woodiesPP - high + low;

            // Add Woodie's pivot points
            pivotLevels.Add(CreatePivotLevel(woodiesPP, true, true, 0.75, "Woodie's PP"));
            pivotLevels.Add(CreatePivotLevel(woodiesR1, false, true, 0.65, "Woodie's R1"));
            pivotLevels.Add(CreatePivotLevel(woodiesR2, false, true, 0.55, "Woodie's R2"));
            pivotLevels.Add(CreatePivotLevel(woodiesS1, true, false, 0.65, "Woodie's S1"));
            pivotLevels.Add(CreatePivotLevel(woodiesS2, true, false, 0.55, "Woodie's S2"));

            // Calculate Camarilla pivot points
            double camarillaR4 = close + (high - low) * 1.1 / 2;
            double camarillaR3 = close + (high - low) * 1.1 / 4;
            double camarillaR2 = close + (high - low) * 1.1 / 6;
            double camarillaR1 = close + (high - low) * 1.1 / 12;
            double camarillaS1 = close - (high - low) * 1.1 / 12;
            double camarillaS2 = close - (high - low) * 1.1 / 6;
            double camarillaS3 = close - (high - low) * 1.1 / 4;
            double camarillaS4 = close - (high - low) * 1.1 / 2;

            // Add Camarilla pivot points
            pivotLevels.Add(CreatePivotLevel(camarillaR3, false, true, 0.7, "Camarilla R3"));
            pivotLevels.Add(CreatePivotLevel(camarillaR2, false, true, 0.6, "Camarilla R2"));
            pivotLevels.Add(CreatePivotLevel(camarillaR1, false, true, 0.5, "Camarilla R1"));
            pivotLevels.Add(CreatePivotLevel(camarillaS1, true, false, 0.5, "Camarilla S1"));
            pivotLevels.Add(CreatePivotLevel(camarillaS2, true, false, 0.6, "Camarilla S2"));
            pivotLevels.Add(CreatePivotLevel(camarillaS3, true, false, 0.7, "Camarilla S3"));

            return pivotLevels;
        }

        /// <summary>
        /// Calculate Fibonacci retracement levels
        /// </summary>
        public List<PriceLevel> CalculateFibonacciLevels(List<HistoricalPrice> prices, int currentIndex)
        {
            var fibLevels = new List<PriceLevel>();

            // Need enough data for analysis
            if (prices == null || prices.Count < 20 || currentIndex < 10)
                return fibLevels;

            // Define Fibonacci ratios
            double[] fibRatios = { 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0 };

            // Find swing high and low for retracement calculation
            int lookbackWindow = Math.Min(100, currentIndex);
            var recentPrices = prices.Skip(currentIndex - lookbackWindow).Take(lookbackWindow).ToList();

            // Determine trend direction
            double firstHalf = recentPrices.Take(recentPrices.Count / 2).Average(p => p.Close);
            double secondHalf = recentPrices.Skip(recentPrices.Count / 2).Average(p => p.Close);
            bool isUptrend = secondHalf > firstHalf;

            // Get swing high and low
            double swingHigh = recentPrices.Max(p => p.High);
            double swingLow = recentPrices.Min(p => p.Low);

            // Range for calculating retracements
            double range = swingHigh - swingLow;

            foreach (double ratio in fibRatios)
            {
                double levelPrice;
                bool isSupport, isResistance;
                string description;

                if (isUptrend)
                {
                    // In uptrend, Fibonacci levels act as support
                    // Level = High - Range * Ratio
                    levelPrice = swingHigh - (range * ratio);
                    isSupport = true;
                    isResistance = false;
                    description = $"Fib {ratio:P0} Support";
                }
                else
                {
                    // In downtrend, Fibonacci levels act as resistance
                    // Level = Low + Range * Ratio
                    levelPrice = swingLow + (range * ratio);
                    isSupport = false;
                    isResistance = true;
                    description = $"Fib {ratio:P0} Resistance";
                }

                // Assign strength based on the Fibonacci ratio
                // The main ratios (0.382, 0.5, 0.618) are considered stronger
                double strength;
                if (ratio == 0.382 || ratio == 0.618)
                    strength = 0.9;
                else if (ratio == 0.5)
                    strength = 0.85;
                else if (ratio == 0.236 || ratio == 0.786)
                    strength = 0.7;
                else
                    strength = 0.5;

                fibLevels.Add(new PriceLevel
                {
                    Price = levelPrice,
                    IsSupport = isSupport,
                    IsResistance = isResistance,
                    TouchCount = 0,  // Fibonacci levels don't track touches directly
                    Strength = strength,
                    LastTouch = DateTime.Now,
                    DetectionMethod = LevelDetectionMethod.Fibonacci,
                    Description = description
                });
            }

            return fibLevels;
        }

        /// <summary>
        /// Detect support and resistance levels based on volume profile
        /// </summary>
        public List<PriceLevel> DetectVolumeLevels(List<HistoricalPrice> prices, int currentIndex)
        {
            var volumeLevels = new List<PriceLevel>();

            // Need enough data for volume analysis
            if (prices == null || currentIndex < 30)
                return volumeLevels;

            // Calculate the start index for analysis
            int startIndex = Math.Max(0, currentIndex - _lookbackPeriods);

            // Extract relevant portion of price data
            var priceSeries = prices.Skip(startIndex).Take(currentIndex - startIndex + 1).ToList();

            // Get min and max prices
            double minPrice = priceSeries.Min(p => p.Low);
            double maxPrice = priceSeries.Max(p => p.High);

            // Create price bins for volume profile
            int numBins = 100;
            double binSize = (maxPrice - minPrice) / numBins;
            var volumeProfile = new Dictionary<int, double>();

            // Initialize bins
            for (int i = 0; i < numBins; i++)
            {
                volumeProfile[i] = 0;
            }

            // Distribute volume across price bins
            foreach (var candle in priceSeries)
            {
                // Calculate fraction of candle in each bin
                double range = candle.High - candle.Low;
                if (range <= 0) continue; // Skip invalid candles

                for (int bin = 0; bin < numBins; bin++)
                {
                    double binLow = minPrice + bin * binSize;
                    double binHigh = binLow + binSize;

                    // Calculate overlap between candle and bin
                    double overlapLow = Math.Max(candle.Low, binLow);
                    double overlapHigh = Math.Min(candle.High, binHigh);

                    if (overlapHigh > overlapLow)
                    {
                        // Allocate volume proportionally to the overlap
                        double overlapRatio = (overlapHigh - overlapLow) / range;
                        volumeProfile[bin] += candle.Volume * overlapRatio;
                    }
                }
            }

            // Calculate average volume and standard deviation
            double avgVolume = volumeProfile.Values.Average();
            double stdDev = Math.Sqrt(volumeProfile.Values.Average(v => Math.Pow(v - avgVolume, 2)));

            // Find high volume nodes
            for (int bin = 1; bin < numBins - 1; bin++)
            {
                double binVolume = volumeProfile[bin];

                // Check if this is a high volume node (> 1.5 standard deviations above average)
                if (binVolume > avgVolume + 1.5 * stdDev)
                {
                    // Check if it's a local maximum
                    if (binVolume > volumeProfile[bin - 1] && binVolume > volumeProfile[bin + 1])
                    {
                        // Calculate price at this volume node
                        double nodePrice = minPrice + (bin + 0.5) * binSize;

                        // Determine if this level acted as support, resistance, or both
                        bool actedAsSupport = false;
                        bool actedAsResistance = false;

                        for (int i = startIndex + 10; i < currentIndex; i++)
                        {
                            double priceTolerance = nodePrice * _levelTolerance / 100;

                            // Check if price bounced off this level
                            if (Math.Abs(prices[i].Low - nodePrice) <= priceTolerance &&
                                prices[i + 1].Close > nodePrice)
                            {
                                actedAsSupport = true;
                            }

                            if (Math.Abs(prices[i].High - nodePrice) <= priceTolerance &&
                                prices[i + 1].Close < nodePrice)
                            {
                                actedAsResistance = true;
                            }
                        }

                        // Calculate strength based on volume intensity
                        double volumeIntensity = (binVolume - avgVolume) / stdDev;
                        double strength = Math.Min(1.0, 0.5 + (volumeIntensity / 10.0));

                        // Add volume-based level
                        volumeLevels.Add(new PriceLevel
                        {
                            Price = nodePrice,
                            IsSupport = actedAsSupport,
                            IsResistance = actedAsResistance,
                            TouchCount = 0,
                            Strength = strength,
                            LastTouch = DateTime.Now,
                            DetectionMethod = LevelDetectionMethod.VolumeProfile,
                            Description = "High Volume Node"
                        });
                    }
                }
            }

            // If no strong support/resistance is found, use POC (Point of Control)
            if (volumeLevels.Count == 0)
            {
                // Find bin with maximum volume
                int maxVolumeBin = volumeProfile.OrderByDescending(kv => kv.Value).First().Key;
                double pocPrice = minPrice + (maxVolumeBin + 0.5) * binSize;

                volumeLevels.Add(new PriceLevel
                {
                    Price = pocPrice,
                    IsSupport = true,
                    IsResistance = true,
                    TouchCount = 0,
                    Strength = 0.7,
                    LastTouch = DateTime.Now,
                    DetectionMethod = LevelDetectionMethod.VolumeProfile,
                    Description = "Point of Control (POC)"
                });
            }

            return volumeLevels;
        }

        /// <summary>
        /// Group similar price levels to avoid duplicates
        /// </summary>
        private List<PriceLevel> GroupSimilarLevels(List<HistoricalPrice> prices, List<PriceLevel> levels)
        {
            if (levels.Count <= 1)
                return levels;

            // Calculate average price for tolerance calculation
            double avgPrice = 0;
            for (int i = Math.Max(0, prices.Count - _lookbackPeriods); i < prices.Count; i++)
            {
                avgPrice += prices[i].Close;
            }
            avgPrice /= Math.Min(prices.Count, _lookbackPeriods);

            // Calculate absolute price tolerance
            double absoluteTolerance = avgPrice * (_levelTolerance / 100.0);

            // Sort levels by price to make grouping easier
            var sortedLevels = levels.OrderBy(l => l.Price).ToList();
            var groupedLevels = new List<PriceLevel>();

            // Current group and its average
            List<PriceLevel> currentGroup = new List<PriceLevel>();

            foreach (var level in sortedLevels)
            {
                if (currentGroup.Count == 0)
                {
                    currentGroup.Add(level);
                }
                else
                {
                    double groupAvg = currentGroup.Average(l => l.Price);

                    if (Math.Abs(level.Price - groupAvg) <= absoluteTolerance)
                    {
                        // Add to current group
                        currentGroup.Add(level);
                    }
                    else
                    {
                        // Process current group
                        groupedLevels.Add(MergeGroupedLevels(currentGroup));

                        // Start new group
                        currentGroup.Clear();
                        currentGroup.Add(level);
                    }
                }
            }

            // Process last group if not empty
            if (currentGroup.Count > 0)
            {
                groupedLevels.Add(MergeGroupedLevels(currentGroup));
            }

            return groupedLevels;
        }

        /// <summary>
        /// Merge a group of similar price levels
        /// </summary>
        private PriceLevel MergeGroupedLevels(List<PriceLevel> group)
        {
            // Calculate weighted average price
            double totalStrength = group.Sum(l => l.Strength);
            double weightedPrice = group.Sum(l => l.Price * l.Strength) / totalStrength;

            // Create merged level
            var mergedLevel = new PriceLevel
            {
                Price = weightedPrice,
                IsSupport = group.Any(l => l.IsSupport),
                IsResistance = group.Any(l => l.IsResistance),
                TouchCount = group.Sum(l => l.TouchCount),
                Strength = Math.Min(1.0, group.Max(l => l.Strength) * (1.0 + 0.1 * (group.Count - 1))),
                LastTouch = group.Max(l => l.LastTouch),
                DetectionMethod = group.OrderByDescending(l => l.Strength).First().DetectionMethod
            };

            // Build description
            var descriptions = group.Select(l => l.DetectionMethod +
                                            (string.IsNullOrEmpty(l.Description) ? "" : ": " + l.Description))
                                   .Distinct()
                                   .Take(2);
            mergedLevel.Description = string.Join(" + ", descriptions);

            return mergedLevel;
        }

        /// <summary>
        /// Group similar price points into support/resistance levels
        /// </summary>
        private void GroupPriceLevels(List<HistoricalPrice> prices, List<int> swingIndexes,
                                     List<PriceLevel> levels, bool isSupport, bool isResistance,
                                     LevelDetectionMethod method)
        {
            // Calculate average price for tolerance calculation
            double avgPrice = 0;
            for (int i = Math.Max(0, prices.Count - _lookbackPeriods); i < prices.Count; i++)
            {
                avgPrice += prices[i].Close;
            }
            avgPrice /= Math.Min(prices.Count, _lookbackPeriods);

            // Calculate absolute price tolerance
            double absoluteTolerance = avgPrice * (_levelTolerance / 100);

            foreach (int swingIndex in swingIndexes)
            {
                double price = isSupport ? prices[swingIndex].Low : prices[swingIndex].High;

                // Check if this price is near an existing level
                var existingLevel = levels.FirstOrDefault(l =>
                    Math.Abs(l.Price - price) <= absoluteTolerance);

                if (existingLevel != null)
                {
                    // Update existing level
                    existingLevel.TouchCount++;
                    existingLevel.LastTouch = prices[swingIndex].Date;
                    existingLevel.IsSupport |= isSupport;
                    existingLevel.IsResistance |= isResistance;

                    // Update price to be the average
                    existingLevel.Price = (existingLevel.Price * (existingLevel.TouchCount - 1) + price)
                                        / existingLevel.TouchCount;

                    // Update strength - more touches and recent touches increase strength
                    existingLevel.Strength = CalculateLevelStrength(existingLevel, prices[swingIndex].Date);
                }
                else if (swingIndexes.Count(i => Math.Abs(price - (isSupport ? prices[i].Low : prices[i].High)) <= absoluteTolerance) >= _minTouchesToConfirm)
                {
                    // Create new level if it has enough touches
                    levels.Add(new PriceLevel
                    {
                        Price = price,
                        TouchCount = 1,
                        IsSupport = isSupport,
                        IsResistance = isResistance,
                        LastTouch = prices[swingIndex].Date,
                        Strength = 0.5, // Initial strength
                        DetectionMethod = method
                    });
                }
            }
        }

        /// <summary>
        /// Helper method to create a pivot point level
        /// </summary>
        private PriceLevel CreatePivotLevel(double price, bool isSupport, bool isResistance,
                                          double strength, string description)
        {
            return new PriceLevel
            {
                Price = price,
                IsSupport = isSupport,
                IsResistance = isResistance,
                TouchCount = 0,
                Strength = strength,
                LastTouch = DateTime.Now,
                DetectionMethod = LevelDetectionMethod.PivotPoint,
                Description = description
            };
        }

        /// <summary>
        /// Calculate strength of a support/resistance level
        /// </summary>
        private double CalculateLevelStrength(PriceLevel level, DateTime currentDate)
        {
            // Base strength from number of touches (0.3-0.8)
            double touchStrength = Math.Min(0.8, 0.3 + (level.TouchCount * 0.1));

            // Recency factor (more recent levels are stronger)
            TimeSpan timeSinceTouch = currentDate - level.LastTouch;
            double recencyStrength = 1.0 - Math.Min(0.5, (timeSinceTouch.TotalDays / 90)); // Decay over 90 days

            // Combined strength
            return touchStrength * recencyStrength;
        }
    }
}