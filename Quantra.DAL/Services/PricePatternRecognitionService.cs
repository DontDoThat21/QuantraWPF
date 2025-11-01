using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for detecting common price action patterns in stock data
    /// </summary>
    public class PricePatternRecognitionService
    {
        private readonly StockDataCacheService _stockDataService;
        
        public PricePatternRecognitionService(StockDataCacheService stockDataService)
        {
            _stockDataService = stockDataService ?? throw new ArgumentNullException(nameof(stockDataService));
        }

        /// <summary>
        /// Represents a price pattern detected in the data
        /// </summary>
        public class PricePattern
        {
            /// <summary>
            /// Type of pattern detected
            /// </summary>
            public PatternType Type { get; set; }
            
            /// <summary>
            /// Confidence level of the pattern (0-100)
            /// </summary>
            public double Confidence { get; set; }
            
            /// <summary>
            /// Key price levels in the pattern (e.g., neckline, support/resistance)
            /// </summary>
            public Dictionary<string, double> KeyLevels { get; set; } = new Dictionary<string, double>();
            
            /// <summary>
            /// Bullish or bearish bias of the pattern
            /// </summary>
            public PatternBias Bias { get; set; }
            
            /// <summary>
            /// Start index of the pattern in the price data
            /// </summary>
            public int StartIndex { get; set; }
            
            /// <summary>
            /// End index of the pattern in the price data
            /// </summary>
            public int EndIndex { get; set; }
            
            /// <summary>
            /// Date when the pattern was detected
            /// </summary>
            public DateTime DetectionDate { get; set; }
            
            /// <summary>
            /// Symbol of the stock
            /// </summary>
            public string Symbol { get; set; }
        }
        
        /// <summary>
        /// Types of patterns that can be detected
        /// </summary>
        public enum PatternType
        {
            DoubleTop,
            DoubleBottom,
            HeadAndShoulders,
            InverseHeadAndShoulders,
            AscendingTriangle,
            DescendingTriangle,
            SymmetricalTriangle,
            BullishEngulfing,
            BearishEngulfing,
            BullishHarami,
            BearishHarami,
            CupAndHandle
        }
        
        /// <summary>
        /// Bias of the detected pattern
        /// </summary>
        public enum PatternBias
        {
            Bullish,
            Bearish,
            Neutral
        }

        /// <summary>
        /// Detects all patterns in the given stock's price history
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range (e.g., "1mo", "6mo", "1y")</param>
        /// <param name="interval">Interval between data points (e.g., "1d", "1h")</param>
        /// <returns>List of detected patterns</returns>
        public async Task<List<PricePattern>> DetectAllPatternsAsync(string symbol, string range = "6mo", string interval = "1d")
        {
            var patterns = new List<PricePattern>();
            
            // Get historical price data
            var historicalData = await _stockDataService.GetStockData(symbol, range, interval);
            
            if (historicalData == null || historicalData.Count < 30)
            {
                //DatabaseMonolith.Log("Warning", $"Insufficient historical data for {symbol} to detect patterns");
                return patterns;
            }
            
            // Detect each pattern type
            patterns.AddRange(await DetectDoubleTopsAsync(symbol, historicalData));
            patterns.AddRange(await DetectDoubleBottomsAsync(symbol, historicalData));
            patterns.AddRange(await DetectHeadAndShouldersAsync(symbol, historicalData));
            patterns.AddRange(await DetectInverseHeadAndShouldersAsync(symbol, historicalData));
            patterns.AddRange(await DetectTrianglePatternsAsync(symbol, historicalData));
            patterns.AddRange(await DetectEngulfingPatternsAsync(symbol, historicalData));
            patterns.AddRange(await DetectCupAndHandlePatternsAsync(symbol, historicalData));
            
            // Sort by confidence and limit results
            return patterns.OrderByDescending(p => p.Confidence).Take(5).ToList();
        }

        /// <summary>
        /// Detects double top patterns in price data
        /// </summary>
        public async Task<List<PricePattern>> DetectDoubleTopsAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 20)
                return patterns;
                
            // Parameters for detection
            const double peakProximityPercent = 2.0; // Price difference between tops
            const int minPeakDistance = 5; // Minimum number of bars between peaks
            const int maxPeakDistance = 30; // Maximum number of bars between peaks
            
            // Find peaks in the data
            var peaks = FindPeaks(priceData, 5);
            
            // Look for pairs of similar peaks
            for (int i = 0; i < peaks.Count - 1; i++)
            {
                for (int j = i + 1; j < peaks.Count; j++)
                {
                    int peak1Index = peaks[i];
                    int peak2Index = peaks[j];
                    
                    int distance = peak2Index - peak1Index;
                    if (distance < minPeakDistance || distance > maxPeakDistance)
                        continue;
                    
                    double peak1Price = priceData[peak1Index].High;
                    double peak2Price = priceData[peak2Index].High;
                    
                    // Calculate price difference as percentage
                    double priceDiffPercent = Math.Abs(peak1Price - peak2Price) / peak1Price * 100;
                    
                    if (priceDiffPercent <= peakProximityPercent)
                    {
                        // Look for a trough between the peaks
                        int midpointIndex = (peak1Index + peak2Index) / 2;
                        int rangeStart = peak1Index + 1;
                        int rangeEnd = peak2Index - 1;
                        
                        if (rangeStart >= rangeEnd)
                            continue;
                        
                        int troughIndex = FindLowestLowIndex(priceData, rangeStart, rangeEnd);
                        double troughPrice = priceData[troughIndex].Low;
                        
                        // Check if there's a decline after the second peak
                        bool hasDeclineAfterSecondPeak = false;
                        if (peak2Index < priceData.Count - 1)
                        {
                            double lastPrice = priceData[priceData.Count - 1].Close;
                            hasDeclineAfterSecondPeak = lastPrice < peak2Price * 0.98; // 2% decline
                        }
                        
                        // Calculate confidence based on peak similarity and post-pattern movement
                        double confidenceBase = 100 - priceDiffPercent * 20; // Penalize for peak difference
                        double confidence = hasDeclineAfterSecondPeak ? confidenceBase : confidenceBase * 0.7;
                        
                        // Only add high confidence patterns
                        if (confidence > 60)
                        {
                            patterns.Add(new PricePattern
                            {
                                Type = PatternType.DoubleTop,
                                Confidence = confidence,
                                Bias = PatternBias.Bearish,
                                StartIndex = peak1Index,
                                EndIndex = peak2Index,
                                Symbol = symbol,
                                DetectionDate = DateTime.Now,
                                KeyLevels = new Dictionary<string, double>
                                {
                                    { "FirstPeak", peak1Price },
                                    { "SecondPeak", peak2Price },
                                    { "Neckline", troughPrice },
                                    { "Target", 2 * troughPrice - (peak1Price + peak2Price) / 2 } // Projected target
                                }
                            });
                        }
                    }
                }
            }
            
            return patterns;
        }
        
        /// <summary>
        /// Detects double bottom patterns in price data
        /// </summary>
        public async Task<List<PricePattern>> DetectDoubleBottomsAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 20)
                return patterns;
                
            // Parameters for detection
            const double bottomProximityPercent = 2.0; // Price difference between bottoms
            const int minBottomDistance = 5; // Minimum number of bars between bottoms
            const int maxBottomDistance = 30; // Maximum number of bars between bottoms
            
            // Find troughs in the data
            var troughs = FindTroughs(priceData, 5);
            
            // Look for pairs of similar bottoms
            for (int i = 0; i < troughs.Count - 1; i++)
            {
                for (int j = i + 1; j < troughs.Count; j++)
                {
                    int trough1Index = troughs[i];
                    int trough2Index = troughs[j];
                    
                    int distance = trough2Index - trough1Index;
                    if (distance < minBottomDistance || distance > maxBottomDistance)
                        continue;
                    
                    double trough1Price = priceData[trough1Index].Low;
                    double trough2Price = priceData[trough2Index].Low;
                    
                    // Calculate price difference as percentage
                    double priceDiffPercent = Math.Abs(trough1Price - trough2Price) / trough1Price * 100;
                    
                    if (priceDiffPercent <= bottomProximityPercent)
                    {
                        // Look for a peak between the troughs
                        int midpointIndex = (trough1Index + trough2Index) / 2;
                        int rangeStart = trough1Index + 1;
                        int rangeEnd = trough2Index - 1;
                        
                        if (rangeStart >= rangeEnd)
                            continue;
                        
                        int peakIndex = FindHighestHighIndex(priceData, rangeStart, rangeEnd);
                        double peakPrice = priceData[peakIndex].High;
                        
                        // Check if there's a rally after the second trough
                        bool hasRallyAfterSecondTrough = false;
                        if (trough2Index < priceData.Count - 1)
                        {
                            double lastPrice = priceData[priceData.Count - 1].Close;
                            hasRallyAfterSecondTrough = lastPrice > trough2Price * 1.02; // 2% rally
                        }
                        
                        // Calculate confidence based on trough similarity and post-pattern movement
                        double confidenceBase = 100 - priceDiffPercent * 20; // Penalize for trough difference
                        double confidence = hasRallyAfterSecondTrough ? confidenceBase : confidenceBase * 0.7;
                        
                        // Only add high confidence patterns
                        if (confidence > 60)
                        {
                            patterns.Add(new PricePattern
                            {
                                Type = PatternType.DoubleBottom,
                                Confidence = confidence,
                                Bias = PatternBias.Bullish,
                                StartIndex = trough1Index,
                                EndIndex = trough2Index,
                                Symbol = symbol,
                                DetectionDate = DateTime.Now,
                                KeyLevels = new Dictionary<string, double>
                                {
                                    { "FirstBottom", trough1Price },
                                    { "SecondBottom", trough2Price },
                                    { "Neckline", peakPrice },
                                    { "Target", 2 * peakPrice - (trough1Price + trough2Price) / 2 } // Projected target
                                }
                            });
                        }
                    }
                }
            }
            
            return patterns;
        }
        
        /// <summary>
        /// Detects head and shoulders patterns in price data
        /// </summary>
        public async Task<List<PricePattern>> DetectHeadAndShouldersAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 30)
                return patterns;
            
            // Find peaks in the data
            var peaks = FindPeaks(priceData, 5);
            
            // Need at least 3 peaks for H&S
            if (peaks.Count < 3)
                return patterns;
            
            // Look for triplets of peaks where the middle one is higher
            for (int i = 0; i < peaks.Count - 2; i++)
            {
                int leftShoulderIdx = peaks[i];
                int headIdx = peaks[i + 1];
                int rightShoulderIdx = peaks[i + 2];
                
                // Head should be higher than shoulders
                if (priceData[headIdx].High <= priceData[leftShoulderIdx].High || 
                    priceData[headIdx].High <= priceData[rightShoulderIdx].High)
                    continue;
                
                // Shoulders should be roughly at the same level
                double leftShoulderPrice = priceData[leftShoulderIdx].High;
                double rightShoulderPrice = priceData[rightShoulderIdx].High;
                double shoulderDiffPercent = Math.Abs(leftShoulderPrice - rightShoulderPrice) / leftShoulderPrice * 100;
                
                if (shoulderDiffPercent > 5.0) // Shoulders should be within 5% of each other
                    continue;
                
                // Find the neckline (connect the troughs between the peaks)
                int trough1Idx = FindLowestLowIndex(priceData, leftShoulderIdx, headIdx);
                int trough2Idx = FindLowestLowIndex(priceData, headIdx, rightShoulderIdx);
                
                double trough1Price = priceData[trough1Idx].Low;
                double trough2Price = priceData[trough2Idx].Low;
                
                // Neckline should be relatively flat
                double necklineDiffPercent = Math.Abs(trough1Price - trough2Price) / trough1Price * 100;
                if (necklineDiffPercent > 3.0) // Neckline should be within 3% flat
                    continue;
                
                // Calculate neckline as average of the two troughs
                double necklinePrice = (trough1Price + trough2Price) / 2;
                
                // Check if there's a break below the neckline after the pattern
                bool hasBreakdown = false;
                if (rightShoulderIdx < priceData.Count - 1)
                {
                    double lastPrice = priceData[priceData.Count - 1].Close;
                    hasBreakdown = lastPrice < necklinePrice;
                }
                
                // Calculate confidence
                double shoulderConfidence = 100 - shoulderDiffPercent * 10;
                double necklineConfidence = 100 - necklineDiffPercent * 10;
                double confidence = (shoulderConfidence + necklineConfidence) / 2;
                
                // Boost confidence if there's a breakdown
                if (hasBreakdown)
                    confidence *= 1.2;
                
                confidence = Math.Min(confidence, 100);
                
                // Only add high confidence patterns
                if (confidence > 65)
                {
                    // Calculate the height of the pattern (for target projection)
                    double headHeight = priceData[headIdx].High - necklinePrice;
                    double targetPrice = necklinePrice - headHeight; // Project down by the height of the pattern
                    
                    patterns.Add(new PricePattern
                    {
                        Type = PatternType.HeadAndShoulders,
                        Confidence = confidence,
                        Bias = PatternBias.Bearish,
                        StartIndex = leftShoulderIdx,
                        EndIndex = rightShoulderIdx,
                        Symbol = symbol,
                        DetectionDate = DateTime.Now,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "LeftShoulder", leftShoulderPrice },
                            { "Head", priceData[headIdx].High },
                            { "RightShoulder", rightShoulderPrice },
                            { "Neckline", necklinePrice },
                            { "Target", targetPrice }
                        }
                    });
                }
            }
            
            return patterns;
        }
        
        /// <summary>
        /// Detects inverse head and shoulders patterns in price data
        /// </summary>
        public async Task<List<PricePattern>> DetectInverseHeadAndShouldersAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 30)
                return patterns;
            
            // Find troughs in the data
            var troughs = FindTroughs(priceData, 5);
            
            // Need at least 3 troughs for inverse H&S
            if (troughs.Count < 3)
                return patterns;
            
            // Look for triplets of troughs where the middle one is lower
            for (int i = 0; i < troughs.Count - 2; i++)
            {
                int leftShoulderIdx = troughs[i];
                int headIdx = troughs[i + 1];
                int rightShoulderIdx = troughs[i + 2];
                
                // Head should be lower than shoulders
                if (priceData[headIdx].Low >= priceData[leftShoulderIdx].Low || 
                    priceData[headIdx].Low >= priceData[rightShoulderIdx].Low)
                    continue;
                
                // Shoulders should be roughly at the same level
                double leftShoulderPrice = priceData[leftShoulderIdx].Low;
                double rightShoulderPrice = priceData[rightShoulderIdx].Low;
                double shoulderDiffPercent = Math.Abs(leftShoulderPrice - rightShoulderPrice) / leftShoulderPrice * 100;
                
                if (shoulderDiffPercent > 5.0) // Shoulders should be within 5% of each other
                    continue;
                
                // Find the neckline (connect the peaks between the troughs)
                int peak1Idx = FindHighestHighIndex(priceData, leftShoulderIdx, headIdx);
                int peak2Idx = FindHighestHighIndex(priceData, headIdx, rightShoulderIdx);
                
                double peak1Price = priceData[peak1Idx].High;
                double peak2Price = priceData[peak2Idx].High;
                
                // Neckline should be relatively flat
                double necklineDiffPercent = Math.Abs(peak1Price - peak2Price) / peak1Price * 100;
                if (necklineDiffPercent > 3.0) // Neckline should be within 3% flat
                    continue;
                
                // Calculate neckline as average of the two peaks
                double necklinePrice = (peak1Price + peak2Price) / 2;
                
                // Check if there's a breakout above the neckline after the pattern
                bool hasBreakout = false;
                if (rightShoulderIdx < priceData.Count - 1)
                {
                    double lastPrice = priceData[priceData.Count - 1].Close;
                    hasBreakout = lastPrice > necklinePrice;
                }
                
                // Calculate confidence
                double shoulderConfidence = 100 - shoulderDiffPercent * 10;
                double necklineConfidence = 100 - necklineDiffPercent * 10;
                double confidence = (shoulderConfidence + necklineConfidence) / 2;
                
                // Boost confidence if there's a breakout
                if (hasBreakout)
                    confidence *= 1.2;
                
                confidence = Math.Min(confidence, 100);
                
                // Only add high confidence patterns
                if (confidence > 65)
                {
                    // Calculate the depth of the pattern (for target projection)
                    double headDepth = necklinePrice - priceData[headIdx].Low;
                    double targetPrice = necklinePrice + headDepth; // Project up by the depth of the pattern
                    
                    patterns.Add(new PricePattern
                    {
                        Type = PatternType.InverseHeadAndShoulders,
                        Confidence = confidence,
                        Bias = PatternBias.Bullish,
                        StartIndex = leftShoulderIdx,
                        EndIndex = rightShoulderIdx,
                        Symbol = symbol,
                        DetectionDate = DateTime.Now,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "LeftShoulder", leftShoulderPrice },
                            { "Head", priceData[headIdx].Low },
                            { "RightShoulder", rightShoulderPrice },
                            { "Neckline", necklinePrice },
                            { "Target", targetPrice }
                        }
                    });
                }
            }
            
            return patterns;
        }
        
        /// <summary>
        /// Detects triangle patterns (ascending, descending, symmetrical)
        /// </summary>
        public async Task<List<PricePattern>> DetectTrianglePatternsAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 20)
                return patterns;
            
            // Triangle patterns require at least 4 touches of trend lines (2 highs + 2 lows)
            const int minPointsRequired = 15; // Minimum number of bars required
            
            // Check for enough data
            if (priceData.Count < minPointsRequired)
                return patterns;
            
            // Get last N bars for pattern detection
            var data = priceData.Skip(Math.Max(0, priceData.Count - 40)).Take(40).ToList();
            
            // Detect ascending triangle (flat top, rising bottom)
            var ascendingTriangle = DetectAscendingTriangle(data);
            if (ascendingTriangle != null)
            {
                ascendingTriangle.Symbol = symbol;
                ascendingTriangle.DetectionDate = DateTime.Now;
                patterns.Add(ascendingTriangle);
            }
            
            // Detect descending triangle (flat bottom, falling top)
            var descendingTriangle = DetectDescendingTriangle(data);
            if (descendingTriangle != null)
            {
                descendingTriangle.Symbol = symbol;
                descendingTriangle.DetectionDate = DateTime.Now;
                patterns.Add(descendingTriangle);
            }
            
            // Detect symmetrical triangle (converging trend lines)
            var symmetricalTriangle = DetectSymmetricalTriangle(data);
            if (symmetricalTriangle != null)
            {
                symmetricalTriangle.Symbol = symbol;
                symmetricalTriangle.DetectionDate = DateTime.Now;
                patterns.Add(symmetricalTriangle);
            }
            
            return patterns;
        }
        
        /// <summary>
        /// Detects engulfing candle patterns (bullish and bearish)
        /// </summary>
        public async Task<List<PricePattern>> DetectEngulfingPatternsAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 10)
                return patterns;
                
            // Get most recent data points
            var recentData = priceData.Skip(Math.Max(0, priceData.Count - 10)).Take(10).ToList();
            
            // Check for bullish engulfing pattern
            for (int i = 1; i < recentData.Count; i++)
            {
                var prevCandle = recentData[i - 1];
                var currCandle = recentData[i];
                
                // Bullish engulfing: previous candle is bearish, current candle is bullish and engulfs previous
                bool prevIsBearish = prevCandle.Close < prevCandle.Open;
                bool currIsBullish = currCandle.Close > currCandle.Open;
                bool engulfsPrevious = currCandle.Open < prevCandle.Close && currCandle.Close > prevCandle.Open;
                
                if (prevIsBearish && currIsBullish && engulfsPrevious)
                {
                    // Calculate confidence based on candle size and volume
                    double candleSizeRatio = (currCandle.Close - currCandle.Open) / (prevCandle.Open - prevCandle.Close);
                    double volumeIncrease = (double)currCandle.Volume / prevCandle.Volume;
                    
                    double confidence = 70 + Math.Min(candleSizeRatio, 2) * 10 + Math.Min(volumeIncrease, 2) * 10;
                    confidence = Math.Min(confidence, 100);
                    
                    patterns.Add(new PricePattern
                    {
                        Type = PatternType.BullishEngulfing,
                        Confidence = confidence,
                        Bias = PatternBias.Bullish,
                        StartIndex = priceData.Count - 10 + i - 1,
                        EndIndex = priceData.Count - 10 + i,
                        Symbol = symbol,
                        DetectionDate = DateTime.Now,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "Support", currCandle.Open },
                            { "Target", currCandle.Close + (currCandle.Close - currCandle.Open) * 2 }
                        }
                    });
                }
                
                // Bearish engulfing: previous candle is bullish, current candle is bearish and engulfs previous
                bool prevIsBullish = prevCandle.Close > prevCandle.Open;
                bool currIsBearish = currCandle.Close < currCandle.Open;
                bool engulfsPreviousBearish = currCandle.Open > prevCandle.Close && currCandle.Close < prevCandle.Open;
                
                if (prevIsBullish && currIsBearish && engulfsPreviousBearish)
                {
                    // Calculate confidence based on candle size and volume
                    double candleSizeRatio = (currCandle.Open - currCandle.Close) / (prevCandle.Close - prevCandle.Open);
                    double volumeIncrease = (double)currCandle.Volume / prevCandle.Volume;
                    
                    double confidence = 70 + Math.Min(candleSizeRatio, 2) * 10 + Math.Min(volumeIncrease, 2) * 10;
                    confidence = Math.Min(confidence, 100);
                    
                    patterns.Add(new PricePattern
                    {
                        Type = PatternType.BearishEngulfing,
                        Confidence = confidence,
                        Bias = PatternBias.Bearish,
                        StartIndex = priceData.Count - 10 + i - 1,
                        EndIndex = priceData.Count - 10 + i,
                        Symbol = symbol,
                        DetectionDate = DateTime.Now,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "Resistance", currCandle.Open },
                            { "Target", currCandle.Close - (currCandle.Open - currCandle.Close) * 2 }
                        }
                    });
                }
            }
            
            return patterns;
        }

        /// <summary>
        /// Detects cup and handle patterns in price data
        /// </summary>
        public async Task<List<PricePattern>> DetectCupAndHandlePatternsAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 50) // Cup and handle requires more data points
                return patterns;
            
            // Cup and handle pattern characteristics:
            // 1. Cup: U-shaped pattern with rounded bottom
            // 2. Handle: Small consolidation after cup formation
            // 3. Breakout: Price breaks above handle resistance
            
            // Look for cup formation in the data
            for (int i = 20; i < priceData.Count - 20; i++) // Need sufficient data on both sides
            {
                var cupPattern = DetectCupFormation(priceData, i);
                if (cupPattern == null)
                    continue;
                
                // Look for handle formation after the cup
                var handlePattern = DetectHandleFormation(priceData, cupPattern.EndIndex);
                if (handlePattern == null)
                    continue;
                
                // Verify the pattern meets cup and handle criteria
                if (ValidateCupAndHandlePattern(cupPattern, handlePattern, priceData))
                {
                    // Calculate confidence based on pattern quality
                    double confidence = CalculateCupAndHandleConfidence(cupPattern, handlePattern, priceData);
                    
                    if (confidence > 70) // Only add high confidence patterns
                    {
                        patterns.Add(new PricePattern
                        {
                            Type = PatternType.CupAndHandle,
                            Confidence = confidence,
                            Bias = PatternBias.Bullish,
                            StartIndex = cupPattern.StartIndex,
                            EndIndex = handlePattern.EndIndex,
                            Symbol = symbol,
                            DetectionDate = DateTime.Now,
                            KeyLevels = new Dictionary<string, double>
                            {
                                { "CupLeft", cupPattern.KeyLevels["LeftRim"] },
                                { "CupRight", cupPattern.KeyLevels["RightRim"] },
                                { "CupBottom", cupPattern.KeyLevels["Bottom"] },
                                { "HandleHigh", handlePattern.KeyLevels["High"] },
                                { "HandleLow", handlePattern.KeyLevels["Low"] },
                                { "Resistance", Math.Max(cupPattern.KeyLevels["RightRim"], handlePattern.KeyLevels["High"]) },
                                { "Target", Math.Max(cupPattern.KeyLevels["RightRim"], handlePattern.KeyLevels["High"]) + 
                                          (Math.Max(cupPattern.KeyLevels["LeftRim"], cupPattern.KeyLevels["RightRim"]) - cupPattern.KeyLevels["Bottom"]) }
                            }
                        });
                    }
                }
            }
            
            return patterns;
        }

        /// <summary>
        /// Detects cup and handle patterns (overloaded version that fetches historical data)
        /// </summary>
        public async Task<List<PricePattern>> DetectCupAndHandlePatternsAsync(string symbol, string range = "6mo", string interval = "1d")
        {
            // Get historical price data
            var historicalData = await _stockDataService.GetStockData(symbol, range, interval);
            
            if (historicalData == null || historicalData.Count < 50)
            {
                //DatabaseMonolith.Log("Warning", $"Insufficient historical data for {symbol} to detect cup and handle patterns");
                return new List<PricePattern>();
            }
            
            return await DetectCupAndHandlePatternsAsync(symbol, historicalData);
        }

        /// <summary>
        /// Detects bearish cup and handle patterns (overloaded version that fetches historical data)
        /// </summary>
        public async Task<List<PricePattern>> DetectBearishCupAndHandlePatternsAsync(string symbol, string range = "6mo", string interval = "1d")
        {
            // Get historical price data
            var historicalData = await _stockDataService.GetStockData(symbol, range, interval);
            
            if (historicalData == null || historicalData.Count < 50)
            {
                //DatabaseMonolith.Log("Warning", $"Insufficient historical data for {symbol} to detect bearish cup and handle patterns");
                return new List<PricePattern>();
            }
            
            return await DetectBearishCupAndHandlePatternsAsync(symbol, historicalData);
        }

        /// <summary>
        /// Detects bearish cup and handle patterns in price data (inverted pattern)
        /// </summary>
        public async Task<List<PricePattern>> DetectBearishCupAndHandlePatternsAsync(string symbol, List<HistoricalPrice> priceData)
        {
            var patterns = new List<PricePattern>();
            
            if (priceData.Count < 50) // Cup and handle requires more data points
                return patterns;
            
            // Bearish cup and handle pattern characteristics (inverted):
            // 1. Cup: Inverted U-shaped pattern with rounded top
            // 2. Handle: Small consolidation after cup formation (downward consolidation)
            // 3. Breakdown: Price breaks below handle support
            
            // Look for inverted cup formation in the data
            for (int i = 20; i < priceData.Count - 20; i++) // Need sufficient data on both sides
            {
                var cupPattern = DetectBearishCupFormation(priceData, i);
                if (cupPattern == null)
                    continue;
                
                // Look for bearish handle formation after the cup
                var handlePattern = DetectBearishHandleFormation(priceData, cupPattern.EndIndex);
                if (handlePattern == null)
                    continue;
                
                // Verify the pattern meets bearish cup and handle criteria
                if (ValidateBearishCupAndHandlePattern(cupPattern, handlePattern, priceData))
                {
                    // Calculate confidence based on pattern quality
                    double confidence = CalculateBearishCupAndHandleConfidence(cupPattern, handlePattern, priceData);
                    
                    if (confidence > 70) // Only add high confidence patterns
                    {
                        patterns.Add(new PricePattern
                        {
                            Type = PatternType.CupAndHandle,
                            Confidence = confidence,
                            Bias = PatternBias.Bearish,
                            StartIndex = cupPattern.StartIndex,
                            EndIndex = handlePattern.EndIndex,
                            Symbol = symbol,
                            DetectionDate = DateTime.Now,
                            KeyLevels = new Dictionary<string, double>
                            {
                                { "CupLeft", cupPattern.KeyLevels["LeftRim"] },
                                { "CupRight", cupPattern.KeyLevels["RightRim"] },
                                { "CupTop", cupPattern.KeyLevels["Top"] },
                                { "HandleHigh", handlePattern.KeyLevels["High"] },
                                { "HandleLow", handlePattern.KeyLevels["Low"] },
                                { "Support", Math.Min(cupPattern.KeyLevels["RightRim"], handlePattern.KeyLevels["Low"]) },
                                { "Target", Math.Min(cupPattern.KeyLevels["RightRim"], handlePattern.KeyLevels["Low"]) - 
                                          (cupPattern.KeyLevels["Top"] - Math.Min(cupPattern.KeyLevels["LeftRim"], cupPattern.KeyLevels["RightRim"])) }
                            }
                        });
                    }
                }
            }
            
            return patterns;
        }

        #region Helper Methods
        
        /// <summary>
        /// Finds peak indices in price data (local maximas)
        /// </summary>
        private List<int> FindPeaks(List<HistoricalPrice> priceData, int lookbackPeriod)
        {
            var peaks = new List<int>();
            
            for (int i = lookbackPeriod; i < priceData.Count - lookbackPeriod; i++)
            {
                bool isPeak = true;
                double currentHigh = priceData[i].High;
                
                // Check if this is higher than surrounding points
                for (int j = i - lookbackPeriod; j <= i + lookbackPeriod; j++)
                {
                    if (j == i)
                        continue;
                        
                    if (priceData[j].High >= currentHigh)
                    {
                        isPeak = false;
                        break;
                    }
                }
                
                if (isPeak)
                    peaks.Add(i);
            }
            
            return peaks;
        }
        
        /// <summary>
        /// Finds trough indices in price data (local minimas)
        /// </summary>
        private List<int> FindTroughs(List<HistoricalPrice> priceData, int lookbackPeriod)
        {
            var troughs = new List<int>();
            
            for (int i = lookbackPeriod; i < priceData.Count - lookbackPeriod; i++)
            {
                bool isTrough = true;
                double currentLow = priceData[i].Low;
                
                // Check if this is lower than surrounding points
                for (int j = i - lookbackPeriod; j <= i + lookbackPeriod; j++)
                {
                    if (j == i)
                        continue;
                        
                    if (priceData[j].Low <= currentLow)
                    {
                        isTrough = false;
                        break;
                    }
                }
                
                if (isTrough)
                    troughs.Add(i);
            }
            
            return troughs;
        }
        
        /// <summary>
        /// Finds the index of the highest high in a range
        /// </summary>
        private int FindHighestHighIndex(List<HistoricalPrice> priceData, int startIdx, int endIdx)
        {
            int highestIdx = startIdx;
            double highestPrice = priceData[startIdx].High;
            
            for (int i = startIdx + 1; i <= endIdx; i++)
            {
                if (priceData[i].High > highestPrice)
                {
                    highestPrice = priceData[i].High;
                    highestIdx = i;
                }
            }
            
            return highestIdx;
        }
        
        /// <summary>
        /// Finds the index of the lowest low in a range
        /// </summary>
        private int FindLowestLowIndex(List<HistoricalPrice> priceData, int startIdx, int endIdx)
        {
            int lowestIdx = startIdx;
            double lowestPrice = priceData[startIdx].Low;
            
            for (int i = startIdx + 1; i <= endIdx; i++)
            {
                if (priceData[i].Low < lowestPrice)
                {
                    lowestPrice = priceData[i].Low;
                    lowestIdx = i;
                }
            }
            
            return lowestIdx;
        }
        
        /// <summary>
        /// Detects ascending triangle pattern
        /// </summary>
        private PricePattern DetectAscendingTriangle(List<HistoricalPrice> data)
        {
            // Find 2+ highs at similar level (flat resistance)
            // Find 2+ ascending lows
            
            // Find highs
            var peaks = FindPeaks(data, 3);
            if (peaks.Count < 2)
                return null;
            
            // Check if we have at least two similar highs (flat resistance)
            double highLevel = 0;
            List<int> resistancePoints = new List<int>();
            
            for (int i = 0; i < peaks.Count - 1; i++)
            {
                double peak1 = data[peaks[i]].High;
                
                for (int j = i + 1; j < peaks.Count; j++)
                {
                    double peak2 = data[peaks[j]].High;
                    double diffPercent = Math.Abs(peak2 - peak1) / peak1 * 100;
                    
                    if (diffPercent < 1.5) // Peaks within 1.5% of each other
                    {
                        resistancePoints.Add(peaks[i]);
                        resistancePoints.Add(peaks[j]);
                        highLevel = (peak1 + peak2) / 2;
                        break;
                    }
                }
                
                if (resistancePoints.Count >= 2)
                    break;
            }
            
            if (resistancePoints.Count < 2)
                return null;
            
            // Find troughs
            var troughs = FindTroughs(data, 3);
            if (troughs.Count < 2)
                return null;
            
            // Filter troughs to those between our resistance points
            int minIndex = resistancePoints.Min();
            int maxIndex = resistancePoints.Max();
            var filteredTroughs = troughs.Where(t => t >= minIndex && t <= maxIndex).ToList();
            
            if (filteredTroughs.Count < 2)
                return null;
            
            // Check for ascending lows
            bool hasAscendingLows = false;
            for (int i = 0; i < filteredTroughs.Count - 1; i++)
            {
                if (data[filteredTroughs[i + 1]].Low > data[filteredTroughs[i]].Low)
                {
                    hasAscendingLows = true;
                    break;
                }
            }
            
            if (!hasAscendingLows)
                return null;
            
            // Check if price is near the apex of the triangle
            double lastPrice = data[data.Count - 1].Close;
            double resistanceLevel = highLevel;
            double supportLevel = data[filteredTroughs.Last()].Low;
            
            double distanceToApex = resistanceLevel - supportLevel;
            double rangePercent = distanceToApex / resistanceLevel * 100;
            
            // Higher confidence when price is closer to the apex (less than 3% range)
            double confidence = rangePercent < 3 ? 80 : 65;
            
            return new PricePattern
            {
                Type = PatternType.AscendingTriangle,
                Bias = PatternBias.Bullish,
                Confidence = confidence,
                StartIndex = minIndex,
                EndIndex = maxIndex,
                KeyLevels = new Dictionary<string, double>
                {
                    { "Resistance", resistanceLevel },
                    { "Support", supportLevel },
                    { "Target", resistanceLevel + (resistanceLevel - supportLevel) } // Project by height of the pattern
                }
            };
        }
        
        /// <summary>
        /// Detects descending triangle pattern
        /// </summary>
        private PricePattern DetectDescendingTriangle(List<HistoricalPrice> data)
        {
            // Find 2+ lows at similar level (flat support)
            // Find 2+ descending highs
            
            // Find lows
            var troughs = FindTroughs(data, 3);
            if (troughs.Count < 2)
                return null;
            
            // Check if we have at least two similar lows (flat support)
            double lowLevel = 0;
            List<int> supportPoints = new List<int>();
            
            for (int i = 0; i < troughs.Count - 1; i++)
            {
                double trough1 = data[troughs[i]].Low;
                
                for (int j = i + 1; j < troughs.Count; j++)
                {
                    double trough2 = data[troughs[j]].Low;
                    double diffPercent = Math.Abs(trough2 - trough1) / trough1 * 100;
                    
                    if (diffPercent < 1.5) // Troughs within 1.5% of each other
                    {
                        supportPoints.Add(troughs[i]);
                        supportPoints.Add(troughs[j]);
                        lowLevel = (trough1 + trough2) / 2;
                        break;
                    }
                }
                
                if (supportPoints.Count >= 2)
                    break;
            }
            
            if (supportPoints.Count < 2)
                return null;
            
            // Find peaks
            var peaks = FindPeaks(data, 3);
            if (peaks.Count < 2)
                return null;
            
            // Filter peaks to those between our support points
            int minIndex = supportPoints.Min();
            int maxIndex = supportPoints.Max();
            var filteredPeaks = peaks.Where(p => p >= minIndex && p <= maxIndex).ToList();
            
            if (filteredPeaks.Count < 2)
                return null;
            
            // Check for descending highs
            bool hasDescendingHighs = false;
            for (int i = 0; i < filteredPeaks.Count - 1; i++)
            {
                if (data[filteredPeaks[i + 1]].High < data[filteredPeaks[i]].High)
                {
                    hasDescendingHighs = true;
                    break;
                }
            }
            
            if (!hasDescendingHighs)
                return null;
            
            // Check if price is near the apex of the triangle
            double lastPrice = data[data.Count - 1].Close;
            double supportLevel = lowLevel;
            double resistanceLevel = data[filteredPeaks.Last()].High;
            
            double distanceToApex = resistanceLevel - supportLevel;
            double rangePercent = distanceToApex / supportLevel * 100;
            
            // Higher confidence when price is closer to the apex (less than 3% range)
            double confidence = rangePercent < 3 ? 80 : 65;
            
            return new PricePattern
            {
                Type = PatternType.DescendingTriangle,
                Bias = PatternBias.Bearish,
                Confidence = confidence,
                StartIndex = minIndex,
                EndIndex = maxIndex,
                KeyLevels = new Dictionary<string, double>
                {
                    { "Resistance", resistanceLevel },
                    { "Support", supportLevel },
                    { "Target", supportLevel - (resistanceLevel - supportLevel) } // Project by height of the pattern
                }
            };
        }
        
        /// <summary>
        /// Detects symmetrical triangle pattern
        /// </summary>
        private PricePattern DetectSymmetricalTriangle(List<HistoricalPrice> data)
        {
            // Find peaks and troughs
            var peaks = FindPeaks(data, 3);
            var troughs = FindTroughs(data, 3);
            
            if (peaks.Count < 2 || troughs.Count < 2)
                return null;
            
            // Need at least 2 descending highs and 2 ascending lows
            bool hasDescendingHighs = false;
            bool hasAscendingLows = false;
            
            // Check for descending highs
            for (int i = 0; i < peaks.Count - 1; i++)
            {
                if (data[peaks[i + 1]].High < data[peaks[i]].High)
                {
                    hasDescendingHighs = true;
                    break;
                }
            }
            
            // Check for ascending lows
            for (int i = 0; i < troughs.Count - 1; i++)
            {
                if (data[troughs[i + 1]].Low > data[troughs[i]].Low)
                {
                    hasAscendingLows = true;
                    break;
                }
            }
            
            if (!hasDescendingHighs || !hasAscendingLows)
                return null;
            
            // Get most recent peak and trough
            int lastPeakIdx = peaks.Max();
            int lastTroughIdx = troughs.Max();
            
            double resistanceLevel = data[lastPeakIdx].High;
            double supportLevel = data[lastTroughIdx].Low;
            
            // Check if price is near the apex of the triangle
            double lastPrice = data[data.Count - 1].Close;
            double distanceToApex = resistanceLevel - supportLevel;
            double rangePercent = distanceToApex / ((resistanceLevel + supportLevel) / 2) * 100;
            
            // Higher confidence when price is closer to the apex (less than 5% range)
            double confidence = rangePercent < 5 ? 75 : 60;
            
            // Direction bias based on recent momentum
            PatternBias bias;
            if (data[data.Count - 1].Close > data[Math.Max(0, data.Count - 10)].Close)
                bias = PatternBias.Bullish;
            else if (data[data.Count - 1].Close < data[Math.Max(0, data.Count - 10)].Close)
                bias = PatternBias.Bearish;
            else
                bias = PatternBias.Neutral;
            
            int startIdx = Math.Min(peaks.Min(), troughs.Min());
            int endIdx = Math.Max(lastPeakIdx, lastTroughIdx);
            
            return new PricePattern
            {
                Type = PatternType.SymmetricalTriangle,
                Bias = bias,
                Confidence = confidence,
                StartIndex = startIdx,
                EndIndex = endIdx,
                KeyLevels = new Dictionary<string, double>
                {
                    { "Resistance", resistanceLevel },
                    { "Support", supportLevel },
                    { "BreakoutTarget", bias == PatternBias.Bullish ? 
                        resistanceLevel + (resistanceLevel - supportLevel) : 
                        supportLevel - (resistanceLevel - supportLevel) }
                }
            };
        }
        
        /// <summary>
        /// Detects cup formation in price data
        /// </summary>
        private PricePattern DetectCupFormation(List<HistoricalPrice> data, int startIndex)
        {
            const int minCupWidth = 20; // Minimum width for cup formation
            const int maxCupWidth = 100; // Maximum width for cup formation
            
            // Find potential cup boundaries
            for (int width = minCupWidth; width <= Math.Min(maxCupWidth, data.Count - startIndex - 10); width++)
            {
                int leftRimIndex = startIndex;
                int rightRimIndex = startIndex + width;
                
                if (rightRimIndex >= data.Count)
                    continue;
                
                double leftRimPrice = data[leftRimIndex].High;
                double rightRimPrice = data[rightRimIndex].High;
                
                // Cup rims should be at similar levels (within 5%)
                double rimDifference = Math.Abs(leftRimPrice - rightRimPrice) / leftRimPrice * 100;
                if (rimDifference > 5.0)
                    continue;
                
                // Find the lowest point in the cup
                int bottomIndex = FindLowestLowIndex(data, leftRimIndex, rightRimIndex);
                double bottomPrice = data[bottomIndex].Low;
                
                // Cup should have significant depth (at least 10% decline from rim)
                double cupDepth = (Math.Min(leftRimPrice, rightRimPrice) - bottomPrice) / Math.Min(leftRimPrice, rightRimPrice) * 100;
                if (cupDepth < 10.0)
                    continue;
                
                // Cup should have a rounded bottom (check for U-shape)
                if (IsRoundedBottom(data, leftRimIndex, rightRimIndex, bottomIndex))
                {
                    return new PricePattern
                    {
                        Type = PatternType.CupAndHandle,
                        StartIndex = leftRimIndex,
                        EndIndex = rightRimIndex,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "LeftRim", leftRimPrice },
                            { "RightRim", rightRimPrice },
                            { "Bottom", bottomPrice }
                        }
                    };
                }
            }
            
            return null;
        }
        
        /// <summary>
        /// Detects handle formation after cup
        /// </summary>
        private PricePattern DetectHandleFormation(List<HistoricalPrice> data, int cupEndIndex)
        {
            const int minHandleWidth = 5; // Minimum width for handle
            const int maxHandleWidth = 20; // Maximum width for handle
            
            if (cupEndIndex + maxHandleWidth >= data.Count)
                return null;
            
            // Handle should start near the cup's right rim
            for (int width = minHandleWidth; width <= maxHandleWidth && cupEndIndex + width < data.Count; width++)
            {
                int handleStartIndex = cupEndIndex;
                int handleEndIndex = cupEndIndex + width;
                
                // Find high and low of handle
                int handleHighIndex = FindHighestHighIndex(data, handleStartIndex, handleEndIndex);
                int handleLowIndex = FindLowestLowIndex(data, handleStartIndex, handleEndIndex);
                
                double handleHigh = data[handleHighIndex].High;
                double handleLow = data[handleLowIndex].Low;
                double cupRightRim = data[cupEndIndex].High;
                
                // Handle should not exceed cup rim and should be a shallow pullback
                double handleDepth = (handleHigh - handleLow) / handleHigh * 100;
                double pullbackFromRim = (cupRightRim - handleLow) / cupRightRim * 100;
                
                if (handleHigh <= cupRightRim * 1.02 && // Handle high close to cup rim (within 2%)
                    handleDepth < 15.0 && // Handle depth less than 15%
                    pullbackFromRim < 25.0) // Pullback from rim less than 25%
                {
                    return new PricePattern
                    {
                        Type = PatternType.CupAndHandle,
                        StartIndex = handleStartIndex,
                        EndIndex = handleEndIndex,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "High", handleHigh },
                            { "Low", handleLow }
                        }
                    };
                }
            }
            
            return null;
        }
        
        /// <summary>
        /// Validates that cup and handle pattern meets criteria
        /// </summary>
        private bool ValidateCupAndHandlePattern(PricePattern cupPattern, PricePattern handlePattern, List<HistoricalPrice> data)
        {
            // Cup should be wide enough relative to overall timeframe
            int cupWidth = cupPattern.EndIndex - cupPattern.StartIndex;
            int totalDataPoints = data.Count;
            
            // Cup should be at least 10% of the total data timeframe
            if (cupWidth < totalDataPoints * 0.1)
                return false;
            
            // Handle should be much shorter than cup (maximum 1/3 of cup width)
            int handleWidth = handlePattern.EndIndex - handlePattern.StartIndex;
            if (handleWidth > cupWidth / 3)
                return false;
            
            // Volume should ideally decrease during handle formation (if volume data available)
            // This is a simplified check - in a full implementation, you'd analyze volume trends
            
            return true;
        }
        
        /// <summary>
        /// Calculates confidence level for cup and handle pattern
        /// </summary>
        private double CalculateCupAndHandleConfidence(PricePattern cupPattern, PricePattern handlePattern, List<HistoricalPrice> data)
        {
            double confidence = 70.0; // Base confidence
            
            // Adjust confidence based on cup shape quality
            if (IsWellFormedCup(cupPattern, data))
                confidence += 10.0;
            
            // Adjust confidence based on handle quality
            if (IsWellFormedHandle(handlePattern, cupPattern, data))
                confidence += 10.0;
            
            // Adjust confidence based on recent price action (proximity to breakout)
            double resistance = Math.Max(cupPattern.KeyLevels["RightRim"], handlePattern.KeyLevels["High"]);
            double currentPrice = data[data.Count - 1].Close;
            double distanceToBreakout = (resistance - currentPrice) / resistance * 100;
            
            if (distanceToBreakout < 2.0) // Very close to breakout
                confidence += 10.0;
            else if (distanceToBreakout < 5.0) // Reasonably close to breakout
                confidence += 5.0;
            
            return Math.Min(confidence, 100.0);
        }
        
        /// <summary>
        /// Checks if the bottom of the cup is rounded (U-shape)
        /// </summary>
        private bool IsRoundedBottom(List<HistoricalPrice> data, int leftIndex, int rightIndex, int bottomIndex)
        {
            // Simple check: bottom should be roughly in the middle third of the cup timeframe
            int cupWidth = rightIndex - leftIndex;
            int bottomPosition = bottomIndex - leftIndex;
            
            // Bottom should be in middle third for a well-rounded cup
            return bottomPosition > cupWidth * 0.33 && bottomPosition < cupWidth * 0.67;
        }
        
        /// <summary>
        /// Checks if cup is well-formed
        /// </summary>
        private bool IsWellFormedCup(PricePattern cupPattern, List<HistoricalPrice> data)
        {
            double leftRim = cupPattern.KeyLevels["LeftRim"];
            double rightRim = cupPattern.KeyLevels["RightRim"];
            double bottom = cupPattern.KeyLevels["Bottom"];
            
            // Good cup characteristics:
            // 1. Rims are very close in price (within 3%)
            double rimDifference = Math.Abs(leftRim - rightRim) / leftRim * 100;
            if (rimDifference > 3.0)
                return false;
            
            // 2. Cup has good depth (15-50% decline)
            double cupDepth = (Math.Min(leftRim, rightRim) - bottom) / Math.Min(leftRim, rightRim) * 100;
            if (cupDepth < 15.0 || cupDepth > 50.0)
                return false;
            
            return true;
        }
        
        /// <summary>
        /// Checks if handle is well-formed
        /// </summary>
        private bool IsWellFormedHandle(PricePattern handlePattern, PricePattern cupPattern, List<HistoricalPrice> data)
        {
            double handleHigh = handlePattern.KeyLevels["High"];
            double handleLow = handlePattern.KeyLevels["Low"];
            double cupRightRim = cupPattern.KeyLevels["RightRim"];
            
            // Good handle characteristics:
            // 1. Handle high is close to cup rim (within 2%)
            double handleToRimDiff = Math.Abs(handleHigh - cupRightRim) / cupRightRim * 100;
            if (handleToRimDiff > 2.0)
                return false;
            
            // 2. Handle pullback is moderate (5-20%)
            double handlePullback = (handleHigh - handleLow) / handleHigh * 100;
            if (handlePullback < 5.0 || handlePullback > 20.0)
                return false;
            
            return true;
        }

        #region Bearish Cup and Handle Helper Methods

        /// <summary>
        /// Detects bearish cup formation (inverted U-shape)
        /// </summary>
        private PricePattern DetectBearishCupFormation(List<HistoricalPrice> data, int centerIndex)
        {
            const int minCupWidth = 20; // Minimum width for a valid cup
            const int maxCupWidth = 100; // Maximum width to maintain pattern integrity
            
            // Look for potential cup boundaries
            for (int width = minCupWidth; width <= maxCupWidth && centerIndex - width/2 >= 0 && centerIndex + width/2 < data.Count; width += 5)
            {
                int leftIndex = centerIndex - width/2;
                int rightIndex = centerIndex + width/2;
                
                // For bearish cup, we need a high point (top) at the center with lower points on the sides
                double centerHigh = data[centerIndex].High;
                double leftLow = data[leftIndex].Low;
                double rightLow = data[rightIndex].Low;
                
                // Check if this forms an inverted U-shape (bearish cup)
                if (centerHigh > leftLow * 1.05 && centerHigh > rightLow * 1.05) // Center should be significantly higher
                {
                    // Find the actual top of the cup around the center
                    int topIndex = centerIndex;
                    double topPrice = centerHigh;
                    
                    for (int i = Math.Max(0, centerIndex - 5); i <= Math.Min(data.Count - 1, centerIndex + 5); i++)
                    {
                        if (data[i].High > topPrice)
                        {
                            topPrice = data[i].High;
                            topIndex = i;
                        }
                    }
                    
                    // Validate the cup shape by checking that it forms a proper inverted U
                    if (IsBearishRoundedTop(data, leftIndex, rightIndex, topIndex))
                    {
                        return new PricePattern
                        {
                            Type = PatternType.CupAndHandle,
                            StartIndex = leftIndex,
                            EndIndex = rightIndex,
                            KeyLevels = new Dictionary<string, double>
                            {
                                { "LeftRim", data[leftIndex].Low },
                                { "RightRim", data[rightIndex].Low },
                                { "Top", topPrice }
                            }
                        };
                    }
                }
            }
            
            return null;
        }

        /// <summary>
        /// Detects bearish handle formation after cup
        /// </summary>
        private PricePattern DetectBearishHandleFormation(List<HistoricalPrice> data, int cupEndIndex)
        {
            const int minHandleWidth = 5; // Minimum width for handle
            const int maxHandleWidth = 20; // Maximum width for handle
            
            if (cupEndIndex + maxHandleWidth >= data.Count)
                return null;
            
            // Handle should start near the cup's right rim
            for (int width = minHandleWidth; width <= maxHandleWidth && cupEndIndex + width < data.Count; width++)
            {
                int handleStartIndex = cupEndIndex;
                int handleEndIndex = cupEndIndex + width;
                
                // Find high and low of handle
                double handleHigh = data[handleStartIndex].High;
                double handleLow = data[handleStartIndex].Low;
                
                for (int i = handleStartIndex; i <= handleEndIndex; i++)
                {
                    if (data[i].High > handleHigh)
                        handleHigh = data[i].High;
                    if (data[i].Low < handleLow)
                        handleLow = data[i].Low;
                }
                
                // Get cup right rim value
                double cupRightRim = data[cupEndIndex].Low;
                
                // Calculate handle characteristics
                double handleRange = handleHigh - handleLow;
                double handleDepth = (handleHigh - handleLow) / handleHigh * 100;
                double riseFromRim = (handleHigh - cupRightRim) / cupRightRim * 100;
                
                if (handleLow >= cupRightRim * 0.98 && // Handle low close to cup rim (within 2%)
                    handleDepth < 15.0 && // Handle depth less than 15%
                    riseFromRim < 25.0) // Rise from rim less than 25%
                {
                    return new PricePattern
                    {
                        Type = PatternType.CupAndHandle,
                        StartIndex = handleStartIndex,
                        EndIndex = handleEndIndex,
                        KeyLevels = new Dictionary<string, double>
                        {
                            { "High", handleHigh },
                            { "Low", handleLow }
                        }
                    };
                }
            }
            
            return null;
        }

        /// <summary>
        /// Validates that bearish cup and handle pattern meets criteria
        /// </summary>
        private bool ValidateBearishCupAndHandlePattern(PricePattern cupPattern, PricePattern handlePattern, List<HistoricalPrice> data)
        {
            // Cup should be wide enough relative to overall timeframe
            int cupWidth = cupPattern.EndIndex - cupPattern.StartIndex;
            int totalDataPoints = data.Count;
            
            // Cup should be at least 10% of the total data timeframe
            if (cupWidth < totalDataPoints * 0.1)
                return false;
            
            // Handle should be much shorter than cup (maximum 1/3 of cup width)
            int handleWidth = handlePattern.EndIndex - handlePattern.StartIndex;
            if (handleWidth > cupWidth / 3)
                return false;
            
            // Volume should ideally decrease during handle formation (if volume data available)
            // This is a simplified check - in a full implementation, you'd analyze volume trends
            
            return true;
        }

        /// <summary>
        /// Calculates confidence level for bearish cup and handle pattern
        /// </summary>
        private double CalculateBearishCupAndHandleConfidence(PricePattern cupPattern, PricePattern handlePattern, List<HistoricalPrice> data)
        {
            double confidence = 70.0; // Base confidence
            
            // Adjust confidence based on cup shape quality
            if (IsBearishWellFormedCup(cupPattern, data))
                confidence += 10.0;
            
            // Adjust confidence based on handle quality
            if (IsBearishWellFormedHandle(handlePattern, cupPattern, data))
                confidence += 10.0;
            
            // Adjust confidence based on recent price action (proximity to breakdown)
            double support = Math.Min(cupPattern.KeyLevels["RightRim"], handlePattern.KeyLevels["Low"]);
            double currentPrice = data[data.Count - 1].Close;
            double distanceToBreakdown = (currentPrice - support) / currentPrice * 100;
            
            if (distanceToBreakdown < 2.0) // Very close to breakdown
                confidence += 10.0;
            else if (distanceToBreakdown < 5.0) // Reasonably close to breakdown
                confidence += 5.0;
            
            return Math.Min(confidence, 100.0);
        }

        /// <summary>
        /// Checks if the top of the cup has an inverted rounded shape
        /// </summary>
        private bool IsBearishRoundedTop(List<HistoricalPrice> data, int leftIndex, int rightIndex, int topIndex)
        {
            // Check if the top is somewhat in the middle and forms a smooth inverted curve
            int midPoint = (leftIndex + rightIndex) / 2;
            
            // Top should be reasonably close to the center
            if (Math.Abs(topIndex - midPoint) > (rightIndex - leftIndex) * 0.3)
                return false;
            
            // Check for gradual rise to the top and fall from the top
            double leftPrice = data[leftIndex].Low;
            double rightPrice = data[rightIndex].Low;
            double topPrice = data[topIndex].High;
            
            // Prices should generally increase towards the top and decrease away from it
            bool risingToTop = true;
            bool fallingFromTop = true;
            
            // Check left side rising to top
            for (int i = leftIndex; i < topIndex - 1; i++)
            {
                if (data[i + 1].High < data[i].High * 0.95) // Allow some fluctuation
                {
                    risingToTop = false;
                    break;
                }
            }
            
            // Check right side falling from top
            for (int i = topIndex; i < rightIndex - 1; i++)
            {
                if (data[i + 1].Low > data[i].Low * 1.05) // Allow some fluctuation
                {
                    fallingFromTop = false;
                    break;
                }
            }
            
            return risingToTop && fallingFromTop;
        }

        /// <summary>
        /// Checks if bearish cup is well-formed
        /// </summary>
        private bool IsBearishWellFormedCup(PricePattern cupPattern, List<HistoricalPrice> data)
        {
            // Check cup symmetry and smoothness
            double leftRim = cupPattern.KeyLevels["LeftRim"];
            double rightRim = cupPattern.KeyLevels["RightRim"];
            double top = cupPattern.KeyLevels["Top"];
            
            // Rims should be at similar levels (within 10% of each other)
            double rimDifference = Math.Abs(leftRim - rightRim) / Math.Max(leftRim, rightRim);
            if (rimDifference > 0.1)
                return false;
            
            // Cup depth should be reasonable (at least 10% from rim to top)
            double cupDepth = (top - Math.Max(leftRim, rightRim)) / top;
            if (cupDepth < 0.1)
                return false;
            
            return true;
        }

        /// <summary>
        /// Checks if bearish handle is well-formed
        /// </summary>
        private bool IsBearishWellFormedHandle(PricePattern handlePattern, PricePattern cupPattern, List<HistoricalPrice> data)
        {
            // Handle should consolidate near the cup's right rim
            double handleHigh = handlePattern.KeyLevels["High"];
            double handleLow = handlePattern.KeyLevels["Low"];
            double cupRightRim = cupPattern.KeyLevels["RightRim"];
            
            // Handle range should be relatively small
            double handleRange = handleHigh - handleLow;
            double handleRangePercent = handleRange / handleHigh * 100;
            
            // Handle should not be too volatile (range should be < 15%)
            if (handleRangePercent > 15.0)
                return false;
            
            // Handle should stay relatively close to the cup rim
            double distanceFromRim = Math.Abs(handleLow - cupRightRim) / cupRightRim * 100;
            if (distanceFromRim > 10.0)
                return false;
            
            return true;
        }

        #endregion
        
        #endregion
    }
}