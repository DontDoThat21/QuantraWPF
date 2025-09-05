using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Modules
{
    /// <summary>
    /// Provides analysis of correlations between technical indicators to identify confirmation patterns
    /// and enhance trading signal reliability.
    /// </summary>
    public class IndicatorCorrelationAnalysis
    {
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly HistoricalDataService _historicalDataService;

        public IndicatorCorrelationAnalysis()
        {
            _indicatorService = ServiceLocator.Resolve<ITechnicalIndicatorService>();
            _historicalDataService = new HistoricalDataService();
        }

        /// <summary>
        /// Calculate correlation matrix between multiple indicators for a given symbol and timeframe
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeframe">Timeframe for analysis (e.g., "1day", "1hour")</param>
        /// <param name="period">Number of periods to analyze</param>
        /// <param name="indicators">List of indicators to include in analysis</param>
        /// <returns>Matrix of correlation coefficients between indicators</returns>
        public async Task<Dictionary<string, Dictionary<string, double>>> CalculateCorrelationMatrix(
            string symbol, 
            string timeframe, 
            int period, 
            List<string> indicators)
        {
            var result = new Dictionary<string, Dictionary<string, double>>();
            
            // Collect historical data for all indicators
            var indicatorHistoricalData = await CollectIndicatorHistoricalData(symbol, timeframe, period, indicators);
            
            // Calculate correlation between each pair of indicators
            foreach (var indicator1 in indicators)
            {
                if (!indicatorHistoricalData.ContainsKey(indicator1) || 
                    indicatorHistoricalData[indicator1].Count < period)
                {
                    continue;
                }
                
                result[indicator1] = new Dictionary<string, double>();
                
                foreach (var indicator2 in indicators)
                {
                    if (indicator1 == indicator2)
                    {
                        // Perfect correlation with self
                        result[indicator1][indicator2] = 1.0;
                        continue;
                    }
                    
                    if (!indicatorHistoricalData.ContainsKey(indicator2) || 
                        indicatorHistoricalData[indicator2].Count < period)
                    {
                        result[indicator1][indicator2] = 0.0;
                        continue;
                    }
                    
                    double correlation = CalculatePearsonCorrelation(
                        indicatorHistoricalData[indicator1], 
                        indicatorHistoricalData[indicator2]);
                    
                    result[indicator1][indicator2] = correlation;
                }
            }
            
            return result;
        }

        /// <summary>
        /// Find confirmation patterns where multiple indicators signal the same trading direction
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="lookbackPeriod">Number of periods to look back</param>
        /// <returns>List of confirmation patterns found</returns>
        public async Task<List<ConfirmationPattern>> FindConfirmationPatterns(
            string symbol, 
            string timeframe, 
            int lookbackPeriod)
        {
            var result = new List<ConfirmationPattern>();
            
            // Define key indicators to analyze for confirmation patterns
            var indicatorsToAnalyze = new List<string> { "RSI", "MACD", "BollingerBands", "VWAP", "ADX", "StochRSI" };
            
            // Get historical data
            string range = MapTimeframeToRange(timeframe);
            string interval = MapTimeframeToInterval(timeframe);
            var priceHistory = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
            
            if (priceHistory.Count < lookbackPeriod)
            {
                return result;
            }
            
            // Analyze each day in the lookback period
            for (int i = priceHistory.Count - lookbackPeriod; i < priceHistory.Count; i++)
            {
                var date = priceHistory[i].Date;
                var price = priceHistory[i].Close;
                
                // Collect indicator values and signals for this date
                var bullishIndicators = new List<string>();
                var bearishIndicators = new List<string>();
                var neutralIndicators = new List<string>();
                
                // Analyze RSI
                var rsi = await _indicatorService.GetRSI(symbol, timeframe);
                if (rsi < 30) bullishIndicators.Add("RSI");
                else if (rsi > 70) bearishIndicators.Add("RSI");
                else neutralIndicators.Add("RSI");
                
                // Analyze MACD
                var (macd, signal) = await _indicatorService.GetMACD(symbol, timeframe);
                if (macd > signal) bullishIndicators.Add("MACD");
                else if (macd < signal) bearishIndicators.Add("MACD");
                else neutralIndicators.Add("MACD");
                
                // Analyze ADX
                var adx = await _indicatorService.GetADX(symbol, timeframe);
                if (adx > 25) // Strong trend, direction determined by other indicators
                {
                    neutralIndicators.Add("ADX");
                }
                else
                {
                    neutralIndicators.Add("ADX");
                }
                
                // Analyze StochRSI
                var stochRsi = await _indicatorService.GetSTOCHRSI(symbol, timeframe);
                if (stochRsi < 0.2) bullishIndicators.Add("StochRSI");
                else if (stochRsi > 0.8) bearishIndicators.Add("StochRSI");
                else neutralIndicators.Add("StochRSI");
                
                // Check for confirmation patterns
                if (bullishIndicators.Count >= 2)
                {
                    result.Add(new ConfirmationPattern
                    {
                        Date = date,
                        Symbol = symbol,
                        Price = price,
                        SignalType = SignalType.Bullish,
                        ConfirmingIndicators = bullishIndicators.ToList(),
                        ConflictingIndicators = bearishIndicators.ToList(),
                        NeutralIndicators = neutralIndicators.ToList(),
                        ConfirmationStrength = CalculateConfirmationStrength(
                            bullishIndicators.Count,
                            bearishIndicators.Count,
                            neutralIndicators.Count)
                    });
                }
                else if (bearishIndicators.Count >= 2)
                {
                    result.Add(new ConfirmationPattern
                    {
                        Date = date,
                        Symbol = symbol,
                        Price = price,
                        SignalType = SignalType.Bearish,
                        ConfirmingIndicators = bearishIndicators.ToList(),
                        ConflictingIndicators = bullishIndicators.ToList(),
                        NeutralIndicators = neutralIndicators.ToList(),
                        ConfirmationStrength = CalculateConfirmationStrength(
                            bearishIndicators.Count,
                            bullishIndicators.Count,
                            neutralIndicators.Count)
                    });
                }
            }
            
            return result;
        }

        /// <summary>
        /// Analyze the relationship between two specific indicators
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="indicator1">First indicator name</param>
        /// <param name="indicator2">Second indicator name</param>
        /// <param name="period">Number of periods to analyze</param>
        /// <returns>Analysis results</returns>
        public async Task<IndicatorRelationshipAnalysis> AnalyzeIndicatorRelationship(
            string symbol,
            string timeframe,
            string indicator1,
            string indicator2,
            int period)
        {
            var indicatorData = await CollectIndicatorHistoricalData(
                symbol, 
                timeframe, 
                period, 
                new List<string> { indicator1, indicator2 });
                
            if (!indicatorData.ContainsKey(indicator1) || 
                !indicatorData.ContainsKey(indicator2) ||
                indicatorData[indicator1].Count < period ||
                indicatorData[indicator2].Count < period)
            {
                return new IndicatorRelationshipAnalysis
                {
                    Indicator1 = indicator1,
                    Indicator2 = indicator2,
                    Correlation = 0,
                    LeadLagRelationship = 0,
                    SignalAgreementRate = 0,
                    ReliabilityScore = 0
                };
            }
            
            var data1 = indicatorData[indicator1];
            var data2 = indicatorData[indicator2];
            
            // Calculate correlation
            double correlation = CalculatePearsonCorrelation(data1, data2);
            
            // Calculate lead/lag relationship (positive means indicator1 leads, negative means indicator2 leads)
            double leadLag = CalculateLeadLagRelationship(data1, data2);
            
            // Calculate signal agreement rate
            double signalAgreementRate = CalculateSignalAgreementRate(data1, data2);
            
            // Calculate overall reliability score
            double reliabilityScore = CalculateReliabilityScore(correlation, signalAgreementRate);
            
            return new IndicatorRelationshipAnalysis
            {
                Indicator1 = indicator1,
                Indicator2 = indicator2,
                Correlation = correlation,
                LeadLagRelationship = leadLag,
                SignalAgreementRate = signalAgreementRate,
                ReliabilityScore = reliabilityScore
            };
        }

        /// <summary>
        /// Get data prepared for visualization of indicator correlations
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="indicators">List of indicators to include</param>
        /// <param name="period">Number of periods to analyze</param>
        /// <returns>Visualization data</returns>
        public async Task<IndicatorCorrelationVisualData> GetVisualData(
            string symbol,
            string timeframe,
            List<string> indicators,
            int period)
        {
            var correlationMatrix = await CalculateCorrelationMatrix(symbol, timeframe, period, indicators);
            var confirmationPatterns = await FindConfirmationPatterns(symbol, timeframe, period);
            
            // Prepare historical data for time series visualization
            var historicalData = await _historicalDataService.GetHistoricalPrices(
                symbol,
                MapTimeframeToRange(timeframe),
                MapTimeframeToInterval(timeframe));
                
            // Get only the relevant subset based on period
            var relevantHistory = historicalData
                .Skip(Math.Max(0, historicalData.Count - period))
                .ToList();
                
            var visualData = new IndicatorCorrelationVisualData
            {
                Symbol = symbol,
                Timeframe = timeframe,
                CorrelationMatrix = correlationMatrix,
                ConfirmationPatterns = confirmationPatterns,
                Dates = relevantHistory.Select(h => h.Date).ToList(),
                Prices = relevantHistory.Select(h => h.Close).ToList(),
                IndicatorValues = new Dictionary<string, List<double>>()
            };
            
            // Collect indicator values for visualization
            var indicatorHistoricalData = await CollectIndicatorHistoricalData(symbol, timeframe, period, indicators);
            foreach (var indicator in indicators)
            {
                if (indicatorHistoricalData.ContainsKey(indicator))
                {
                    visualData.IndicatorValues[indicator] = indicatorHistoricalData[indicator];
                }
            }
            
            return visualData;
        }

        #region Helper Methods

        /// <summary>
        /// Collect historical data for multiple indicators
        /// </summary>
        private async Task<Dictionary<string, List<double>>> CollectIndicatorHistoricalData(
            string symbol,
            string timeframe,
            int period,
            List<string> indicators)
        {
            var result = new Dictionary<string, List<double>>();
            
            // Get price history
            string range = MapTimeframeToRange(timeframe);
            string interval = MapTimeframeToInterval(timeframe);
            var priceHistory = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
            
            if (priceHistory.Count < period)
            {
                return result;
            }
            
            // Take the relevant subset of price history
            var relevantHistory = priceHistory
                .Skip(Math.Max(0, priceHistory.Count - period))
                .ToList();
                
            // Collect data for each indicator
            foreach (var indicator in indicators)
            {
                var values = new List<double>();
                
                switch (indicator)
                {
                    case "RSI":
                        values = await GetHistoricalRSI(symbol, timeframe, period);
                        break;
                    case "MACD":
                        values = await GetHistoricalMACD(symbol, timeframe, period);
                        break;
                    case "BollingerBands":
                        values = await GetHistoricalBollingerBands(symbol, timeframe, period);
                        break;
                    case "VWAP":
                        values = await GetHistoricalVWAP(symbol, timeframe, period);
                        break;
                    case "ADX":
                        values = await GetHistoricalADX(symbol, timeframe, period);
                        break;
                    case "StochRSI":
                        values = await GetHistoricalStochRSI(symbol, timeframe, period);
                        break;
                    default:
                        continue;
                }
                
                if (values.Count >= period)
                {
                    result[indicator] = values;
                }
            }
            
            return result;
        }

        /// <summary>
        /// Calculate Pearson correlation coefficient between two series
        /// </summary>
        private double CalculatePearsonCorrelation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count == 0)
            {
                return 0;
            }
            
            int n = x.Count;
            
            // Calculate means
            double meanX = x.Average();
            double meanY = y.Average();
            
            // Calculate covariance and variances
            double covariance = 0;
            double varianceX = 0;
            double varianceY = 0;
            
            for (int i = 0; i < n; i++)
            {
                double deltaX = x[i] - meanX;
                double deltaY = y[i] - meanY;
                
                covariance += deltaX * deltaY;
                varianceX += deltaX * deltaX;
                varianceY += deltaY * deltaY;
            }
            
            if (varianceX == 0 || varianceY == 0)
            {
                return 0;
            }
            
            return covariance / Math.Sqrt(varianceX * varianceY);
        }

        /// <summary>
        /// Calculate lead/lag relationship between two indicator series
        /// </summary>
        private double CalculateLeadLagRelationship(List<double> series1, List<double> series2)
        {
            if (series1.Count != series2.Count || series1.Count < 3)
            {
                return 0;
            }
            
            // Calculate correlations with different lags
            double maxCorrelation = double.MinValue;
            int bestLag = 0;
            
            for (int lag = -3; lag <= 3; lag++)
            {
                if (lag == 0) continue;
                
                var laggedCorrelation = CalculateLaggedCorrelation(series1, series2, lag);
                if (laggedCorrelation > maxCorrelation)
                {
                    maxCorrelation = laggedCorrelation;
                    bestLag = lag;
                }
            }
            
            return bestLag;
        }

        /// <summary>
        /// Calculate correlation with a specific lag
        /// </summary>
        private double CalculateLaggedCorrelation(List<double> series1, List<double> series2, int lag)
        {
            int n = series1.Count;
            if (Math.Abs(lag) >= n)
            {
                return 0;
            }
            
            var x = new List<double>();
            var y = new List<double>();
            
            if (lag > 0)
            {
                // series1 leads series2
                x.AddRange(series1.Take(n - lag));
                y.AddRange(series2.Skip(lag));
            }
            else
            {
                // series2 leads series1
                x.AddRange(series1.Skip(-lag));
                y.AddRange(series2.Take(n + lag));
            }
            
            return CalculatePearsonCorrelation(x, y);
        }

        /// <summary>
        /// Calculate signal agreement rate between two indicators
        /// </summary>
        private double CalculateSignalAgreementRate(List<double> series1, List<double> series2)
        {
            if (series1.Count != series2.Count || series1.Count < 2)
            {
                return 0;
            }
            
            int agreements = 0;
            int total = series1.Count - 1;
            
            for (int i = 1; i < series1.Count; i++)
            {
                bool direction1 = series1[i] > series1[i - 1];
                bool direction2 = series2[i] > series2[i - 1];
                
                if (direction1 == direction2)
                {
                    agreements++;
                }
            }
            
            return (double)agreements / total;
        }

        /// <summary>
        /// Calculate overall reliability score
        /// </summary>
        private double CalculateReliabilityScore(double correlation, double signalAgreementRate)
        {
            // Weight the absolute correlation and signal agreement equally
            return (Math.Abs(correlation) + signalAgreementRate) / 2;
        }

        /// <summary>
        /// Calculate confirmation strength based on number of confirming/conflicting indicators
        /// </summary>
        private double CalculateConfirmationStrength(
            int confirmingCount,
            int conflictingCount,
            int neutralCount)
        {
            double total = confirmingCount + conflictingCount + neutralCount;
            if (total == 0) return 0;
            
            // Calculate basic strength as ratio of confirming indicators
            double baseStrength = confirmingCount / total;
            
            // Penalize for conflicting indicators
            double conflictPenalty = conflictingCount / total;
            
            // Final strength score (0-1 range)
            return Math.Max(0, Math.Min(1, baseStrength - (conflictPenalty * 0.5)));
        }

        /// <summary>
        /// Map timeframe to range for historical data retrieval
        /// </summary>
        private string MapTimeframeToRange(string timeframe)
        {
            switch (timeframe.ToLower())
            {
                case "1min":
                case "5min":
                case "15min":
                case "30min":
                    return "1d";
                case "1hour":
                    return "5d";
                case "4hour":
                    return "1mo";
                case "1day":
                    return "1y";
                case "1week":
                    return "2y";
                case "1month":
                    return "5y";
                default:
                    return "1mo";
            }
        }

        /// <summary>
        /// Map timeframe to interval for historical data retrieval
        /// </summary>
        private string MapTimeframeToInterval(string timeframe)
        {
            switch (timeframe.ToLower())
            {
                case "1min":
                    return "1m";
                case "5min":
                    return "5m";
                case "15min":
                    return "15m";
                case "30min":
                    return "30m";
                case "1hour":
                    return "60m";
                case "4hour":
                    return "1d";
                case "1day":
                    return "1d";
                case "1week":
                    return "1wk";
                case "1month":
                    return "1mo";
                default:
                    return "1d";
            }
        }

        /// <summary>
        /// Get historical RSI values
        /// </summary>
        private async Task<List<double>> GetHistoricalRSI(string symbol, string timeframe, int period)
        {
            // Implementation would depend on historical data sources
            // This is a simplified version that returns only the current RSI multiple times
            var result = new List<double>();
            double rsi = await _indicatorService.GetRSI(symbol, timeframe);
            
            // Fill with duplicate values for now - in a real implementation,
            // we would get actual historical RSI values for each point in time
            for (int i = 0; i < period; i++)
            {
                result.Add(rsi);
            }
            
            return result;
        }

        /// <summary>
        /// Get historical MACD values
        /// </summary>
        private async Task<List<double>> GetHistoricalMACD(string symbol, string timeframe, int period)
        {
            var result = new List<double>();
            var (macd, _) = await _indicatorService.GetMACD(symbol, timeframe);
            
            // Fill with duplicate values
            for (int i = 0; i < period; i++)
            {
                result.Add(macd);
            }
            
            return result;
        }

        /// <summary>
        /// Get historical Bollinger Bands values (using %B value)
        /// </summary>
        private async Task<List<double>> GetHistoricalBollingerBands(string symbol, string timeframe, int period)
        {
            // Simplified implementation - in real code, would calculate actual historical values
            var result = new List<double>();
            
            // Fill with placeholder values
            for (int i = 0; i < period; i++)
            {
                result.Add(0.5); // Default mid-band value
            }
            
            return result;
        }

        /// <summary>
        /// Get historical VWAP values
        /// </summary>
        private async Task<List<double>> GetHistoricalVWAP(string symbol, string timeframe, int period)
        {
            var result = new List<double>();
            double vwap = await _indicatorService.GetVWAP(symbol, timeframe);
            
            // Fill with duplicate values
            for (int i = 0; i < period; i++)
            {
                result.Add(vwap);
            }
            
            return result;
        }

        /// <summary>
        /// Get historical ADX values
        /// </summary>
        private async Task<List<double>> GetHistoricalADX(string symbol, string timeframe, int period)
        {
            var result = new List<double>();
            double adx = await _indicatorService.GetADX(symbol, timeframe);
            
            // Fill with duplicate values
            for (int i = 0; i < period; i++)
            {
                result.Add(adx);
            }
            
            return result;
        }

        /// <summary>
        /// Get historical StochRSI values
        /// </summary>
        private async Task<List<double>> GetHistoricalStochRSI(string symbol, string timeframe, int period)
        {
            var result = new List<double>();
            double stochRsi = await _indicatorService.GetSTOCHRSI(symbol, timeframe);
            
            // Fill with duplicate values
            for (int i = 0; i < period; i++)
            {
                result.Add(stochRsi);
            }
            
            return result;
        }

        #endregion
    }

    #region Model Classes

    /// <summary>
    /// Represents a pattern where multiple indicators confirm the same trading signal
    /// </summary>
    public class ConfirmationPattern
    {
        /// <summary>
        /// Date when the pattern occurred
        /// </summary>
        public DateTime Date { get; set; }
        
        /// <summary>
        /// Symbol being analyzed
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// Price at the time of the pattern
        /// </summary>
        public double Price { get; set; }
        
        /// <summary>
        /// Type of signal (bullish/bearish)
        /// </summary>
        public SignalType SignalType { get; set; }
        
        /// <summary>
        /// List of indicators confirming the signal
        /// </summary>
        public List<string> ConfirmingIndicators { get; set; } = new List<string>();
        
        /// <summary>
        /// List of indicators conflicting with the signal
        /// </summary>
        public List<string> ConflictingIndicators { get; set; } = new List<string>();
        
        /// <summary>
        /// List of indicators that are neutral
        /// </summary>
        public List<string> NeutralIndicators { get; set; } = new List<string>();
        
        /// <summary>
        /// Strength of the confirmation (0-1)
        /// </summary>
        public double ConfirmationStrength { get; set; }
    }

    /// <summary>
    /// Signal type for confirmation patterns
    /// </summary>
    public enum SignalType
    {
        Bullish,
        Bearish,
        Neutral
    }

    /// <summary>
    /// Analysis of the relationship between two indicators
    /// </summary>
    public class IndicatorRelationshipAnalysis
    {
        /// <summary>
        /// First indicator name
        /// </summary>
        public string Indicator1 { get; set; }
        
        /// <summary>
        /// Second indicator name
        /// </summary>
        public string Indicator2 { get; set; }
        
        /// <summary>
        /// Correlation coefficient between the indicators
        /// </summary>
        public double Correlation { get; set; }
        
        /// <summary>
        /// Lead/lag relationship (positive if indicator1 leads, negative if indicator2 leads)
        /// </summary>
        public double LeadLagRelationship { get; set; }
        
        /// <summary>
        /// Rate at which the indicators agree on signal direction
        /// </summary>
        public double SignalAgreementRate { get; set; }
        
        /// <summary>
        /// Overall reliability score (0-1)
        /// </summary>
        public double ReliabilityScore { get; set; }
    }

    /// <summary>
    /// Data structure for visualizing indicator correlations
    /// </summary>
    public class IndicatorCorrelationVisualData
    {
        /// <summary>
        /// Symbol being analyzed
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// Timeframe of the analysis
        /// </summary>
        public string Timeframe { get; set; }
        
        /// <summary>
        /// Correlation matrix between indicators
        /// </summary>
        public Dictionary<string, Dictionary<string, double>> CorrelationMatrix { get; set; }
        
        /// <summary>
        /// List of confirmed patterns found
        /// </summary>
        public List<ConfirmationPattern> ConfirmationPatterns { get; set; } = new List<ConfirmationPattern>();
        
        /// <summary>
        /// Dates for the time series data
        /// </summary>
        public List<DateTime> Dates { get; set; } = new List<DateTime>();
        
        /// <summary>
        /// Price data for the time series
        /// </summary>
        public List<double> Prices { get; set; } = new List<double>();
        
        /// <summary>
        /// Historical values for each indicator
        /// </summary>
        public Dictionary<string, List<double>> IndicatorValues { get; set; } = new Dictionary<string, List<double>>();
    }

    #endregion
}