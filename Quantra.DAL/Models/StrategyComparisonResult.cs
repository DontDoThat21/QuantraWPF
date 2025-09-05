using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Models
{
    /// <summary>
    /// Class for storing the results of a multi-strategy backtest comparison
    /// </summary>
    public class StrategyComparisonResult
    {
        /// <summary>
        /// Symbol used in the backtest
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// Time frame used for the backtest
        /// </summary>
        public string TimeFrame { get; set; }
        
        /// <summary>
        /// Asset class used in the backtest
        /// </summary>
        public string AssetClass { get; set; }
        
        /// <summary>
        /// Start date of the backtest period
        /// </summary>
        public DateTime StartDate { get; set; }
        
        /// <summary>
        /// End date of the backtest period
        /// </summary>
        public DateTime EndDate { get; set; }
        
        /// <summary>
        /// Initial capital used for each strategy
        /// </summary>
        public double InitialCapital { get; set; }
        
        /// <summary>
        /// List of individual strategy backtest results
        /// </summary>
        public List<StrategyBacktestResult> StrategyResults { get; set; } = new List<StrategyBacktestResult>();
        
        /// <summary>
        /// Correlation matrix between strategy returns
        /// [i,j] contains correlation between strategy i and strategy j
        /// </summary>
        public double[,] CorrelationMatrix { get; set; }
        
        /// <summary>
        /// Strategies ranked by total return (best to worst)
        /// </summary>
        public List<string> TotalReturnRanking { get; set; } = new List<string>();
        
        /// <summary>
        /// Strategies ranked by Sharpe ratio (best to worst)
        /// </summary>
        public List<string> SharpeRatioRanking { get; set; } = new List<string>();
        
        /// <summary>
        /// Strategies ranked by maximum drawdown (best to worst)
        /// </summary>
        public List<string> MaxDrawdownRanking { get; set; } = new List<string>();
        
        /// <summary>
        /// Strategies ranked by profit factor (best to worst)
        /// </summary>
        public List<string> ProfitFactorRanking { get; set; } = new List<string>();
        
        /// <summary>
        /// Strategies ranked by win rate (best to worst)
        /// </summary>
        public List<string> WinRateRanking { get; set; } = new List<string>();
        
        /// <summary>
        /// Identifies the best strategy overall based on a composite score
        /// </summary>
        /// <returns>Name of the best overall strategy</returns>
        public string GetBestOverallStrategy()
        {
            if (StrategyResults.Count == 0)
                return string.Empty;
                
            if (StrategyResults.Count == 1)
                return StrategyResults[0].StrategyName;
                
            // Create a scoring system based on rankings
            var scores = new Dictionary<string, int>();
            
            foreach (var strategy in StrategyResults)
            {
                scores[strategy.StrategyName] = 0;
            }
            
            // Award points based on rankings (higher = better)
            // Give most points to top performer, down to 1 point for last place
            foreach (var strategyName in TotalReturnRanking)
            {
                scores[strategyName] += TotalReturnRanking.Count - TotalReturnRanking.IndexOf(strategyName);
            }
            
            foreach (var strategyName in SharpeRatioRanking)
            {
                scores[strategyName] += SharpeRatioRanking.Count - SharpeRatioRanking.IndexOf(strategyName);
            }
            
            foreach (var strategyName in MaxDrawdownRanking)
            {
                scores[strategyName] += MaxDrawdownRanking.Count - MaxDrawdownRanking.IndexOf(strategyName);
            }
            
            foreach (var strategyName in ProfitFactorRanking)
            {
                scores[strategyName] += ProfitFactorRanking.Count - ProfitFactorRanking.IndexOf(strategyName);
            }
            
            foreach (var strategyName in WinRateRanking)
            {
                scores[strategyName] += WinRateRanking.Count - WinRateRanking.IndexOf(strategyName);
            }
            
            // Find the strategy with the highest score
            string bestStrategy = string.Empty;
            int highestScore = -1;
            
            foreach (var kvp in scores)
            {
                if (kvp.Value > highestScore)
                {
                    highestScore = kvp.Value;
                    bestStrategy = kvp.Key;
                }
            }
            
            return bestStrategy;
        }
        
        /// <summary>
        /// Calculates optimal portfolio weights for the strategies based on a risk-adjusted approach
        /// </summary>
        /// <param name="riskAversion">Risk aversion parameter (higher = more risk averse)</param>
        /// <returns>Dictionary mapping strategy names to their optimal weights</returns>
        public Dictionary<string, double> CalculateOptimalPortfolioWeights(double riskAversion = 1.0)
        {
            var weights = new Dictionary<string, double>();
            int n = StrategyResults.Count;
            
            if (n == 0)
                return weights;
                
            if (n == 1)
            {
                weights[StrategyResults[0].StrategyName] = 1.0;
                return weights;
            }
            
            // Create a simple heuristic for portfolio allocation based on Sharpe ratio and correlation
            double totalSharpe = StrategyResults.Sum(s => Math.Max(0.1, s.Result.SharpeRatio));
            
            // Initially allocate based on relative Sharpe ratios
            foreach (var strategy in StrategyResults)
            {
                double weight = Math.Max(0.1, strategy.Result.SharpeRatio) / totalSharpe;
                weights[strategy.StrategyName] = weight;
            }
            
            // Adjust weights based on correlation matrix
            // Strategies with low correlation should get higher weights
            if (CorrelationMatrix != null)
            {
                for (int i = 0; i < n; i++)
                {
                    double averageCorrelation = 0;
                    for (int j = 0; j < n; j++)
                    {
                        if (i != j)
                        {
                            averageCorrelation += CorrelationMatrix[i, j];
                        }
                    }
                    
                    if (n > 1)
                    {
                        averageCorrelation /= (n - 1);
                        
                        // Adjust weight - lower correlation = higher weight
                        string strategyName = StrategyResults[i].StrategyName;
                        double correlationFactor = 1.0 - (averageCorrelation / 2.0); // Range from 0.5 to 1.5
                        weights[strategyName] *= correlationFactor;
                    }
                }
                
                // Normalize weights to sum to 1.0
                double totalWeight = weights.Values.Sum();
                if (totalWeight > 0)
                {
                    foreach (var key in weights.Keys.ToList())
                    {
                        weights[key] /= totalWeight;
                    }
                }
            }
            
            return weights;
        }
        
        /// <summary>
        /// Simulates a portfolio combining multiple strategies with the given weights
        /// </summary>
        /// <param name="weights">Dictionary mapping strategy names to weights (should sum to 1.0)</param>
        /// <returns>Backtest result for the combined portfolio</returns>
        public BacktestingEngine.BacktestResult SimulateCombinedPortfolio(Dictionary<string, double> weights)
        {
            // Validate inputs
            if (StrategyResults.Count == 0 || weights.Count == 0)
                return null;
                
            // Normalize weights if they don't sum to 1.0
            double totalWeight = weights.Values.Sum();
            if (Math.Abs(totalWeight - 1.0) > 0.001 && totalWeight > 0)
            {
                weights = weights.ToDictionary(kv => kv.Key, kv => kv.Value / totalWeight);
            }
            
            // Create a combined equity curve
            var combinedEquity = new List<BacktestingEngine.EquityPoint>();
            var combinedDrawdown = new List<BacktestingEngine.DrawdownPoint>();
            
            // Get the first strategy to use as template for dates
            var firstStrategy = StrategyResults.First();
            var datePoints = firstStrategy.Result.EquityCurve.Select(e => e.Date).ToList();
            
            // Create map from strategy name to result
            var strategyMap = StrategyResults.ToDictionary(s => s.StrategyName, s => s.Result);
            
            // Verify all strategies have the same dates in their equity curves
            foreach (var strategyName in weights.Keys)
            {
                if (!strategyMap.ContainsKey(strategyName))
                    continue;
                    
                var currentDates = strategyMap[strategyName].EquityCurve.Select(e => e.Date).ToList();
                if (!datePoints.SequenceEqual(currentDates))
                {
                    // Dates don't match exactly - this would require more complex alignment logic
                    // For this implementation, we'll return null to indicate failure
                    return null;
                }
            }
            
            // Create the combined equity curve
            double peakEquity = InitialCapital;
            for (int i = 0; i < datePoints.Count; i++)
            {
                double totalEquity = 0;
                
                // Sum weighted equity for each strategy at this point in time
                foreach (var kvp in weights)
                {
                    string strategyName = kvp.Key;
                    double weight = kvp.Value;
                    
                    if (!strategyMap.ContainsKey(strategyName) || strategyMap[strategyName].EquityCurve.Count <= i)
                        continue;
                        
                    double equity = strategyMap[strategyName].EquityCurve[i].Equity;
                    totalEquity += equity * weight;
                }
                
                // Create equity point
                combinedEquity.Add(new BacktestingEngine.EquityPoint
                {
                    Date = datePoints[i],
                    Equity = totalEquity
                });
                
                // Calculate drawdown
                if (totalEquity > peakEquity)
                    peakEquity = totalEquity;
                    
                double drawdown = (peakEquity - totalEquity) / peakEquity;
                combinedDrawdown.Add(new BacktestingEngine.DrawdownPoint
                {
                    Date = datePoints[i],
                    Drawdown = drawdown
                });
            }
            
            // Create combined result
            var result = new BacktestingEngine.BacktestResult
            {
                Symbol = Symbol + " (Combined Portfolio)",
                TimeFrame = TimeFrame,
                StartDate = StartDate,
                EndDate = EndDate,
                EquityCurve = combinedEquity,
                DrawdownCurve = combinedDrawdown,
                TotalReturn = (combinedEquity.Last().Equity - InitialCapital) / InitialCapital,
                MaxDrawdown = combinedDrawdown.Max(d => d.Drawdown),
                AssetClass = AssetClass
            };
            
            // Calculate advanced metrics for the combined result
            CalculateAdvancedMetricsForCombinedResult(result);
            
            return result;
        }
        
        /// <summary>
        /// Calculate advanced metrics for a combined portfolio result
        /// </summary>
        private void CalculateAdvancedMetricsForCombinedResult(BacktestingEngine.BacktestResult result)
        {
            // 1. Calculate daily returns from equity curve
            List<double> dailyReturns = new List<double>();
            List<double> dailyDownsideReturns = new List<double>();
            
            for (int i = 1; i < result.EquityCurve.Count; i++)
            {
                double previousValue = result.EquityCurve[i - 1].Equity;
                double currentValue = result.EquityCurve[i].Equity;
                double dailyReturn = (currentValue - previousValue) / previousValue;
                
                dailyReturns.Add(dailyReturn);
                
                if (dailyReturn < 0)
                {
                    dailyDownsideReturns.Add(dailyReturn);
                }
            }
            
            // 2. Calculate metrics
            double riskFreeRate = 0.0;
            double averageReturn = dailyReturns.Count > 0 ? dailyReturns.Average() : 0;
            double returnStdDev = CalculateStandardDeviation(dailyReturns);
            
            // Sharpe ratio
            result.SharpeRatio = returnStdDev > 0 ? 
                (averageReturn - riskFreeRate) / returnStdDev * Math.Sqrt(252) : 0;
            
            // Sortino ratio
            double downsideDeviation = CalculateStandardDeviation(dailyDownsideReturns);
            result.SortinoRatio = downsideDeviation > 0 ? 
                (averageReturn - riskFreeRate) / downsideDeviation * Math.Sqrt(252) : 0;
            
            // CAGR
            double totalDays = (result.EndDate - result.StartDate).TotalDays;
            if (totalDays > 0)
            {
                double startValue = InitialCapital;
                double endValue = result.EquityCurve.Last().Equity;
                result.CAGR = Math.Pow(endValue / startValue, 365.0 / totalDays) - 1;
            }
            
            // Calmar ratio
            result.CalmarRatio = result.MaxDrawdown > 0 ? result.CAGR / result.MaxDrawdown : 0;
            
            // Information ratio (approximated - no true benchmark in this case)
            result.InformationRatio = returnStdDev > 0 ? 
                (averageReturn - riskFreeRate) / returnStdDev * Math.Sqrt(252) : 0;
        }
        
        private double CalculateStandardDeviation(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0;
                
            double avg = values.Average();
            double sumOfSquaresOfDifferences = values.Sum(val => Math.Pow(val - avg, 2));
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
        }
    }
    
    /// <summary>
    /// Class for storing individual strategy results in a backtest comparison
    /// </summary>
    public class StrategyBacktestResult
    {
        /// <summary>
        /// Name of the strategy
        /// </summary>
        public string StrategyName { get; set; }
        
        /// <summary>
        /// Type/class name of the strategy
        /// </summary>
        public string StrategyType { get; set; }
        
        /// <summary>
        /// Backtest result for this strategy
        /// </summary>
        public BacktestingEngine.BacktestResult Result { get; set; }
        
        /// <summary>
        /// Daily return volatility for the strategy (standard deviation of daily returns)
        /// </summary>
        public double DailyReturnVolatility { get; set; }
        
        /// <summary>
        /// Annualized volatility for the strategy
        /// </summary>
        public double AnnualizedVolatility { get; set; }
        
        /// <summary>
        /// Consistency score - a measure of how consistent the strategy is
        /// Higher values indicate more consistent positive returns
        /// </summary>
        public double ConsistencyScore { get; set; }
    }
}