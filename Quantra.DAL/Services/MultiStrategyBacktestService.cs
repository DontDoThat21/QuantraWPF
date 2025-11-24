using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for running and comparing multiple trading strategies in backtests
    /// </summary>
    public class MultiStrategyBacktestService
    {
        private readonly BacktestingEngine _backtestingEngine;
        private readonly HistoricalDataService _historicalDataService;

        public MultiStrategyBacktestService(HistoricalDataService historicalDataService)
        {
            _historicalDataService = historicalDataService;
            _backtestingEngine = new BacktestingEngine(historicalDataService);
        }

        /// <summary>
        /// Runs backtest comparison for multiple strategies using the same historical data
        /// </summary>
        /// <param name="symbol">Symbol to test strategies on</param>
        /// <param name="strategies">List of strategies to test</param>
        /// <param name="interval">Time interval for the data</param>
        /// <param name="assetClass">Asset class (stock, forex, crypto)</param>
        /// <param name="initialCapital">Initial capital for each backtest</param>
        /// <param name="tradeSize">Size per trade for each backtest</param>
        /// <returns>StrategyComparisonResult containing results of all strategies</returns>
        public async Task<StrategyComparisonResult> RunComparisonAsync(
            string symbol,
            List<StrategyProfile> strategies,
            string interval = "daily",
            string assetClass = "auto",
            double initialCapital = 10000,
            int tradeSize = 1)
        {
            if (strategies == null || !strategies.Any())
                throw new ArgumentException("At least one strategy must be provided", nameof(strategies));

            // Get historical data (retrieve once for all strategies)
            var historicalData = await _historicalDataService.GetComprehensiveHistoricalData(symbol, interval, assetClass);

            if (historicalData == null || !historicalData.Any())
                throw new InvalidOperationException($"Failed to retrieve historical data for {symbol}");

            // Create result container
            var result = new StrategyComparisonResult
            {
                Symbol = symbol,
                TimeFrame = interval,
                StartDate = historicalData.First().Date,
                EndDate = historicalData.Last().Date,
                AssetClass = assetClass == "auto" ? DetermineAssetClass(symbol) : assetClass,
                InitialCapital = initialCapital
            };

            // Run backtests in parallel for better performance
            var backTestTasks = strategies.Select(strategy =>
                _backtestingEngine.RunBacktestAsync(symbol, historicalData, strategy, initialCapital, tradeSize)
            ).ToList();

            // Wait for all backtest tasks to complete
            var backTestResults = await Task.WhenAll(backTestTasks);

            // Store results with strategy information
            for (int i = 0; i < strategies.Count; i++)
            {
                var strategyResult = backTestResults[i];
                var strategy = strategies[i];

                result.StrategyResults.Add(new StrategyBacktestResult
                {
                    StrategyName = strategy.Name,
                    StrategyType = strategy.GetType().Name,
                    Result = strategyResult
                });
            }

            // Calculate comparative metrics
            CalculateComparativeMetrics(result);

            return result;
        }

        /// <summary>
        /// Calculate metrics that compare strategies against each other
        /// </summary>
        private void CalculateComparativeMetrics(StrategyComparisonResult result)
        {
            if (result.StrategyResults.Count <= 1)
                return;

            // Rank strategies by various metrics
            result.TotalReturnRanking = result.StrategyResults
                .OrderByDescending(s => s.Result.TotalReturn)
                .Select(s => s.StrategyName)
                .ToList();

            result.SharpeRatioRanking = result.StrategyResults
                .OrderByDescending(s => s.Result.SharpeRatio)
                .Select(s => s.StrategyName)
                .ToList();

            result.MaxDrawdownRanking = result.StrategyResults
                .OrderBy(s => s.Result.MaxDrawdown) // Lower is better for drawdown
                .Select(s => s.StrategyName)
                .ToList();

            result.ProfitFactorRanking = result.StrategyResults
                .OrderByDescending(s => s.Result.ProfitFactor)
                .Select(s => s.StrategyName)
                .ToList();

            // Calculate correlation matrix between strategy returns
            CalculateCorrelationMatrix(result);

            // Calculate win rate comparisons
            result.WinRateRanking = result.StrategyResults
                .OrderByDescending(s => s.Result.WinRate)
                .Select(s => s.StrategyName)
                .ToList();

            // Calculate consistency and volatility metrics
            CalculateConsistencyAndVolatility(result);
        }

        /// <summary>
        /// Calculate correlation matrix between strategies
        /// </summary>
        private void CalculateCorrelationMatrix(StrategyComparisonResult result)
        {
            int strategyCount = result.StrategyResults.Count;
            if (strategyCount <= 1)
                return;

            // Initialize the matrix
            result.CorrelationMatrix = new double[strategyCount, strategyCount];

            // For each pair of strategies
            for (int i = 0; i < strategyCount; i++)
            {
                var strategy1Returns = CalculateDailyReturns(result.StrategyResults[i].Result.EquityCurve);

                // Diagonal is always 1.0 (correlation with self)
                result.CorrelationMatrix[i, i] = 1.0;

                // Calculate correlation with other strategies
                for (int j = i + 1; j < strategyCount; j++)
                {
                    var strategy2Returns = CalculateDailyReturns(result.StrategyResults[j].Result.EquityCurve);

                    // Align return series to ensure we're comparing the same dates
                    var alignedReturns = AlignReturnSeries(strategy1Returns, strategy2Returns);
                    double correlation = CalculateCorrelation(alignedReturns.series1, alignedReturns.series2);

                    // Correlation matrix is symmetric
                    result.CorrelationMatrix[i, j] = correlation;
                    result.CorrelationMatrix[j, i] = correlation;
                }
            }
        }

        /// <summary>
        /// Calculate daily returns from equity curve
        /// </summary>
        private List<(DateTime date, double returnValue)> CalculateDailyReturns(List<BacktestingEngine.EquityPoint> equityCurve)
        {
            var returns = new List<(DateTime date, double returnValue)>();

            for (int i = 1; i < equityCurve.Count; i++)
            {
                var prev = equityCurve[i - 1].Equity;
                var curr = equityCurve[i].Equity;
                var dailyReturn = (curr - prev) / prev;

                returns.Add((equityCurve[i].Date, dailyReturn));
            }

            return returns;
        }

        /// <summary>
        /// Align two return series to make sure they have the same dates
        /// </summary>
        private (List<double> series1, List<double> series2) AlignReturnSeries(
            List<(DateTime date, double returnValue)> returns1,
            List<(DateTime date, double returnValue)> returns2)
        {
            // Create lookup dictionaries by date
            var dict1 = returns1.ToDictionary(r => r.date.Date, r => r.returnValue);
            var dict2 = returns2.ToDictionary(r => r.date.Date, r => r.returnValue);

            // Find common dates
            var commonDates = dict1.Keys.Intersect(dict2.Keys).OrderBy(d => d).ToList();

            // Create aligned series
            var series1 = commonDates.Select(d => dict1[d]).ToList();
            var series2 = commonDates.Select(d => dict2[d]).ToList();

            return (series1, series2);
        }

        /// <summary>
        /// Calculate Pearson correlation coefficient between two series
        /// </summary>
        private double CalculateCorrelation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count <= 1)
                return 0;

            double xMean = x.Average();
            double yMean = y.Average();

            double numerator = 0;
            double denomX = 0;
            double denomY = 0;

            for (int i = 0; i < x.Count; i++)
            {
                double xDiff = x[i] - xMean;
                double yDiff = y[i] - yMean;

                numerator += xDiff * yDiff;
                denomX += xDiff * xDiff;
                denomY += yDiff * yDiff;
            }

            if (denomX <= 0 || denomY <= 0)
                return 0;

            return numerator / Math.Sqrt(denomX * denomY);
        }

        /// <summary>
        /// Calculate consistency and volatility metrics for strategies
        /// </summary>
        private void CalculateConsistencyAndVolatility(StrategyComparisonResult result)
        {
            foreach (var strategyResult in result.StrategyResults)
            {
                // Calculate volatility of daily returns
                var returns = CalculateDailyReturns(strategyResult.Result.EquityCurve);
                var returnValues = returns.Select(r => r.returnValue).ToList();

                if (returnValues.Count > 0)
                {
                    // Daily volatility
                    double volatility = CalculateStandardDeviation(returnValues);
                    strategyResult.DailyReturnVolatility = volatility;

                    // Annualized volatility (assuming 252 trading days)
                    strategyResult.AnnualizedVolatility = volatility * Math.Sqrt(252);

                    // Calculate consistency score (higher is better)
                    // This is the ratio of positive to negative days, adjusted by the win rate
                    int positiveDays = returnValues.Count(r => r > 0);
                    int negativeDays = returnValues.Count(r => r < 0);

                    strategyResult.ConsistencyScore = negativeDays > 0
                        ? (double)positiveDays / negativeDays * strategyResult.Result.WinRate
                        : strategyResult.Result.WinRate * 10; // Very high if no negative days
                }
            }
        }

        /// <summary>
        /// Calculate standard deviation of a list of values
        /// </summary>
        private double CalculateStandardDeviation(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0;

            double avg = values.Average();
            double sumOfSquaresOfDifferences = values.Sum(val => Math.Pow(val - avg, 2));
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
        }

        /// <summary>
        /// Determines the asset class from a symbol
        /// </summary>
        private string DetermineAssetClass(string symbol)
        {
            if (symbol.Contains("/"))
            {
                return "forex";
            }

            string[] cryptos = { "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "XLM", "UNI" };
            if (cryptos.Contains(symbol))
            {
                return "crypto";
            }

            return "stock";
        }
    }
}