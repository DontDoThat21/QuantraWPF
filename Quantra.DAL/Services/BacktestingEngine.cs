using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class BacktestingEngine
    {
        public class BacktestResult
        {
            public List<SimulatedTrade> Trades { get; set; } = new();
            public List<EquityPoint> EquityCurve { get; set; } = new();
            public List<DrawdownPoint> DrawdownCurve { get; set; } = new();
            public double TotalReturn { get; set; }
            public double MaxDrawdown { get; set; }
            public string Symbol { get; set; }
            public string TimeFrame { get; set; }
            public DateTime StartDate { get; set; }
            public DateTime EndDate { get; set; }
            public string AssetClass { get; set; } = "stock";
            public int TotalTrades { get => Trades.Count; }
            public int WinningTrades
            {
                get => Trades.Count(t => t.ExitPrice.HasValue &&
                t.ProfitLoss > 0);
            }
            public double WinRate { get => TotalTrades > 0 ? (double)WinningTrades / TotalTrades : 0; }

            // Transaction cost metrics
            /// <summary>
            /// Total transaction costs across all trades
            /// </summary>
            public double TotalTransactionCosts { get; set; }

            /// <summary>
            /// Transaction costs as a percentage of total trading volume
            /// </summary>
            public double TransactionCostsPercentage { get; set; }

            /// <summary>
            /// Gross return before transaction costs
            /// </summary>
            public double GrossReturn { get; set; }

            /// <summary>
            /// Net return after transaction costs
            /// </summary>
            public double NetReturn => TotalReturn;

            /// <summary>
            /// Impact of transaction costs on total return (GrossReturn - NetReturn)
            /// </summary>
            public double CostsImpact => GrossReturn - NetReturn;

            // Advanced performance metrics
            /// <summary>
            /// Sharpe Ratio - Risk-adjusted return measure (higher is better)
            /// </summary>
            public double SharpeRatio { get; set; }

            /// <summary>
            /// Sortino Ratio - Risk-adjusted return measure focusing on downside risk (higher is better)
            /// </summary>
            public double SortinoRatio { get; set; }

            /// <summary>
            /// CAGR - Compound Annual Growth Rate as a percentage (annualized return)
            /// </summary>
            public double CAGR { get; set; }

            /// <summary>
            /// Calmar Ratio - Annual return divided by maximum drawdown (higher is better)
            /// </summary>
            public double CalmarRatio { get; set; }

            /// <summary>
            /// Profit Factor - Gross profit divided by gross loss (higher than 1 is profitable)
            /// </summary>
            public double ProfitFactor { get; set; }

            /// <summary>
            /// Information Ratio - Measures the risk-adjusted excess return relative to a benchmark
            /// </summary>
            public double InformationRatio { get; set; }

            // Monte Carlo simulation results
            /// <summary>
            /// Monte Carlo simulation results for risk assessment
            /// </summary>
            public MonteCarloSimulationResult MonteCarloResults { get; set; }

            /// <summary>
            /// Check if Monte Carlo simulation was performed on this backtest result
            /// </summary>
            public bool HasMonteCarloResults => MonteCarloResults != null;

            /// <summary>
            /// Transaction cost model used for this backtest
            /// </summary>
            public TransactionCostModel CostModel { get; set; }

            /// <summary>
            /// Spread-specific results if this is a spread strategy backtest
            /// </summary>
            public SpreadBacktestResult SpreadResults { get; set; }

            /// <summary>
            /// Check if this is a spread strategy backtest
            /// </summary>
            public bool IsSpreadBacktest => SpreadResults != null;
        }

        public class SimulatedTrade
        {
            public DateTime EntryDate { get; set; }
            public double EntryPrice { get; set; }
            public DateTime? ExitDate { get; set; }
            public double? ExitPrice { get; set; }
            public string Action { get; set; } // BUY/SELL
            public int Quantity { get; set; }

            // Transaction costs
            public double EntryCosts { get; set; } = 0;
            public double ExitCosts { get; set; } = 0;
            public double TotalCosts => EntryCosts + ExitCosts;

            // Profit calculation including transaction costs
            public double ProfitLoss => ExitPrice.HasValue ?
                (ExitPrice.Value - EntryPrice) * (Action == "BUY" ? 1 : -1) * Quantity - TotalCosts :
                -EntryCosts; // If position is still open, only entry costs are realized

            // Profit without transaction costs (for comparison)
            public double GrossProfitLoss => ExitPrice.HasValue ?
                (ExitPrice.Value - EntryPrice) * (Action == "BUY" ? 1 : -1) * Quantity :
                0;
        }

        public class EquityPoint
        {
            public DateTime Date { get; set; }
            public double Equity { get; set; }
        }

        public class DrawdownPoint
        {
            public DateTime Date { get; set; }
            public double Drawdown { get; set; }
        }

        public class MonteCarloSimulationResult
        {
            /// <summary>
            /// Number of simulations run
            /// </summary>
            public int SimulationCount { get; set; }

            /// <summary>
            /// List of all simulated equity curves (each inner list is one simulation path)
            /// </summary>
            public List<List<EquityPoint>> SimulatedEquityCurves { get; set; } = new();

            /// <summary>
            /// Final values of all simulations
            /// </summary>
            public List<double> FinalValues { get; set; } = new();

            /// <summary>
            /// Maximum drawdown of each simulation
            /// </summary>
            public List<double> MaxDrawdowns { get; set; } = new();

            /// <summary>
            /// Percentiles of final equity values [5%, 25%, 50% (median), 75%, 95%]
            /// </summary>
            public Dictionary<string, double> ReturnPercentiles { get; set; } = new();

            /// <summary>
            /// Percentiles of maximum drawdowns [5%, 25%, 50% (median), 75%, 95%]
            /// </summary>
            public Dictionary<string, double> DrawdownPercentiles { get; set; } = new();

            /// <summary>
            /// Value at Risk (VaR) at 95% confidence level
            /// </summary>
            public double ValueAtRisk95 { get; set; }

            /// <summary>
            /// Value at Risk (VaR) at 99% confidence level
            /// </summary>
            public double ValueAtRisk99 { get; set; }

            /// <summary>
            /// Conditional Value at Risk (CVaR/Expected Shortfall) at 95% confidence level
            /// </summary>
            public double ConditionalValueAtRisk95 { get; set; }

            /// <summary>
            /// Probability of profit (percentage of simulations with positive returns)
            /// </summary>
            public double ProbabilityOfProfit { get; set; }

            /// <summary>
            /// Probability of exceeding the backtest return
            /// </summary>
            public double ProbabilityOfExceedingBacktestReturn { get; set; }

            /// <summary>
            /// Representative equity curves for each percentile (5%, 25%, 50%, 75%, 95%)
            /// </summary>
            public Dictionary<string, List<EquityPoint>> PercentileEquityCurves { get; set; } = new();
        }

        public class SpreadBacktestResult
        {
            /// <summary>
            /// List of simulated spread trades
            /// </summary>
            public List<SimulatedSpreadTrade> SpreadTrades { get; set; } = new();

            /// <summary>
            /// Rolling P&L for each spread position over time
            /// </summary>
            public List<SpreadPnLPoint> RollingPnL { get; set; } = new();

            /// <summary>
            /// Maximum profit achieved
            /// </summary>
            public double MaxProfit { get; set; }

            /// <summary>
            /// Maximum loss incurred
            /// </summary>
            public double MaxLoss { get; set; }

            /// <summary>
            /// Average time in trade (days)
            /// </summary>
            public double AverageTimeInTrade { get; set; }

            /// <summary>
            /// Percentage of trades that were profitable
            /// </summary>
            public double ProfitableTradePercentage { get; set; }

            /// <summary>
            /// Average profit per winning trade
            /// </summary>
            public double AverageWinningTrade { get; set; }

            /// <summary>
            /// Average loss per losing trade
            /// </summary>
            public double AverageLosingTrade { get; set; }

            /// <summary>
            /// Total premium collected (for credit spreads) or paid (for debit spreads)
            /// </summary>
            public double TotalPremium { get; set; }

            /// <summary>
            /// Comparison to buy-and-hold equity strategy return
            /// </summary>
            public double OutperformanceVsEquity { get; set; }
        }

        public class SimulatedSpreadTrade
        {
            /// <summary>
            /// Entry date for the spread position
            /// </summary>
            public DateTime EntryDate { get; set; }

            /// <summary>
            /// Exit date for the spread position
            /// </summary>
            public DateTime? ExitDate { get; set; }

            /// <summary>
            /// Underlying price at entry
            /// </summary>
            public double UnderlyingPriceAtEntry { get; set; }

            /// <summary>
            /// Underlying price at exit
            /// </summary>
            public double? UnderlyingPriceAtExit { get; set; }

            /// <summary>
            /// Net premium paid or received for the spread
            /// </summary>
            public double NetPremium { get; set; }

            /// <summary>
            /// Final profit or loss of the spread trade
            /// </summary>
            public double ProfitLoss { get; set; }

            /// <summary>
            /// Reason for exit (EXPIRATION, TARGET_PROFIT, STOP_LOSS, TIME_DECAY)
            /// </summary>
            public string ExitReason { get; set; }

            /// <summary>
            /// The spread configuration used for this trade
            /// </summary>
            public SpreadConfiguration SpreadConfig { get; set; }

            /// <summary>
            /// Days held in the trade
            /// </summary>
            public double DaysHeld => ExitDate.HasValue ?
                (ExitDate.Value - EntryDate).TotalDays : 0;
        }

        public class SpreadPnLPoint
        {
            /// <summary>
            /// Date of the P&L calculation
            /// </summary>
            public DateTime Date { get; set; }

            /// <summary>
            /// Cumulative P&L up to this date
            /// </summary>
            public double CumulativePnL { get; set; }

            /// <summary>
            /// P&L for the current period
            /// </summary>
            public double PeriodPnL { get; set; }

            /// <summary>
            /// Underlying price at this date
            /// </summary>
            public double UnderlyingPrice { get; set; }

            /// <summary>
            /// Number of open positions
            /// </summary>
            public int OpenPositions { get; set; }
        }

        private readonly HistoricalDataService _historicalDataService;

        public BacktestingEngine(UserSettingsService userSettingsService, LoggingService loggingService)
        {
            _historicalDataService = new HistoricalDataService(userSettingsService, loggingService);
        }

        public BacktestingEngine(HistoricalDataService historicalDataService)
        {
            _historicalDataService = historicalDataService;
        }

        /// <summary>
        /// Runs a backtest using the specified strategy and symbol
        /// </summary>
        /// <param name="symbol">The asset symbol to backtest</param>
        /// <param name="historical">Historical price data</param>
        /// <param name="strategy">The strategy to test</param>
        /// <param name="initialCapital">Starting capital amount</param>
        /// <param name="tradeSize">Size of each trade</param>
        /// <param name="costModel">Transaction cost model (if null, zero costs are assumed)</param>
        /// <returns>Backtest results</returns>
        public async Task<BacktestResult> RunBacktestAsync(
            string symbol,
            List<HistoricalPrice> historical,
            StrategyProfile strategy,
            double initialCapital = 10000,
            int tradeSize = 1,
            TransactionCostModel costModel = null)
        {
            // Use zero cost model if none provided
            costModel ??= TransactionCostModel.CreateZeroCostModel();

            var result = new BacktestResult
            {
                Symbol = symbol,
                TimeFrame = "daily", // Default
                CostModel = costModel
            };

            if (historical.Any())
            {
                result.StartDate = historical.First().Date;
                result.EndDate = historical.Last().Date;
            }

            double cash = initialCapital;
            double grossCash = initialCapital; // For tracking returns without costs
            double position = 0;
            double entryPrice = 0;
            double peakEquity = initialCapital;
            double maxDrawdown = 0;
            double totalTransactionCosts = 0;
            double totalTradingVolume = 0;

            for (int i = 1; i < historical.Count; i++)
            {
                var prev = historical[i - 1];
                var curr = historical[i];
                var signal = strategy.GenerateSignal(historical, i);

                // Entry
                if (position == 0 && (signal == "BUY" || signal == "SELL"))
                {
                    bool isBuy = signal == "BUY";
                    position = isBuy ? tradeSize : -tradeSize;

                    // Calculate execution price with slippage, spread
                    var (entryCost, effectiveEntryPrice) = costModel.CalculateAllCosts(
                        (int)position, curr.Close, isBuy, curr.Volume);

                    entryPrice = effectiveEntryPrice;
                    totalTransactionCosts += entryCost;
                    totalTradingVolume += Math.Abs(position) * entryPrice;

                    // Deduct costs from cash
                    cash -= entryCost;

                    var trade = new SimulatedTrade
                    {
                        EntryDate = curr.Date,
                        EntryPrice = entryPrice,
                        Action = signal,
                        Quantity = tradeSize,
                        EntryCosts = entryCost
                    };

                    result.Trades.Add(trade);
                }
                // Exit
                else if (position != 0 && (signal == "EXIT" || position > 0 && signal == "SELL" || position < 0 && signal == "BUY"))
                {
                    var trade = result.Trades.LastOrDefault(t => !t.ExitDate.HasValue);
                    if (trade != null)
                    {
                        bool isBuy = position < 0; // If position is negative, we're covering (buying)

                        // Calculate exit price with slippage, spread
                        var (exitCost, effectiveExitPrice) = costModel.CalculateAllCosts(
                            (int)-position, curr.Close, isBuy, curr.Volume);

                        trade.ExitDate = curr.Date;
                        trade.ExitPrice = effectiveExitPrice;
                        trade.ExitCosts = exitCost;

                        totalTransactionCosts += exitCost;
                        totalTradingVolume += Math.Abs(position) * effectiveExitPrice;

                        // Update cash positions
                        cash += (effectiveExitPrice - entryPrice) * position - exitCost;
                        grossCash += (curr.Close - entryPrice) * position; // Without costs

                        position = 0;
                    }
                }

                // Equity curve
                double equity = cash + position * curr.Close;
                result.EquityCurve.Add(new EquityPoint { Date = curr.Date, Equity = equity });

                if (equity > peakEquity) peakEquity = equity;
                double drawdown = (peakEquity - equity) / peakEquity;
                if (drawdown > maxDrawdown) maxDrawdown = drawdown;
                result.DrawdownCurve.Add(new DrawdownPoint { Date = curr.Date, Drawdown = drawdown });
            }

            // Close open position at end
            if (position != 0)
            {
                var last = historical.Last();
                var trade = result.Trades.LastOrDefault(t => !t.ExitDate.HasValue);
                if (trade != null)
                {
                    bool isBuy = position < 0; // If position is negative, we're covering (buying)

                    // Calculate exit price with slippage, spread
                    var (exitCost, effectiveExitPrice) = costModel.CalculateAllCosts(
                        (int)-position, last.Close, isBuy, last.Volume);

                    trade.ExitDate = last.Date;
                    trade.ExitPrice = effectiveExitPrice;
                    trade.ExitCosts = exitCost;

                    totalTransactionCosts += exitCost;
                    totalTradingVolume += Math.Abs(position) * effectiveExitPrice;

                    // Update cash positions
                    cash += (effectiveExitPrice - entryPrice) * position - exitCost;
                    grossCash += (last.Close - entryPrice) * position; // Without costs
                }
            }

            // Calculate returns
            result.GrossReturn = (grossCash - initialCapital) / initialCapital;
            result.TotalReturn = (cash - initialCapital) / initialCapital;
            result.MaxDrawdown = maxDrawdown;
            result.TotalTransactionCosts = totalTransactionCosts;
            result.TransactionCostsPercentage = totalTradingVolume > 0 ?
                totalTransactionCosts / totalTradingVolume : 0;

            // Calculate advanced performance metrics
            CalculateAdvancedMetrics(result, initialCapital);

            return result;
        }

        /// <summary>
        /// Runs a backtest for spread strategies using historical option chain simulation
        /// </summary>
        /// <param name="symbol">The underlying symbol</param>
        /// <param name="historical">Historical price data</param>
        /// <param name="spreadStrategy">The spread strategy to test</param>
        /// <param name="initialCapital">Starting capital amount</param>
        /// <param name="costModel">Transaction cost model (if null, zero costs are assumed)</param>
        /// <returns>Backtest results with spread-specific metrics</returns>
        public async Task<BacktestResult> RunSpreadBacktestAsync(
            string symbol,
            List<HistoricalPrice> historical,
            SpreadStrategyProfile spreadStrategy,
            double initialCapital = 10000,
            TransactionCostModel costModel = null)
        {
            // Use zero cost model if none provided
            costModel ??= TransactionCostModel.CreateZeroCostModel();

            var result = new BacktestResult
            {
                Symbol = symbol,
                TimeFrame = "daily",
                CostModel = costModel,
                AssetClass = "options_spread"
            };

            if (historical.Any())
            {
                result.StartDate = historical.First().Date;
                result.EndDate = historical.Last().Date;
            }

            // Initialize spread-specific results
            var spreadResults = new SpreadBacktestResult();
            result.SpreadResults = spreadResults;

            double cash = initialCapital;
            double peakEquity = initialCapital;
            double maxDrawdown = 0;
            double totalTransactionCosts = 0;

            // Track open spread positions
            var openPositions = new List<SimulatedSpreadTrade>();
            var cumulativePnL = 0.0;

            // Calculate buy-and-hold equity return for comparison
            double equityBuyHoldReturn = historical.Count > 1 ?
                (historical.Last().Close - historical.First().Close) / historical.First().Close : 0;

            for (int i = 1; i < historical.Count; i++)
            {
                var curr = historical[i];
                var signal = spreadStrategy.GenerateSignal(historical, i);

                // Check for new spread entries
                if (signal == "ENTER" && openPositions.Count < 5) // Limit concurrent positions
                {
                    var spreadTrade = CreateSpreadTrade(spreadStrategy.SpreadConfig, curr, spreadStrategy.RiskFreeRate);
                    if (spreadTrade != null && Math.Abs(spreadTrade.NetPremium) <= cash * 0.1) // Risk management
                    {
                        openPositions.Add(spreadTrade);
                        cash -= Math.Abs(spreadTrade.NetPremium);
                        totalTransactionCosts += CalculateSpreadTransactionCosts(spreadTrade, costModel);
                        spreadResults.TotalPremium += spreadTrade.NetPremium;
                    }
                }

                // Evaluate existing positions for exit conditions
                var positionsToClose = new List<SimulatedSpreadTrade>();
                foreach (var position in openPositions.ToList())
                {
                    var daysHeld = (curr.Date - position.EntryDate).TotalDays;
                    var currentPnL = CalculateSpreadPnL(position, curr.Close, spreadStrategy.RiskFreeRate, daysHeld);

                    bool shouldExit = false;
                    string exitReason = "";

                    // Check exit conditions
                    if (daysHeld >= 30) // Max hold period
                    {
                        shouldExit = true;
                        exitReason = "TIME_DECAY";
                    }
                    else if (currentPnL >= position.NetPremium * spreadStrategy.TargetProfitPercentage)
                    {
                        shouldExit = true;
                        exitReason = "TARGET_PROFIT";
                    }
                    else if (currentPnL <= position.NetPremium * spreadStrategy.StopLossPercentage)
                    {
                        shouldExit = true;
                        exitReason = "STOP_LOSS";
                    }

                    if (shouldExit)
                    {
                        position.ExitDate = curr.Date;
                        position.UnderlyingPriceAtExit = curr.Close;
                        position.ProfitLoss = currentPnL;
                        position.ExitReason = exitReason;

                        cash += position.NetPremium + currentPnL;
                        totalTransactionCosts += CalculateSpreadTransactionCosts(position, costModel);

                        positionsToClose.Add(position);
                        spreadResults.SpreadTrades.Add(position);
                    }
                }

                // Remove closed positions
                foreach (var closedPosition in positionsToClose)
                {
                    openPositions.Remove(closedPosition);
                }

                // Calculate current portfolio value (cash + unrealized P&L of open positions)
                double unrealizedPnL = 0;
                foreach (var position in openPositions)
                {
                    var daysHeld = (curr.Date - position.EntryDate).TotalDays;
                    unrealizedPnL += CalculateSpreadPnL(position, curr.Close, spreadStrategy.RiskFreeRate, daysHeld);
                }

                double currentEquity = cash + unrealizedPnL;
                result.EquityCurve.Add(new EquityPoint { Date = curr.Date, Equity = currentEquity });

                // Track cumulative P&L
                cumulativePnL = currentEquity - initialCapital;
                spreadResults.RollingPnL.Add(new SpreadPnLPoint
                {
                    Date = curr.Date,
                    CumulativePnL = cumulativePnL,
                    PeriodPnL = i > 1 ? cumulativePnL - spreadResults.RollingPnL.LastOrDefault()?.CumulativePnL ?? 0 : cumulativePnL,
                    UnderlyingPrice = curr.Close,
                    OpenPositions = openPositions.Count
                });

                // Update peak equity and drawdown
                if (currentEquity > peakEquity) peakEquity = currentEquity;
                double drawdown = (peakEquity - currentEquity) / peakEquity;
                if (drawdown > maxDrawdown) maxDrawdown = drawdown;
                result.DrawdownCurve.Add(new DrawdownPoint { Date = curr.Date, Drawdown = drawdown });
            }

            // Close any remaining open positions at expiration
            foreach (var position in openPositions)
            {
                var lastPrice = historical.Last();
                var daysHeld = (lastPrice.Date - position.EntryDate).TotalDays;
                position.ExitDate = lastPrice.Date;
                position.UnderlyingPriceAtExit = lastPrice.Close;
                position.ProfitLoss = CalculateSpreadPnL(position, lastPrice.Close, spreadStrategy.RiskFreeRate, daysHeld);
                position.ExitReason = "EXPIRATION";

                cash += position.NetPremium + position.ProfitLoss;
                spreadResults.SpreadTrades.Add(position);
            }

            // Calculate final returns and metrics
            result.TotalReturn = (cash - initialCapital) / initialCapital;
            result.MaxDrawdown = maxDrawdown;
            result.TotalTransactionCosts = totalTransactionCosts;

            // Calculate spread-specific metrics
            CalculateSpreadMetrics(spreadResults, equityBuyHoldReturn);

            // Calculate standard advanced metrics
            CalculateAdvancedMetrics(result, initialCapital);

            return result;
        }

        /// <summary>
        /// Runs a comprehensive backtest with extended historical data
        /// </summary>
        /// <param name="symbol">Asset symbol</param>
        /// <param name="strategy">Strategy to test</param>
        /// <param name="interval">Time interval</param>
        /// <param name="assetClass">Asset class (stock, forex, crypto, or auto)</param>
        /// <param name="initialCapital">Starting capital</param>
        /// <param name="tradeSize">Size per trade</param>
        /// <param name="costModel">Transaction cost model (null for zero costs)</param>
        /// <returns>Backtest results</returns>
        public async Task<BacktestResult> RunComprehensiveBacktestAsync(
            string symbol,
            StrategyProfile strategy,
            string interval = "daily",
            string assetClass = "auto",
            double initialCapital = 10000,
            int tradeSize = 1,
            TransactionCostModel costModel = null)
        {
            // Get the appropriate historical data based on asset class
            List<HistoricalPrice> historical = await _historicalDataService.GetComprehensiveHistoricalData(symbol, interval, assetClass);

            // Run the backtest with the data
            var result = await RunBacktestAsync(symbol, historical, strategy, initialCapital, tradeSize, costModel);

            // Set additional properties
            result.TimeFrame = interval;
            result.AssetClass = assetClass == "auto" ? DetermineAssetClass(symbol) : assetClass;

            return result;
        }

        /// <summary>
        /// Attempts to determine the asset class from the symbol
        /// </summary>
        /// <param name="symbol">Asset symbol</param>
        /// <returns>Asset class (stock, forex, crypto)</returns>
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

        /// <summary>
        /// Calculate advanced performance metrics for a backtest result
        /// </summary>
        /// <param name="result">The backtest result to analyze</param>
        /// <param name="initialCapital">Initial investment amount</param>
        private void CalculateAdvancedMetrics(BacktestResult result, double initialCapital)
        {
            // Debug logging
            System.Diagnostics.Debug.WriteLine($"CalculateAdvancedMetrics called:");
            System.Diagnostics.Debug.WriteLine($"  Equity curve count: {result.EquityCurve.Count}");
            System.Diagnostics.Debug.WriteLine($"  Trades count: {result.Trades.Count}");
            System.Diagnostics.Debug.WriteLine($"  Completed trades: {result.Trades.Count(t => t.ExitPrice.HasValue)}");
            
            // 1. Calculate daily returns from equity curve
            List<double> dailyReturns = new List<double>();
            List<double> dailyDownsideReturns = new List<double>(); // For Sortino ratio (only negative returns)

            for (int i = 1; i < result.EquityCurve.Count; i++)
            {
                double previousValue = result.EquityCurve[i - 1].Equity;
                double currentValue = result.EquityCurve[i].Equity;
                
                // Skip if previous value is 0 to avoid division by zero
                if (previousValue == 0)
                {
                    System.Diagnostics.Debug.WriteLine($"  WARNING: Previous equity value is 0 at index {i}");
                    continue;
                }
                
                double dailyReturn = (currentValue - previousValue) / previousValue;

                dailyReturns.Add(dailyReturn);

                // Store negative returns for Sortino calculation
                if (dailyReturn < 0)
                {
                    dailyDownsideReturns.Add(dailyReturn);
                }
            }
            
            System.Diagnostics.Debug.WriteLine($"  Daily returns calculated: {dailyReturns.Count}");
            if (dailyReturns.Count > 0)
            {
                System.Diagnostics.Debug.WriteLine($"  Average daily return: {dailyReturns.Average():F6}");
                System.Diagnostics.Debug.WriteLine($"  Min daily return: {dailyReturns.Min():F6}");
                System.Diagnostics.Debug.WriteLine($"  Max daily return: {dailyReturns.Max():F6}");
            }

            // 2. Calculate metrics

            // Sharpe Ratio = (Average return - Risk free rate) / Standard deviation of returns
            // Using 0% as risk-free rate for simplicity
            double riskFreeRate = 0.0;
            double averageReturn = dailyReturns.Count > 0 ? dailyReturns.Average() : 0;
            double returnStdDev = CalculateStandardDeviation(dailyReturns);
            
            System.Diagnostics.Debug.WriteLine($"  Return Std Dev: {returnStdDev:F6}");
            System.Diagnostics.Debug.WriteLine($"  Average return: {averageReturn:F6}");

            // Annualize Sharpe ratio (assuming 252 trading days per year for stocks)
            if (returnStdDev > 0.0001)  // Use small epsilon instead of exact zero check
            {
                result.SharpeRatio = (averageReturn - riskFreeRate) / returnStdDev * Math.Sqrt(252);
                System.Diagnostics.Debug.WriteLine($"  Calculated Sharpe Ratio: {result.SharpeRatio:F2}");
            }
            else
            {
                result.SharpeRatio = 0;
                System.Diagnostics.Debug.WriteLine($"  Sharpe Ratio set to 0 (stddev too low: {returnStdDev})");
            }

            // Sortino Ratio - Similar to Sharpe but only considers downside risk
            double downsideDeviation = CalculateStandardDeviation(dailyDownsideReturns);
            
            System.Diagnostics.Debug.WriteLine($"  Downside returns count: {dailyDownsideReturns.Count}");
            System.Diagnostics.Debug.WriteLine($"  Downside deviation: {downsideDeviation:F6}");
            
            if (downsideDeviation > 0.0001)  // Use small epsilon
            {
                result.SortinoRatio = (averageReturn - riskFreeRate) / downsideDeviation * Math.Sqrt(252);
                System.Diagnostics.Debug.WriteLine($"  Calculated Sortino Ratio: {result.SortinoRatio:F2}");
            }
            else
            {
                result.SortinoRatio = 0;
                System.Diagnostics.Debug.WriteLine($"  Sortino Ratio set to 0 (no downside or too low deviation)");
            }

            // CAGR (Compound Annual Growth Rate)
            double totalDays = (result.EndDate - result.StartDate).TotalDays;
            System.Diagnostics.Debug.WriteLine($"  Total days: {totalDays}");
            
            if (totalDays > 0 && result.EquityCurve.Count > 0)
            {
                double startValue = initialCapital;
                double endValue = result.EquityCurve.Last().Equity;
                
                System.Diagnostics.Debug.WriteLine($"  Start value: {startValue:C2}");
                System.Diagnostics.Debug.WriteLine($"  End value: {endValue:C2}");
                
                if (startValue > 0)
                {
                    result.CAGR = Math.Pow(endValue / startValue, 365.0 / totalDays) - 1;
                    System.Diagnostics.Debug.WriteLine($"  Calculated CAGR: {result.CAGR:P2}");
                }
                else
                {
                    result.CAGR = 0;
                    System.Diagnostics.Debug.WriteLine($"  CAGR set to 0 (start value is 0)");
                }
            }
            else
            {
                result.CAGR = 0;
                System.Diagnostics.Debug.WriteLine($"  CAGR set to 0 (no time period or no equity curve)");
            }

            // Calmar Ratio = CAGR / Maximum Drawdown
            System.Diagnostics.Debug.WriteLine($"  Max Drawdown: {result.MaxDrawdown:P2}");
            
            if (result.MaxDrawdown > 0.0001)  // Use small epsilon
            {
                result.CalmarRatio = result.CAGR / result.MaxDrawdown;
                System.Diagnostics.Debug.WriteLine($"  Calculated Calmar Ratio: {result.CalmarRatio:F2}");
            }
            else
            {
                result.CalmarRatio = 0;
                System.Diagnostics.Debug.WriteLine($"  Calmar Ratio set to 0 (no max drawdown)");
            }

            // Profit Factor = Gross profit divided by gross loss
            double grossProfit = result.Trades
                .Where(t => t.ExitPrice.HasValue && t.ProfitLoss > 0)
                .Sum(t => t.ProfitLoss);

            double grossLoss = Math.Abs(result.Trades
                .Where(t => t.ExitPrice.HasValue && t.ProfitLoss < 0)
                .Sum(t => t.ProfitLoss));
            
            System.Diagnostics.Debug.WriteLine($"  Gross profit: {grossProfit:C2}");
            System.Diagnostics.Debug.WriteLine($"  Gross loss: {grossLoss:C2}");

            if (grossLoss > 0.01)  // Small threshold to avoid meaningless ratios
            {
                result.ProfitFactor = grossProfit / grossLoss;
                System.Diagnostics.Debug.WriteLine($"  Calculated Profit Factor: {result.ProfitFactor:F2}");
            }
            else if (grossProfit > 0)
            {
                result.ProfitFactor = double.MaxValue;
                System.Diagnostics.Debug.WriteLine($"  Profit Factor set to MaxValue (no losses)");
            }
            else
            {
                result.ProfitFactor = 0;
                System.Diagnostics.Debug.WriteLine($"  Profit Factor set to 0 (no profit or loss)");
            }

            // Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error
            // Since we don't have a direct benchmark here, we'll use the risk-free rate as a simple benchmark
            // Tracking error is the standard deviation of the difference between portfolio returns and benchmark returns
            double excessReturn = averageReturn - riskFreeRate; // Excess return over risk-free rate
            
            if (returnStdDev > 0.0001)  // Use small epsilon
            {
                result.InformationRatio = excessReturn / returnStdDev * Math.Sqrt(252);
                System.Diagnostics.Debug.WriteLine($"  Calculated Information Ratio: {result.InformationRatio:F2}");
            }
            else
            {
                result.InformationRatio = 0;
                System.Diagnostics.Debug.WriteLine($"  Information Ratio set to 0 (no volatility)");
            }
            
            System.Diagnostics.Debug.WriteLine($"\nFinal Metrics Summary:");
            System.Diagnostics.Debug.WriteLine($"  Sharpe Ratio: {result.SharpeRatio:F2}");
            System.Diagnostics.Debug.WriteLine($"  Sortino Ratio: {result.SortinoRatio:F2}");
            System.Diagnostics.Debug.WriteLine($"  CAGR: {result.CAGR:P2}");
            System.Diagnostics.Debug.WriteLine($"  Calmar Ratio: {result.CalmarRatio:F2}");
            System.Diagnostics.Debug.WriteLine($"  Profit Factor: {result.ProfitFactor:F2}");
            System.Diagnostics.Debug.WriteLine($"  Information Ratio: {result.InformationRatio:F2}");
            System.Diagnostics.Debug.WriteLine($"  Win Rate: {result.WinRate:P2}");
            System.Diagnostics.Debug.WriteLine($"  Max Drawdown: {result.MaxDrawdown:P2}");
        }

        /// <summary>
        /// Calculate the standard deviation of a list of values
        /// </summary>
        /// <param name="values">List of numeric values</param>
        /// <returns>Standard deviation</returns>
        private double CalculateStandardDeviation(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0;

            double avg = values.Average();
            double sumOfSquaresOfDifferences = values.Sum(val => Math.Pow(val - avg, 2));
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
        }

        /// <summary>
        /// Run Monte Carlo simulation on a backtest result to assess risk
        /// </summary>
        /// <param name="result">The original backtest result</param>
        /// <param name="simulationCount">Number of simulations to run</param>
        /// <param name="initialCapital">Initial investment capital</param>
        /// <returns>The original backtest result with added Monte Carlo simulation results</returns>
        public BacktestResult RunMonteCarloSimulation(BacktestResult result, int simulationCount = 1000, double initialCapital = 10000)
        {
            if (result == null || result.EquityCurve.Count <= 1)
            {
                return result;
            }

            // Create a new Monte Carlo simulation result
            var mcResult = new MonteCarloSimulationResult
            {
                SimulationCount = simulationCount
            };

            // 1. Calculate daily returns from the original equity curve
            List<double> dailyReturns = new List<double>();
            for (int i = 1; i < result.EquityCurve.Count; i++)
            {
                double previousValue = result.EquityCurve[i - 1].Equity;
                double currentValue = result.EquityCurve[i].Equity;
                double dailyReturn = (currentValue - previousValue) / previousValue;
                dailyReturns.Add(dailyReturn);
            }

            // Record the dates for all simulated paths
            List<DateTime> dates = result.EquityCurve.Select(e => e.Date).ToList();

            // 2. Generate simulated equity curves using bootstrap resampling of daily returns
            Random random = new Random(42); // Fixed seed for reproducibility

            for (int sim = 0; sim < simulationCount; sim++)
            {
                double equity = initialCapital;
                List<EquityPoint> equityCurve = new List<EquityPoint>
                {
                    new EquityPoint { Date = dates[0], Equity = equity }
                };

                double peakEquity = equity;
                double maxDrawdown = 0;

                // Generate the path by randomly sampling from the original daily returns
                for (int day = 1; day < dates.Count; day++)
                {
                    // Randomly select a return from the original series (bootstrap resampling)
                    int randomIndex = random.Next(dailyReturns.Count);
                    double dailyReturn = dailyReturns[randomIndex];

                    // Apply the return to current equity
                    equity *= 1 + dailyReturn;

                    // Update peak equity and drawdown
                    if (equity > peakEquity)
                    {
                        peakEquity = equity;
                    }

                    double drawdown = (peakEquity - equity) / peakEquity;
                    if (drawdown > maxDrawdown)
                    {
                        maxDrawdown = drawdown;
                    }

                    equityCurve.Add(new EquityPoint { Date = dates[day], Equity = equity });
                }

                // Store this simulation's results
                mcResult.SimulatedEquityCurves.Add(equityCurve);
                mcResult.FinalValues.Add(equity);
                mcResult.MaxDrawdowns.Add(maxDrawdown);
            }

            // 3. Calculate statistics from all simulations

            // Sort final values to compute percentiles
            var sortedFinalValues = mcResult.FinalValues.OrderBy(v => v).ToList();
            var sortedMaxDrawdowns = mcResult.MaxDrawdowns.OrderBy(d => d).ToList();

            // Calculate return percentiles
            mcResult.ReturnPercentiles["5%"] = CalculatePercentile(sortedFinalValues, 5);
            mcResult.ReturnPercentiles["25%"] = CalculatePercentile(sortedFinalValues, 25);
            mcResult.ReturnPercentiles["50%"] = CalculatePercentile(sortedFinalValues, 50); // Median
            mcResult.ReturnPercentiles["75%"] = CalculatePercentile(sortedFinalValues, 75);
            mcResult.ReturnPercentiles["95%"] = CalculatePercentile(sortedFinalValues, 95);

            // Calculate drawdown percentiles
            mcResult.DrawdownPercentiles["5%"] = CalculatePercentile(sortedMaxDrawdowns, 5);
            mcResult.DrawdownPercentiles["25%"] = CalculatePercentile(sortedMaxDrawdowns, 25);
            mcResult.DrawdownPercentiles["50%"] = CalculatePercentile(sortedMaxDrawdowns, 50); // Median
            mcResult.DrawdownPercentiles["75%"] = CalculatePercentile(sortedMaxDrawdowns, 75);
            mcResult.DrawdownPercentiles["95%"] = CalculatePercentile(sortedMaxDrawdowns, 95);

            // Calculate risk metrics (VaR and CVaR)
            // Convert to percentage returns for VaR calculation
            List<double> percentReturns = mcResult.FinalValues.Select(v => (v - initialCapital) / initialCapital).ToList();
            percentReturns.Sort();

            int var95Index = (int)Math.Ceiling(simulationCount * 0.05) - 1;
            int var99Index = (int)Math.Ceiling(simulationCount * 0.01) - 1;

            // Ensure indices are within bounds
            var95Index = Math.Max(0, Math.Min(var95Index, percentReturns.Count - 1));
            var99Index = Math.Max(0, Math.Min(var99Index, percentReturns.Count - 1));

            // VaR is expressed as a positive value representing potential loss percentage
            mcResult.ValueAtRisk95 = Math.Abs(Math.Min(0, percentReturns[var95Index]));
            mcResult.ValueAtRisk99 = Math.Abs(Math.Min(0, percentReturns[var99Index]));

            // Calculate CVaR as the average of losses beyond VaR
            double sum95 = 0;
            int count95 = 0;
            for (int i = 0; i <= var95Index; i++)
            {
                if (percentReturns[i] < 0) // Only consider losses
                {
                    sum95 += percentReturns[i];
                    count95++;
                }
            }
            mcResult.ConditionalValueAtRisk95 = count95 > 0 ? Math.Abs(sum95 / count95) : 0;

            // Calculate probability metrics
            mcResult.ProbabilityOfProfit = (double)percentReturns.Count(r => r > 0) / simulationCount;
            double backtestReturn = (result.EquityCurve.Last().Equity - initialCapital) / initialCapital;
            mcResult.ProbabilityOfExceedingBacktestReturn = (double)percentReturns.Count(r => r >= backtestReturn) / simulationCount;

            // 4. Find representative paths for each percentile
            FindPercentileEquityCurves(mcResult, initialCapital);

            // 5. Attach Monte Carlo results to the backtest result
            result.MonteCarloResults = mcResult;

            return result;
        }

        /// <summary>
        /// Find representative equity curves for each percentile
        /// </summary>
        private void FindPercentileEquityCurves(MonteCarloSimulationResult mcResult, double initialCapital)
        {
            // For each percentile, find the simulation path that's closest to that percentile's final value
            string[] percentiles = new[] { "5%", "25%", "50%", "75%", "95%" };

            foreach (string percentile in percentiles)
            {
                double targetValue = mcResult.ReturnPercentiles[percentile];
                int closestIndex = 0;
                double closestDiff = double.MaxValue;

                // Find the simulation with final value closest to the percentile
                for (int i = 0; i < mcResult.SimulatedEquityCurves.Count; i++)
                {
                    double finalValue = mcResult.SimulatedEquityCurves[i].Last().Equity;
                    double diff = Math.Abs(finalValue - targetValue);

                    if (diff < closestDiff)
                    {
                        closestDiff = diff;
                        closestIndex = i;
                    }
                }

                // Store the representative path
                mcResult.PercentileEquityCurves[percentile] = mcResult.SimulatedEquityCurves[closestIndex];
            }
        }

        /// <summary>
        /// Calculate a specific percentile from a sorted list of values
        /// </summary>
        private double CalculatePercentile(List<double> sortedValues, int percentile)
        {
            if (sortedValues == null || !sortedValues.Any())
            {
                return 0;
            }

            if (percentile <= 0)
            {
                return sortedValues.First();
            }

            if (percentile >= 100)
            {
                return sortedValues.Last();
            }

            // Calculate the index in the sorted array
            double index = percentile / 100.0 * (sortedValues.Count - 1);
            int lowerIndex = (int)Math.Floor(index);
            int upperIndex = (int)Math.Ceiling(index);

            // Interpolate if necessary
            if (lowerIndex == upperIndex)
            {
                return sortedValues[lowerIndex];
            }
            else
            {
                double weight = index - lowerIndex;
                return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
            }
        }

        /// <summary>
        /// Creates a simulated spread trade based on the spread configuration
        /// </summary>
        private SimulatedSpreadTrade CreateSpreadTrade(SpreadConfiguration spreadConfig, HistoricalPrice currentPrice, double riskFreeRate)
        {
            if (spreadConfig?.Legs == null || !spreadConfig.Legs.Any())
                return null;

            // Create simulated option legs with estimated pricing
            var simulatedLegs = new List<OptionLeg>();
            double netPremium = 0;

            foreach (var leg in spreadConfig.Legs)
            {
                // Estimate option price using simplified Black-Scholes
                double timeToExpiry = 30.0 / 365.0; // Assume 30 days to expiry
                double volatility = 0.25; // Assume 25% implied volatility

                double optionPrice = EstimateOptionPrice(
                    currentPrice.Close,
                    leg.Option.StrikePrice,
                    timeToExpiry,
                    riskFreeRate,
                    volatility,
                    leg.Option.OptionType == "CALL");

                // Apply bid-ask spread (assume 5% spread)
                if (leg.Action == "BUY")
                    optionPrice *= 1.025; // Pay ask
                else
                    optionPrice *= 0.975; // Receive bid

                var simulatedLeg = new OptionLeg
                {
                    Option = new OptionData
                    {
                        StrikePrice = leg.Option.StrikePrice,
                        OptionType = leg.Option.OptionType,
                        ExpirationDate = currentPrice.Date.AddDays(30)
                    },
                    Action = leg.Action,
                    Quantity = leg.Quantity,
                    Price = optionPrice
                };

                simulatedLegs.Add(simulatedLeg);

                // Calculate net premium (positive = debit, negative = credit)
                double legPremium = optionPrice * leg.Quantity * 100; // Options are per 100 shares
                if (leg.Action == "BUY")
                    netPremium += legPremium;
                else
                    netPremium -= legPremium;
            }

            return new SimulatedSpreadTrade
            {
                EntryDate = currentPrice.Date,
                UnderlyingPriceAtEntry = currentPrice.Close,
                NetPremium = netPremium,
                SpreadConfig = new SpreadConfiguration
                {
                    SpreadType = spreadConfig.SpreadType,
                    UnderlyingSymbol = spreadConfig.UnderlyingSymbol,
                    Legs = simulatedLegs
                }
            };
        }

        /// <summary>
        /// Estimates option price using simplified Black-Scholes formula
        /// </summary>
        private double EstimateOptionPrice(double underlyingPrice, double strikePrice, double timeToExpiry, double riskFreeRate, double volatility, bool isCall)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
                return 0;

            double S = underlyingPrice;
            double K = strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double d2 = d1 - sigma * Math.Sqrt(T);

            if (isCall)
            {
                return S * CumulativeNormalDistribution(d1) - K * Math.Exp(-r * T) * CumulativeNormalDistribution(d2);
            }
            else
            {
                return K * Math.Exp(-r * T) * CumulativeNormalDistribution(-d2) - S * CumulativeNormalDistribution(-d1);
            }
        }

        /// <summary>
        /// Calculates current P&L for a spread position
        /// </summary>
        private double CalculateSpreadPnL(SimulatedSpreadTrade spreadTrade, double currentUnderlyingPrice, double riskFreeRate, double daysHeld)
        {
            if (spreadTrade?.SpreadConfig?.Legs == null)
                return 0;

            double totalPnL = 0;
            double timeToExpiry = Math.Max(0.01, (30 - daysHeld) / 365.0); // Remaining time
            double volatility = 0.25; // Assume constant volatility

            foreach (var leg in spreadTrade.SpreadConfig.Legs)
            {
                // Calculate current option value
                double currentOptionPrice = EstimateOptionPrice(
                    currentUnderlyingPrice,
                    leg.Option.StrikePrice,
                    timeToExpiry,
                    riskFreeRate,
                    volatility,
                    leg.Option.OptionType == "CALL");

                // Calculate P&L for this leg
                double pnl = (currentOptionPrice - leg.Price) * leg.Quantity * 100;

                // Reverse sign for short positions
                if (leg.Action == "SELL")
                    pnl = -pnl;

                totalPnL += pnl;
            }

            return totalPnL;
        }

        /// <summary>
        /// Calculates transaction costs for spread trades
        /// </summary>
        private double CalculateSpreadTransactionCosts(SimulatedSpreadTrade spreadTrade, TransactionCostModel costModel)
        {
            if (spreadTrade?.SpreadConfig?.Legs == null || costModel == null)
                return 0;

            double totalCosts = 0;

            foreach (var leg in spreadTrade.SpreadConfig.Legs)
            {
                // For options, typically charged per contract
                // Assume each leg has transaction costs
                var (cost, _) = costModel.CalculateAllCosts(leg.Quantity, leg.Price, leg.Action == "BUY", 1000);
                totalCosts += cost;
            }

            return totalCosts;
        }

        /// <summary>
        /// Calculates spread-specific metrics
        /// </summary>
        private void CalculateSpreadMetrics(SpreadBacktestResult spreadResults, double equityBuyHoldReturn)
        {
            if (spreadResults?.SpreadTrades == null || !spreadResults.SpreadTrades.Any())
                return;

            var completedTrades = spreadResults.SpreadTrades.Where(t => t.ExitDate.HasValue).ToList();

            if (completedTrades.Any())
            {
                // Calculate basic metrics
                spreadResults.MaxProfit = completedTrades.Max(t => t.ProfitLoss);
                spreadResults.MaxLoss = completedTrades.Min(t => t.ProfitLoss);
                spreadResults.AverageTimeInTrade = completedTrades.Average(t => t.DaysHeld);

                var profitableTrades = completedTrades.Where(t => t.ProfitLoss > 0).ToList();
                var losingTrades = completedTrades.Where(t => t.ProfitLoss < 0).ToList();

                spreadResults.ProfitableTradePercentage = (double)profitableTrades.Count / completedTrades.Count;
                spreadResults.AverageWinningTrade = profitableTrades.Any() ? profitableTrades.Average(t => t.ProfitLoss) : 0;
                spreadResults.AverageLosingTrade = losingTrades.Any() ? losingTrades.Average(t => t.ProfitLoss) : 0;

                // Calculate outperformance vs equity
                double totalSpreadReturn = completedTrades.Sum(t => t.ProfitLoss) / Math.Abs(spreadResults.TotalPremium);
                spreadResults.OutperformanceVsEquity = totalSpreadReturn - equityBuyHoldReturn;
            }
        }

        /// <summary>
        /// Cumulative normal distribution function for Black-Scholes
        /// </summary>
        private double CumulativeNormalDistribution(double x)
        {
            // Approximation using error function
            return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
        }

        /// <summary>
        /// Error function approximation
        /// </summary>
        private double Erf(double x)
        {
            // Abramowitz and Stegun approximation
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            int sign = x < 0 ? -1 : 1;
            x = Math.Abs(x);

            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

            return sign * y;
        }
    }
}
