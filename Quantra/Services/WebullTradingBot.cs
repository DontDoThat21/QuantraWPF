using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Quantra.Enums;
using Quantra.Models;
using Newtonsoft.Json;
using WebSocketSharp;
using Quantra.Services;
using System.Timers;

namespace Quantra.Services
{
    public class WebullTradingBot
    {
        private TradingMode tradingMode = TradingMode.Market;
        private RiskMode riskMode = RiskMode.Normal;
        private Dictionary<string, double> paperPortfolio = new Dictionary<string, double>();
        private static readonly HttpClient client = new HttpClient { Timeout = TimeSpan.FromSeconds(25) };
        private string accessToken;
        private List<string> volatileStocks = new List<string>();
        private List<string> personalSymbols = new List<string>();
        private string symbolFilePath = "symbols.txt";
        private Dictionary<string, double> trailingStopLoss = new Dictionary<string, double>();
        private Dictionary<string, double> takeProfitTargets = new Dictionary<string, double>();
        private Dictionary<string, (double price, DateTime timestamp)> marketPriceCache = new Dictionary<string, (double, DateTime)>();
        private WebSocket marketDataSocket;
        private TechnicalIndicatorService technicalIndicatorService;
        private HistoricalDataService historicalDataService;
        private readonly AlphaVantageService alphaVantageService;
        private Task _monitoringTask;
        
        // Trading hour and market session related fields
        private List<TimeOnly> _tradingHourRestrictions = new List<TimeOnly>();
        private MarketSession _enabledMarketSessions = MarketSession.All;
        private bool _emergencyStopActive = false;
        private Dictionary<string, List<ScheduledOrder>> _scheduledOrders = new Dictionary<string, List<ScheduledOrder>>();
        
        // Default market session times
        private TimeOnly _preMarketOpenTime = new TimeOnly(4, 0, 0); // 4:00 AM
        private TimeOnly _regularMarketOpenTime = new TimeOnly(9, 30, 0); // 9:30 AM
        private TimeOnly _regularMarketCloseTime = new TimeOnly(16, 0, 0); // 4:00 PM
        private TimeOnly _afterHoursCloseTime = new TimeOnly(20, 0, 0); // 8:00 PM
        
        // Target allocations for portfolio rebalancing
        private Dictionary<string, double> _targetAllocations = new Dictionary<string, double>();
        private Dictionary<string, TrailingStopInfo> _trailingStops = new Dictionary<string, TrailingStopInfo>();
        private Dictionary<string, DateTime> _timeBasedExits = new Dictionary<string, DateTime>();
        private Dictionary<string, TimeBasedExit> _timeBasedExitStrategies = new Dictionary<string, TimeBasedExit>();
        private Dictionary<string, DCAStrategy> _dcaStrategies = new Dictionary<string, DCAStrategy>();
        private Dictionary<string, DCAStrategy> _dollarCostAveraging = new Dictionary<string, DCAStrategy>();
        private CancellationTokenSource _monitoringCancellationTokenSource;
        private CancellationTokenSource _emergencyStopTokenSource = new CancellationTokenSource();
        
        // Rebalancing related fields
        private Dictionary<string, RebalancingProfile> _rebalancingProfiles = new Dictionary<string, RebalancingProfile>();
        private string _activeRebalancingProfileId = null;
        private System.Timers.Timer _rebalancingScheduleTimer = null;
        private DateTime _lastRebalanceCheckTime = DateTime.Now;

        // Remove IConfiguration from constructor
        public WebullTradingBot()
        {
            LoadSymbols();
            // Initialize the services without configuration
            technicalIndicatorService = new TechnicalIndicatorService();
            historicalDataService = new HistoricalDataService();
            alphaVantageService = new AlphaVantageService();
            
            // Initialize rebalancing profiles
            InitializeRebalancingProfiles();
            
            // Start the rebalancing scheduler
            StartRebalancingScheduler();
        }
        
        /// <summary>
        /// Initializes the default rebalancing profiles
        /// </summary>
        private void InitializeRebalancingProfiles()
        {
            try
            {
                // Create a conservative allocation profile (lower risk, income-focused)
                var conservativeProfile = new RebalancingProfile
                {
                    Name = "Conservative Allocation",
                    TargetAllocations = new Dictionary<string, double>
                    {
                        { "AGG", 0.60 },   // 60% bonds
                        { "VTI", 0.20 },   // 20% stocks
                        { "GLD", 0.10 },   // 10% gold
                        { "SCHD", 0.10 }   // 10% dividend stocks
                    },
                    TolerancePercentage = 0.03,
                    RiskLevel = RebalancingRiskLevel.Conservative,
                    Schedule = RebalancingSchedule.Monthly
                };
                
                // Create a balanced allocation profile (moderate risk)
                var balancedProfile = new RebalancingProfile
                {
                    Name = "Balanced Allocation",
                    TargetAllocations = new Dictionary<string, double>
                    {
                        { "VTI", 0.40 },   // 40% stocks
                        { "AGG", 0.40 },   // 40% bonds
                        { "QQQ", 0.10 },   // 10% tech
                        { "GLD", 0.10 }    // 10% gold
                    },
                    TolerancePercentage = 0.04,
                    RiskLevel = RebalancingRiskLevel.Balanced,
                    Schedule = RebalancingSchedule.Monthly
                };
                
                // Create a growth allocation profile (higher risk, growth-focused)
                var growthProfile = new RebalancingProfile
                {
                    Name = "Growth Allocation",
                    TargetAllocations = new Dictionary<string, double>
                    {
                        { "VTI", 0.50 },   // 50% stocks
                        { "QQQ", 0.25 },   // 25% tech
                        { "AGG", 0.15 },   // 15% bonds
                        { "REET", 0.10 }   // 10% real estate
                    },
                    TolerancePercentage = 0.05,
                    RiskLevel = RebalancingRiskLevel.Growth,
                    Schedule = RebalancingSchedule.Quarterly
                };
                
                // Add profiles to the dictionary
                _rebalancingProfiles[conservativeProfile.ProfileId] = conservativeProfile;
                _rebalancingProfiles[balancedProfile.ProfileId] = balancedProfile;
                _rebalancingProfiles[growthProfile.ProfileId] = growthProfile;
                
                // Set the balanced profile as active by default
                _activeRebalancingProfileId = balancedProfile.ProfileId;
                
                DatabaseMonolith.Log("Info", $"Initialized default rebalancing profiles: Conservative, Balanced, Growth");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to initialize rebalancing profiles", ex.ToString());
            }
        }
        
        /// <summary>
        /// Starts the background monitoring of positions for trailing stops and other time-based events
        /// </summary>
        private void StartMonitoring()
        {
            try
            {
                _monitoringCancellationTokenSource = new CancellationTokenSource();
                _monitoringTask = Task.Run(async () => 
                {
                    await MonitorPositions(_monitoringCancellationTokenSource.Token);
                });
                
                DatabaseMonolith.Log("Info", "Position monitoring started successfully");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to start position monitoring", ex.ToString());
            }
        }
        
        /// <summary>
        /// Stops the background monitoring of positions
        /// </summary>
        private void StopMonitoring()
        {
            try
            {
                if (_monitoringCancellationTokenSource != null)
                {
                    _monitoringCancellationTokenSource.Cancel();
                    _monitoringCancellationTokenSource = null;
                }
                
                DatabaseMonolith.Log("Info", "Position monitoring stopped");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error stopping position monitoring", ex.ToString());
            }
        }
        
        /// <summary>
        /// Starts the scheduler to check for automated rebalancing opportunities
        /// </summary>
        private void StartRebalancingScheduler()
        {
            try
            {
                // Stop any existing timer
                if (_rebalancingScheduleTimer != null)
                {
                    _rebalancingScheduleTimer.Stop();
                    _rebalancingScheduleTimer.Dispose();
                }
                
                // Create a new timer that checks every hour
                _rebalancingScheduleTimer = new System.Timers.Timer(60 * 60 * 1000); // 1 hour
                _rebalancingScheduleTimer.Elapsed += OnRebalancingTimerElapsed;
                _rebalancingScheduleTimer.AutoReset = true;
                _rebalancingScheduleTimer.Start();
                
                DatabaseMonolith.Log("Info", "Portfolio rebalancing scheduler started (checking hourly)");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to start rebalancing scheduler", ex.ToString());
            }
        }
        
        /// <summary>
        /// Handles the rebalancing timer elapsed event to check for scheduled rebalancing
        /// </summary>
        private void OnRebalancingTimerElapsed(object sender, ElapsedEventArgs e)
        {
            // Prevent reentrancy
            if ((DateTime.Now - _lastRebalanceCheckTime).TotalMinutes < 30)
                return;
                
            _lastRebalanceCheckTime = DateTime.Now;
            
            // Check if rebalancing is needed
            CheckScheduledRebalancing();
        }
        
        /// <summary>
        /// Checks if a scheduled rebalancing should occur based on active profile
        /// </summary>
        private async void CheckScheduledRebalancing()
        {
            try
            {
                if (_activeRebalancingProfileId == null || !_rebalancingProfiles.ContainsKey(_activeRebalancingProfileId))
                    return;
                    
                var activeProfile = _rebalancingProfiles[_activeRebalancingProfileId];
                
                // Check if rebalancing is scheduled for now
                if (activeProfile.NextScheduledRebalance.HasValue && 
                    DateTime.Now >= activeProfile.NextScheduledRebalance.Value)
                {
                    // Check market session
                    if (!IsTradingAllowed())
                    {
                        DatabaseMonolith.Log("Info", "Scheduled rebalancing deferred: Trading not allowed at this time based on market session");
                        return;
                    }
                    
                    DatabaseMonolith.Log("Info", $"Executing scheduled portfolio rebalancing for profile: {activeProfile.Name}");
                    
                    // Execute rebalancing using the active profile
                    bool result = await RebalancePortfolioWithProfile(activeProfile);
                    
                    if (result)
                    {
                        // Update last rebalance date and calculate next scheduled date
                        activeProfile.LastRebalanceDate = DateTime.Now;
                        DatabaseMonolith.Log("Info", $"Scheduled rebalancing completed successfully. Next scheduled: {activeProfile.NextScheduledRebalance?.ToString("g")}");
                    }
                    else
                    {
                        DatabaseMonolith.Log("Warning", "Scheduled rebalancing failed");
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error during scheduled rebalancing check", ex.ToString());
            }
        }

        public void SetTradingMode(TradingMode mode)
        {
            tradingMode = mode;
        }

        public void SetRiskMode(RiskMode mode)
        {
            riskMode = mode;
        }

        public async Task ExecuteGoodFaithValueTrading()
        {
            // Implement Good Faith Value trading logic
        }

        public async Task<bool> Authenticate(string username, string password)
        {
            // I'm just here to hold your hand when you die
            // And to show you around imaginary places
            // Put money lumps in my bloody stump
            return true;
        }

        public virtual async Task<double> GetMarketPrice(string symbol)
        {
            // Use Alpha Vantage for market price
            var alphaVantageService = new AlphaVantageService();
            var price = await alphaVantageService.GetQuoteData(symbol);
            return price;
        }

        public async Task<List<string>> GetMostVolatileStocks()
        {
            try
            {
                return await alphaVantageService.GetMostVolatileStocksAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error scraping most volatile stocks: {ex.Message}");
                // Fallback to static error list
                return new List<string> { "GME", "AMC", "NIO" };
            }
        }

        /// <summary>
        /// Checks if trading is currently allowed based on time restrictions
        /// </summary>
        /// <returns>True if trading is allowed</returns>
        public bool IsTradingAllowed()
        {
            // Check emergency stop first
            if (_emergencyStopActive)
            {
                return false;
            }
            
            // Get current time
            TimeOnly now = TimeOnly.FromDateTime(DateTime.Now);
            
            // If no market sessions are enabled, then trading is not allowed
            if (_enabledMarketSessions == MarketSession.None)
            {
                return false;
            }
            
            // Check if current time falls within any enabled market session
            bool inEnabledSession = IsInEnabledMarketSession(now);
            
            // If custom trading hour restrictions are set, check those too
            if (_tradingHourRestrictions.Count > 0)
            {
                TimeOnly marketOpen = _tradingHourRestrictions[0];
                TimeOnly marketClose = _tradingHourRestrictions[1];
                
                // Must be within both custom hours AND an enabled session
                if (now < marketOpen || now > marketClose)
                {
                    return false;
                }
            }
            
            return inEnabledSession;
        }

        public async Task ExecuteOptimalRiskTrading()
        {
            // Check if trading is allowed based on our market session and time restrictions
            if (!IsTradingAllowed())
            {
                DatabaseMonolith.Log("Warning", $"ExecuteOptimalRiskTrading called but trading is not allowed at this time based on market session filters");
                return; // Exit early if trading is not allowed
            }
            
            volatileStocks = await GetMostVolatileStocks();
            volatileStocks = volatileStocks.Where(stock => IsHighLiquidity(stock) && HasRecentNewsCatalyst(stock)).ToList();

            var tasks = volatileStocks.Select(async stock =>
            {
                string[] timeframes = { "1min", "5min", "15min" };
                foreach (var timeframe in timeframes)
                {
                    double price = await GetMarketPrice(stock);
                    var (macd, signal) = await technicalIndicatorService.GetMACD(stock, timeframe);
                    double vwap = await technicalIndicatorService.GetVWAP(stock, timeframe);
                    double adx = await technicalIndicatorService.GetADX(stock, timeframe);
                    double rsi = await technicalIndicatorService.GetRSI(stock, timeframe);
                    double roc = await technicalIndicatorService.GetROC(stock, timeframe);
                    var (high, low) = await technicalIndicatorService.GetHighsLows(stock, timeframe);
                    double ultimateOscillator = await technicalIndicatorService.GetUltimateOscillator(stock, timeframe);
                    var (bullPower, bearPower) = await technicalIndicatorService.GetBullBearPower(stock, timeframe);
                    double cci = await technicalIndicatorService.GetCCI(stock, timeframe);
                    double atr = await technicalIndicatorService.GetATR(stock, timeframe);
                    double williamsR = await technicalIndicatorService.GetWilliamsR(stock, timeframe);
                    double stochrsi = await technicalIndicatorService.GetSTOCHRSI(stock, timeframe);
                    var (stochK, stochD) = await technicalIndicatorService.GetSTOCH(stock, timeframe);

                    if (macd > signal && price > vwap && adx > 25 && rsi < 30 && roc > 0 && high > low && ultimateOscillator > 50 && bullPower > bearPower && cci > 100 && atr > 1 && williamsR < -80 && stochrsi < 0.2 && stochK > stochD)
                    {
                        await PlaceLimitOrder(stock, CalculateOrderSize(price, macd, signal, adx), "BUY", price);
                        SetTrailingStopLoss(stock, price);
                    }
                    else if (macd < signal && price < vwap && adx > 25 && rsi > 70 && roc < 0 && high < low && ultimateOscillator < 50 && bullPower < bearPower && cci < -100 && atr > 1 && williamsR > -20 && stochrsi > 0.8 && stochK < stochD)
                    {
                        await PlaceLimitOrder(stock, CalculateOrderSize(price, macd, signal, adx), "SELL", price);
                        SetTrailingStopLoss(stock, price);
                    }
                }
            });
            await Task.WhenAll(tasks);
        }

        private void HandleMarketData(string data)
        {
            Console.WriteLine($"Real-time data received: {data}");
        }

        private int CalculateOrderSize(double price, double macd, double signal, double adx)
        {
            double confidence = Math.Abs(macd - signal) * adx / 50;
            return (int)(confidence * 100);
        }

        /// <summary>
        /// Calculates position size based on risk parameters
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="price">Current market price</param>
        /// <param name="stopLossPrice">Stop loss price</param>
        /// <param name="riskPercentage">Percentage of account to risk (0.01 = 1%)</param>
        /// <param name="accountSize">Total account size in dollars</param>
        /// <returns>Position size in shares</returns>
        public int CalculatePositionSizeByRisk(string symbol, double price, double stopLossPrice, double riskPercentage, double accountSize)
        {
            try
            {
                // Create position sizing parameters with default method (FixedRisk)
                var parameters = new PositionSizingParameters
                {
                    Symbol = symbol,
                    Price = price,
                    StopLossPrice = stopLossPrice,
                    RiskPercentage = riskPercentage,
                    AccountSize = accountSize,
                    RiskMode = this.riskMode,
                    Method = PositionSizingMethod.FixedRisk
                };
                
                // Calculate position size using the default method
                return CalculatePositionSize(parameters);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate position size for {symbol}", ex.ToString());
                return 0;
            }
        }
        
        /// <summary>
        /// Calculates position size using the AdaptiveRisk method with market conditions
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="price">Current market price</param>
        /// <param name="stopLossPrice">Stop loss price</param>
        /// <param name="basePositionPercentage">Base position percentage (e.g., 0.01 = 1%)</param>
        /// <param name="accountSize">Account size in dollars</param>
        /// <param name="volatilityFactor">Market volatility factor (-1.0 to 1.0)</param>
        /// <param name="performanceFactor">Recent performance factor (-1.0 to 1.0)</param>
        /// <param name="trendStrengthFactor">Trend strength factor (0.0 to 1.0)</param>
        /// <returns>Position size in shares</returns>
        public int CalculatePositionSizeByAdaptiveRisk(string symbol, double price, double stopLossPrice, 
            double basePositionPercentage, double accountSize, double volatilityFactor = 0.0, 
            double performanceFactor = 0.0, double trendStrengthFactor = 0.5)
        {
            try
            {
                // Create position sizing parameters for AdaptiveRisk method
                var parameters = new PositionSizingParameters
                {
                    Symbol = symbol,
                    Price = price,
                    StopLossPrice = stopLossPrice,
                    BasePositionPercentage = basePositionPercentage,
                    AccountSize = accountSize,
                    RiskMode = this.riskMode,
                    Method = PositionSizingMethod.AdaptiveRisk,
                    MarketVolatilityFactor = volatilityFactor,
                    PerformanceFactor = performanceFactor,
                    TrendStrengthFactor = trendStrengthFactor
                };
                
                // Calculate position size using the adaptive risk method
                return CalculatePositionSize(parameters);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate adaptive position size for {symbol}", ex.ToString());
                return 0;
            }
        }
        
        /// <summary>
        /// Calculates position size using specified parameters and sizing method
        /// </summary>
        /// <param name="parameters">Position sizing parameters</param>
        /// <returns>Position size in shares</returns>
        public int CalculatePositionSize(PositionSizingParameters parameters)
        {
            try
            {
                // Validate common parameters
                if (parameters == null || string.IsNullOrWhiteSpace(parameters.Symbol) || 
                    parameters.Price <= 0 || parameters.AccountSize <= 0)
                {
                    DatabaseMonolith.Log("Warning", "Invalid parameters for position sizing calculation");
                    return 0;
                }
                
                // Apply risk mode adjustments
                AdjustParametersForRiskMode(parameters);
                
                // Calculate position size based on the specified method
                int shares;
                switch (parameters.Method)
                {
                    case PositionSizingMethod.PercentageOfEquity:
                        shares = CalculatePositionSizeByEquityPercentage(parameters);
                        break;
                    case PositionSizingMethod.VolatilityBased:
                        shares = CalculatePositionSizeByVolatility(parameters);
                        break;
                    case PositionSizingMethod.KellyFormula:
                        shares = CalculatePositionSizeByKellyFormula(parameters);
                        break;
                    case PositionSizingMethod.FixedAmount:
                        shares = CalculatePositionSizeByFixedAmount(parameters);
                        break;
                    case PositionSizingMethod.TierBased:
                        shares = CalculatePositionSizeByTiers(parameters);
                        break;
                    case PositionSizingMethod.AdaptiveRisk:
                        shares = CalculatePositionSizeByAdaptiveRisk(parameters);
                        break;
                    case PositionSizingMethod.FixedRisk:
                    default:
                        shares = CalculatePositionSizeByFixedRisk(parameters);
                        break;
                }
                
                // Apply maximum position size constraint
                int maxShares = CalculateMaxPositionSize(parameters);
                shares = Math.Min(shares, maxShares);
                
                // Log the calculation
                string methodName = parameters.Method.ToString();
                DatabaseMonolith.Log("Info", $"Position sizing for {parameters.Symbol} using {methodName}: " +
                    $"{shares} shares at {parameters.Price:C2} with {parameters.RiskMode} risk mode");
                
                return shares;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate position size for {parameters.Symbol}", ex.ToString());
                return 0;
            }
        }
        
        /// <summary>
        /// Adjusts position sizing parameters based on risk mode
        /// </summary>
        private void AdjustParametersForRiskMode(PositionSizingParameters parameters)
        {
            switch (parameters.RiskMode)
            {
                case RiskMode.Conservative:
                    // Conservative mode - lower risk percentage and lower maximums
                    parameters.RiskPercentage *= 0.5; // Half the risk percentage
                    parameters.MaxPositionSizePercent *= 0.7; // 70% of normal maximum size
                    parameters.KellyFractionMultiplier = 0.3; // Lower Kelly fraction
                    parameters.ATRMultiple *= 0.7; // Tighter ATR multiple
                    break;
                
                case RiskMode.Moderate:
                    // Moderate mode - slightly reduced risk factors
                    parameters.RiskPercentage *= 0.8; // 80% of normal risk
                    parameters.KellyFractionMultiplier = 0.4; // Moderate Kelly fraction
                    break;
                
                case RiskMode.Aggressive:
                    // Aggressive mode - increased risk factors
                    parameters.RiskPercentage *= 1.5; // 150% of normal risk
                    parameters.MaxPositionSizePercent *= 1.3; // 130% of normal maximum size
                    parameters.KellyFractionMultiplier = 0.7; // Higher Kelly fraction
                    parameters.ATRMultiple *= 1.2; // Wider ATR multiple
                    break;
                
                case RiskMode.GoodFaithValue:
                    // Good Faith Value uses available cash rather than full account size
                    // This is just a placeholder - in a real implementation, we would get actual GFV
                    parameters.AccountSize *= 0.9; // Use 90% of account size as a proxy for GFV
                    break;
                    
                case RiskMode.Normal:
                default:
                    // No adjustments for normal mode
                    break;
            }
        }
        
        /// <summary>
        /// Calculates maximum allowed position size based on account size and maximum percentage
        /// </summary>
        private int CalculateMaxPositionSize(PositionSizingParameters parameters)
        {
            // Calculate maximum dollar amount allowed for the position
            double maxDollarAmount = parameters.AccountSize * parameters.MaxPositionSizePercent;
            
            // Calculate maximum shares based on price
            int maxShares = (int)Math.Floor(maxDollarAmount / parameters.Price);
            
            return maxShares;
        }
        
        /// <summary>
        /// Calculates position size based on fixed risk percentage of account
        /// </summary>
        private int CalculatePositionSizeByFixedRisk(PositionSizingParameters parameters)
        {
            // Calculate risk per share
            double riskPerShare = Math.Abs(parameters.Price - parameters.StopLossPrice);
            
            if (riskPerShare <= 0)
            {
                DatabaseMonolith.Log("Warning", $"Invalid risk parameters for {parameters.Symbol}: " +
                    $"Price {parameters.Price:C2}, Stop Loss {parameters.StopLossPrice:C2}");
                return 0;
            }
            
            // Calculate dollar amount to risk
            double riskAmount = parameters.AccountSize * parameters.RiskPercentage;
            
            // Calculate number of shares
            int shares = (int)Math.Floor(riskAmount / riskPerShare);
            
            return shares;
        }
        
        /// <summary>
        /// Calculates position size based on a fixed percentage of account equity
        /// </summary>
        private int CalculatePositionSizeByEquityPercentage(PositionSizingParameters parameters)
        {
            // Calculate dollar amount to allocate based on equity percentage
            double positionAmount = parameters.AccountSize * parameters.RiskPercentage;
            
            // Calculate number of shares based on current price
            int shares = (int)Math.Floor(positionAmount / parameters.Price);
            
            return shares;
        }
        
        /// <summary>
        /// Calculates position size based on volatility (ATR)
        /// </summary>
        private int CalculatePositionSizeByVolatility(PositionSizingParameters parameters)
        {
            // If ATR is not provided, try to retrieve it
            double atr = parameters.ATR ?? 0;
            
            if (atr <= 0)
            {
                try
                {
                    // Try to get ATR from technical indicator service
                    atr = technicalIndicatorService.GetATR(parameters.Symbol, "1d").Result;
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Warning", $"Failed to retrieve ATR for {parameters.Symbol}, using estimate", ex.ToString());
                    
                    // Estimate ATR as a percentage of price if retrieval fails (e.g., 2% of price)
                    atr = parameters.Price * 0.02;
                }
            }
            
            // Calculate dollar amount to risk based on risk percentage
            double riskAmount = parameters.AccountSize * parameters.RiskPercentage;
            
            // Calculate position size based on ATR multiple for the stop loss distance
            double riskPerShare = atr * parameters.ATRMultiple;
            
            if (riskPerShare <= 0)
            {
                DatabaseMonolith.Log("Warning", $"Invalid risk per share for {parameters.Symbol}: ATR={atr}, Multiple={parameters.ATRMultiple}");
                return 0;
            }
            
            // Calculate number of shares
            int shares = (int)Math.Floor(riskAmount / riskPerShare);
            
            return shares;
        }
        
        /// <summary>
        /// Calculates position size based on Kelly formula
        /// </summary>
        private int CalculatePositionSizeByKellyFormula(PositionSizingParameters parameters)
        {
            // Kelly fraction = (winRate * rewardRiskRatio - (1 - winRate)) / rewardRiskRatio
            double winRate = parameters.WinRate;
            double rewardRiskRatio = parameters.RewardRiskRatio;
            
            // Calculate the Kelly fraction (percentage of capital to risk)
            double kellyFraction = (winRate * rewardRiskRatio - (1 - winRate)) / rewardRiskRatio;
            
            // Apply the Kelly fraction multiplier to be more conservative (usually 0.25 to 0.5)
            kellyFraction *= parameters.KellyFractionMultiplier;
            
            // Ensure the Kelly fraction is positive and reasonable
            kellyFraction = Math.Max(0, Math.Min(kellyFraction, parameters.MaxPositionSizePercent));
            
            // Use the Kelly fraction to calculate dollar amount to risk
            double kellyAmount = parameters.AccountSize * kellyFraction;
            
            // Calculate risk per share
            double riskPerShare = Math.Abs(parameters.Price - parameters.StopLossPrice);
            
            if (riskPerShare <= 0)
            {
                // Default to a percentage of price (e.g., 2%) if stop loss isn't defined
                riskPerShare = parameters.Price * 0.02;
            }
            
            // Calculate number of shares
            int shares = (int)Math.Floor(kellyAmount / riskPerShare);
            
            return shares;
        }
        
        /// <summary>
        /// Calculates position size based on a fixed dollar amount per trade
        /// </summary>
        private int CalculatePositionSizeByFixedAmount(PositionSizingParameters parameters)
        {
            // Calculate number of shares based on the fixed amount allocated per trade
            int shares = (int)Math.Floor(parameters.FixedAmount / parameters.Price);
            
            return shares;
        }
        
        /// <summary>
        /// Calculates position size using a tiered approach based on signal confidence
        /// </summary>
        private int CalculatePositionSizeByTiers(PositionSizingParameters parameters)
        {
            double confidence = parameters.Confidence;
            double baseRiskPercentage = parameters.RiskPercentage;
            
            // Adjust risk percentage based on confidence tier
            double adjustedRiskPercentage;
            
            if (confidence >= 0.9)
            {
                // Highest tier - full risk percentage
                adjustedRiskPercentage = baseRiskPercentage * 1.2; // 120% of base
            }
            else if (confidence >= 0.75)
            {
                // High tier - 80% of risk percentage
                adjustedRiskPercentage = baseRiskPercentage;
            }
            else if (confidence >= 0.6)
            {
                // Medium tier - 60% of risk percentage
                adjustedRiskPercentage = baseRiskPercentage * 0.75;
            }
            else
            {
                // Low tier - 40% of risk percentage
                adjustedRiskPercentage = baseRiskPercentage * 0.5;
            }
            
            // Update the risk percentage and calculate using fixed risk method
            parameters.RiskPercentage = adjustedRiskPercentage;
            return CalculatePositionSizeByFixedRisk(parameters);
        }
        
        /// <summary>
        /// Calculates position size using adaptive risk sizing based on multiple factors
        /// including market volatility, recent performance, and trend strength
        /// </summary>
        private int CalculatePositionSizeByAdaptiveRisk(PositionSizingParameters parameters)
        {
            // Start with base position percentage
            double basePositionPercentage = parameters.BasePositionPercentage;
            
            // Extract factors from parameters
            double volatilityFactor = parameters.MarketVolatilityFactor;
            double performanceFactor = parameters.PerformanceFactor;
            double trendStrengthFactor = parameters.TrendStrengthFactor;
            
            // 1. Adjust for market volatility
            // Reduce position size in high/increasing volatility, increase in low/decreasing volatility
            double volatilityAdjustment = 1.0;
            if (volatilityFactor > 0.3)
            {
                // High/increasing volatility - reduce position size
                volatilityAdjustment = 1.0 - (volatilityFactor * 0.5);
            }
            else if (volatilityFactor < -0.3)
            {
                // Low/decreasing volatility - potentially increase position size
                volatilityAdjustment = 1.0 + (Math.Abs(volatilityFactor) * 0.3);
            }
            
            // 2. Adjust for recent performance
            // Reduce position size after losses, carefully increase after gains
            double performanceAdjustment = 1.0;
            if (performanceFactor < -0.2)
            {
                // Recent losses - reduce position size for risk management
                performanceAdjustment = 1.0 - (Math.Abs(performanceFactor) * 0.6);
            }
            else if (performanceFactor > 0.2)
            {
                // Recent gains - modest increase in position size
                performanceAdjustment = 1.0 + (performanceFactor * 0.3);
            }
            
            // 3. Adjust for trend strength
            // Increase position size in stronger trends, reduce in weaker trends
            double trendAdjustment = 0.8 + (trendStrengthFactor * 0.4); // Range from 0.8 to 1.2
            
            // Calculate adjusted risk percentage
            double adjustedRiskPercentage = basePositionPercentage * volatilityAdjustment * performanceAdjustment * trendAdjustment;
            
            // Apply min/max constraints for safety
            double minRiskPercentage = parameters.BasePositionPercentage * 0.3; // Never go below 30% of base
            double maxRiskPercentage = parameters.BasePositionPercentage * 2.0; // Never go above 200% of base
            adjustedRiskPercentage = Math.Max(minRiskPercentage, Math.Min(adjustedRiskPercentage, maxRiskPercentage));
            
            // Log the adaptive calculation factors
            DatabaseMonolith.Log("Info", $"Adaptive sizing for {parameters.Symbol}: Base={parameters.BasePositionPercentage:P2}, " +
                $"Vol={volatilityFactor:F2} (adj={volatilityAdjustment:F2}), " +
                $"Perf={performanceFactor:F2} (adj={performanceAdjustment:F2}), " +
                $"Trend={trendStrengthFactor:F2} (adj={trendAdjustment:F2}), " +
                $"Final Risk={adjustedRiskPercentage:P2}");
            
            // Update risk percentage and use fixed risk calculation
            parameters.RiskPercentage = adjustedRiskPercentage;
            return CalculatePositionSizeByFixedRisk(parameters);
        }
        
        /// <summary>
        /// Sets up a share-based dollar-cost averaging strategy for a symbol
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="totalShares">Total shares to acquire</param>
        /// <param name="numberOfOrders">Number of orders to split into</param>
        /// <param name="intervalDays">Days between each order</param>
        /// <param name="strategyType">Distribution strategy type</param>
        /// <returns>Strategy ID if successful, null if failed</returns>
        public string SetupDollarCostAveraging(string symbol, int totalShares, int numberOfOrders, int intervalDays, 
            DCAStrategyType strategyType = DCAStrategyType.Equal)
        {
            try
            {
                // Validate parameters
                if (string.IsNullOrWhiteSpace(symbol) || totalShares <= 0 || numberOfOrders <= 0 || intervalDays < 0)
                {
                    DatabaseMonolith.Log("Error", $"Invalid parameters for DCA strategy for {symbol}: shares={totalShares}, orders={numberOfOrders}, interval={intervalDays}");
                    return null;
                }
                
                int sharesPerOrder = totalShares / numberOfOrders;
                if (sharesPerOrder <= 0) sharesPerOrder = 1;
                
                // Create new DCA strategy
                var strategy = new DCAStrategy
                {
                    Symbol = symbol,
                    Name = $"{symbol} Share-Based DCA",
                    IsShareBased = true,
                    TotalShares = totalShares,
                    SharesPerOrder = sharesPerOrder,
                    OrdersRemaining = numberOfOrders,
                    IntervalDays = intervalDays,
                    StrategyType = strategyType
                };
                
                // Add strategy to dictionary using strategy ID
                _dollarCostAveraging[strategy.StrategyId] = strategy;
                
                // Schedule the first order
                ScheduleDollarCostAveragingOrder(strategy.StrategyId);
                
                DatabaseMonolith.Log("Info", $"Share-based dollar-cost averaging set up for {symbol} (ID: {strategy.StrategyId}): " +
                    $"{totalShares} shares over {numberOfOrders} orders every {intervalDays} days using {strategyType} distribution");
                
                return strategy.StrategyId;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to set up dollar-cost averaging for {symbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Sets up a dollar-based dollar-cost averaging strategy for a symbol
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="totalAmount">Total dollar amount to invest</param>
        /// <param name="numberOfOrders">Number of orders to split into</param>
        /// <param name="intervalDays">Days between each order</param>
        /// <param name="strategyType">Distribution strategy type</param>
        /// <returns>Strategy ID if successful, null if failed</returns>
        public string SetupDollarCostAveraging(string symbol, double totalAmount, int numberOfOrders, int intervalDays, 
            DCAStrategyType strategyType = DCAStrategyType.Equal)
        {
            try
            {
                // Validate parameters
                if (string.IsNullOrWhiteSpace(symbol) || totalAmount <= 0 || numberOfOrders <= 0 || intervalDays < 0)
                {
                    DatabaseMonolith.Log("Error", $"Invalid parameters for DCA strategy for {symbol}: amount=${totalAmount}, orders={numberOfOrders}, interval={intervalDays}");
                    return null;
                }
                
                double amountPerOrder = totalAmount / numberOfOrders;
                if (amountPerOrder <= 0) amountPerOrder = 1;
                
                // Create new DCA strategy
                var strategy = new DCAStrategy
                {
                    Symbol = symbol,
                    Name = $"{symbol} Dollar-Based DCA",
                    IsShareBased = false,
                    TotalAmount = totalAmount,
                    AmountPerOrder = amountPerOrder,
                    OrdersRemaining = numberOfOrders,
                    IntervalDays = intervalDays,
                    StrategyType = strategyType
                };
                
                // Add strategy to dictionary using strategy ID
                _dollarCostAveraging[strategy.StrategyId] = strategy;
                
                // Schedule the first order
                ScheduleDollarCostAveragingOrder(strategy.StrategyId);
                
                DatabaseMonolith.Log("Info", $"Dollar-based dollar-cost averaging set up for {symbol} (ID: {strategy.StrategyId}): " +
                    $"${totalAmount:N2} over {numberOfOrders} orders every {intervalDays} days using {strategyType} distribution");
                
                return strategy.StrategyId;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to set up dollar-cost averaging for {symbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Pauses a dollar-cost averaging strategy
        /// </summary>
        /// <param name="strategyId">The ID of the DCA strategy to pause</param>
        /// <returns>True if paused successfully, false otherwise</returns>
        public bool PauseDollarCostAveragingStrategy(string strategyId)
        {
            try
            {
                if (!_dollarCostAveraging.ContainsKey(strategyId))
                {
                    DatabaseMonolith.Log("Warning", $"Cannot pause DCA strategy: Strategy ID {strategyId} not found");
                    return false;
                }
                
                var strategy = _dollarCostAveraging[strategyId];
                
                if (strategy.IsPaused)
                {
                    DatabaseMonolith.Log("Warning", $"DCA strategy {strategyId} for {strategy.Symbol} is already paused");
                    return true;
                }
                
                strategy.IsPaused = true;
                strategy.PausedAt = DateTime.Now;
                
                DatabaseMonolith.Log("Info", $"DCA strategy {strategyId} for {strategy.Symbol} has been paused");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to pause DCA strategy {strategyId}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Resumes a paused dollar-cost averaging strategy
        /// </summary>
        /// <param name="strategyId">The ID of the DCA strategy to resume</param>
        /// <returns>True if resumed successfully, false otherwise</returns>
        public bool ResumeDollarCostAveragingStrategy(string strategyId)
        {
            try
            {
                if (!_dollarCostAveraging.ContainsKey(strategyId))
                {
                    DatabaseMonolith.Log("Warning", $"Cannot resume DCA strategy: Strategy ID {strategyId} not found");
                    return false;
                }
                
                var strategy = _dollarCostAveraging[strategyId];
                
                if (!strategy.IsPaused)
                {
                    DatabaseMonolith.Log("Warning", $"DCA strategy {strategyId} for {strategy.Symbol} is not paused");
                    return true;
                }
                
                strategy.IsPaused = false;
                
                // If there are no scheduled orders for this strategy, schedule the next one
                bool hasScheduledOrders = false;
                if (_scheduledOrders.ContainsKey(strategy.Symbol))
                {
                    hasScheduledOrders = _scheduledOrders[strategy.Symbol]
                        .Any(o => o.IsDollarCostAveraging && o.Symbol == strategy.Symbol);
                }
                
                if (!hasScheduledOrders && strategy.OrdersRemaining > 0)
                {
                    ScheduleDollarCostAveragingOrder(strategyId);
                }
                
                DatabaseMonolith.Log("Info", $"DCA strategy {strategyId} for {strategy.Symbol} has been resumed");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to resume DCA strategy {strategyId}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Cancels a dollar-cost averaging strategy
        /// </summary>
        /// <param name="strategyId">The ID of the DCA strategy to cancel</param>
        /// <returns>True if cancelled successfully, false otherwise</returns>
        public bool CancelDollarCostAveragingStrategy(string strategyId)
        {
            try
            {
                if (!_dollarCostAveraging.ContainsKey(strategyId))
                {
                    DatabaseMonolith.Log("Warning", $"Cannot cancel DCA strategy: Strategy ID {strategyId} not found");
                    return false;
                }
                
                var strategy = _dollarCostAveraging[strategyId];
                string symbol = strategy.Symbol;
                
                // Remove the strategy from the dictionary
                _dollarCostAveraging.Remove(strategyId);
                
                // Cancel any pending scheduled orders for this strategy
                if (_scheduledOrders.ContainsKey(symbol))
                {
                    var ordersToRemove = _scheduledOrders[symbol]
                        .Where(o => o.IsDollarCostAveraging && o.Symbol == symbol)
                        .ToList();
                        
                    foreach (var order in ordersToRemove)
                    {
                        _scheduledOrders[symbol].Remove(order);
                    }
                    
                    if (_scheduledOrders[symbol].Count == 0)
                    {
                        _scheduledOrders.Remove(symbol);
                    }
                    
                    DatabaseMonolith.Log("Info", $"Cancelled {ordersToRemove.Count} pending orders for DCA strategy {strategyId}");
                }
                
                DatabaseMonolith.Log("Info", $"DCA strategy {strategyId} for {symbol} has been cancelled");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to cancel DCA strategy {strategyId}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Gets information about all active dollar-cost averaging strategies
        /// </summary>
        /// <returns>List of active DCA strategies</returns>
        public List<DCAStrategy> GetActiveDCAStrategies()
        {
            return _dollarCostAveraging.Values.ToList();
        }
        
        /// <summary>
        /// Gets information about a specific dollar-cost averaging strategy
        /// </summary>
        /// <param name="strategyId">The ID of the DCA strategy</param>
        /// <returns>The DCA strategy object if found, null otherwise</returns>
        public DCAStrategy GetDCAStrategyInfo(string strategyId)
        {
            if (_dollarCostAveraging.TryGetValue(strategyId, out var strategy))
            {
                return strategy;
            }
            
            return null;
        }
        
        /// <summary>
        /// Schedules the next dollar-cost averaging order for a strategy
        /// </summary>
        /// <param name="strategyId">The ID of the DCA strategy</param>
        private async Task ScheduleDollarCostAveragingOrder(string strategyId)
        {
            try
            {
                if (!_dollarCostAveraging.ContainsKey(strategyId))
                    return;
                
                var strategy = _dollarCostAveraging[strategyId];
                
                // Check if strategy is paused or completed
                if (strategy.IsPaused)
                {
                    DatabaseMonolith.Log("Info", $"Skipped scheduling DCA order for {strategy.Symbol}: Strategy is paused");
                    return;
                }
                
                if (strategy.OrdersRemaining <= 0)
                {
                    DatabaseMonolith.Log("Info", $"DCA strategy completed for {strategy.Symbol} (ID: {strategyId}): " + 
                        $"{strategy.OrdersExecuted} orders executed, {strategy.SharesAcquired} shares acquired, ${strategy.AmountInvested:N2} invested");
                    _dollarCostAveraging.Remove(strategyId);
                    return;
                }
                
                // Get current market price
                double price = await GetMarketPrice(strategy.Symbol);
                
                // Calculate quantity based on strategy type
                int quantity = 0;
                double orderAmount = 0;
                
                if (strategy.IsShareBased)
                {
                    // Share-based DCA: Calculate shares based on distribution type
                    quantity = CalculateDCAShares(strategy);
                    orderAmount = quantity * price;
                }
                else
                {
                    // Dollar-based DCA: Calculate amount based on distribution type
                    orderAmount = CalculateDCAAmount(strategy);
                    quantity = (int)Math.Floor(orderAmount / price);
                    
                    // Ensure at least 1 share
                    if (quantity < 1) quantity = 1;
                }
                
                // Create scheduled order
                var order = new ScheduledOrder
                {
                    Symbol = strategy.Symbol,
                    Quantity = quantity,
                    OrderType = "BUY",
                    Price = price,
                    ExecutionTime = DateTime.Now.AddDays(strategy.IntervalDays),
                    IsDollarCostAveraging = true
                };
                
                // Add to scheduled orders
                if (!_scheduledOrders.ContainsKey(strategy.Symbol))
                {
                    _scheduledOrders[strategy.Symbol] = new List<ScheduledOrder>();
                }
                _scheduledOrders[strategy.Symbol].Add(order);
                
                // Update strategy information
                strategy.OrdersRemaining--;
                strategy.NextExecutionAt = order.ExecutionTime;
                
                // Update tracking after execution (will be applied when order is actually executed)
                // These values get updated when the scheduled order is executed in MonitorScheduledOrders
                
                DatabaseMonolith.Log("Info", $"Dollar-cost averaging order scheduled for {strategy.Symbol}: " + 
                    $"{quantity} shares (${orderAmount:N2}) at {price:C2} on {order.ExecutionTime}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to schedule dollar-cost averaging order for strategy {strategyId}", ex.ToString());
            }
        }
        
        /// <summary>
        /// Calculates the number of shares to buy for a share-based DCA strategy based on the distribution type
        /// </summary>
        /// <param name="strategy">The DCA strategy</param>
        /// <returns>Number of shares to buy</returns>
        private int CalculateDCAShares(DCAStrategy strategy)
        {
            int totalOrderCount = strategy.OrdersExecuted + strategy.OrdersRemaining;
            int currentOrderIndex = strategy.OrdersExecuted;
            double weightFactor = 0;
            
            switch (strategy.StrategyType)
            {
                case DCAStrategyType.FrontLoaded:
                    // Front-loaded: Larger chunks at the beginning, tapering off
                    weightFactor = (double)(totalOrderCount - currentOrderIndex) / totalOrderCount;
                    break;
                    
                case DCAStrategyType.BackLoaded:
                    // Back-loaded: Smaller chunks at the beginning, larger at the end
                    weightFactor = (double)(currentOrderIndex + 1) / totalOrderCount;
                    break;
                    
                case DCAStrategyType.Normal:
                    // Normal (bell curve): Middle chunks are larger
                    double mean = totalOrderCount / 2.0;
                    double stdDev = totalOrderCount / 6.0;
                    weightFactor = Math.Exp(-0.5 * Math.Pow((currentOrderIndex - mean) / stdDev, 2));
                    weightFactor = weightFactor / Math.Exp(-0.5 * Math.Pow((mean - mean) / stdDev, 2)); // Normalize to max 1.0
                    break;
                
                // Additional distribution types
                case DCAStrategyType.ValueBased:
                case DCAStrategyType.VolatilityBased:
                    // For these advanced strategies, we'll use Equal distribution for now
                    // but these would be implemented with price and volatility data in a real scenario
                    weightFactor = 1.0;
                    break;
                    
                case DCAStrategyType.Custom:
                    // For custom, we'd look up from a custom weights array. Using Equal for now.
                    weightFactor = 1.0;
                    break;
                    
                case DCAStrategyType.Equal:
                default:
                    // Equal distribution: Same amount for all orders
                    weightFactor = 1.0;
                    break;
            }
            
            // Calculate shares for this order
            int baseShares = strategy.SharesPerOrder;
            int adjustedShares = (int)Math.Round(baseShares * weightFactor);
            
            // Make sure we don't exceed the total remaining shares
            int remainingShares = strategy.TotalShares - strategy.SharesAcquired;
            if (adjustedShares > remainingShares)
                adjustedShares = remainingShares;
            
            // Ensure at least 1 share
            if (adjustedShares < 1) adjustedShares = 1;
            
            return adjustedShares;
        }
        
        /// <summary>
        /// Calculates the dollar amount to invest for a dollar-based DCA strategy based on the distribution type
        /// </summary>
        /// <param name="strategy">The DCA strategy</param>
        /// <returns>Dollar amount to invest</returns>
        private double CalculateDCAAmount(DCAStrategy strategy)
        {
            int totalOrderCount = strategy.OrdersExecuted + strategy.OrdersRemaining;
            int currentOrderIndex = strategy.OrdersExecuted;
            double weightFactor = 0;
            
            switch (strategy.StrategyType)
            {
                case DCAStrategyType.FrontLoaded:
                    // Front-loaded: Larger chunks at the beginning, tapering off
                    weightFactor = (double)(totalOrderCount - currentOrderIndex) / totalOrderCount;
                    break;
                    
                case DCAStrategyType.BackLoaded:
                    // Back-loaded: Smaller chunks at the beginning, larger at the end
                    weightFactor = (double)(currentOrderIndex + 1) / totalOrderCount;
                    break;
                    
                case DCAStrategyType.Normal:
                    // Normal (bell curve): Middle chunks are larger
                    double mean = totalOrderCount / 2.0;
                    double stdDev = totalOrderCount / 6.0;
                    weightFactor = Math.Exp(-0.5 * Math.Pow((currentOrderIndex - mean) / stdDev, 2));
                    weightFactor = weightFactor / Math.Exp(-0.5 * Math.Pow((mean - mean) / stdDev, 2)); // Normalize to max 1.0
                    break;
                
                // Additional distribution types
                case DCAStrategyType.ValueBased:
                case DCAStrategyType.VolatilityBased:
                    // For these advanced strategies, we'll use Equal distribution for now
                    // but these would be implemented with price and volatility data in a real scenario
                    weightFactor = 1.0;
                    break;
                    
                case DCAStrategyType.Custom:
                    // For custom, we'd look up from a custom weights array. Using Equal for now.
                    weightFactor = 1.0;
                    break;
                    
                case DCAStrategyType.Equal:
                default:
                    // Equal distribution: Same amount for all orders
                    weightFactor = 1.0;
                    break;
            }
            
            // Calculate amount for this order
            double baseAmount = strategy.AmountPerOrder;
            double adjustedAmount = baseAmount * weightFactor;
            
            // Make sure we don't exceed the total remaining amount
            double remainingAmount = strategy.TotalAmount - strategy.AmountInvested;
            if (adjustedAmount > remainingAmount)
                adjustedAmount = remainingAmount;
            
            // Ensure a minimum investment amount (e.g. $1)
            if (adjustedAmount < 1) adjustedAmount = 1;
            
            return adjustedAmount;
        }
        
        /// <summary>
        /// Sets up portfolio target allocations for rebalancing
        /// </summary>
        /// <param name="allocations">Dictionary of symbols and their target percentages (0.05 = 5%)</param>
        /// <returns>True if allocations were set successfully</returns>
        public bool SetPortfolioAllocations(Dictionary<string, double> allocations)
        {
            try
            {
                // Validate allocations
                double total = allocations.Values.Sum();
                if (Math.Abs(total - 1.0) > 0.0001)
                {
                    DatabaseMonolith.Log("Warning", $"Portfolio allocations do not sum to 100%: {total:P2}");
                    return false;
                }
                
                _targetAllocations = new Dictionary<string, double>(allocations);
                DatabaseMonolith.Log("Info", $"Portfolio allocations set with {_targetAllocations.Count} symbols");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to set portfolio allocations", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Adds a rebalancing profile to the available profiles
        /// </summary>
        /// <param name="profile">Rebalancing profile to add</param>
        /// <returns>True if added successfully</returns>
        public bool AddRebalancingProfile(RebalancingProfile profile)
        {
            try
            {
                if (profile == null)
                {
                    DatabaseMonolith.Log("Warning", "Cannot add null rebalancing profile");
                    return false;
                }
                
                if (!profile.ValidateAllocations())
                {
                    DatabaseMonolith.Log("Warning", $"Invalid allocations in rebalancing profile: {profile.Name}");
                    return false;
                }
                
                _rebalancingProfiles[profile.ProfileId] = profile;
                DatabaseMonolith.Log("Info", $"Added rebalancing profile: {profile.Name} ({profile.ProfileId})");
                
                // If this is the first profile, set it as active
                if (_activeRebalancingProfileId == null)
                {
                    _activeRebalancingProfileId = profile.ProfileId;
                }
                
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to add rebalancing profile: {profile?.Name}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Gets all available rebalancing profiles
        /// </summary>
        /// <returns>List of rebalancing profiles</returns>
        public List<RebalancingProfile> GetRebalancingProfiles()
        {
            return _rebalancingProfiles.Values.ToList();
        }
        
        /// <summary>
        /// Gets the active rebalancing profile
        /// </summary>
        /// <returns>Active profile or null if none set</returns>
        public RebalancingProfile GetActiveRebalancingProfile()
        {
            if (_activeRebalancingProfileId != null && _rebalancingProfiles.ContainsKey(_activeRebalancingProfileId))
            {
                return _rebalancingProfiles[_activeRebalancingProfileId];
            }
            return null;
        }
        
        /// <summary>
        /// Sets the active rebalancing profile
        /// </summary>
        /// <param name="profileId">ID of the profile to make active</param>
        /// <returns>True if set successfully</returns>
        public bool SetActiveRebalancingProfile(string profileId)
        {
            if (string.IsNullOrEmpty(profileId))
            {
                DatabaseMonolith.Log("Warning", "Cannot set null or empty rebalancing profile ID as active");
                return false;
            }
            
            if (_rebalancingProfiles.ContainsKey(profileId))
            {
                _activeRebalancingProfileId = profileId;
                var profile = _rebalancingProfiles[profileId];
                
                // Update the target allocations based on the profile
                _targetAllocations = new Dictionary<string, double>(profile.TargetAllocations);
                
                DatabaseMonolith.Log("Info", $"Set active rebalancing profile to: {profile.Name} ({profileId})");
                return true;
            }
            
            DatabaseMonolith.Log("Warning", $"Rebalancing profile not found with ID: {profileId}");
            return false;
        }
        
        /// <summary>
        /// Gets the current portfolio allocations based on market prices
        /// </summary>
        /// <returns>Dictionary mapping symbols to current allocation percentages</returns>
        public async Task<Dictionary<string, double>> GetCurrentPortfolioAllocations()
        {
            var result = new Dictionary<string, double>();
            
            try
            {
                // Calculate current portfolio value and allocations
                double portfolioValue = 0;
                var currentValues = new Dictionary<string, double>();
                
                // Use all symbols from paper portfolio
                var symbols = new HashSet<string>(paperPortfolio.Keys);
                
                // Also include any symbols from target allocations that aren't in the portfolio yet
                if (_targetAllocations != null)
                {
                    foreach (var symbol in _targetAllocations.Keys)
                    {
                        symbols.Add(symbol);
                    }
                }
                
                foreach (var symbol in symbols)
                {
                    double shares = paperPortfolio.ContainsKey(symbol) ? paperPortfolio[symbol] : 0;
                    double price = await GetMarketPrice(symbol);
                    double value = shares * price;
                    
                    currentValues[symbol] = value;
                    portfolioValue += value;
                }
                
                // Calculate percentages if portfolio has value
                if (portfolioValue > 0)
                {
                    foreach (var symbol in currentValues.Keys)
                    {
                        result[symbol] = currentValues[symbol] / portfolioValue;
                    }
                }
                
                return result;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to get current portfolio allocations", ex.ToString());
                return result;
            }
        }
        
        /// <summary>
        /// Gets the current market conditions for rebalancing decisions
        /// </summary>
        /// <returns>Market conditions</returns>
        private async Task<Quantra.Models.MarketConditions> GetMarketConditions()
        {
            var conditions = new Quantra.Models.MarketConditions();
            
            try
            {
                // Get VIX index value as volatility indicator
                const string vixSymbol = "^VIX";
                double vixValue = 15.0; // Default moderate volatility
                
                try
                {
                    var vixPrice = await GetMarketPrice(vixSymbol);
                    if (vixPrice > 0)
                    {
                        vixValue = vixPrice;
                    }
                }
                catch
                {
                    // Fallback to default if we can't get VIX
                }
                
                conditions.VolatilityIndex = vixValue;
                
                // Determine market trend from SPY
                const string spySymbol = "SPY";
                double marketTrend = 0; // Default neutral
                
                try
                {
                    // Calculate 5-day vs 50-day moving average for trend
                    var historicalPrices = await historicalDataService.GetHistoricalPrices(spySymbol, "3mo", "1d");
                    if (historicalPrices != null && historicalPrices.Count >= 50)
                    {
                        double sma5 = historicalPrices.Take(5).Average(p => p.Close);
                        double sma50 = historicalPrices.Take(50).Average(p => p.Close);
                        
                        // Scale to -1 to 1 range
                        marketTrend = Math.Min(1.0, Math.Max(-1.0, (sma5 / sma50) - 1));
                    }
                }
                catch
                {
                    // Fallback to default if we can't calculate trend
                }
                
                conditions.MarketTrend = marketTrend;
                
                // Other conditions could be set here from external data sources
                
                return conditions;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", "Error getting market conditions, using defaults", ex.ToString());
                return conditions; // Return default conditions on error
            }
        }
        
        /// <summary>
        /// Rebalances the portfolio to match target allocations
        /// </summary>
        /// <param name="tolerancePercentage">Tolerance before rebalancing (0.02 = 2%)</param>
        /// <returns>True if rebalancing was initiated</returns>
        public async Task<bool> RebalancePortfolio(double tolerancePercentage = 0.02)
        {
            // Check if we have an active rebalancing profile
            if (_activeRebalancingProfileId != null && _rebalancingProfiles.ContainsKey(_activeRebalancingProfileId))
            {
                // Use the active profile for rebalancing
                return await RebalancePortfolioWithProfile(_rebalancingProfiles[_activeRebalancingProfileId]);
            }
            
            // Fall back to basic rebalancing with the provided tolerance
            return await RebalancePortfolioBasic(tolerancePercentage);
        }
        
        /// <summary>
        /// Rebalances the portfolio using a specific profile with advanced settings
        /// </summary>
        /// <param name="profile">The rebalancing profile to use</param>
        /// <returns>True if rebalancing was initiated</returns>
        public async Task<bool> RebalancePortfolioWithProfile(RebalancingProfile profile)
        {
            try
            {
                if (profile == null)
                {
                    DatabaseMonolith.Log("Warning", "Cannot rebalance: Null profile provided");
                    return false;
                }
                
                // Check if trading is allowed based on market session and time restrictions
                if (!IsTradingAllowed())
                {
                    DatabaseMonolith.Log("Warning", "Portfolio rebalance rejected: Trading not allowed at this time based on market session filters");
                    return false;
                }
                
                if (!profile.ValidateAllocations())
                {
                    DatabaseMonolith.Log("Warning", $"Cannot rebalance: Invalid target allocations in profile {profile.Name}");
                    return false;
                }
                
                // Get market conditions if adjustments are enabled
                Dictionary<string, double> targetAllocations;
                if (profile.EnableMarketConditionAdjustments)
                {
                    var marketConditions = await GetMarketConditions();
                    targetAllocations = profile.GetMarketAdjustedAllocations(marketConditions);
                    
                    // Log market conditions and any adjustments
                    DatabaseMonolith.Log("Info", $"Market conditions for rebalancing: " +
                        $"VIX={marketConditions.VolatilityIndex:F1}, " +
                        $"Trend={marketConditions.MarketTrend:F2}, " +
                        $"Risk={marketConditions.OverallRiskLevel:F2}");
                }
                else
                {
                    targetAllocations = new Dictionary<string, double>(profile.TargetAllocations);
                }
                
                // Calculate current portfolio value and allocations
                double portfolioValue = 0;
                Dictionary<string, double> currentValues = new Dictionary<string, double>();
                
                foreach (var symbol in targetAllocations.Keys)
                {
                    // For simplicity, we'll use paper portfolio positions
                    // In a real implementation, you'd get actual positions
                    double shares = paperPortfolio.ContainsKey(symbol) ? paperPortfolio[symbol] : 0;
                    double price = await GetMarketPrice(symbol);
                    double value = shares * price;
                    
                    currentValues[symbol] = value;
                    portfolioValue += value;
                }
                
                if (portfolioValue <= 0)
                {
                    DatabaseMonolith.Log("Warning", "Cannot rebalance: Portfolio value is zero");
                    return false;
                }
                
                // Calculate and schedule rebalancing trades based on profile's tolerance
                Dictionary<string, int> sharesToAdjust = new Dictionary<string, int>();
                Dictionary<string, double> pricesForAdjustment = new Dictionary<string, double>();
                Dictionary<string, string> orderTypesForAdjustment = new Dictionary<string, string>();
                
                bool anyRebalanceNeeded = false;
                
                foreach (var symbol in targetAllocations.Keys)
                {
                    double targetValue = portfolioValue * targetAllocations[symbol];
                    double currentValue = currentValues.ContainsKey(symbol) ? currentValues[symbol] : 0;
                    
                    // Calculate the difference and see if it exceeds tolerance
                    double difference = targetValue - currentValue;
                    double differencePercentage = Math.Abs(difference) / portfolioValue;
                    
                    if (differencePercentage > profile.TolerancePercentage)
                    {
                        // Need to rebalance this position
                        double price = await GetMarketPrice(symbol);
                        int sharesToAdjustForSymbol = (int)Math.Floor(Math.Abs(difference) / price);
                        
                        if (sharesToAdjustForSymbol > 0)
                        {
                            string orderType = difference > 0 ? "BUY" : "SELL";
                            
                            sharesToAdjust[symbol] = sharesToAdjustForSymbol;
                            pricesForAdjustment[symbol] = price;
                            orderTypesForAdjustment[symbol] = orderType;
                            anyRebalanceNeeded = true;
                        }
                    }
                }
                
                // If no rebalancing is needed, return early
                if (!anyRebalanceNeeded)
                {
                    DatabaseMonolith.Log("Info", $"No rebalancing needed - all positions within tolerance of {profile.TolerancePercentage:P2}");
                    return true;
                }
                
                // Create combined multi-leg strategy for rebalancing
                var strategy = CreateBasketOrder(
                    sharesToAdjust.Keys.ToList(),
                    sharesToAdjust.Values.ToList(),
                    orderTypesForAdjustment.Values.ToList(),
                    targetAllocations.Values.ToList()
                );
                
                if (strategy != null)
                {
                    strategy.Name = $"Portfolio Rebalance - {profile.Name}";
                    
                    // Execute each leg of the rebalancing
                    foreach (var order in strategy.Legs)
                    {
                        // Create a rebalancing order with the portfolio rebalancing flag set
                        var rebalanceOrder = new ScheduledOrder
                        {
                            Symbol = order.Symbol,
                            Quantity = order.Quantity,
                            OrderType = order.OrderType,
                            Price = order.Price,
                            ExecutionTime = DateTime.Now,
                            IsRebalancing = true,
                            IsMultiLegStrategy = true,
                            MultiLegStrategyId = strategy.StrategyId
                        };
                        
                        // Add to scheduled orders for immediate execution
                        if (!_scheduledOrders.ContainsKey(order.Symbol))
                        {
                            _scheduledOrders[order.Symbol] = new List<ScheduledOrder>();
                        }
                        _scheduledOrders[order.Symbol].Add(rebalanceOrder);
                        
                        DatabaseMonolith.Log("Info", $"Rebalancing order scheduled for {order.Symbol}: {order.OrderType} {order.Quantity} shares at {order.Price:C2}");
                    }
                    
                    // Update the last rebalance date on the profile
                    profile.LastRebalanceDate = DateTime.Now;
                    
                    return true;
                }
                
                return false;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to rebalance portfolio with profile", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Basic portfolio rebalancing without advanced profile features (original implementation)
        /// </summary>
        /// <param name="tolerancePercentage">Tolerance before rebalancing (0.02 = 2%)</param>
        /// <returns>True if rebalancing was initiated</returns>
        public async Task<bool> RebalancePortfolioBasic(double tolerancePercentage = 0.02)
        {
            try
            {
                // Check if trading is allowed based on market session and time restrictions
                if (!IsTradingAllowed())
                {
                    DatabaseMonolith.Log("Warning", "Portfolio rebalance rejected: Trading not allowed at this time based on market session filters");
                    return false;
                }
                
                if (_targetAllocations.Count == 0)
                {
                    DatabaseMonolith.Log("Warning", "Cannot rebalance: No target allocations set");
                    return false;
                }
                
                // Calculate current portfolio value and allocations
                double portfolioValue = 0;
                Dictionary<string, double> currentValues = new Dictionary<string, double>();
                
                foreach (var symbol in _targetAllocations.Keys)
                {
                    // For simplicity, we'll use paper portfolio positions
                    // In a real implementation, you'd get actual positions
                    double shares = paperPortfolio.ContainsKey(symbol) ? paperPortfolio[symbol] : 0;
                    double price = await GetMarketPrice(symbol);
                    double value = shares * price;
                    
                    currentValues[symbol] = value;
                    portfolioValue += value;
                }
                
                if (portfolioValue <= 0)
                {
                    DatabaseMonolith.Log("Warning", "Cannot rebalance: Portfolio value is zero");
                    return false;
                }
                
                // Calculate and schedule rebalancing trades
                foreach (var symbol in _targetAllocations.Keys)
                {
                    double targetValue = portfolioValue * _targetAllocations[symbol];
                    double currentValue = currentValues.ContainsKey(symbol) ? currentValues[symbol] : 0;
                    
                    // Calculate the difference and see if it exceeds tolerance
                    double difference = targetValue - currentValue;
                    double differencePercentage = Math.Abs(difference) / portfolioValue;
                    
                    if (differencePercentage > tolerancePercentage)
                    {
                        // Need to rebalance this position
                        double price = await GetMarketPrice(symbol);
                        int sharesToAdjust = (int)Math.Floor(Math.Abs(difference) / price);
                        
                        if (sharesToAdjust > 0)
                        {
                            string orderType = difference > 0 ? "BUY" : "SELL";
                            
                            // Create a rebalancing order
                            var order = new ScheduledOrder
                            {
                                Symbol = symbol,
                                Quantity = sharesToAdjust,
                                OrderType = orderType,
                                Price = price,
                                ExecutionTime = DateTime.Now,
                                IsRebalancing = true
                            };
                            
                            // Add to scheduled orders for immediate execution
                            if (!_scheduledOrders.ContainsKey(symbol))
                            {
                                _scheduledOrders[symbol] = new List<ScheduledOrder>();
                            }
                            _scheduledOrders[symbol].Add(order);
                            
                            DatabaseMonolith.Log("Info", $"Rebalancing order scheduled for {symbol}: {orderType} {sharesToAdjust} shares at {price:C2}");
                        }
                    }
                }
                
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to rebalance portfolio", ex.ToString());
                return false;
            }
        }

        private void SetTrailingStopLoss(string symbol, double price)
        {
            trailingStopLoss[symbol] = price * 0.95;
            takeProfitTargets[symbol] = price * 1.05;
        }

        /// <summary>
        /// Places a bracket order which includes an entry order with automatic stop loss and take profit orders
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="quantity">Number of shares to trade</param>
        /// <param name="orderType">BUY or SELL</param>
        /// <param name="price">Entry price</param>
        /// <param name="stopLossPrice">Stop loss price</param>
        /// <param name="takeProfitPrice">Take profit price</param>
        /// <returns>True if the order was placed successfully</returns>
        public async Task<bool> PlaceBracketOrder(string symbol, int quantity, string orderType, double price, double stopLossPrice, double takeProfitPrice)
        {
            try
            {
                // Check if emergency stop is active
                if (_emergencyStopActive)
                {
                    DatabaseMonolith.Log("Warning", $"Bracket order rejected: Emergency stop is active. {orderType} {quantity} {symbol} @ {price:C2}");
                    return false;
                }
                
                // Place the main order first
                await PlaceLimitOrder(symbol, quantity, orderType, price);
                
                // Store the stop loss and take profit values for monitoring
                trailingStopLoss[symbol] = stopLossPrice;
                takeProfitTargets[symbol] = takeProfitPrice;
                
                // For WebULL integration - create true bracket orders
                // Create an OCO (One-Cancels-Other) pair of orders for stop loss and take profit
                if (tradingMode == TradingMode.Paper)
                {
                    try
                    {
                        // Create the stop loss order - opposite direction of entry order
                        string exitOrderType = orderType == "BUY" ? "SELL" : "BUY";
                        
                        // Determine quantity for the exit orders
                        int exitQuantity = quantity;
                        
                        // Create the stop loss order structure
                        var stopLossOrderData = new
                        {
                            symbol = symbol,
                            qty = exitQuantity,
                            side = exitOrderType,
                            type = "STP", // Stop order
                            time_in_force = "GTC",
                            stop_price = stopLossPrice,
                            order_id = Guid.NewGuid().ToString(), // Generate unique ID for order
                            parent_order_id = Guid.NewGuid().ToString() // Track relationship to main order
                        };
                        
                        // Create the take profit order structure
                        var takeProfitOrderData = new
                        {
                            symbol = symbol,
                            qty = exitQuantity,
                            side = exitOrderType,
                            type = "LMT", // Limit order
                            time_in_force = "GTC",
                            limit_price = takeProfitPrice,
                            order_id = Guid.NewGuid().ToString(), // Generate unique ID for order
                            parent_order_id = symbol + DateTime.Now.Ticks // Track relationship to main order
                        };
                        
                        // Example endpoint for Webull paper trading (replace with actual endpoint)
                        // string webullPaperEndpoint = "https://paper-api.webull.com/api/trade/order";
                        
                        // NOTE: In a real implementation, you would send these orders to the broker
                        // For now, we're just storing them locally and monitoring them ourselves
                        
                        // Add a scheduled order for monitoring that combines both stop loss and take profit  
                        var scheduledOrder = new ScheduledOrder
                        {
                            Symbol = symbol,
                            Quantity = exitQuantity,
                            OrderType = exitOrderType,
                            Price = orderType == "BUY" ? takeProfitPrice : stopLossPrice, // Default price
                            ExecutionTime = DateTime.Now, // Not time-based but price-based
                            StopLoss = stopLossPrice,
                            TakeProfit = takeProfitPrice
                        };
                        
                        // Add to scheduled orders for monitoring
                        if (!_scheduledOrders.ContainsKey(symbol))
                        {
                            _scheduledOrders[symbol] = new List<ScheduledOrder>();
                        }
                        _scheduledOrders[symbol].Add(scheduledOrder);
                    }
                    catch (Exception ex)
                    {
                        DatabaseMonolith.Log("Warning", $"Failed to create exit orders for bracket order on {symbol}", ex.ToString());
                        // Main order was placed, continue despite exit order failure
                    }
                }
                
                DatabaseMonolith.Log("Info", $"Bracket order placed for {symbol}: Entry at {price:C2}, Stop Loss at {stopLossPrice:C2}, Take Profit at {takeProfitPrice:C2}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to place bracket order for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Sets a trailing stop for a position that moves up as the price increases (for long positions)
        /// or moves down as the price decreases (for short positions)
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="initialPrice">Current price when setting the stop</param>
        /// <param name="trailingDistance">Distance to maintain below/above the highest/lowest price (percentage)</param>
        /// <param name="orderType">BUY (for short positions) or SELL (for long positions)</param>
        /// <returns>True if the trailing stop was set successfully</returns>
        public bool SetTrailingStop(string symbol, double initialPrice, double trailingDistance, string orderType = "SELL")
        {
            try
            {
                // Validate parameters
                if (string.IsNullOrWhiteSpace(symbol))
                {
                    DatabaseMonolith.Log("Error", "Failed to set trailing stop: Symbol cannot be empty");
                    return false;
                }
                
                if (initialPrice <= 0)
                {
                    DatabaseMonolith.Log("Error", $"Failed to set trailing stop for {symbol}: Initial price must be positive");
                    return false;
                }
                
                if (trailingDistance <= 0 || trailingDistance >= 1)
                {
                    DatabaseMonolith.Log("Error", $"Failed to set trailing stop for {symbol}: Trailing distance must be between 0 and 1");
                    return false;
                }
                
                if (orderType != "BUY" && orderType != "SELL")
                {
                    DatabaseMonolith.Log("Error", $"Failed to set trailing stop for {symbol}: Order type must be BUY or SELL");
                    return false;
                }
                
                // Calculate initial trigger price based on order type
                double initialTriggerPrice;
                if (orderType == "SELL") // For long positions
                {
                    // Stop below current price
                    initialTriggerPrice = initialPrice * (1 - trailingDistance);
                }
                else // For short positions
                {
                    // Stop above current price
                    initialTriggerPrice = initialPrice * (1 + trailingDistance);
                }
                
                // Add or update trailing stop
                _trailingStops[symbol] = new TrailingStopInfo(symbol, initialPrice, trailingDistance);
                
                DatabaseMonolith.Log("Info", $"Trailing stop ({orderType}) set for {symbol}: Initial price {initialPrice:C2}, Distance {trailingDistance:P2}, Trigger at {initialPrice * (1 - trailingDistance):C2}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to set trailing stop for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Monitors positions for trailing stops, time-based exits, and scheduled orders
        /// </summary>
        /// <param name="cancellationToken">Token to cancel the monitoring</param>
        private async Task MonitorPositions(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // Monitor trailing stops
                    await MonitorTrailingStops();
                    
                    // Monitor bracket orders (stop loss and take profit)
                    await MonitorBracketOrders();
                    
                    // Monitor time-based exits
                    await MonitorTimeBasedExits();
                    
                    // Monitor scheduled orders
                    await MonitorScheduledOrders();
                    
                    // Wait before checking again
                    //await Task.Delay(5000, cancellationToken);
                }
                catch (TaskCanceledException)
                {
                    // Normal cancellation
                    break;
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", "Error in position monitoring", ex.ToString());
                    // Wait a bit longer on error to avoid spamming logs
                    try
                    {
                        await Task.Delay(30000, cancellationToken);
                    }
                    catch (TaskCanceledException)
                    {
                        break;
                    }
                }
            }
        }
        
        /// <summary>
        /// Monitors trailing stops and adjusts them based on price movements
        /// </summary>
        private async Task MonitorTrailingStops()
        {
            // Get symbols with trailing stops
            var symbols = _trailingStops.Keys.ToList();
            
            foreach (var symbol in symbols)
            {
                try
                {
                    // Get current market price
                    double currentPrice = await GetMarketPrice(symbol);
                    
                    // Get trailing stop information
                    var stopInfo = _trailingStops[symbol];
                    var initialPrice = stopInfo.InitialPrice;
                    var trailingDistance = stopInfo.TrailingDistance;
                    var currentTriggerPrice = stopInfo.CurrentStopPrice;
                    
                    // Check if we have a valid price
                    if (currentPrice <= 0)
                    {
                        DatabaseMonolith.Log("Warning", $"Invalid market price for {symbol}: {currentPrice}");
                        continue;
                    }
                    
                    // If price has moved favorably, adjust the trigger price and check if triggered
                    bool triggered = stopInfo.UpdateStopPrice(currentPrice);
                    
                    if (triggered)
                    {
                        // Determine the appropriate order type (for selling long positions)
                        string orderType = "SELL";
                        
                        // Look up quantity in paper portfolio or use default
                        int quantity = 100; // Default quantity
                        if (paperPortfolio.TryGetValue(symbol, out double shares))
                        {
                            quantity = Math.Max(1, (int)Math.Round(shares));
                        }
                        
                        // Place the order
                        await PlaceLimitOrder(symbol, quantity, orderType, currentPrice);
                        
                        // Remove the trailing stop
                        _trailingStops.Remove(symbol);
                        
                        DatabaseMonolith.Log("Info", $"Trailing stop triggered for {symbol} at {currentPrice:C2}: Executed {orderType} order for {quantity} shares");
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error monitoring trailing stop for {symbol}", ex.ToString());
                }
            }
        }
        
        /// <summary>
        /// Monitors bracket orders for stop loss and take profit conditions
        /// </summary>
        private async Task MonitorBracketOrders()
        {
            foreach (var stock in volatileStocks.Concat(paperPortfolio.Keys).Distinct().ToList())
            {
                try
                {
                    double currentPrice = await GetMarketPrice(stock);
                    
                    // Check for stop loss and take profit conditions from bracket orders
                    if (trailingStopLoss.ContainsKey(stock) && currentPrice <= trailingStopLoss[stock])
                    {
                        // Stop loss triggered - execute exit order
                        await PlaceLimitOrder(stock, 100, "SELL", currentPrice);
                        DatabaseMonolith.Log("Info", $"Stop loss triggered for {stock} at {currentPrice:C2}");
                        
                        // Clean up after executing the stop loss
                        trailingStopLoss.Remove(stock);
                        takeProfitTargets.Remove(stock);
                        
                        // Also remove any time-based exit strategies for this symbol
                        if (_timeBasedExits.ContainsKey(stock))
                            _timeBasedExits.Remove(stock);
                            
                        if (_timeBasedExitStrategies.ContainsKey(stock))
                            _timeBasedExitStrategies.Remove(stock);
                        
                        // If we have scheduled orders for this symbol that are part of the bracket, remove them
                        if (_scheduledOrders.ContainsKey(stock))
                        {
                            var bracketsToRemove = _scheduledOrders[stock]
                                .Where(o => o.StopLoss.HasValue && o.TakeProfit.HasValue).ToList();
                                
                            foreach (var order in bracketsToRemove)
                            {
                                _scheduledOrders[stock].Remove(order);
                            }
                            
                            if (_scheduledOrders[stock].Count == 0)
                            {
                                _scheduledOrders.Remove(stock);
                            }
                        }
                    }
                    else if (takeProfitTargets.ContainsKey(stock) && currentPrice >= takeProfitTargets[stock])
                    {
                        // Take profit triggered - execute exit order
                        await PlaceLimitOrder(stock, 100, "SELL", currentPrice);
                        DatabaseMonolith.Log("Info", $"Take profit triggered for {stock} at {currentPrice:C2}");
                        
                        // Clean up after executing the take profit
                        trailingStopLoss.Remove(stock);
                        takeProfitTargets.Remove(stock);
                        
                        // Also remove any time-based exit strategies for this symbol
                        if (_timeBasedExits.ContainsKey(stock))
                            _timeBasedExits.Remove(stock);
                            
                        if (_timeBasedExitStrategies.ContainsKey(stock))
                            _timeBasedExitStrategies.Remove(stock);
                        
                        // If we have scheduled orders for this symbol that are part of the bracket, remove them
                        if (_scheduledOrders.ContainsKey(stock))
                        {
                            var bracketsToRemove = _scheduledOrders[stock]
                                .Where(o => o.StopLoss.HasValue && o.TakeProfit.HasValue).ToList();
                                
                            foreach (var order in bracketsToRemove)
                            {
                                _scheduledOrders[stock].Remove(order);
                            }
                            
                            if (_scheduledOrders[stock].Count == 0)
                            {
                                _scheduledOrders.Remove(stock);
                            }
                        }
                    }
                    // If the price has moved up, update the trailing stop loss
                    else if (trailingStopLoss.ContainsKey(stock) && currentPrice > trailingStopLoss[stock] * 1.05)
                    {
                        double oldStopLoss = trailingStopLoss[stock];
                        trailingStopLoss[stock] = currentPrice * 0.95;
                        DatabaseMonolith.Log("Info", $"Trailing stop updated for {stock}: {oldStopLoss:C2} -> {trailingStopLoss[stock]:C2}");
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error monitoring bracket orders for {stock}", ex.ToString());
                }
            }
        }
        
        /// <summary>
        /// Monitors time-based exits and executes them when their exit time is reached
        /// </summary>
        private async Task MonitorTimeBasedExits()
        {
            // Get a list of symbols with time-based exit strategies
            var symbolsWithTimeBasedExits = _timeBasedExitStrategies.Keys.ToList();
            
            foreach (var symbol in symbolsWithTimeBasedExits)
            {
                try
                {
                    var exitStrategy = _timeBasedExitStrategies[symbol];
                    var currentTime = DateTime.Now;
                    
                    // Check if it's time to execute the exit strategy
                    if (currentTime >= exitStrategy.ExitTime)
                    {
                        // Get current price for the symbol
                        double currentPrice = await GetMarketPrice(symbol);
                        
                        // Get quantity from paper portfolio or use default
                        int quantity = 100; // Default quantity if not found in portfolio
                        if (paperPortfolio.TryGetValue(symbol, out double shares))
                        {
                            quantity = Math.Max(1, (int)Math.Round(shares));
                        }
                        
                        // Execute the exit order
                        await PlaceLimitOrder(symbol, quantity, "SELL", currentPrice);
                        
                        // Log the execution with the specific strategy type
                        DatabaseMonolith.Log("Info", $"Time-based exit ({exitStrategy.Strategy}) executed for {symbol} at {currentPrice:C2}");
                        
                        // Clean up after executing the exit
                        _timeBasedExits.Remove(symbol);
                        _timeBasedExitStrategies.Remove(symbol);
                    }
                    else if (exitStrategy.Strategy == TimeBasedExitStrategy.Duration && 
                             exitStrategy.EntryTime.HasValue && 
                             exitStrategy.DurationMinutes.HasValue)
                    {
                        // Check if the position has been re-entered recently
                        // (Simple heuristic: price has moved significantly)
                        bool positionReentered = false;
                        
                        // If detected re-entry, update entry time and exit time
                        if (positionReentered)
                        {
                            exitStrategy.EntryTime = DateTime.Now;
                            exitStrategy.ExitTime = DateTime.Now.AddMinutes(exitStrategy.DurationMinutes.Value);
                            _timeBasedExits[symbol] = exitStrategy.ExitTime;
                            
                            DatabaseMonolith.Log("Info", $"Position re-entry detected for {symbol}, time-based exit updated to {exitStrategy.ExitTime}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error monitoring time-based exit for {symbol}", ex.ToString());
                }
            }
            
            // Check the legacy/simple time-based exits
            var symbolsWithBasicExits = _timeBasedExits.Keys.Where(k => !_timeBasedExitStrategies.ContainsKey(k)).ToList();
            
            foreach (var symbol in symbolsWithBasicExits)
            {
                try
                {
                    if (DateTime.Now >= _timeBasedExits[symbol])
                    {
                        // Get current price for the symbol
                        double currentPrice = await GetMarketPrice(symbol);
                        
                        // Get quantity from paper portfolio or use default
                        int quantity = 100;
                        if (paperPortfolio.TryGetValue(symbol, out double shares))
                        {
                            quantity = Math.Max(1, (int)Math.Round(shares));
                        }
                        
                        // Execute the exit order
                        await PlaceLimitOrder(symbol, quantity, "SELL", currentPrice);
                        
                        // Log the execution
                        DatabaseMonolith.Log("Info", $"Basic time-based exit executed for {symbol} at {currentPrice:C2}");
                        
                        // Clean up after executing the exit
                        _timeBasedExits.Remove(symbol);
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error monitoring basic time-based exit for {symbol}", ex.ToString());
                }
            }
        }
        
        /// <summary>
        /// Monitors and executes any scheduled orders
        /// </summary>
        private async Task MonitorScheduledOrders()
        {
            // Check for scheduled orders
            foreach (var symbol in _scheduledOrders.Keys.ToList())
            {
                try
                {
                    var orders = _scheduledOrders[symbol].ToList();
                    foreach (var order in orders.Where(o => o.ExecutionTime <= DateTime.Now).ToList())
                    {
                        await PlaceLimitOrder(order.Symbol, order.Quantity, order.OrderType, order.Price);
                        _scheduledOrders[symbol].Remove(order);
                        
                        // If it's a bracketed order, set stop loss and take profit
                        if (order.StopLoss.HasValue && order.TakeProfit.HasValue)
                        {
                            trailingStopLoss[symbol] = order.StopLoss.Value;
                            takeProfitTargets[symbol] = order.TakeProfit.Value;
                        }
                        
                        // If it's a DCA order, update strategy stats and schedule the next one
                        if (order.IsDollarCostAveraging)
                        {
                            // Find the associated DCA strategy for this symbol
                            var dcaStrategy = _dollarCostAveraging.Values
                                .FirstOrDefault(s => s.Symbol == symbol && !s.IsPaused);
                                
                            if (dcaStrategy != null)
                            {
                                // Update strategy stats
                                dcaStrategy.OrdersExecuted++;
                                dcaStrategy.SharesAcquired += order.Quantity;
                                dcaStrategy.AmountInvested += order.Quantity * order.Price;
                                dcaStrategy.LastExecutedAt = DateTime.Now;
                                
                                // If orders remain, schedule the next one
                                if (dcaStrategy.OrdersRemaining > 0)
                                {
                                    await ScheduleDollarCostAveragingOrder(dcaStrategy.StrategyId);
                                }
                                else
                                {
                                    DatabaseMonolith.Log("Info", $"DCA strategy completed for {symbol} (ID: {dcaStrategy.StrategyId}): " +
                                        $"{dcaStrategy.OrdersExecuted} orders executed, {dcaStrategy.SharesAcquired} shares acquired, " +
                                        $"${dcaStrategy.AmountInvested:N2} invested, avg price: ${dcaStrategy.AveragePricePerShare:N2}");
                                }
                            }
                        }
                        
                        DatabaseMonolith.Log("Info", $"Scheduled order executed for {symbol}: {order.OrderType} {order.Quantity} shares at {order.Price:C2}");
                    }
                    
                    if (_scheduledOrders[symbol].Count == 0)
                    {
                        _scheduledOrders.Remove(symbol);
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error monitoring scheduled orders for {symbol}", ex.ToString());
                }
            }
        }

        /// <summary>
        /// Sets a time-based exit for a position
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="exitTime">When to exit the position</param>
        /// <returns>True if the time-based exit was set successfully</returns>


        /// <summary>
        /// Sets a time-based exit strategy for a position
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="strategy">The exit strategy to use</param>
        /// <param name="durationMinutes">Duration in minutes (for Duration strategy)</param>
        /// <param name="specificTime">Specific time of day (for SpecificTimeOfDay strategy)</param>
        /// <returns>True if the exit strategy was set successfully</returns>
        public bool SetTimeBasedExitStrategy(string symbol, TimeBasedExitStrategy strategy, int? durationMinutes = null, TimeOnly? specificTime = null)
        {
            try
            {
                DateTime exitTime;
                TimeBasedExit exitStrategy = new TimeBasedExit
                {
                    Strategy = strategy,
                    DurationMinutes = durationMinutes,
                    EntryTime = DateTime.Now, // Record entry time for Duration-based strategies
                    SpecificTime = specificTime
                };
                
                // Calculate the appropriate exit time based on strategy
                switch (strategy)
                {
                    case TimeBasedExitStrategy.EndOfDay:
                        // Exit at regular market close (4:00 PM)
                        exitTime = DateTime.Today.AddHours(16);
                        // If it's already past 4 PM, set for next day
                        if (DateTime.Now > exitTime)
                            exitTime = exitTime.AddDays(1);
                        break;
                        
                    case TimeBasedExitStrategy.EndOfWeek:
                        // Exit at market close on Friday
                        DateTime friday = DateTime.Today;
                        // Find the coming Friday
                        while (friday.DayOfWeek != DayOfWeek.Friday)
                            friday = friday.AddDays(1);
                        exitTime = friday.AddHours(16); // 4:00 PM
                        break;
                        
                    case TimeBasedExitStrategy.Duration:
                        if (!durationMinutes.HasValue)
                        {
                            DatabaseMonolith.Log("Error", $"Duration minutes must be specified for Duration exit strategy on {symbol}");
                            return false;
                        }
                        exitTime = DateTime.Now.AddMinutes(durationMinutes.Value);
                        break;
                        
                    case TimeBasedExitStrategy.SpecificTimeOfDay:
                        if (!specificTime.HasValue)
                        {
                            DatabaseMonolith.Log("Error", $"Specific time must be specified for SpecificTimeOfDay exit strategy on {symbol}");
                            return false;
                        }
                        
                        // Calculate the next occurrence of the specific time
                        DateTime today = DateTime.Today;
                        DateTime specificDateTime = today.Add(specificTime.Value.ToTimeSpan());
                        
                        // If the time has already passed today, set for tomorrow
                        if (DateTime.Now > specificDateTime)
                            specificDateTime = specificDateTime.AddDays(1);
                            
                        exitTime = specificDateTime;
                        break;
                        
                    default:
                        DatabaseMonolith.Log("Error", $"Unknown time-based exit strategy: {strategy}");
                        return false;
                }
                
                // Store the exit strategy and time
                exitStrategy.ExitTime = exitTime;
                _timeBasedExitStrategies[symbol] = exitStrategy;
                _timeBasedExits[symbol] = exitTime; // For backward compatibility
                
                DatabaseMonolith.Log("Info", $"Time-based exit strategy set for {symbol}: {strategy}, Exit time: {exitTime}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to set time-based exit strategy for {symbol}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Sets a time-based exit strategy for a position with a specific duration
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="durationMinutes">Duration in minutes before exit</param>
        /// <returns>True if the duration-based exit was set successfully</returns>
        public bool SetDurationBasedExit(string symbol, int durationMinutes)
        {
            return SetTimeBasedExitStrategy(symbol, TimeBasedExitStrategy.Duration, durationMinutes);
        }
        
        /// <summary>
        /// Sets a time-based exit strategy for a position at the end of the trading day
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <returns>True if the end-of-day exit was set successfully</returns>
        public bool SetEndOfDayExit(string symbol)
        {
            return SetTimeBasedExitStrategy(symbol, TimeBasedExitStrategy.EndOfDay);
        }
        
        /// <summary>
        /// Sets a time-based exit strategy for a position at the end of the trading week
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <returns>True if the end-of-week exit was set successfully</returns>
        public bool SetEndOfWeekExit(string symbol)
        {
            return SetTimeBasedExitStrategy(symbol, TimeBasedExitStrategy.EndOfWeek);
        }
        
        /// <summary>
        /// Sets a time-based exit strategy for a position at a specific time of day
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="time">The specific time of day to exit</param>
        /// <returns>True if the specific-time exit was set successfully</returns>
        public bool SetSpecificTimeExit(string symbol, TimeOnly time)
        {
            return SetTimeBasedExitStrategy(symbol, TimeBasedExitStrategy.SpecificTimeOfDay, null, time);
        }
        
        /// <summary>
        /// Removes a time-based exit strategy for a position
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <returns>True if the time-based exit was removed successfully</returns>
        public bool RemoveTimeBasedExit(string symbol)
        {
            try
            {
                bool removed = false;
                
                // Remove from the exit strategies dictionary
                if (_timeBasedExitStrategies.ContainsKey(symbol))
                {
                    _timeBasedExitStrategies.Remove(symbol);
                    removed = true;
                }
                
                // Also remove from the basic exits dictionary for compatibility
                if (_timeBasedExits.ContainsKey(symbol))
                {
                    _timeBasedExits.Remove(symbol);
                    removed = true;
                }
                
                if (removed)
                {
                    DatabaseMonolith.Log("Info", $"Time-based exit removed for {symbol}");
                }
                
                return removed;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to remove time-based exit for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Activates emergency stop to halt all trading
        /// </summary>
        /// <returns>True if emergency stop was activated</returns>
        public bool ActivateEmergencyStop()
        {
            try
            {
                // Set emergency stop flag and cancel any operations using the token
                _emergencyStopActive = true;
                _emergencyStopTokenSource.Cancel();
                
                // Stop monitoring positions
                StopMonitoring();
                
                // Create a detailed log of all pending operations
                StringBuilder logBuilder = new StringBuilder();
                logBuilder.AppendLine("Emergency Stop Activated - Cancelling operations:");
                
                // Cancel all scheduled orders
                int canceledOrderCount = 0;
                foreach (var symbol in _scheduledOrders.Keys.ToList())
                {
                    canceledOrderCount += _scheduledOrders[symbol].Count;
                    logBuilder.AppendLine($"  - Cancelling {_scheduledOrders[symbol].Count} pending orders for {symbol}");
                    
                    // Log details of each canceled order
                    foreach (var order in _scheduledOrders[symbol])
                    {
                        logBuilder.AppendLine($"    * {order.OrderType} {order.Quantity} {order.Symbol} @ {order.Price:C2} scheduled for {order.ExecutionTime}");
                    }
                    
                    _scheduledOrders[symbol].Clear();
                }
                _scheduledOrders.Clear();
                
                // Cancel all dollar cost averaging strategies
                int dcaStrategiesCount = _dollarCostAveraging.Count;
                if (dcaStrategiesCount > 0)
                {
                    logBuilder.AppendLine($"  - Cancelling {dcaStrategiesCount} dollar-cost averaging strategies");
                    foreach (var symbol in _dollarCostAveraging.Keys)
                    {
                        var strategy = _dollarCostAveraging[symbol];
                        logBuilder.AppendLine($"    * {symbol}: {strategy.OrdersRemaining} remaining orders of {strategy.SharesPerOrder} shares each");
                    }
                }
                _dollarCostAveraging.Clear();
                
                // Cancel any rebalancing operations
                int allocationTargetsCount = _targetAllocations.Count;
                if (allocationTargetsCount > 0)
                {
                    logBuilder.AppendLine($"  - Cancelling portfolio rebalancing for {allocationTargetsCount} symbols");
                }
                _targetAllocations.Clear();
                
                // Log all trailing stops being canceled
                int trailingStopsCount = _trailingStops.Count;
                if (trailingStopsCount > 0) 
                {
                    logBuilder.AppendLine($"  - Cancelling {trailingStopsCount} active trailing stops");
                    foreach (var symbol in _trailingStops.Keys.ToList())
                    {
                        var stopInfo = _trailingStops[symbol];
                        logBuilder.AppendLine($"    * {symbol}: Initial price {stopInfo.InitialPrice:C2}, Distance {stopInfo.TrailingDistance:P2}, Current trigger {stopInfo.CurrentStopPrice:C2}");
                    }
                    _trailingStops.Clear();
                }
                
                // Log time-based exits being canceled
                int timeBasedExitsCount = _timeBasedExits.Count;
                if (timeBasedExitsCount > 0)
                {
                    logBuilder.AppendLine($"  - Cancelling {timeBasedExitsCount} time-based exits");
                    
                    foreach (var symbol in _timeBasedExits.Keys)
                    {
                        string strategy = _timeBasedExitStrategies.ContainsKey(symbol) 
                            ? $", Strategy: {_timeBasedExitStrategies[symbol].Strategy}" 
                            : "";
                        
                        logBuilder.AppendLine($"    * {symbol}: Exit at {_timeBasedExits[symbol]}{strategy}");
                    }
                    
                    _timeBasedExits.Clear();
                }
                
                // Clear time-based exit strategies too
                if (_timeBasedExitStrategies.Count > 0)
                {
                    logBuilder.AppendLine($"  - Cancelling {_timeBasedExitStrategies.Count} time-based exit strategies");
                    _timeBasedExitStrategies.Clear();
                }
                
                // Log the entire emergency stop operation
                DatabaseMonolith.Log("Warning", $"EMERGENCY STOP ACTIVATED - All trading halted. Cancelled {canceledOrderCount} pending orders.");
                DatabaseMonolith.Log("Info", logBuilder.ToString());
                
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to activate emergency stop", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Deactivates emergency stop to resume trading
        /// </summary>
        /// <returns>True if emergency stop was deactivated</returns>
        public bool DeactivateEmergencyStop()
        {
            try
            {
                _emergencyStopActive = false;
                _emergencyStopTokenSource = new CancellationTokenSource();
                
                // Restart position monitoring
                StartMonitoring();
                
                DatabaseMonolith.Log("Info", "Emergency stop deactivated - Trading resumed");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to deactivate emergency stop", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Checks if emergency stop is currently active
        /// </summary>
        /// <returns>True if emergency stop is active</returns>
        public bool IsEmergencyStopActive()
        {
            return _emergencyStopActive;
        }

        /// <summary>
        /// Checks if the current time is within any enabled market session
        /// </summary>
        /// <param name="time">The time to check</param>
        /// <returns>True if within an enabled market session</returns>
        private bool IsInEnabledMarketSession(TimeOnly time)
        {
            // Check if current time falls within any enabled session
            if ((_enabledMarketSessions.HasFlag(MarketSession.PreMarket)) && 
                (time >= _preMarketOpenTime && time < _regularMarketOpenTime))
            {
                return true;
            }
            
            if ((_enabledMarketSessions.HasFlag(MarketSession.Regular)) && 
                (time >= _regularMarketOpenTime && time < _regularMarketCloseTime))
            {
                return true;
            }
            
            if ((_enabledMarketSessions.HasFlag(MarketSession.AfterHours)) && 
                (time >= _regularMarketCloseTime && time < _afterHoursCloseTime))
            {
                return true;
            }
            
            return false;
        }
        
        /// <summary>
        /// Sets trading hour restrictions to limit when trades can be executed
        /// </summary>
        /// <param name="marketOpen">Market opening time</param>
        /// <param name="marketClose">Market closing time</param>
        /// <returns>True if restrictions were set successfully</returns>
        public bool SetTradingHourRestrictions(TimeOnly marketOpen, TimeOnly marketClose)
        {
            try
            {
                // Validate times
                if (marketClose <= marketOpen)
                {
                    DatabaseMonolith.Log("Error", $"Invalid trading hours: Market close ({marketClose}) must be after market open ({marketOpen})");
                    return false;
                }
                
                // Clear existing restrictions
                _tradingHourRestrictions.Clear();
                
                // Add new restrictions
                _tradingHourRestrictions.Add(marketOpen);
                _tradingHourRestrictions.Add(marketClose);
                
                DatabaseMonolith.Log("Info", $"Trading hour restrictions set: {marketOpen} - {marketClose}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to set trading hour restrictions", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Sets which market sessions are enabled for trading
        /// </summary>
        /// <param name="sessions">The market sessions to enable</param>
        /// <returns>True if sessions were set successfully</returns>
        public bool SetEnabledMarketSessions(MarketSession sessions)
        {
            try
            {
                _enabledMarketSessions = sessions;
                
                string enabledSessions = GetEnabledSessionsDescription(sessions);
                DatabaseMonolith.Log("Info", $"Enabled market sessions for trading: {enabledSessions}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to set enabled market sessions", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Gets which market sessions are currently enabled for trading
        /// </summary>
        /// <returns>The currently enabled market sessions</returns>
        public MarketSession GetEnabledMarketSessions()
        {
            return _enabledMarketSessions;
        }
        
        /// <summary>
        /// Sets the time boundaries for different market sessions
        /// </summary>
        /// <param name="preMarketOpenTime">Pre-market opening time (default: 4:00 AM)</param>
        /// <param name="regularMarketOpenTime">Regular market opening time (default: 9:30 AM)</param>
        /// <param name="regularMarketCloseTime">Regular market closing time (default: 4:00 PM)</param>
        /// <param name="afterHoursCloseTime">After-hours closing time (default: 8:00 PM)</param>
        /// <returns>True if session times were set successfully</returns>
        public bool SetMarketSessionTimes(
            TimeOnly preMarketOpenTime,
            TimeOnly regularMarketOpenTime, 
            TimeOnly regularMarketCloseTime,
            TimeOnly afterHoursCloseTime)
        {
            try
            {
                // Validate times
                if (preMarketOpenTime >= regularMarketOpenTime || 
                    regularMarketOpenTime >= regularMarketCloseTime ||
                    regularMarketCloseTime >= afterHoursCloseTime)
                {
                    DatabaseMonolith.Log("Error", "Invalid market session times: Times must be in sequence (preMarket < regularOpen < regularClose < afterHoursClose)");
                    return false;
                }
                
                _preMarketOpenTime = preMarketOpenTime;
                _regularMarketOpenTime = regularMarketOpenTime;
                _regularMarketCloseTime = regularMarketCloseTime;
                _afterHoursCloseTime = afterHoursCloseTime;
                
                DatabaseMonolith.Log("Info", $"Market session times set: Pre-market {preMarketOpenTime} to {regularMarketOpenTime}, " +
                    $"Regular {regularMarketOpenTime} to {regularMarketCloseTime}, After-hours {regularMarketCloseTime} to {afterHoursCloseTime}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to set market session times", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Gets the current market session time boundaries
        /// </summary>
        /// <returns>A tuple containing the session time boundaries (preMarketOpen, regularMarketOpen, regularMarketClose, afterHoursClose)</returns>
        public (TimeOnly preMarketOpen, TimeOnly regularMarketOpen, TimeOnly regularMarketClose, TimeOnly afterHoursClose) GetMarketSessionTimes()
        {
            return (_preMarketOpenTime, _regularMarketOpenTime, _regularMarketCloseTime, _afterHoursCloseTime);
        }
        
        /// <summary>
        /// Gets a description of the enabled market sessions
        /// </summary>
        /// <param name="sessions">The market sessions to describe</param>
        /// <returns>A string describing the enabled sessions</returns>
        private string GetEnabledSessionsDescription(MarketSession sessions)
        {
            if (sessions == MarketSession.None)
                return "None";
            
            if (sessions == MarketSession.All)
                return "All (Pre-Market, Regular, After-Hours)";
                
            var enabledList = new List<string>();
            
            if (sessions.HasFlag(MarketSession.PreMarket))
                enabledList.Add("Pre-Market");
                
            if (sessions.HasFlag(MarketSession.Regular))
                enabledList.Add("Regular");
                
            if (sessions.HasFlag(MarketSession.AfterHours))
                enabledList.Add("After-Hours");
                
            return string.Join(", ", enabledList);
        }
        
        /// <summary>
        /// Places multiple orders as part of a multi-leg strategy
        /// </summary>
        /// <param name="orders">List of orders to place</param>
        /// <returns>True if all orders were placed successfully</returns>
        public async Task<bool> PlaceMultiLegOrder(List<ScheduledOrder> orders)
        {
            try
            {
                if (orders == null || orders.Count == 0)
                {
                    return false;
                }
                
                // Check if trading is allowed based on market session and time restrictions
                if (!IsTradingAllowed())
                {
                    DatabaseMonolith.Log("Warning", $"Multi-leg order rejected: Trading not allowed at this time based on market session filters. Order count: {orders.Count}");
                    return false;
                }
                
                // Place all orders in sequence
                foreach (var order in orders)
                {
                    await PlaceLimitOrder(order.Symbol, order.Quantity, order.OrderType, order.Price);
                    
                    // If it has stop loss/take profit, set those as well
                    if (order.StopLoss.HasValue && order.TakeProfit.HasValue)
                    {
                        trailingStopLoss[order.Symbol] = order.StopLoss.Value;
                        takeProfitTargets[order.Symbol] = order.TakeProfit.Value;
                    }
                }
                
                DatabaseMonolith.Log("Info", $"Multi-leg order executed with {orders.Count} legs");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to place multi-leg order", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Places a multi-leg strategy with strategy-specific handling
        /// </summary>
        /// <param name="strategy">The multi-leg strategy to execute</param>
        /// <returns>True if the strategy was executed successfully</returns>
        public async Task<bool> PlaceMultiLegStrategy(MultiLegStrategy strategy)
        {
            try
            {
                if (strategy == null || strategy.Legs == null || strategy.Legs.Count == 0)
                {
                    DatabaseMonolith.Log("Error", "Cannot place multi-leg strategy: Strategy or legs are null or empty");
                    return false;
                }
                
                // Check if trading is allowed based on market session and time restrictions
                if (!IsTradingAllowed())
                {
                    DatabaseMonolith.Log("Warning", $"Multi-leg strategy rejected: Trading not allowed at this time based on market session filters. Strategy: {strategy.Name}");
                    return false;
                }
                
                // Validate the strategy
                if (!strategy.Validate())
                {
                    DatabaseMonolith.Log("Error", $"Multi-leg strategy validation failed: {strategy.Name} ({strategy.StrategyType})");
                    return false;
                }
                
                // Ensure all legs have the strategy ID and IsMultiLegStrategy flag set
                foreach (var leg in strategy.Legs)
                {
                    leg.IsMultiLegStrategy = true;
                    leg.MultiLegStrategyId = strategy.StrategyId;
                }
                
                // Additional strategy-specific pre-execution logic
                switch (strategy.StrategyType)
                {
                    case MultiLegStrategyType.VerticalSpread:
                    case MultiLegStrategyType.CalendarSpread:
                    case MultiLegStrategyType.DiagonalSpread:
                    case MultiLegStrategyType.Straddle:
                    case MultiLegStrategyType.Strangle:
                    case MultiLegStrategyType.IronCondor:
                    case MultiLegStrategyType.ButterflySpread:
                        // These strategies typically need All-or-None execution
                        strategy.AllOrNone = true;
                        break;
                }
                
                // If the strategy is "All or None", we need to ensure we can execute all legs
                if (strategy.AllOrNone)
                {
                    // In a real implementation, we would verify availability, market conditions, etc.
                    // For now, we'll just log that we're running in All-or-None mode
                    DatabaseMonolith.Log("Info", $"Executing multi-leg strategy {strategy.Name} in All-or-None mode");
                }
                
                bool success = false;
                
                // Execute the legs based on the ExecuteSimultaneously flag
                if (strategy.ExecuteSimultaneously)
                {
                    // In a real implementation, we would use a broker API that supports simultaneous execution
                    // For now, we'll just place the orders in rapid succession
                    success = await PlaceMultiLegOrder(strategy.Legs);
                }
                else
                {
                    // Place orders sequentially with slight delays
                    foreach (var leg in strategy.Legs.OrderBy(l => l.LegPosition))
                    {
                        await PlaceLimitOrder(leg.Symbol, leg.Quantity, leg.OrderType, leg.Price);
                        
                        // Add a small delay between orders
                        await Task.Delay(500);
                    }
                    success = true;
                }
                
                // Additional strategy-specific post-execution logic
                if (success)
                {
                    DatabaseMonolith.Log("Info", $"Multi-leg strategy executed successfully: {strategy.Name} ({strategy.StrategyType}) with {strategy.Legs.Count} legs");
                    
                    // For certain strategies, set up monitoring or additional management
                    switch (strategy.StrategyType)
                    {
                        case MultiLegStrategyType.VerticalSpread:
                        case MultiLegStrategyType.Straddle:
                        case MultiLegStrategyType.Strangle:
                            // Set up monitoring for these strategies
                            SetupStrategyMonitoring(strategy);
                            break;
                    }
                }
                
                return success;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to place multi-leg strategy: {strategy?.Name}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Sets up monitoring for a multi-leg strategy
        /// </summary>
        /// <param name="strategy">The strategy to monitor</param>
        private void SetupStrategyMonitoring(MultiLegStrategy strategy)
        {
            // In a real implementation, this would set up monitoring tasks for the strategy
            // tracking overall P&L, approaching expiration dates, etc.
            
            // For now, we'll just log that we're setting up monitoring
            DatabaseMonolith.Log("Info", $"Setting up monitoring for strategy {strategy.Name} ({strategy.StrategyType})");
        }
        
        /// <summary>
        /// Creates a vertical spread strategy (bull call spread or bear put spread)
        /// </summary>
        /// <param name="symbol">The underlying symbol</param>
        /// <param name="quantity">Number of contracts</param>
        /// <param name="isBullish">True for bull call spread, false for bear put spread</param>
        /// <param name="lowerStrike">Lower strike price</param>
        /// <param name="upperStrike">Upper strike price</param>
        /// <param name="expiration">Option expiration date</param>
        /// <param name="totalPrice">Optional limit price for the entire spread</param>
        /// <returns>A configured vertical spread strategy</returns>
        public MultiLegStrategy CreateVerticalSpread(
            string symbol, 
            int quantity, 
            bool isBullish, 
            double lowerStrike, 
            double upperStrike, 
            DateTime expiration, 
            double? totalPrice = null)
        {
            try
            {
                // Validate inputs
                if (string.IsNullOrEmpty(symbol) || quantity <= 0 || lowerStrike >= upperStrike)
                {
                    DatabaseMonolith.Log("Error", "Invalid parameters for vertical spread");
                    return null;
                }
                
                var strategy = new MultiLegStrategy
                {
                    Name = isBullish ? $"{symbol} Bull Call Spread" : $"{symbol} Bear Put Spread",
                    StrategyType = MultiLegStrategyType.VerticalSpread,
                    AllOrNone = true,
                    ExecuteSimultaneously = true,
                    RiskLevel = 4 // Moderate risk
                };
                
                if (isBullish) // Bull Call Spread
                {
                    // For bull call spread: Buy lower strike call, sell higher strike call
                    // Get current market prices (in a real implementation)
                    double lowerCallPrice = GetEstimatedOptionPrice(symbol, "CALL", lowerStrike, expiration);
                    double upperCallPrice = GetEstimatedOptionPrice(symbol, "CALL", upperStrike, expiration);
                    
                    // First leg: Buy the lower strike call
                    var leg1 = new ScheduledOrder
                    {
                        Symbol = symbol,
                        Quantity = quantity,
                        OrderType = "BUY",
                        Price = lowerCallPrice,
                        ExecutionTime = DateTime.Now,
                        IsMultiLegStrategy = true,
                        MultiLegStrategyId = strategy.StrategyId,
                        LegPosition = 1,
                        StrikePrice = lowerStrike,
                        ExpirationDate = expiration,
                        OptionType = "CALL",
                        IsOption = true
                    };
                    
                    // Second leg: Sell the higher strike call
                    var leg2 = new ScheduledOrder
                    {
                        Symbol = symbol,
                        Quantity = quantity,
                        OrderType = "SELL",
                        Price = upperCallPrice,
                        ExecutionTime = DateTime.Now,
                        IsMultiLegStrategy = true,
                        MultiLegStrategyId = strategy.StrategyId,
                        LegPosition = 2,
                        StrikePrice = upperStrike,
                        ExpirationDate = expiration,
                        OptionType = "CALL",
                        IsOption = true
                    };
                    
                    strategy.Legs.Add(leg1);
                    strategy.Legs.Add(leg2);
                    
                    // Calculate maximum profit and loss
                    strategy.MaximumLoss = (lowerCallPrice - upperCallPrice) * quantity * 100; // Net debit paid
                    strategy.MaximumProfit = (upperStrike - lowerStrike - (lowerCallPrice - upperCallPrice)) * quantity * 100;
                }
                else // Bear Put Spread
                {
                    // For bear put spread: Buy higher strike put, sell lower strike put
                    // Get current market prices (in a real implementation)
                    double lowerPutPrice = GetEstimatedOptionPrice(symbol, "PUT", lowerStrike, expiration);
                    double upperPutPrice = GetEstimatedOptionPrice(symbol, "PUT", upperStrike, expiration);
                    
                    // First leg: Buy the higher strike put
                    var leg1 = new ScheduledOrder
                    {
                        Symbol = symbol,
                        Quantity = quantity,
                        OrderType = "BUY",
                        Price = upperPutPrice,
                        ExecutionTime = DateTime.Now,
                        IsMultiLegStrategy = true,
                        MultiLegStrategyId = strategy.StrategyId,
                        LegPosition = 1,
                        StrikePrice = upperStrike,
                        ExpirationDate = expiration,
                        OptionType = "PUT",
                        IsOption = true
                    };
                    
                    // Second leg: Sell the lower strike put
                    var leg2 = new ScheduledOrder
                    {
                        Symbol = symbol,
                        Quantity = quantity,
                        OrderType = "SELL",
                        Price = lowerPutPrice,
                        ExecutionTime = DateTime.Now,
                        IsMultiLegStrategy = true,
                        MultiLegStrategyId = strategy.StrategyId,
                        LegPosition = 2,
                        StrikePrice = lowerStrike,
                        ExpirationDate = expiration,
                        OptionType = "PUT",
                        IsOption = true
                    };
                    
                    strategy.Legs.Add(leg1);
                    strategy.Legs.Add(leg2);
                    
                    // Calculate maximum profit and loss
                    strategy.MaximumLoss = (upperPutPrice - lowerPutPrice) * quantity * 100; // Net debit paid
                    strategy.MaximumProfit = (upperStrike - lowerStrike - (upperPutPrice - lowerPutPrice)) * quantity * 100;
                }
                
                // If a total price for the spread was specified, adjust the individual leg prices
                if (totalPrice.HasValue)
                {
                    double currentPrice = strategy.CalculateNetCost();
                    if (Math.Abs(currentPrice) > 0)
                    {
                        double ratio = totalPrice.Value / Math.Abs(currentPrice);
                        foreach (var leg in strategy.Legs)
                        {
                            leg.Price *= ratio;
                        }
                    }
                }
                
                return strategy;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to create vertical spread for {symbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Creates a straddle strategy (buying a call and put at the same strike)
        /// </summary>
        /// <param name="symbol">The underlying symbol</param>
        /// <param name="quantity">Number of contracts</param>
        /// <param name="strikePrice">Strike price for both call and put</param>
        /// <param name="expiration">Option expiration date</param>
        /// <returns>A configured straddle strategy</returns>
        public MultiLegStrategy CreateStraddle(string symbol, int quantity, double strikePrice, DateTime expiration)
        {
            try
            {
                // Validate inputs
                if (string.IsNullOrEmpty(symbol) || quantity <= 0 || strikePrice <= 0)
                {
                    DatabaseMonolith.Log("Error", "Invalid parameters for straddle");
                    return null;
                }
                
                var strategy = new MultiLegStrategy
                {
                    Name = $"{symbol} {strikePrice} Straddle",
                    StrategyType = MultiLegStrategyType.Straddle,
                    AllOrNone = true,
                    ExecuteSimultaneously = true,
                    RiskLevel = 6 // Higher risk due to premium paid for both options
                };
                
                // Get current market prices (in a real implementation)
                double callPrice = GetEstimatedOptionPrice(symbol, "CALL", strikePrice, expiration);
                double putPrice = GetEstimatedOptionPrice(symbol, "PUT", strikePrice, expiration);
                
                // First leg: Buy the call
                var callLeg = new ScheduledOrder
                {
                    Symbol = symbol,
                    Quantity = quantity,
                    OrderType = "BUY",
                    Price = callPrice,
                    ExecutionTime = DateTime.Now,
                    IsMultiLegStrategy = true,
                    MultiLegStrategyId = strategy.StrategyId,
                    LegPosition = 1,
                    StrikePrice = strikePrice,
                    ExpirationDate = expiration,
                    OptionType = "CALL",
                    IsOption = true
                };
                
                // Second leg: Buy the put
                var putLeg = new ScheduledOrder
                {
                    Symbol = symbol,
                    Quantity = quantity,
                    OrderType = "BUY",
                    Price = putPrice,
                    ExecutionTime = DateTime.Now,
                    IsMultiLegStrategy = true,
                    MultiLegStrategyId = strategy.StrategyId,
                    LegPosition = 2,
                    StrikePrice = strikePrice,
                    ExpirationDate = expiration,
                    OptionType = "PUT",
                    IsOption = true
                };
                
                strategy.Legs.Add(callLeg);
                strategy.Legs.Add(putLeg);
                
                // Calculate maximum loss (total premium paid)
                strategy.MaximumLoss = (callPrice + putPrice) * quantity * 100;
                strategy.MaximumProfit = double.PositiveInfinity; // Theoretically unlimited profit potential
                
                return strategy;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to create straddle for {symbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Creates a pairs trade strategy (long one security, short a correlated one)
        /// </summary>
        /// <param name="longSymbol">Symbol to buy</param>
        /// <param name="shortSymbol">Symbol to short</param>
        /// <param name="longQuantity">Quantity to buy</param>
        /// <param name="shortQuantity">Quantity to short</param>
        /// <param name="correlation">Correlation coefficient between the securities</param>
        /// <returns>A configured pairs trade strategy</returns>
        public MultiLegStrategy CreatePairsTrade(
            string longSymbol, 
            string shortSymbol, 
            int longQuantity, 
            int shortQuantity, 
            double correlation)
        {
            try
            {
                // Validate inputs
                if (string.IsNullOrEmpty(longSymbol) || string.IsNullOrEmpty(shortSymbol) || 
                    longQuantity <= 0 || shortQuantity <= 0)
                {
                    DatabaseMonolith.Log("Error", "Invalid parameters for pairs trade");
                    return null;
                }
                
                var strategy = new MultiLegStrategy
                {
                    Name = $"{longSymbol}/{shortSymbol} Pairs Trade",
                    StrategyType = MultiLegStrategyType.PairsTrade,
                    AllOrNone = true,
                    ExecuteSimultaneously = false, // Often better to execute sequentially to avoid market impact
                    RiskLevel = 5 // Moderate risk
                };
                
                // Get market prices (in a real implementation)
                double longPrice = GetMarketPrice(longSymbol).Result;
                double shortPrice = GetMarketPrice(shortSymbol).Result;
                
                // First leg: Buy the first symbol
                var longLeg = new ScheduledOrder
                {
                    Symbol = longSymbol,
                    Quantity = longQuantity,
                    OrderType = "BUY",
                    Price = longPrice,
                    ExecutionTime = DateTime.Now,
                    IsMultiLegStrategy = true,
                    MultiLegStrategyId = strategy.StrategyId,
                    LegPosition = 1,
                    IsOption = false
                };
                
                // Second leg: Short the second symbol
                var shortLeg = new ScheduledOrder
                {
                    Symbol = shortSymbol,
                    Quantity = shortQuantity,
                    OrderType = "SELL",
                    Price = shortPrice,
                    ExecutionTime = DateTime.Now,
                    IsMultiLegStrategy = true,
                    MultiLegStrategyId = strategy.StrategyId,
                    LegPosition = 2,
                    IsOption = false
                };
                
                strategy.Legs.Add(longLeg);
                strategy.Legs.Add(shortLeg);
                
                // Add correlation information to strategy notes
                strategy.Notes = $"Correlation: {correlation:F2}. Long: {longSymbol} ({longQuantity} shares), " +
                                 $"Short: {shortSymbol} ({shortQuantity} shares).";
                
                return strategy;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to create pairs trade: {longSymbol}/{shortSymbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Creates a basket order strategy (multiple securities traded together)
        /// </summary>
        /// <param name="symbols">List of symbols</param>
        /// <param name="quantities">Corresponding quantities</param>
        /// <param name="orderTypes">Corresponding order types (BUY/SELL)</param>
        /// <param name="weights">Optional portfolio weights</param>
        /// <returns>A configured basket order strategy</returns>
        public MultiLegStrategy CreateBasketOrder(
            List<string> symbols, 
            List<int> quantities, 
            List<string> orderTypes,
            List<double> weights = null)
        {
            try
            {
                // Validate inputs
                if (symbols == null || quantities == null || orderTypes == null ||
                    symbols.Count != quantities.Count || symbols.Count != orderTypes.Count)
                {
                    DatabaseMonolith.Log("Error", "Invalid parameters for basket order");
                    return null;
                }
                
                var strategy = new MultiLegStrategy
                {
                    Name = "Basket Order",
                    StrategyType = MultiLegStrategyType.BasketOrder,
                    AllOrNone = true,
                    ExecuteSimultaneously = false, // Sequential execution to minimize market impact
                    RiskLevel = 3 // Lower risk due to diversification
                };
                
                // Build description of the basket
                StringBuilder descBuilder = new StringBuilder();
                descBuilder.AppendLine("Basket composition:");
                
                // Add each security to the basket
                for (int i = 0; i < symbols.Count; i++)
                {
                    double price = GetMarketPrice(symbols[i]).Result;
                    
                    var order = new ScheduledOrder
                    {
                        Symbol = symbols[i],
                        Quantity = quantities[i],
                        OrderType = orderTypes[i],
                        Price = price,
                        ExecutionTime = DateTime.Now,
                        IsMultiLegStrategy = true,
                        MultiLegStrategyId = strategy.StrategyId,
                        LegPosition = i + 1,
                        IsOption = false
                    };
                    
                    strategy.Legs.Add(order);
                    
                    // Add description
                    string weightStr = weights != null && i < weights.Count ? $" (Weight: {weights[i]:P2})" : "";
                    descBuilder.AppendLine($"- {orderTypes[i]} {quantities[i]} {symbols[i]} @ {price:C2}{weightStr}");
                }
                
                strategy.Notes = descBuilder.ToString();
                
                // If this is a rebalance, note that in the name
                if (weights != null)
                {
                    strategy.Name = "Portfolio Rebalance";
                }
                
                return strategy;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to create basket order", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Validates a multi-leg strategy before execution
        /// </summary>
        /// <param name="strategy">The strategy to validate</param>
        /// <returns>True if the strategy is valid for execution</returns>
        public bool ValidateMultiLegStrategy(MultiLegStrategy strategy)
        {
            try
            {
                if (strategy == null)
                {
                    DatabaseMonolith.Log("Error", "Cannot validate null strategy");
                    return false;
                }
                
                // Check if trading is allowed
                if (!IsTradingAllowed())
                {
                    DatabaseMonolith.Log("Warning", $"Strategy validation failed: Trading not allowed at this time");
                    return false;
                }
                
                // Check legs existence
                if (strategy.Legs == null || strategy.Legs.Count == 0)
                {
                    DatabaseMonolith.Log("Error", "Strategy validation failed: No legs defined");
                    return false;
                }
                
                // Risk check
                bool isHighRisk = strategy.RiskLevel >= 8;
                if (isHighRisk && riskMode != RiskMode.Aggressive)
                {
                    DatabaseMonolith.Log("Warning", 
                        $"Strategy validation failed: Risk level {strategy.RiskLevel} exceeds current risk mode {riskMode}");
                    return false;
                }
                
                // Perform strategy-specific validation
                if (!strategy.Validate())
                {
                    DatabaseMonolith.Log("Error", 
                        $"Strategy validation failed: Failed strategy-specific validation for {strategy.StrategyType}");
                    return false;
                }
                
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error validating multi-leg strategy", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Gets an estimated option price (simplified model for demonstration)
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="optionType">CALL or PUT</param>
        /// <param name="strikePrice">Strike price</param>
        /// <param name="expiration">Expiration date</param>
        /// <returns>Estimated option price</returns>
        private double GetEstimatedOptionPrice(string symbol, string optionType, double strikePrice, DateTime expiration)
        {
            try
            {
                // This is a very simplified option pricing model for demonstration
                // In reality, you'd use a real options API or Black-Scholes model
                
                // Get underlying price
                double underlyingPrice = GetMarketPrice(symbol).Result;
                
                // Calculate days to expiration
                double daysToExpiration = (expiration - DateTime.Now).TotalDays;
                if (daysToExpiration <= 0) daysToExpiration = 1; // Avoid division by zero
                
                // Simple volatility estimate (would use historical volatility in reality)
                double volatility = 0.3; // 30% annual volatility
                
                // Simple time decay factor
                double timeFactor = Math.Sqrt(daysToExpiration / 365.0);
                
                // Intrinsic value
                double intrinsicValue = 0;
                if (optionType == "CALL")
                {
                    intrinsicValue = Math.Max(0, underlyingPrice - strikePrice);
                }
                else // PUT
                {
                    intrinsicValue = Math.Max(0, strikePrice - underlyingPrice);
                }
                
                // Time value (very simplified)
                double timeValue = underlyingPrice * volatility * timeFactor;
                
                // Adjust time value based on how far in/out of the money
                double moneyness = Math.Abs(1 - (strikePrice / underlyingPrice));
                timeValue *= Math.Exp(-2 * moneyness); // Reduce time value for deep ITM/OTM options
                
                // Total estimated price
                double estimatedPrice = intrinsicValue + timeValue;
                
                // Minimum price
                double minimumPrice = 0.05;
                
                return Math.Max(minimumPrice, Math.Round(estimatedPrice, 2));
            }
            catch
            {
                // If there's any error, return a default estimate
                return 1.00;
            }
        }
        
        /// <summary>
        /// Calculates Greek metrics for an option position including Theta
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="optionType">CALL or PUT</param>
        /// <param name="strikePrice">Strike price</param>
        /// <param name="expiration">Expiration date</param>
        /// <param name="quantity">Number of contracts (positive for long, negative for short)</param>
        /// <returns>Greek metrics including Theta for time decay analysis</returns>
        private async Task<GreekMetrics> GetOptionGreeks(string symbol, string optionType, double strikePrice, DateTime expiration, int quantity = 1)
        {
            try
            {
                var greekEngine = new GreekCalculationEngine();
                
                // Get underlying price
                double underlyingPrice = await GetMarketPrice(symbol);
                
                // Calculate days to expiration
                double daysToExpiration = (expiration - DateTime.Now).TotalDays;
                if (daysToExpiration <= 0) daysToExpiration = 1; // Avoid division by zero
                
                // Get current market conditions
                var marketConditions = new MarketConditions
                {
                    InterestRate = 0.05,
                    VolatilityIndex = 20.0,
                    MarketTrend = 0.0,
                    EconomicGrowth = 0.0
                };
                
                // Create position for Greek calculation
                var position = new Position
                {
                    Symbol = symbol,
                    UnderlyingPrice = underlyingPrice,
                    StrikePrice = strikePrice,
                    TimeToExpiration = daysToExpiration / 365.0, // Convert to years
                    RiskFreeRate = marketConditions.InterestRate, // Dynamic risk-free rate sourced from market conditions
                    Volatility = 0.3, // 30% annual volatility (could be calculated from historical data)
                    IsCall = optionType.ToUpper() == "CALL",
                    Quantity = quantity,
                    OptionPrice = GetEstimatedOptionPrice(symbol, optionType, strikePrice, expiration)
                };
                
                // Calculate all Greeks including Theta
                var greeks = greekEngine.CalculateGreeks(position, marketConditions);
                
                DatabaseMonolith.Log("Info", $"Option Greeks calculated for {symbol} {optionType} {strikePrice}: " +
                    $"Theta={greeks.Theta:F4} (time decay per day), Delta={greeks.Delta:F4}, Gamma={greeks.Gamma:F4}");
                
                return greeks;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Error calculating option Greeks for {symbol}", ex.ToString());
                
                // Return default Greeks on error
                return new GreekMetrics
                {
                    Theta = 0.0,
                    Delta = 0.5,
                    Gamma = 0.1,
                    Vega = 0.2,
                    Rho = 0.05
                };
            }
        }
        
        /// <summary>
        /// Analyzes time decay for option positions and provides trading recommendations
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="optionType">CALL or PUT</param>
        /// <param name="strikePrice">Strike price</param>
        /// <param name="expiration">Expiration date</param>
        /// <returns>Time decay analysis and recommendations</returns>
        public async Task<string> AnalyzeOptionTimeDecay(string symbol, string optionType, double strikePrice, DateTime expiration)
        {
            try
            {
                var greeksTask = GetOptionGreeks(symbol, optionType, strikePrice, expiration, 1);
                var shortGreeksTask = GetOptionGreeks(symbol, optionType, strikePrice, expiration, -1);
                await Task.WhenAll(greeksTask, shortGreeksTask);
                
                var greeks = greeksTask.Result;
                var shortGreeks = shortGreeksTask.Result;
                
                double daysToExpiration = (expiration - DateTime.Now).TotalDays;
                double weeklyDecay = greeks.Theta * 7; // Weekly time decay
                double totalDecay = greeks.Theta * daysToExpiration; // Total decay to expiration
                
                var analysis = new StringBuilder();
                analysis.AppendLine($"=== THETA ANALYSIS FOR {symbol} {optionType} ${strikePrice} ===");
                analysis.AppendLine($"Days to Expiration: {daysToExpiration:F0}");
                analysis.AppendLine($"Current Theta: {greeks.Theta:F4} (daily time decay)");
                analysis.AppendLine($"Weekly Time Decay: ${Math.Abs(weeklyDecay):F2}");
                analysis.AppendLine($"Total Decay to Expiration: ${Math.Abs(totalDecay):F2}");
                analysis.AppendLine();
                
                // Long position analysis
                analysis.AppendLine("LONG POSITION IMPACT:");
                analysis.AppendLine($"• Daily P&L from time decay: ${greeks.Theta:F2}");
                analysis.AppendLine($"• Weekly P&L from time decay: ${weeklyDecay:F2}");
                
                // Short position analysis  
                analysis.AppendLine();
                analysis.AppendLine("SHORT POSITION IMPACT:");
                analysis.AppendLine($"• Daily P&L from time decay: ${shortGreeks.Theta:F2}");
                analysis.AppendLine($"• Weekly P&L from time decay: ${shortGreeks.Theta * 7:F2}");
                analysis.AppendLine();
                
                // Trading recommendations based on Theta
                analysis.AppendLine("THETA TRADING RECOMMENDATIONS:");
                
                if (daysToExpiration < 30)
                {
                    analysis.AppendLine("⚠️  HIGH TIME DECAY ZONE (< 30 days)");
                    analysis.AppendLine("• Consider theta harvesting strategies (sell options)");
                    analysis.AppendLine("• Avoid buying options unless strong conviction");
                    analysis.AppendLine("• Monitor positions closely - time decay accelerating");
                }
                else if (daysToExpiration < 60)
                {
                    analysis.AppendLine("📈 MODERATE TIME DECAY ZONE (30-60 days)");
                    analysis.AppendLine("• Optimal zone for selling premium");
                    analysis.AppendLine("• Consider covered calls or cash-secured puts");
                    analysis.AppendLine("• Good balance between time decay and time value");
                }
                else
                {
                    analysis.AppendLine("🕐 LOW TIME DECAY ZONE (> 60 days)");
                    analysis.AppendLine("• Time decay impact minimal");
                    analysis.AppendLine("• Suitable for buying options with directional bias");
                    analysis.AppendLine("• Focus on delta and gamma rather than theta");
                }
                
                // Additional theta-specific insights
                analysis.AppendLine();
                if (Math.Abs(greeks.Theta) > 0.05)
                {
                    analysis.AppendLine("🔥 HIGH THETA: Significant time decay - ideal for income strategies");
                }
                else if (Math.Abs(greeks.Theta) < 0.01)
                {
                    analysis.AppendLine("❄️  LOW THETA: Minimal time decay - suitable for directional plays");
                }
                
                return analysis.ToString();
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error analyzing time decay for {symbol}", ex.ToString());
                return $"Error analyzing time decay: {ex.Message}";
            }
        }
        
        /// <summary>
        /// Splits a large order into smaller chunks to minimize market impact
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="quantity">Total quantity to trade</param>
        /// <param name="orderType">BUY or SELL</param>
        /// <param name="price">Limit price</param>
        /// <param name="chunks">Number of chunks to split into</param>
        /// <param name="intervalMinutes">Minutes between each chunk</param>
        /// <returns>True if the split order was scheduled successfully</returns>
        public bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes)
        {
            // Call the enhanced version with default parameters
            return SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes, 
                priceVariancePercent: 0, 
                randomizeIntervals: false, 
                distribution: OrderDistributionType.Equal);
        }
        
        /// <summary>
        /// Enhanced version that splits a large order into smaller chunks with additional options to minimize market impact
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="quantity">Total quantity to trade</param>
        /// <param name="orderType">BUY or SELL</param>
        /// <param name="price">Base limit price</param>
        /// <param name="chunks">Number of chunks to split into</param>
        /// <param name="intervalMinutes">Base minutes between each chunk</param>
        /// <param name="priceVariancePercent">Percentage to vary price between chunks (0-5%)</param>
        /// <param name="randomizeIntervals">Whether to randomize time intervals between chunks</param>
        /// <param name="distribution">How to distribute quantity across chunks</param>
        /// <returns>True if the split order was scheduled successfully</returns>
        public bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes,
            double priceVariancePercent, bool randomizeIntervals, OrderDistributionType distribution)
        {
            try
            {
                // Validate parameters
                if (chunks <= 0 || quantity <= 0 || string.IsNullOrEmpty(symbol))
                {
                    DatabaseMonolith.Log("Error", $"Invalid parameters for split order: Symbol={symbol}, Quantity={quantity}, Chunks={chunks}");
                    return false;
                }
                
                // Ensure price variance is within a reasonable range (0-5%)
                priceVariancePercent = Math.Max(0, Math.Min(5, priceVariancePercent));
                
                // Generate a unique ID for this group of split orders
                string splitOrderGroupId = $"{symbol}-{Guid.NewGuid():N}";
                
                // Calculate order quantities based on distribution type
                List<int> chunkSizes = CalculateChunkSizes(quantity, chunks, distribution);
                
                // Create a random number generator for variance calculations
                Random random = new Random();
                
                // Calculate base intervals based on randomization setting
                List<int> intervals = CalculateIntervals(chunks, intervalMinutes, randomizeIntervals);
                
                // Keep track of cumulative time for scheduling
                int cumulativeMinutes = 0;
                
                // Schedule each chunk
                for (int i = 0; i < chunks; i++)
                {
                    int chunkShares = chunkSizes[i];
                    
                    // Apply price variance if specified
                    double chunkPrice = price;
                    if (priceVariancePercent > 0)
                    {
                        // Calculate variance within the specified percentage range
                        double varianceFactor = 1.0 + ((random.NextDouble() * 2 - 1) * priceVariancePercent / 100.0);
                        chunkPrice = Math.Round(price * varianceFactor, 2);
                    }
                    
                    // Add interval for this chunk to cumulative time
                    cumulativeMinutes += intervals[i];
                    
                    // Create the scheduled order
                    var order = new ScheduledOrder
                    {
                        Symbol = symbol,
                        Quantity = chunkShares,
                        OrderType = orderType,
                        Price = chunkPrice,
                        ExecutionTime = DateTime.Now.AddMinutes(cumulativeMinutes),
                        IsSplitOrder = true,
                        SplitOrderGroupId = splitOrderGroupId,
                        SplitOrderSequence = i + 1,
                        SplitOrderTotalChunks = chunks
                    };
                    
                    // Add to scheduled orders
                    if (!_scheduledOrders.ContainsKey(symbol))
                    {
                        _scheduledOrders[symbol] = new List<ScheduledOrder>();
                    }
                    _scheduledOrders[symbol].Add(order);
                }
                
                // Log details of the split order
                string distributionName = distribution.ToString();
                string intervalType = randomizeIntervals ? "randomized" : "fixed";
                string priceVariance = priceVariancePercent > 0 ? $" with price variance of ±{priceVariancePercent:F1}%" : "";
                
                DatabaseMonolith.Log("Info", $"Enhanced order split for {symbol}: {quantity} {orderType} shares into {chunks} chunks " +
                    $"using {distributionName} distribution, {intervalType} intervals{priceVariance}. " +
                    $"Group ID: {splitOrderGroupId}");
                
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to split large order for {symbol}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Calculates the size of each chunk based on the distribution type
        /// </summary>
        private List<int> CalculateChunkSizes(int quantity, int chunks, OrderDistributionType distribution)
        {
            List<int> chunkSizes = new List<int>();
            
            switch (distribution)
            {
                case OrderDistributionType.FrontLoaded:
                    // Front-loaded: Larger chunks at the beginning, tapering off
                    double totalWeight = chunks * (chunks + 1) / 2.0; // Sum of 1 to chunks
                    
                    for (int i = chunks; i >= 1; i--)
                    {
                        int chunkSize = (int)Math.Round((i / totalWeight) * quantity);
                        chunkSizes.Add(chunkSize);
                    }
                    break;
                    
                case OrderDistributionType.BackLoaded:
                    // Back-loaded: Smaller chunks at the beginning, larger at the end
                    totalWeight = chunks * (chunks + 1) / 2.0; // Sum of 1 to chunks
                    
                    for (int i = 1; i <= chunks; i++)
                    {
                        int chunkSize = (int)Math.Round((i / totalWeight) * quantity);
                        chunkSizes.Add(chunkSize);
                    }
                    break;
                    
                case OrderDistributionType.Normal:
                    // Normal (bell curve): Middle chunks are larger
                    double mean = (chunks - 1) / 2.0;
                    double stdDev = chunks / 6.0; // ~99% within the range
                    double[] weights = new double[chunks];
                    double weightSum = 0;
                    
                    for (int i = 0; i < chunks; i++)
                    {
                        weights[i] = Math.Exp(-0.5 * Math.Pow((i - mean) / stdDev, 2));
                        weightSum += weights[i];
                    }
                    
                    for (int i = 0; i < chunks; i++)
                    {
                        int chunkSize = (int)Math.Round((weights[i] / weightSum) * quantity);
                        chunkSizes.Add(chunkSize);
                    }
                    break;
                    
                case OrderDistributionType.Equal:
                default:
                    // Equal distribution (with remainder added to first chunk)
                    int sharesPerChunk = quantity / chunks;
                    int remainder = quantity % chunks;
                    
                    for (int i = 0; i < chunks; i++)
                    {
                        int chunkSize = sharesPerChunk;
                        if (i == 0)
                        {
                            chunkSize += remainder;
                        }
                        chunkSizes.Add(chunkSize);
                    }
                    break;
            }
            
            // Ensure we distribute exactly the requested quantity
            int totalAllocated = chunkSizes.Sum();
            if (totalAllocated != quantity)
            {
                int diff = quantity - totalAllocated;
                chunkSizes[0] += diff;
            }
            
            return chunkSizes;
        }
        
        /// <summary>
        /// Calculates the time intervals between chunks
        /// </summary>
        private List<int> CalculateIntervals(int chunks, int baseIntervalMinutes, bool randomize)
        {
            List<int> intervals = new List<int>();
            Random random = new Random();
            
            for (int i = 0; i < chunks; i++)
            {
                if (i == 0)
                {
                    // First chunk is executed immediately
                    intervals.Add(0);
                }
                else if (randomize)
                {
                    // Randomize interval between 50% and 150% of base interval
                    int minInterval = Math.Max(1, (int)(baseIntervalMinutes * 0.5));
                    int maxInterval = (int)(baseIntervalMinutes * 1.5);
                    intervals.Add(random.Next(minInterval, maxInterval + 1));
                }
                else
                {
                    // Use fixed interval
                    intervals.Add(baseIntervalMinutes);
                }
            }
            
            return intervals;
        }
        
        /// <summary>
        /// Cancels all remaining chunks of a split order group
        /// </summary>
        /// <param name="splitOrderGroupId">The unique identifier of the split order group</param>
        /// <returns>Number of chunks cancelled</returns>
        public int CancelSplitOrderGroup(string splitOrderGroupId)
        {
            try
            {
                int cancelCount = 0;
                
                foreach (var symbol in _scheduledOrders.Keys.ToList())
                {
                    var ordersToRemove = _scheduledOrders[symbol]
                        .Where(o => o.IsSplitOrder && o.SplitOrderGroupId == splitOrderGroupId)
                        .ToList();
                    
                    if (ordersToRemove.Any())
                    {
                        foreach (var order in ordersToRemove)
                        {
                            _scheduledOrders[symbol].Remove(order);
                            cancelCount++;
                        }
                        
                        // Clean up if no orders left for this symbol
                        if (_scheduledOrders[symbol].Count == 0)
                        {
                            _scheduledOrders.Remove(symbol);
                        }
                    }
                }
                
                if (cancelCount > 0)
                {
                    DatabaseMonolith.Log("Info", $"Cancelled {cancelCount} remaining chunks of split order group {splitOrderGroupId}");
                }
                
                return cancelCount;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to cancel split order group {splitOrderGroupId}", ex.ToString());
                return 0;
            }
        }
        

        
        /// <summary>
        /// Gets a list of all symbols with active trailing stops
        /// </summary>
        /// <returns>List of symbols with active trailing stops</returns>
        public List<string> GetSymbolsWithTrailingStops()
        {
            return _trailingStops.Keys.ToList();
        }
        
        /// <summary>
        /// Removes a trailing stop for a symbol
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <returns>True if the trailing stop was removed successfully</returns>
        public bool RemoveTrailingStop(string symbol)
        {
            try
            {
                if (_trailingStops.ContainsKey(symbol))
                {
                    _trailingStops.Remove(symbol);
                    DatabaseMonolith.Log("Info", $"Trailing stop removed for {symbol}");
                    return true;
                }
                
                return false;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error removing trailing stop for {symbol}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Checks if emergency stop is currently active
        /// </summary>
        /// <returns>True if emergency stop is active</returns>

        private bool IsHighLiquidity(string stock)
        {
            return GetMarketPrice(stock).Result > 5 && technicalIndicatorService.GetVWAP(stock, "1min").Result > 1;
        }

        private bool HasRecentNewsCatalyst(string stock)
        {
            return new Random().Next(0, 2) == 1;
        }

        private void LoadSymbols()
        {
            if (File.Exists(symbolFilePath))
            {
                personalSymbols = File.ReadAllLines(symbolFilePath).ToList();
            }
        }

        private void SaveSymbols()
        {
            File.WriteAllLines(symbolFilePath, personalSymbols);
        }

        public void AddSymbol(string symbol)
        {
            if (!personalSymbols.Contains(symbol))
            {
                personalSymbols.Add(symbol);
                SaveSymbols();
            }
        }

        public void RemoveSymbol(string symbol)
        {
            if (personalSymbols.Contains(symbol))
            {
                personalSymbols.Remove(symbol);
                SaveSymbols();
            }
        }

        private async Task<List<double>> GetPrices(string symbol, string timeRange)
        {
            // Implement logic to fetch prices based on the time range
            // This is a placeholder implementation
            return new List<double> { 696969 };
        }

        private async Task<List<double>> GetRSIValues(string symbol, string timeRange)
        {
            // Implement logic to fetch RSI values based on the time range
            // This is a placeholder implementation
            return new List<double> { 3.14 };
        }

        // ...existing code...
        public async Task<QuoteData> GetQuoteData(string symbol)
        {
            try
            {
                // Use CEF-based Yahoo Finance data retrieval
                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                if (quote != null)
                {
                    return quote;
                }
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching quote data for {symbol} via Yahoo Finance: {ex.Message}");
                throw new Exception($"Error fetching quote data: {ex.Message}");
            }
        }

        public async Task<StockData> GetStockData(string symbol, string timeRange)
        {
            try
            {
                // Map the timeRange to Yahoo parameters
                string range = "1mo";
                string interval = "1d";
                
                switch (timeRange)
                {
                    case "1day":
                        range = "1d";
                        interval = "5m";
                        break;
                    case "1week":
                        range = "5d";
                        interval = "30m";
                        break;
                    case "1month":
                        range = "1mo";
                        interval = "1d";
                        break;
                    case "3month":
                        range = "3mo";
                        interval = "1d";
                        break;
                    case "1year":
                        range = "1y";
                        interval = "1wk";
                        break;
                }
                
                // Use the HistoricalDataService to get the data
                var historicalPrices = await historicalDataService.GetHistoricalPrices(symbol, range, interval);
                // Await the conversion to StockData, since ConvertToStockData is async and returns Task<StockData>
                return await historicalDataService.ConvertToStockData(historicalPrices, symbol, range, interval);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching stock data for {symbol}: {ex.Message}");
                throw;
            }
        }

        private (List<double> upperBand, List<double> middleBand, List<double> lowerBand) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier)
        {
            var middleBand = new List<double>();
            var upperBand = new List<double>();
            var lowerBand = new List<double>();

            // Add empty values for initial periods where we can't calculate
            for (int i = 0; i < period - 1; i++)
            {
                middleBand.Add(double.NaN);
                upperBand.Add(double.NaN);
                lowerBand.Add(double.NaN);
            }

            for (int i = period - 1; i < prices.Count; i++)
            {
                var periodPrices = prices.Skip(i - period + 1).Take(period).ToList();
                var average = periodPrices.Average();
                var stdDev = Math.Sqrt(periodPrices.Average(v => Math.Pow(v - average, 2)));

                middleBand.Add(average);
                upperBand.Add(average + stdDevMultiplier * stdDev);
                lowerBand.Add(average - stdDevMultiplier * stdDev);
            }

            return (upperBand, middleBand, lowerBand);
        }

        private List<double> CalculateRSI(List<double> prices, int period)
        {
            var rsiValues = new List<double>();

            // Add empty values for initial periods where we can't calculate
            for (int i = 0; i < period; i++)
            {
                rsiValues.Add(double.NaN);
            }

            if (prices.Count <= period)
            {
                return rsiValues;
            }

            List<double> gains = new List<double>();
            List<double> losses = new List<double>();

            // Calculate price changes
            for (int i = 1; i < prices.Count; i++)
            {
                double change = prices[i] - prices[i - 1];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? -change : 0);
            }

            // Calculate initial average gain and loss
            double avgGain = gains.Take(period).Average();
            double avgLoss = losses.Take(period).Average();

            // Calculate first RSI
            double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            double rsi = 100 - (100 / (1 + rs));
            rsiValues.Add(rsi);

            // Calculate remaining RSI values
            for (int i = period + 1; i < prices.Count; i++)
            {
                avgGain = ((avgGain * (period - 1)) + gains[i - 1]) / period;
                avgLoss = ((avgLoss * (period - 1)) + losses[i - 1]) / period;

                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiValues.Add(rsi);
            }

            return rsiValues;
        }

        public async Task<QuoteData> FetchQuoteData(string symbol)
        {
            try
            {
                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                if (quote != null)
                {
                    return quote;
                }
                return null;
            }
            catch (Exception ex)
            {
                throw new Exception($"Error fetching quote data: {ex.Message}");
            }
        }

        public async Task<StockData> FetchChartData(string symbol, string timeRange)
        {
            try
            {
                // Map the timeRange to number of data points (approximate)
                int dataPoints = 30;
                switch (timeRange)
                {
                    case "1day":
                        dataPoints = 78; // 6.5 hours * 12 (5min intervals)
                        break;
                    case "1week":
                        dataPoints = 65; // 5 days * 13 (30min intervals)
                        break;
                    case "1month":
                        dataPoints = 30;
                        break;
                    case "3month":
                        dataPoints = 90;
                        break;
                    case "1year":
                        dataPoints = 52;
                        break;
                }

                var prices = await alphaVantageService.GetHistoricalClosingPricesAsync(symbol, dataPoints);

                if (prices == null || prices.Count == 0)
                    throw new Exception($"No historical data available for {symbol}");

                // Calculate Bollinger Bands
                var period = Math.Min(20, prices.Count);
                var (upperBand, middleBand, lowerBand) = CalculateBollingerBandsForChart(prices, period, 2.0);

                // Calculate RSI values
                var rsiValues = CalculateRSIForChart(prices, 14);

                // Generate date labels as List<DateTime> (chronological order, most recent last)
                var today = DateTime.Now.Date;
                var dates = Enumerable.Range(0, prices.Count)
                                      .Select(i => today.AddDays(-(prices.Count - 1 - i)))
                                      .ToList();

                return new StockData
                {
                    Prices = prices,
                    UpperBand = upperBand,
                    MiddleBand = middleBand,
                    LowerBand = lowerBand,
                    RSI = rsiValues,
                    Dates = dates
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching stock data for {symbol}: {ex.Message}");
                throw;
            }
        }

        // Helper methods with unique names to avoid conflicts
        private (List<double> upperBand, List<double> middleBand, List<double> lowerBand) CalculateBollingerBandsForChart(List<double> prices, int period, double stdDevMultiplier)
        {
            var middleBand = new List<double>();
            var upperBand = new List<double>();
            var lowerBand = new List<double>();
            
            // Add empty values for initial periods where we can't calculate
            for (int i = 0; i < period - 1; i++)
            {
                middleBand.Add(double.NaN);
                upperBand.Add(double.NaN);
                lowerBand.Add(double.NaN);
            }

            for (int i = period - 1; i < prices.Count; i++)
            {
                var periodPrices = prices.Skip(i - period + 1).Take(period).ToList();
                var average = periodPrices.Average();
                var stdDev = Math.Sqrt(periodPrices.Average(v => Math.Pow(v - average, 2)));

                middleBand.Add(average);
                upperBand.Add(average + stdDevMultiplier * stdDev);
                lowerBand.Add(average - stdDevMultiplier * stdDev);
            }

            return (upperBand, middleBand, lowerBand);
        }

        private List<double> CalculateRSIForChart(List<double> prices, int period)
        {
            var rsiValues = new List<double>();
            
            // Add empty values for initial periods where we can't calculate
            for (int i = 0; i < period; i++)
            {
                rsiValues.Add(double.NaN);
            }
            
            if (prices.Count <= period)
            {
                return rsiValues;
            }
            
            List<double> gains = new List<double>();
            List<double> losses = new List<double>();
            
            // Calculate price changes
            for (int i = 1; i < prices.Count; i++)
            {
                double change = prices[i] - prices[i - 1];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? -change : 0);
            }
            
            // Calculate initial average gain and loss
            double avgGain = gains.Take(period).Average();
            double avgLoss = losses.Take(period).Average();
            
            // Calculate first RSI
            double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            double rsi = 100 - (100 / (1 + rs));
            rsiValues.Add(rsi);
            
            // Calculate remaining RSI values
            for (int i = period + 1; i < prices.Count; i++)
            {
                avgGain = ((avgGain * (period - 1)) + gains[i - 1]) / period;
                avgLoss = ((avgLoss * (period - 1)) + losses[i - 1]) / period;
                
                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiValues.Add(rsi);
            }
            
            return rsiValues;
        }

        /// <summary>
        /// Gets current market conditions for decision making
        /// </summary>
        /// <returns>MarketConditions object with current market metrics</returns>
        public Quantra.Models.MarketConditions GetCurrentMarketConditions()
        {
            // Placeholder implementation - would return actual market conditions
            return new Quantra.Models.MarketConditions();
        }
        
        /// <summary>
        /// Updates market condition indicators based on latest data
        /// </summary>
        private async Task UpdateMarketConditions()
        {
            try
            {
                // In a real implementation, this would query external services or APIs
                // Here we're just logging the call
                DatabaseMonolith.Log("Info", "Market conditions updated");
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to update market conditions", ex.ToString());
            }
        }
        
        /// <summary>
        /// Schedules an order as part of a dollar-cost averaging strategy
        /// </summary>
        /// <param name="symbol">Symbol to trade</param>
        /// <param name="quantity">Number of shares</param>
        /// <param name="executionTime">When to execute the order</param>
        /// <param name="strategyId">ID of the DCA strategy</param>
        /// <returns>True if scheduled successfully, false otherwise</returns>
        private bool ScheduleDollarCostAveragingOrder(string symbol, int quantity, DateTime executionTime, string strategyId)
        {
            try
            {
                // Create a scheduled order
                var order = new ScheduledOrder
                {
                    Symbol = symbol,
                    Quantity = quantity,
                    OrderType = "BUY", // DCA is typically for buying
                    ExecutionTime = executionTime,
                    IsDollarCostAveraging = true
                };
                
                // Add to scheduled orders
                if (!_scheduledOrders.ContainsKey(symbol))
                {
                    _scheduledOrders[symbol] = new List<ScheduledOrder>();
                }
                
                _scheduledOrders[symbol].Add(order);
                
                DatabaseMonolith.Log("Info", $"Scheduled DCA order for {quantity} shares of {symbol} at {executionTime:g}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to schedule DCA order for {symbol}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Cancels all remaining orders in a DCA strategy
        /// </summary>
        /// <param name="strategyId">ID of the DCA strategy to cancel</param>
        /// <returns>Number of orders canceled</returns>
        public int CancelDollarCostAveraging(string strategyId)
        {
            try
            {
                if (!_dcaStrategies.TryGetValue(strategyId, out var strategy))
                {
                    return 0;
                }
                
                // Remove the strategy
                _dcaStrategies.Remove(strategyId);
                
                // Find and remove all scheduled orders for this strategy
                int removedCount = 0;
                if (_scheduledOrders.TryGetValue(strategy.Symbol, out var orders))
                {
                    // Find orders that are part of this DCA strategy
                    var dcaOrders = orders.Where(o => o.IsDollarCostAveraging).ToList();
                    foreach (var order in dcaOrders)
                    {
                        orders.Remove(order);
                        removedCount++;
                    }
                    
                    // If no orders left for this symbol, remove the symbol key
                    if (orders.Count == 0)
                    {
                        _scheduledOrders.Remove(strategy.Symbol);
                    }
                }
                
                DatabaseMonolith.Log("Info", $"Canceled DCA strategy for {strategy.Symbol}, removed {removedCount} scheduled orders");
                return removedCount;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to cancel DCA strategy {strategyId}", ex.ToString());
                return 0;
            }
        }
        /// <summary>
        /// Rebalances portfolio according to a specific rebalancing profile
        /// </summary>
        /// <param name="profileId">ID of the rebalancing profile to use</param>
        /// <returns>True if rebalancing was initiated, false otherwise</returns>
        public async Task<bool> RebalancePortfolioWithProfile(string profileId)
        {
            // Find the specified profile
            var profile = _rebalancingProfiles.FirstOrDefault(p => p.Value.ProfileId == profileId).Value;
            if (profile == null)
            {
                DatabaseMonolith.Log("Warning", $"Rebalancing profile not found: {profileId}");
                return false;
            }
            
            return await RebalancePortfolioWithProfile(profile);
        }
        

        /// <summary>
        /// Gets trailing stop information for a symbol
        /// </summary>
        /// <param name="symbol">Symbol to get trailing stop for</param>
        /// <returns>TrailingStopInfo or null if not found</returns>
        public TrailingStopInfo GetTrailingStopInfo(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return null;
                
            if (_trailingStops.TryGetValue(symbol, out var stopInfo))
                return stopInfo;
                
            return null;
        }
        
        /// <summary>
        /// Set time-based exit for a symbol
        /// </summary>
        /// <param name="symbol">Symbol to set exit for</param>
        /// <param name="exitTime">Time to exit the position</param>
        /// <returns>True if successful, false otherwise</returns>
        public bool SetTimeBasedExit(string symbol, DateTime exitTime)
        {
            try
            {
                if (string.IsNullOrEmpty(symbol) || exitTime <= DateTime.Now)
                    return false;
                
                _timeBasedExits[symbol] = exitTime;
                _timeBasedExitStrategies[symbol] = new TimeBasedExit 
                { 
                    ExitTime = exitTime,
                    Strategy = TimeBasedExitStrategy.Custom
                };
                
                DatabaseMonolith.Log("Info", $"Time-based exit set for {symbol} at {exitTime:g}");
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to set time-based exit for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Provides an easily testable method to check if a trailing stop has been triggered
        /// </summary>
        /// <param name="symbol">The symbol to check</param>
        /// <param name="currentPrice">The current market price</param>
        /// <returns>True if the stop is triggered, false otherwise</returns>
        public bool IsTrailingStopTriggered(string symbol, double currentPrice)
        {
            var stopInfo = GetTrailingStopInfo(symbol);
            if (stopInfo == null)
                return false;
                
            return currentPrice <= stopInfo.CurrentStopPrice;
        }



        public TechnicalIndicatorService GetTechnicalIndicatorService()
        {
            return technicalIndicatorService;
        }

        // Add this new public method to the WebullTradingBot class to make PlaceLimitOrder accessible from the OrdersPage
        public async Task PlaceLimitOrder(string symbol, int quantity, string orderType, double limitPrice)
        {
            // Check if emergency stop is active
            if (_emergencyStopActive)
            {
                DatabaseMonolith.Log("Warning", $"Order rejected: Emergency stop is active. {orderType} {quantity} {symbol} @ {limitPrice:C2}");
                throw new InvalidOperationException("Cannot place order: Emergency stop is active");
            }
            
            // Check if trading is allowed based on market session and time restrictions
            if (!IsTradingAllowed())
            {
                DatabaseMonolith.Log("Warning", $"Order rejected: Trading not allowed at this time based on market session filters. {orderType} {quantity} {symbol} @ {limitPrice:C2}");
                throw new InvalidOperationException("Cannot place order: Trading not allowed at this time");
            }
            
            if (tradingMode == TradingMode.Paper)
            {
                // Simulate in local paper portfolio
                if (!paperPortfolio.ContainsKey(symbol))
                    paperPortfolio[symbol] = 0;
                if (orderType == "BUY")
                {
                    paperPortfolio[symbol] += quantity;
                }
                else if (orderType == "SELL" && paperPortfolio.ContainsKey(symbol))
                {
                    paperPortfolio[symbol] -= quantity;
                    if (paperPortfolio[symbol] <= 0)
                        paperPortfolio.Remove(symbol);
                }

                // --- Place paper trade in Webull paper account ---
                try
                {
                    // NOTE: This is a placeholder endpoint and payload for Webull paper trading.
                    // Replace with actual Webull paper trading API endpoint and authentication as needed.
                    var paperOrderData = new
                    {
                        symbol = symbol,
                        qty = quantity,
                        side = orderType.ToUpper(), // "BUY" or "SELL"
                        type = "LMT",
                        time_in_force = "GTC",
                        limit_price = limitPrice
                    };
                    var content = new StringContent(JsonConvert.SerializeObject(paperOrderData), Encoding.UTF8, "application/json");

                    // Example endpoint for Webull paper trading (replace with actual endpoint)
                    string webullPaperEndpoint = "https://paper-api.webull.com/api/trade/order";
                    // Add authentication headers if required by Webull API
                    // client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);

                    var response = await client.PostAsync(webullPaperEndpoint, content);
                    if (!response.IsSuccessStatusCode)
                    {
                        var resp = await response.Content.ReadAsStringAsync();
                        DatabaseMonolith.Log("Error", $"Webull paper trade failed: {response.StatusCode} {resp}");
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", "Exception placing paper trade in Webull account", ex.ToString());
                }
            }
            else
            {
                var orderData = new { symbol, quantity, orderType, limitPrice };
                var content = new StringContent(JsonConvert.SerializeObject(orderData), Encoding.UTF8, "application/json");
                var response = await client.PostAsync("https://api.webull.com/api/trade/order", content);
            }
        }

        public async Task<double> GetRSI(string symbol, string timeframe)
        {
            try
            {
                // Map timeframe to number of data points (approximate)
                int period = 14;
                int dataPoints = 50; // enough for RSI calculation

                // Optionally adjust dataPoints based on timeframe
                switch (timeframe)
                {
                    case "1min":
                    case "5min":
                    case "15min":
                        dataPoints = 100;
                        break;
                    case "1d":
                    case "1day":
                        dataPoints = 50;
                        break;
                    case "1wk":
                    case "1week":
                        dataPoints = 30;
                        break;
                    default:
                        dataPoints = 50;
                        break;
                }

                // Fetch historical closing prices using CEF/Yahoo Finance
                var closingPrices = await alphaVantageService.GetHistoricalClosingPricesAsync(symbol, dataPoints);

                if (closingPrices == null || closingPrices.Count < period + 1)
                {
                    Console.WriteLine($"Not enough data to calculate RSI for {symbol}.");
                    return 50.0; // Default value
                }

                // Calculate RSI using the most recent period
                var rsiValues = CalculateRSI(closingPrices, period);
                // Get the last valid RSI value
                var lastRsi = rsiValues.LastOrDefault(r => !double.IsNaN(r));
                return double.IsNaN(lastRsi) ? 50.0 : lastRsi;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching RSI for {symbol} via Yahoo Finance: {ex.Message}");
                return 50.0; // Default value on error
            }
        }

        // Add this public method to make Bollinger Bands accessible to the UI
        public async Task<(List<double> upperBand, List<double> middleBand, List<double> lowerBand)> GetBollingerBands(string symbol, int period, double stdDevMultiplier)
        {
            // Fetch historical price data (e.g., for "5min" or "1d" as appropriate)
            var stockData = await FetchChartData(symbol, "5min");
            if (stockData?.Prices == null || stockData.Prices.Count < period)
                return (new List<double>(), new List<double>(), new List<double>());

            // Use the existing CalculateBollingerBands method
            return CalculateBollingerBands(stockData.Prices, period, stdDevMultiplier);
        }
    }
}