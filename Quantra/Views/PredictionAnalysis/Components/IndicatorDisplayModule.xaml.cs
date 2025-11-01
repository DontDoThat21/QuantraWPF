using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Controls;
using System.Windows.Threading;
using MaterialDesignThemes.Wpf;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using Quantra.Views.PredictionAnalysis.Components;
//using System.Data.SQLite;
using Quantra.Controls;  // Added for AlertsControl
using Quantra.CrossCutting.Monitoring;
using Quantra.DAL.Services;
using Quantra.Utilities;

namespace Quantra.Controls.Components
{
    /// <summary>
    /// Interaction logic for IndicatorDisplayModule.xaml
    /// </summary>
    public partial class IndicatorDisplayModule : UserControl, INotifyPropertyChanged, IDisposable
    {
        private Dictionary<string, double> _indicators;
        private const double DefaultValue = 0.0;
        private readonly object _indicatorLock = new object();
        private CancellationTokenSource _cancellationTokenSource;
        private readonly int _maxRetryAttempts = 3;
        private readonly TimeSpan _retryDelay = TimeSpan.FromSeconds(2);
        private readonly WebullTradingBot _tradingBot;
        private readonly Dictionary<string, DateTime> _lastTradeTime;
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly INotificationService _notificationService;
        private readonly IEmailService _emailService;
        private readonly DatabaseSettingsProfile _settingsProfile;
        private readonly IMonitoringManager _monitoringManager;
        private bool _disposed;

        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(e.Property.Name));
        }

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public Dictionary<string, double> Indicators
        {
            get
            {
                lock (_indicatorLock)
                {
                    return _indicators;
                }
            }
            private set
            {
                lock (_indicatorLock)
                {
                    _indicators = value;
                }
                OnPropertyChanged(nameof(Indicators));
                if (!_disposed)
                {
                    // Ensure UI updates happen on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        // Already on UI thread, update directly
                        UpdateIndicatorDisplay();
                        
                        // Check for auto trading conditions when indicators update
                        if (IsAutoTradingEnabled && SelectedTradingRule != null)
                        {
                            _ = CheckAndExecuteTradingConditions();
                        }
                    }
                    else
                    {
                        // Not on UI thread, dispatch the UI updates
                        _ = Dispatcher.InvokeAsync(() =>
                        {
                            UpdateIndicatorDisplay();
                            
                            // Check for auto trading conditions when indicators update
                            if (IsAutoTradingEnabled && SelectedTradingRule != null)
                            {
                                _ = CheckAndExecuteTradingConditions();
                            }
                        });
                    }
                }
            }
        }

        private string _symbol;
        public string Symbol
        {
            get => _symbol;
            set
            {
                if (_symbol != value)
                {
                    _symbol = value;
                    OnSymbolChanged();
                }
            }
        }

        private string _timeframe;
        public string Timeframe
        {
            get => _timeframe;
            set
            {
                if (_timeframe != value)
                {
                    _timeframe = value;
                    OnTimeframeChanged();
                }
            }
        }

        private bool _isAutoTradingEnabled;
        public bool IsAutoTradingEnabled
        {
            get => _isAutoTradingEnabled;
            set
            {
                if (_isAutoTradingEnabled != value)
                {
                    _isAutoTradingEnabled = value;
                    OnAutoTradingChanged();
                    OnPropertyChanged(nameof(IsAutoTradingEnabled));
                }
            }
        }

        private TradingRule _selectedTradingRule;
        public TradingRule SelectedTradingRule
        {
            get => _selectedTradingRule;
            set
            {
                if (_selectedTradingRule != value)
                {
                    _selectedTradingRule = value;
                    OnPropertyChanged(nameof(SelectedTradingRule));
                    ValidateAutoTrading();
                }
            }
        }

        private ObservableCollection<TradingRule> _availableTradingRules;
        public ObservableCollection<TradingRule> AvailableTradingRules
        {
            get => _availableTradingRules;
            private set
            {
                if (_availableTradingRules != value)
                {
                    _availableTradingRules = value;
                    OnPropertyChanged(nameof(AvailableTradingRules));
                }
            }
        }

        public IndicatorDisplayModule(
            ISettingsService settingsService,
            ITechnicalIndicatorService indicatorService,
            INotificationService notificationService,
            IEmailService emailService)
        {
            InitializeComponent();
            DataContext = this;

            _indicatorService = indicatorService;
            _notificationService = notificationService;
            _emailService = emailService;
            _settingsProfile = settingsService.GetDefaultSettingsProfile();
            _monitoringManager = MonitoringManager.Instance;
            _tradingBot = new WebullTradingBot();
            _cancellationTokenSource = new CancellationTokenSource();
            _lastTradeTime = new Dictionary<string, DateTime>();
            
            Indicators = new Dictionary<string, double>();
            AvailableTradingRules = new ObservableCollection<TradingRule>();
            
            LoadTradingRules();
        }

        public void LoadTradingRules()
        {
            try
            {
                if (string.IsNullOrEmpty(Symbol))
                {
                    // Ensure notification is shown on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        _notificationService.ShowInfo("No symbol selected for trading rules");
                    }
                    else
                    {
                        _ = Dispatcher.InvokeAsync(() => _notificationService.ShowInfo("No symbol selected for trading rules"));
                    }
                    return;
                }

                // Use ExecuteQuery to get trading rules
                using (var connection = ConnectionHelper.GetConnection())
                {
                    connection.Open();
                    string query = @"
                        SELECT * FROM TradingRules 
                        WHERE Symbol = @Symbol AND IsActive = 1 
                        ORDER BY Name";

                    var rules = new List<TradingRule>();
                    using (var cmd = new SQLiteCommand(query, connection))
                    {
                        cmd.Parameters.AddWithValue("@Symbol", Symbol);
                        using (var reader = cmd.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                var rule = new TradingRule
                                {
                                    Id = reader.GetInt32(0),
                                    Name = reader.GetString(1),
                                    Symbol = reader.GetString(2),
                                    OrderType = reader.GetString(3),
                                    MinConfidence = reader.GetDouble(4),
                                    EntryPrice = reader.GetDouble(5),
                                    ExitPrice = reader.GetDouble(6),
                                    StopLoss = reader.GetDouble(7),
                                    Quantity = reader.GetInt32(8),
                                    IsActive = true,
                                    CreatedDate = reader.GetDateTime(9)
                                };

                                if (IsValidTradingRule(rule))
                                {
                                    rules.Add(rule);
                                }
                                else
                                {
                                    //DatabaseMonolith.Log("Warning", $"Invalid trading rule skipped: {rule.Name}", 
                                        //$"Symbol: {rule.Symbol}, OrderType: {rule.OrderType}");
                                }
                            }
                        }
                    }

                    // Ensure UI updates happen on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        UpdateTradingRulesUI(rules);
                    }
                    else
                    {
                        _ = Dispatcher.InvokeAsync(() => UpdateTradingRulesUI(rules));
                    }

                    ValidateAutoTrading();
                }
            }
            catch (Exception ex)
            {
                // Ensure error notification is shown on UI thread
                if (Dispatcher.CheckAccess())
                {
                    _notificationService.ShowError($"Error loading trading rules: {ex.Message}");
                }
                else
                {
                    _ = Dispatcher.InvokeAsync(() => _notificationService.ShowError($"Error loading trading rules: {ex.Message}"));
                }
                //DatabaseMonolith.Log("Error", "Failed to load trading rules", ex.ToString());
            }
        }

        private void UpdateTradingRulesUI(List<TradingRule> rules)
        {
            // This method should only be called from UI thread
            AvailableTradingRules.Clear();

            if (rules.Count == 0)
            {
                _notificationService.ShowInfo($"No valid trading rules found for {Symbol}");
            }
            else
            {
                foreach (var rule in rules)
                {
                    AvailableTradingRules.Add(rule);
                }

                _notificationService.ShowInfo($"Loaded {AvailableTradingRules.Count} valid trading rules");
                
                if (SelectedTradingRule != null)
                {
                    var previousRule = AvailableTradingRules.FirstOrDefault(r => r.Id == SelectedTradingRule.Id);
                    if (previousRule != null)
                    {
                        SelectedTradingRule = previousRule;
                        _notificationService.ShowInfo($"Restored previous trading rule: {previousRule.Name}");
                    }
                    else
                    {
                        SelectedTradingRule = null;
                    }
                }
                
                if (SelectedTradingRule == null && AvailableTradingRules.Count > 0)
                {
                    SelectedTradingRule = AvailableTradingRules[0];
                    _notificationService.ShowInfo($"Selected default trading rule: {SelectedTradingRule.Name}");
                }
            }
        }

        protected virtual void OnSymbolChanged()
        {
            _ = RefreshIndicatorsWithRetry();
            LoadTradingRules();
        }

        protected virtual void OnTimeframeChanged()
        {
            _ = RefreshIndicatorsWithRetry();
        }

        protected virtual void OnAutoTradingChanged()
        {
            if (IsAutoTradingEnabled)
            {
                if (SelectedTradingRule == null)
                {
                    // Ensure notification is shown on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        _notificationService.ShowWarning("Please select a trading rule before enabling auto-trading");
                    }
                    else
                    {
                        _ = Dispatcher.InvokeAsync(() => _notificationService.ShowWarning("Please select a trading rule before enabling auto-trading"));
                    }
                    IsAutoTradingEnabled = false;
                    return;
                }

                // Ensure notification is shown on UI thread
                if (Dispatcher.CheckAccess())
                {
                    _notificationService.ShowInfo($"Auto-trading enabled with rule: {SelectedTradingRule.Name}");
                }
                else
                {
                    _ = Dispatcher.InvokeAsync(() => _notificationService.ShowInfo($"Auto-trading enabled with rule: {SelectedTradingRule.Name}"));
                }
                _ = CheckAndExecuteTradingConditions();
            }
            else
            {
                // Ensure notification is shown on UI thread
                if (Dispatcher.CheckAccess())
                {
                    _notificationService.ShowInfo("Auto-trading disabled");
                }
                else
                {
                    _ = Dispatcher.InvokeAsync(() => _notificationService.ShowInfo("Auto-trading disabled"));
                }
            }
        }

        private async Task RefreshIndicatorsWithRetry()
        {
            int attempts = 0;
            bool success = false;

            while (!success && attempts < _maxRetryAttempts && !_disposed)
            {
                try
                {
                    attempts++;
                    await RefreshIndicators();
                    success = true;
                }
                catch (Exception) when (attempts < _maxRetryAttempts)
                {
                    await Task.Delay(_retryDelay);
                }
            }
        }

        private async Task RefreshIndicators()
        {
            await _monitoringManager.RecordExecutionTimeAsync($"RefreshIndicators_{Symbol}_{Timeframe}", async () =>
            {
                if (string.IsNullOrEmpty(Symbol) || _disposed)
                    return;

                // Cancel any existing operations
                _cancellationTokenSource?.Cancel();
                _cancellationTokenSource = new CancellationTokenSource();
                var token = _cancellationTokenSource.Token;

                try
                {
                    // Set loading state on UI thread with timing
                    await _monitoringManager.RecordExecutionTimeAsync($"UIUpdate_SetLoading_{Symbol}", async () =>
                    {
                        await Dispatcher.InvokeAsync(() => IsLoading = true);
                    });

                    // Load indicators directly with async pattern
                    token.ThrowIfCancellationRequested();
                    var indicatorSet = await _indicatorService.GetIndicatorsForPrediction(Symbol, Timeframe);
                    token.ThrowIfCancellationRequested();
                    var algorithmicSignals = await _indicatorService.GetAlgorithmicTradingSignals(Symbol);
                    token.ThrowIfCancellationRequested();

                    foreach (var signal in algorithmicSignals)
                    {
                        if (!indicatorSet.ContainsKey(signal.Key))
                        {
                            indicatorSet[signal.Key] = signal.Value;
                        }
                    }

                    // Update indicators on UI thread with timing
                    if (!_disposed)
                    {
                        await _monitoringManager.RecordExecutionTimeAsync($"UIUpdate_SetIndicators_{Symbol}", async () =>
                        {
                            await Dispatcher.InvokeAsync(() => Indicators = indicatorSet);
                        });
                    }
                }
                catch (OperationCanceledException)
                {
                    // Operation was cancelled, ignore
                }
                catch (Exception ex)
                {
                    if (!_disposed)
                    {
                        // Show notification on UI thread with timing
                        await _monitoringManager.RecordExecutionTimeAsync($"UIUpdate_ShowError_{Symbol}", async () =>
                        {
                            await Dispatcher.InvokeAsync(() =>
                            {
                                _notificationService.ShowNotification(
                                    $"Error loading indicators: {ex.Message}",
                                    PackIconKind.Error,
                                    Colors.Red);
                            });
                        });
                    }
                }
                finally
                {
                    if (!token.IsCancellationRequested && !_disposed)
                    {
                        // Reset loading state on UI thread with timing
                        await _monitoringManager.RecordExecutionTimeAsync($"UIUpdate_ClearLoading_{Symbol}", async () =>
                        {
                            await Dispatcher.InvokeAsync(() => IsLoading = false);
                        });
                    }
                }
            });
        }

        private bool _isLoading;
        public bool IsLoading
        {
            get => _isLoading;
            private set
            {
                if (_isLoading != value)
                {
                    _isLoading = value;
                    OnPropertyChanged(nameof(IsLoading));
                }
            }
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _cancellationTokenSource?.Cancel();
                    _cancellationTokenSource?.Dispose();
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~IndicatorDisplayModule()
        {
            Dispose(false);
        }

        private void UpdateIndicatorDisplay()
        {
            if (Indicators == null)
                return;

            Dictionary<string, double> currentIndicators;
            lock (_indicatorLock)
            {
                currentIndicators = new Dictionary<string, double>(_indicators);
            }

            // Trend Indicators
            UpdateIndicator("ADX", "adxValue", GetIndicatorColor("ADX"));
            UpdateIndicator("VWAP", "vwapValue", GetIndicatorColor("VWAP"));
            UpdateIndicator("ROC", "rocValue", GetIndicatorColor("ROC"));

            // Momentum Indicators
            UpdateIndicator("RSI", "rsiValue", GetIndicatorColor("RSI"));
            UpdateIndicator("CCI", "cciValue", GetIndicatorColor("CCI"));
            UpdateIndicator("StochRSI", "stochRsiValue", GetIndicatorColor("StochRSI"));
            UpdateIndicator("WilliamsR", "williamsRValue", GetIndicatorColor("WilliamsR"));

            // Volatility Indicators
            UpdateIndicator("ATR", "atrValue", GetIndicatorColor("ATR"));
            UpdateIndicator("BullPower", "bullPowerValue", GetIndicatorColor("BullPower"));
            UpdateIndicator("BearPower", "bearPowerValue", GetIndicatorColor("BearPower"));

            // MACD Components
            if (currentIndicators.TryGetValue("MACD", out double macd) && 
                currentIndicators.TryGetValue("MACDSignal", out double signal))
            {
                UpdateIndicator("MACD", "macdValue", GetIndicatorColor("MACD"));
                UpdateIndicator("MACDSignal", "macdSignalValue", GetIndicatorColor("MACDSignal"));
                UpdateIndicator("MACDHistogram", "macdHistValue", GetIndicatorColor("MACDHistogram"));
            }

            // Stochastic Oscillator
            if (currentIndicators.TryGetValue("StochK", out double k) && 
                currentIndicators.TryGetValue("StochD", out double d))
            {
                UpdateIndicator("StochK", "stochKValue", GetIndicatorColor("StochK"));
                UpdateIndicator("StochD", "stochDValue", GetIndicatorColor("StochD"));
            }
        }

        private void UpdateIndicator(string indicatorName, string controlName, Brush colorBrush)
        {
            Dictionary<string, double> currentIndicators;
            lock (_indicatorLock)
            {
                currentIndicators = new Dictionary<string, double>(_indicators);
            }

            if (currentIndicators.TryGetValue(indicatorName, out double value))
            {
                if (this.FindName(controlName) is TextBlock textBlock)
                {
                    textBlock.Text = FormatIndicatorValue(indicatorName, value);
                    textBlock.Foreground = colorBrush;
                }
            }
        }

        private string FormatIndicatorValue(string indicatorName, double value)
        {
            return indicatorName switch
            {
                "RSI" or "StochRSI" or "StochK" or "StochD" => $"{value:F1}",
                "MACD" or "MACDSignal" or "MACDHistogram" => $"{value:F3}",
                "ADX" => $"{value:F1}",
                "ATR" => $"${value:F2}",
                "VWAP" => $"${value:F2}",
                "ROC" => $"{value:F1}%",
                "BullPower" or "BearPower" => $"${value:F2}",
                "MomentumScore" => $"{value:F0}",
                "TradingSignal" => $"{value:F0}",
                "BuySellSignal" => FormatBuySellSignal(value),
                _ => $"{value:F2}"
            };
        }

        private string FormatBuySellSignal(double value)
        {
            return value switch
            {
                1 => "Strong Buy",
                0.5 => "Buy",
                0 => "Neutral",
                -0.5 => "Sell",
                -1 => "Strong Sell",
                _ => "Neutral"
            };
        }

        private Brush GetIndicatorColor(string indicatorName)
        {
            Dictionary<string, double> currentIndicators;
            lock (_indicatorLock)
            {
                currentIndicators = new Dictionary<string, double>(_indicators);
            }

            if (!currentIndicators.TryGetValue(indicatorName, out double value))
                return Brushes.Gray;

            return indicatorName switch
            {
                "RSI" => GetRSIColor(value),
                "StochRSI" or "StochK" or "StochD" => GetStochasticColor(value),
                "MACD" or "MACDHistogram" => GetMACDColor(value),
                "ADX" => GetADXColor(value),
                "CCI" => GetCCIColor(value),
                "WilliamsR" => GetWilliamsRColor(value),
                "ROC" => GetROCColor(value),
                "BullPower" => value > 0 ? Brushes.Green : Brushes.Red,
                "BearPower" => value < 0 ? Brushes.Green : Brushes.Red,
                "MomentumScore" => GetMomentumColor(value),
                "TradingSignal" => GetTradingSignalColor(value),
                "BuySellSignal" => GetBuySellSignalColor(value),
                _ => Brushes.Gray
            };
        }

        private Brush GetRSIColor(double value)
        {
            // Special case: RSI = 0 should display as white
            if (value == 0)
                return Brushes.White;
                
            return value >= 70 ? Brushes.Red :
                   value <= 30 ? Brushes.Green :
                   Brushes.Gray;
        }

        private Brush GetStochasticColor(double value) =>
            value >= 80 ? Brushes.Red :
            value <= 20 ? Brushes.Green :
            Brushes.Gray;

        private Brush GetMACDColor(double value) =>
            value > 0 ? Brushes.Green :
            value < 0 ? Brushes.Red :
            Brushes.Gray;

        private Brush GetADXColor(double value) =>
            value >= 25 ? Brushes.Green :
            Brushes.Gray;

        private Brush GetCCIColor(double value) =>
            value >= 100 ? Brushes.Red :
            value <= -100 ? Brushes.Green :
            Brushes.Gray;

        private Brush GetWilliamsRColor(double value) =>
            value <= -80 ? Brushes.Green :
            value >= -20 ? Brushes.Red :
            Brushes.Gray;

        private Brush GetROCColor(double value) =>
            value > 0 ? Brushes.Green :
            value < 0 ? Brushes.Red :
            Brushes.Gray;

        private Brush GetMomentumColor(double value) =>
            value >= 70 ? Brushes.DarkGreen :
            value >= 50 ? Brushes.Green :
            value <= 30 ? Brushes.DarkRed :
            value <= 50 ? Brushes.Red :
            Brushes.Gray;

        private Brush GetTradingSignalColor(double value) =>
            value >= 70 ? Brushes.DarkGreen :
            value >= 30 ? Brushes.Green :
            value <= -70 ? Brushes.DarkRed :
            value <= -30 ? Brushes.Red :
            Brushes.Gray;

        private Brush GetBuySellSignalColor(double value) =>
            value >= 0.5 ? Brushes.Green :
            value <= -0.5 ? Brushes.Red :
            Brushes.Gray;

        private bool IsValidTradingRule(TradingRule rule)
        {
            if (rule == null) return false;

            // Basic validation checks
            if (string.IsNullOrWhiteSpace(rule.Symbol) || 
                string.IsNullOrWhiteSpace(rule.OrderType) ||
                rule.Quantity <= 0 ||
                rule.MinConfidence <= 0 || 
                rule.MinConfidence > 1)
            {
                return false;
            }

            // Validate price levels based on order type
            if (rule.OrderType == "BUY")
            {
                // For buy orders: Entry > Stop Loss, Exit > Entry
                return rule.EntryPrice > rule.StopLoss && 
                       rule.ExitPrice > rule.EntryPrice;
            }
            else if (rule.OrderType == "SELL")
            {
                // For sell orders: Entry < Stop Loss, Exit < Entry
                return rule.EntryPrice < rule.StopLoss && 
                       rule.ExitPrice < rule.EntryPrice;
            }

            return false;
        }

        private void ValidateAutoTrading()
        {
            if (IsAutoTradingEnabled)
            {
                bool isValid = true;
                var validationErrors = new List<string>();

                // Check for selected trading rule
                if (SelectedTradingRule == null)
                {
                    isValid = false;
                    validationErrors.Add("No trading rule selected");
                }
                // Validate selected rule parameters
                else
                {
                    if (SelectedTradingRule.Quantity <= 0)
                    {
                        isValid = false;
                        validationErrors.Add("Invalid quantity in trading rule");
                    }

                    if (SelectedTradingRule.MinConfidence <= 0 || SelectedTradingRule.MinConfidence > 1)
                    {
                        isValid = false;
                        validationErrors.Add("Invalid confidence threshold in trading rule");
                    }

                    // Validate price levels based on order type
                    if (SelectedTradingRule.OrderType == "BUY")
                    {
                        if (SelectedTradingRule.EntryPrice <= SelectedTradingRule.StopLoss)
                        {
                            isValid = false;
                            validationErrors.Add("Buy rule: Entry price must be higher than stop loss");
                        }
                        if (SelectedTradingRule.ExitPrice <= SelectedTradingRule.EntryPrice)
                        {
                            isValid = false;
                            validationErrors.Add("Buy rule: Exit price must be higher than entry price");
                        }
                    }
                    else if (SelectedTradingRule.OrderType == "SELL")
                    {
                        if (SelectedTradingRule.EntryPrice >= SelectedTradingRule.StopLoss)
                        {
                            isValid = false;
                            validationErrors.Add("Sell rule: Entry price must be lower than stop loss");
                        }
                        if (SelectedTradingRule.ExitPrice >= SelectedTradingRule.EntryPrice)
                        {
                            isValid = false;
                            validationErrors.Add("Sell rule: Exit price must be lower than entry price");
                        }
                    }
                }

                // If validation fails, disable auto-trading and show errors
                if (!isValid)
                {
                    IsAutoTradingEnabled = false;
                    // Ensure notification is shown on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        _notificationService.ShowWarning($"Auto-trading disabled: {string.Join(", ", validationErrors)}");
                    }
                    else
                    {
                        _ = Dispatcher.InvokeAsync(() => _notificationService.ShowWarning($"Auto-trading disabled: {string.Join(", ", validationErrors)}"));
                    }
                    //DatabaseMonolith.Log("Warning", "Auto-trading validation failed", 
                        //$"Rule: {SelectedTradingRule?.Name}\nErrors: {string.Join("\n", validationErrors)}");
                }
                else
                {
                    // Log successful validation
                    //DatabaseMonolith.Log("Info", "Auto-trading validation passed", 
                        //$"Rule: {SelectedTradingRule.Name}\nSymbol: {Symbol}");
                }
            }
        }

        private async Task CheckAndExecuteTradingConditions()
        {
            if (!IsAutoTradingEnabled || SelectedTradingRule == null || Indicators == null || HasTradedRecently(Symbol))
                return;

            try
            {
                // Get current market data from indicators
                var currentPrice = Indicators.GetValueOrDefault("VWAP", 0);
                var tradingSignal = Indicators.GetValueOrDefault("TradingSignal", 0);
                var momentumScore = Indicators.GetValueOrDefault("MomentumScore", 0);
                var confidenceScore = momentumScore / 100.0;
                var rsi = Indicators.GetValueOrDefault("RSI", 50);
                var macdSignal = Indicators.GetValueOrDefault("MACDSignal", 0);
                var adx = Indicators.GetValueOrDefault("ADX", 0);

                // Check confidence threshold
                if (confidenceScore < SelectedTradingRule.MinConfidence)
                {
                    return;
                }

                bool shouldTrade = false;
                string action = "";

                // Enhanced buy conditions
                if (SelectedTradingRule.OrderType == "BUY")
                {
                    shouldTrade = tradingSignal > 0 &&                  // Positive trading signal
                        currentPrice <= SelectedTradingRule.EntryPrice && // Price below or at entry point
                        rsi < 70 &&                                     // Not overbought
                        macdSignal > 0 &&                              // MACD confirms uptrend 
                        adx > 25;                                      // Strong trend                        
                    action = "BUY";
                }
                // Enhanced sell conditions
                else if (SelectedTradingRule.OrderType == "SELL")
                {
                    shouldTrade = tradingSignal < 0 &&                   // Negative trading signal
                        currentPrice >= SelectedTradingRule.EntryPrice && // Price above or at entry point
                        rsi > 30 &&                                      // Not oversold
                        macdSignal < 0 &&                               // MACD confirms downtrend
                        adx > 25;                                       // Strong trend
                    action = "SELL";
                }

                if (shouldTrade)
                {
                    // Create prediction model for trade execution
                    var prediction = new PredictionModel
                    {
                        Symbol = Symbol,
                        CurrentPrice = currentPrice,
                        Confidence = confidenceScore,
                        PredictedAction = action,
                        TradingRule = SelectedTradingRule.Condition,
                        TargetPrice = SelectedTradingRule.ExitPrice,
                        Indicators = new Dictionary<string, double>(Indicators)
                    };

                    // Create alert before trade execution
                    var alert = new AlertModel
                    {
                        Symbol = Symbol,
                        Name = $"{action} Signal: {Symbol}",
                        Condition = $"{action} at ${currentPrice:F2}",
                        AlertType = "Trading Signal",
                        Category = AlertCategory.Opportunity,
                        Priority = confidenceScore > 0.8 ? 1 : 2,
                        CreatedDate = DateTime.Now,
                        IsActive = true,
                        Notes = $"Confidence: {confidenceScore:P0}\nRSI: {rsi:F1}\nADX: {adx:F1}\nMACD Signal: {macdSignal:F3}"
                    };

                    // Show UI notification on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        _notificationService.ShowNotification(
                            $"Trading signal detected for {Symbol}",
                            PackIconKind.TrendingUp,
                            action == "BUY" ? Colors.Green : Colors.OrangeRed
                        );
                    }
                    else
                    {
                        _ = Dispatcher.InvokeAsync(() => _notificationService.ShowNotification(
                            $"Trading signal detected for {Symbol}",
                            PackIconKind.TrendingUp,
                            action == "BUY" ? Colors.Green : Colors.OrangeRed
                        ));
                    }

                    // Send email notification if enabled
                    if (_settingsProfile.EnableEmailAlerts && _settingsProfile.EnableOpportunityAlertEmails)
                    {
                        try
                        {
                            string subject = $"Quantra: {action} Signal for {Symbol}";
                            string body = $"Trading Signal Details:\n" +
                                        $"Symbol: {Symbol}\n" +
                                        $"Action: {action}\n" +
                                        $"Current Price: ${currentPrice:F2}\n" +
                                        $"Entry Price: ${SelectedTradingRule.EntryPrice:F2}\n" +
                                        $"Target Price: ${SelectedTradingRule.ExitPrice:F2}\n" +
                                        $"Stop Loss: ${SelectedTradingRule.StopLoss:F2}\n" +
                                        $"Confidence: {confidenceScore:P0}\n\n" +
                                        $"Technical Indicators:\n" +
                                        $"RSI: {rsi:F1}\n" +
                                        $"ADX: {adx:F1}\n" +
                                        $"MACD Signal: {macdSignal:F3}\n" +
                                        $"Momentum Score: {momentumScore:F0}\n\n" +
                                        $"Trading Rule: {SelectedTradingRule.Name}\n" +
                                        $"Time: {DateTime.Now}\n";

                            await _emailService.SendEmailAsync(_settingsProfile.AlertEmail, subject, body);
                            //DatabaseMonolith.Log("Info", $"Trading signal email sent for {Symbol}", subject);
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Error", "Failed to send trading signal email", ex.ToString());
                        }
                    }

                    // Execute the trade
                    await ExecuteTrade(prediction);

                    // Log the alert (using AlertsControl.EmitGlobalAlert correctly)
                    Alerting.EmitGlobalAlert(alert);
                }
            }
            catch (Exception ex)
            {
                // Ensure error notification is shown on UI thread
                if (Dispatcher.CheckAccess())
                {
                    _notificationService.ShowError($"Error in auto-trading: {ex.Message}");
                }
                else
                {
                    _ = Dispatcher.InvokeAsync(() => _notificationService.ShowError($"Error in auto-trading: {ex.Message}"));
                }
                //DatabaseMonolith.Log("Error", "Auto-trading error", ex.ToString());
            }
        }

        private bool HasTradedRecently(string symbol)
        {
            if (_lastTradeTime.TryGetValue(symbol, out DateTime lastTradeTime))
            {
                TimeSpan timeSinceLastTrade = DateTime.Now - lastTradeTime;
                return timeSinceLastTrade.TotalHours < 1;
            }
            return false;
        }

        private async Task ExecuteTrade(PredictionModel prediction)
        {
            try
            {
                // Validate all required values are valid
                if (prediction.CurrentPrice <= 0 || 
                    SelectedTradingRule.Quantity <= 0 || 
                    SelectedTradingRule.StopLoss <= 0 || 
                    SelectedTradingRule.ExitPrice <= 0)
                {
                    // Ensure warning notification is shown on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        _notificationService.ShowWarning("Invalid trade parameters - check price and quantity values");
                    }
                    else
                    {
                        _ = Dispatcher.InvokeAsync(() => _notificationService.ShowWarning("Invalid trade parameters - check price and quantity values"));
                    }
                    return;
                }

                // Calculate risk parameters
                double potentialLoss = Math.Abs(prediction.CurrentPrice - SelectedTradingRule.StopLoss) * SelectedTradingRule.Quantity;
                double potentialGain = Math.Abs(SelectedTradingRule.ExitPrice - prediction.CurrentPrice) * SelectedTradingRule.Quantity;
                double riskRewardRatio = potentialGain / (potentialLoss > 0 ? potentialLoss : 1);

                // Validate risk-reward ratio (minimum 2:1)
                if (riskRewardRatio < 2.0)
                {
                    //DatabaseMonolith.Log("Warning", $"Trade rejected for {prediction.Symbol} - Risk/Reward ratio too low: {riskRewardRatio:F2}");
                    return;
                }

                // Execute trade via trading bot
                await _tradingBot.PlaceLimitOrder(
                    prediction.Symbol,
                    SelectedTradingRule.Quantity,
                    prediction.PredictedAction,
                    prediction.CurrentPrice
                );

                // Record trade time
                _lastTradeTime[prediction.Symbol] = DateTime.Now;

                // Create and save order record
                var order = new OrderModel
                {
                    Symbol = prediction.Symbol,
                    OrderType = prediction.PredictedAction,
                    Quantity = SelectedTradingRule.Quantity,
                    Price = prediction.CurrentPrice,
                    StopLoss = SelectedTradingRule.StopLoss,
                    TakeProfit = SelectedTradingRule.ExitPrice,
                    IsPaperTrade = true,
                    PredictionSource = $"Auto: {prediction.Symbol} ({prediction.Confidence:P0}) - Rule: {SelectedTradingRule.Name}",
                    Status = "Executed",
                    Timestamp = DateTime.Now
                };

                DatabaseMonolith.AddOrderToHistory(order);

                // Show detailed notification on UI thread
                if (Dispatcher.CheckAccess())
                {
                    _notificationService.ShowSuccess(
                        $"Auto-trade executed: {prediction.PredictedAction} {SelectedTradingRule.Quantity} {prediction.Symbol} @ {prediction.CurrentPrice:C}\n" +
                        $"Stop Loss: {SelectedTradingRule.StopLoss:C} | Take Profit: {SelectedTradingRule.ExitPrice:C}\n" +
                        $"R/R Ratio: {riskRewardRatio:F2}"
                    );
                }
                else
                {
                    _ = Dispatcher.InvokeAsync(() => _notificationService.ShowSuccess(
                        $"Auto-trade executed: {prediction.PredictedAction} {SelectedTradingRule.Quantity} {prediction.Symbol} @ {prediction.CurrentPrice:C}\n" +
                        $"Stop Loss: {SelectedTradingRule.StopLoss:C} | Take Profit: {SelectedTradingRule.ExitPrice:C}\n" +
                        $"R/R Ratio: {riskRewardRatio:F2}"
                    ));
                }

                // Send trade execution email if enabled
                if (_settingsProfile.EnableEmailAlerts && _settingsProfile.EnableTradeNotifications)
                {
                    try
                    {
                        string subject = $"Quantra: Trade Executed - {prediction.PredictedAction} {prediction.Symbol}";
                        string body = $"Trade Execution Details:\n" +
                                    $"Symbol: {prediction.Symbol}\n" +
                                    $"Action: {prediction.PredictedAction}\n" +
                                    $"Quantity: {SelectedTradingRule.Quantity}\n" +
                                    $"Price: ${prediction.CurrentPrice:F2}\n" +
                                    $"Stop Loss: ${SelectedTradingRule.StopLoss:F2}\n" +
                                    $"Take Profit: ${SelectedTradingRule.ExitPrice:F2}\n" +
                                    $"Risk/Reward Ratio: {riskRewardRatio:F2}\n" +
                                    $"Confidence: {prediction.Confidence:P0}\n\n" +
                                    $"Position Value: ${prediction.CurrentPrice * SelectedTradingRule.Quantity:F2}\n" +
                                    $"Max Risk: ${potentialLoss:F2}\n" +
                                    $"Potential Gain: ${potentialGain:F2}\n\n" +
                                    $"Trading Rule: {SelectedTradingRule.Name}\n" +
                                    $"Execution Time: {DateTime.Now}\n";

                        await _emailService.SendEmailAsync(_settingsProfile.AlertEmail, subject, body);
                        //DatabaseMonolith.Log("Info", $"Trade execution email sent for {prediction.Symbol}", subject);
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", "Failed to send trade execution email", ex.ToString());
                    }
                }

                // Create and emit a trade execution alert
                var alert = new AlertModel
                {
                    Symbol = prediction.Symbol,
                    Name = $"Trade Executed: {prediction.Symbol}",
                    Condition = $"{prediction.PredictedAction} {SelectedTradingRule.Quantity} @ ${prediction.CurrentPrice:F2}",
                    AlertType = "Trade Execution",
                    Category = AlertCategory.Opportunity,  // Changed from Trade to Opportunity since that's a valid value
                    Priority = 1,
                    CreatedDate = DateTime.Now,
                    IsActive = true,
                    Notes = $"Rule: {SelectedTradingRule.Name}\nConfidence: {prediction.Confidence:P0}"
                };

                // Emit the global alert using AlertsControl instead of AlertManager
                Alerting.EmitGlobalAlert(alert);

                // Log successful trade
                //DatabaseMonolith.Log("Info", $"Trade executed for {prediction.Symbol}", alert.Notes);
            }
            catch (Exception ex)
            {
                // Ensure error notification is shown on UI thread
                if (Dispatcher.CheckAccess())
                {
                    _notificationService.ShowError($"Error in auto-trading: {ex.Message}");
                }
                else
                {
                    _ = Dispatcher.InvokeAsync(() => _notificationService.ShowError($"Error in auto-trading: {ex.Message}"));
                }
                //DatabaseMonolith.Log("Error", "Auto-trading error", ex.ToString());
            }
        }

        public void UpdateIndicatorValues(Dictionary<string, double> indicators)
        {
            if (indicators == null)
                return;

            lock (_indicatorLock)
            {
                _indicators = new Dictionary<string, double>(indicators);
            }
            
            // Ensure UI updates happen on UI thread
            if (Dispatcher.CheckAccess())
            {
                UpdateIndicatorDisplay();
            }
            else
            {
                _ = Dispatcher.InvokeAsync(() => UpdateIndicatorDisplay());
            }
        }
    }
}