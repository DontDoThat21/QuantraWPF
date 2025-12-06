using LiveCharts;
using LiveCharts.Defaults;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows.Input;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;
using System.Linq;
using System.Windows.Data;
using System.Collections.Generic;
using Quantra.Commands;
using Quantra.Repositories;
using Quantra.DAL.Data;
using Microsoft.EntityFrameworkCore;

namespace Quantra.ViewModels
{
    public class PredictionAnalysisViewModel : INotifyPropertyChanged
    {
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly PredictionAnalysisRepository _analysisRepository;
        private readonly IAlphaVantageService _alphaVantageService;
        private readonly ISettingsService _settingsService;
        private readonly IEmailService _emailService;
        private readonly ITradingRuleService _tradingRuleService;
        private readonly PredictionAnalysisService _predictionAnalysisService;
        private double _currentPrice;
        private string _symbol;
        private object _selectedStrategyProfile;

        // Make trading service public
        public ITradingService TradingService { get; }

        // Chart data
        public ChartValues<ObservablePoint> PriceValues { get; set; } = new();
        public ChartValues<ObservablePoint> VwapValues { get; set; } = new();
        public ChartValues<ObservablePoint> PredictionValues { get; set; } = new();

        // Multi-horizon prediction data for TFT visualization
        public ChartValues<ObservablePoint> HistoricalPrices { get; set; } = new();
        public ChartValues<ObservablePoint> PredictedPrices { get; set; } = new();
        public ChartValues<ObservablePoint> UpperBandPrices { get; set; } = new();
        public ChartValues<ObservablePoint> LowerBandPrices { get; set; } = new();
        public List<string> DateLabels { get; set; } = new();
        public Func<double, string> PriceFormatter { get; set; } = value => value.ToString("C2");

        // Temporal attention weights for heatmap
        public ChartValues<double> AttentionWeights { get; set; } = new();
        public List<string> AttentionLabels { get; set; } = new();

        // Feature importance data
        public ChartValues<double> FeatureImportances { get; set; } = new();
        public List<string> FeatureNames { get; set; } = new();

        // Prediction horizon selections
        private bool _horizon1DayChecked = true;
        public bool Horizon1DayChecked
        {
            get => _horizon1DayChecked;
            set { _horizon1DayChecked = value; OnPropertyChanged(nameof(Horizon1DayChecked)); }
        }

        private bool _horizon3DayChecked = true;
        public bool Horizon3DayChecked
        {
            get => _horizon3DayChecked;
            set { _horizon3DayChecked = value; OnPropertyChanged(nameof(Horizon3DayChecked)); }
        }

        private bool _horizon5DayChecked = true;
        public bool Horizon5DayChecked
        {
            get => _horizon5DayChecked;
            set { _horizon5DayChecked = value; OnPropertyChanged(nameof(Horizon5DayChecked)); }
        }

        private bool _horizon10DayChecked = false;
        public bool Horizon10DayChecked
        {
            get => _horizon10DayChecked;
            set { _horizon10DayChecked = value; OnPropertyChanged(nameof(Horizon10DayChecked)); }
        }

        // Prediction data
        public ObservableCollection<PredictionModel> Predictions { get; set; } = new();
        private ObservableCollection<PredictionModel> _models = new();
        public ObservableCollection<PredictionModel> Models
        {
            get => _models;
            private set
            {
                _models = value;
                OnPropertyChanged(nameof(Models));
            }
        }

        // Top Predictions from database (StockPredictions table)
        private ObservableCollection<PredictionModel> _topPredictions = new();
        public ObservableCollection<PredictionModel> TopPredictions
        {
            get => _topPredictions;
            private set
            {
                _topPredictions = value;
                OnPropertyChanged(nameof(TopPredictions));
                OnPropertyChanged(nameof(TopPredictionsCountText));
            }
        }

        // Count text for UI binding
        public string TopPredictionsCountText => TopPredictions.Count > 0 
            ? $"({TopPredictions.Count} predictions)" 
            : "";

        // Filtering and status
        private string _statusText = "Ready";
        public string StatusText { get => _statusText; set { _statusText = value; OnPropertyChanged(nameof(StatusText)); } }

        private string _lastUpdatedText = "Last updated: Never";
        public string LastUpdatedText { get => _lastUpdatedText; set { _lastUpdatedText = value; OnPropertyChanged(nameof(LastUpdatedText)); } }

        private string _symbolFilter = "All Symbols";
        public string SymbolFilter { get => _symbolFilter; set { _symbolFilter = value; OnPropertyChanged(nameof(SymbolFilter)); ApplyFilter(); } }

        private double _minConfidence = 0.6;
        public double MinConfidence { get => _minConfidence; set { _minConfidence = value; OnPropertyChanged(nameof(MinConfidence)); ApplyFilter(); } }

        private bool _isAutomatedMode;
        public bool IsAutomatedMode { get => _isAutomatedMode; set { _isAutomatedMode = value; OnPropertyChanged(nameof(IsAutomatedMode)); } }

        // Trading related members
        private ObservableCollection<TradingRule> _tradingRules = new();
        public ObservableCollection<TradingRule> TradingRules
        {
            get => _tradingRules;
            set { _tradingRules = value; OnPropertyChanged(nameof(TradingRules)); }
        }

        // Add a property to hold the selected strategy profile
        public Models.TradingStrategyProfile SelectedStrategyProfile { get; set; }

        // Commands
        public ICommand AnalyzeCommand { get; }
        public ICommand RefreshCommand { get; }

        public ObservableCollection<string> Timeframes { get; } = new ObservableCollection<string>
        {
            "1min", "5min", "15min", "30min", "1hour", "4hour", "1day", "1week", "1month"
        };

        public string Symbol
        {
            get => _symbol;
            set
            {
                if (_symbol != value)
                {
                    _symbol = value;
                    OnPropertyChanged(nameof(Symbol));
                    RefreshAsync().ConfigureAwait(false);
                }
            }
        }

        private string _selectedTimeframe = "1day";
        public string SelectedTimeframe
        {
            get => _selectedTimeframe;
            set
            {
                if (_selectedTimeframe != value)
                {
                    _selectedTimeframe = value;
                    OnPropertyChanged(nameof(SelectedTimeframe));
                    RefreshAsync().ConfigureAwait(false);
                }
            }
        }

        private bool _isLoading;
        public bool IsLoading
        {
            get => _isLoading;
            private set
            {
                _isLoading = value;
                OnPropertyChanged(nameof(IsLoading));
            }
        }

        public double CurrentPrice
        {
            get => _currentPrice;
            set
            {
                if (_currentPrice != value)
                {
                    _currentPrice = value;
                    OnPropertyChanged(nameof(CurrentPrice));
                }
            }
        }

        public PredictionAnalysisViewModel(
            ITechnicalIndicatorService indicatorService,
            PredictionAnalysisRepository analysisRepository,
            ITradingService tradingService,
            ISettingsService settingsService,
            IAlphaVantageService alphaVantageService,
            IEmailService emailService,
            ITradingRuleService tradingRuleService)
        {
            _indicatorService = indicatorService ?? throw new ArgumentNullException(nameof(indicatorService));
            _analysisRepository = analysisRepository ?? throw new ArgumentNullException(nameof(analysisRepository));
            TradingService = tradingService ?? throw new ArgumentNullException(nameof(tradingService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _emailService = emailService ?? throw new ArgumentNullException(nameof(emailService));
            _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
            _tradingRuleService = tradingRuleService ?? throw new ArgumentNullException(nameof(tradingRuleService));
            
            // Initialize PredictionAnalysisService for database operations
            _predictionAnalysisService = new PredictionAnalysisService();
            
            AnalyzeCommand = new RelayCommand(async _ => await AnalyzeAsync(), _ => true);
            RefreshCommand = new RelayCommand(async _ => await RefreshAsync(), _ => true);

            LoadCachedPredictions();
            BindingOperations.EnableCollectionSynchronization(Predictions, new object());
            BindingOperations.EnableCollectionSynchronization(TopPredictions, new object());
        }

        public async Task AnalyzeAsync()
        {
            StatusText = "Starting analysis...";
            Predictions.Clear();
            var symbols = await Task.Run(() => _analysisRepository.GetSymbols());
            int processed = 0;
            foreach (var symbol in symbols)
            {
                // Use the selected strategy profile if set, otherwise default to SmaCrossover
                var strategy = SelectedStrategyProfile ?? new Models.SmaCrossoverStrategy();
                // The repository now expects Models.StrategyProfile which is the base type of TradingStrategyProfile
                var predictionResult = await Task.Run(() => _analysisRepository.AnalyzeSymbol(symbol, strategy));
                if (predictionResult != null)
                {
                    var prediction = predictionResult.ToPredictionModel();
                    Predictions.Add(prediction);
                    if (prediction.PredictedAction == "BUY" && prediction.Confidence >= 0.8)
                    {
                        try
                        {
                            await _emailService.SendEmailAsync(
                                "test@gmail.com",
                                $"Buy Opportunity: {prediction.Symbol}",
                                $"A high-confidence BUY signal was detected for {prediction.Symbol}.\nConfidence: {prediction.Confidence:P0}\nCurrent Price: {prediction.CurrentPrice}\nTarget Price: {prediction.TargetPrice}"
                            );
                        }
                        catch (Exception ex)
                        {
                            StatusText = $"Error sending email alert: {ex.Message}";
                        }
                    }
                }
                processed++;
                StatusText = $"Analyzing... ({processed}/{symbols.Count})";
            }
            StatusText = $"Analysis complete. {Predictions.Count} predictions.";
            LastUpdatedText = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
            ApplyFilter();
        }

        public async Task RefreshAsync()
        {
            if (string.IsNullOrEmpty(Symbol))
                return;

            try
            {
                IsLoading = true;

                // Get latest indicators
                var indicators = await _indicatorService.GetIndicatorsForPrediction(Symbol, SelectedTimeframe);

                // Get algorithmic trading signals
                var signals = await _indicatorService.GetAlgorithmicTradingSignals(Symbol);

                // Save analysis results
                var result = new PredictionAnalysisResult
                {
                    AnalysisTime = DateTime.Now,
                    Indicators = indicators
                };
                _analysisRepository.SaveAnalysisResults(new[] { result });

                // Update predictions collection
                var latestPrice = indicators["Price"];
                var prediction = new PredictionModel
                {
                    Symbol = Symbol,
                    CurrentPrice = latestPrice,
                    Confidence = signals["MomentumScore"] / 100,
                    PredictedAction = signals["TradingSignal"] > 0 ? "Buy" : "Sell",
                    TargetPrice = latestPrice * (1 + (signals["TradingSignal"] / 100)),
                    Indicators = indicators
                };

                Predictions.Clear();
                Predictions.Add(prediction);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error in PredictionAnalysisViewModel.RefreshAsync: {ex.Message}", ex.ToString());
            }
            finally
            {
                IsLoading = false;
            }
        }

        private void LoadCachedPredictions()
        {
            Predictions.Clear();
            var cached = _analysisRepository.GetLatestAnalyses().Select(r => r.ToPredictionModel());
            foreach (var p in cached)
                Predictions.Add(p);
            LastUpdatedText = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
        }

        // Call this on control load
        public async Task OnLoaded()
        {
            LoadCachedPredictions();
            await LoadTopPredictionsAsync();
            await LoadTradingRulesAsync();
            await RefreshAsync();
        }

        /// <summary>
        /// Loads top predictions from the StockPredictions database table
        /// </summary>
        public async Task LoadTopPredictionsAsync()
        {
            try
            {
                StatusText = "Loading top predictions from database...";
                
                var predictions = await _predictionAnalysisService.GetAllPredictionsAsync(1000);
                
                // Create a new collection for efficiency
                var newCollection = new ObservableCollection<PredictionModel>(predictions);
                TopPredictions = newCollection;
                
                OnPropertyChanged(nameof(TopPredictionsCountText));
                StatusText = $"Loaded {TopPredictions.Count} predictions from database.";
            }
            catch (Exception ex)
            {
                StatusText = $"Error loading predictions: {ex.Message}";
            }
        }

        /// <summary>
        /// Adds a prediction to the TopPredictions collection (for when analyzing a ticker)
        /// </summary>
        /// <param name="prediction">The prediction to add</param>
        public void AddToTopPredictions(PredictionModel prediction)
        {
            if (prediction == null) return;
            
            // Insert at the beginning of the collection (most recent first)
            TopPredictions.Insert(0, prediction);
            OnPropertyChanged(nameof(TopPredictionsCountText));
        }

        private void ApplyFilter()
        {
            ICollectionView view = CollectionViewSource.GetDefaultView(Predictions);
            if (view == null) return;
            view.Filter = item =>
            {
                if (item is not PredictionModel p) return false;
                bool confidenceMatch = p.Confidence >= MinConfidence;
                bool symbolMatch = SymbolFilter == "All Symbols" || p.Symbol.Equals(SymbolFilter, StringComparison.OrdinalIgnoreCase);
                return confidenceMatch && symbolMatch;
            };
            StatusText = $"Filtered: {view.Cast<object>().Count()} predictions.";
        }

        private async Task LoadTradingRulesAsync()
        {
            try
            {
                List<TradingRule> rules;
                if (Symbol != null)
                {
                    rules = await _tradingRuleService.GetActiveTradingRulesAsync(Symbol);
                    StatusText = $"Loaded {rules.Count} trading rules for {Symbol}";
                }
                else
                {
                    rules = await _tradingRuleService.GetTradingRulesAsync();
                    StatusText = $"Loaded {rules.Count} trading rules";
                }

                TradingRules.Clear();
                foreach (var rule in rules)
                {
                    TradingRules.Add(rule);
                }
            }
            catch (Exception ex)
            {
                StatusText = $"Error loading trading rules: {ex.Message}";
                // Log error if needed
            }
        }

        public async Task SaveTradingRuleAsync(TradingRule rule)
        {
            try
            {
                await _tradingRuleService.SaveTradingRuleAsync(rule);

                // Only add to collection if it's a new rule
                if (!TradingRules.Contains(rule))
                {
                    TradingRules.Add(rule);
                }

                StatusText = $"Trading rule saved for {rule.Symbol}";
            }
            catch (Exception ex)
            {
                StatusText = $"Error saving trading rule: {ex.Message}";
                throw;
            }
        }

        public List<TradingRule> GetTradingRules(string symbol)
        {
            try
            {
                return TradingRules.Where(r => r.Symbol == symbol && r.IsActive).ToList();
            }
            catch (Exception ex)
            {
                StatusText = $"Error getting trading rules: {ex.Message}";
                throw;
            }
        }

        public async Task<bool> ExecuteTrade(PredictionModel prediction)
        {
            try
            {
                if (prediction == null)
                    return false;

                // Get latest technical indicators
                var indicators = await _indicatorService.GetIndicatorsForPrediction(
                    prediction.Symbol, "1day");

                // Validate trading conditions
                if (!ValidateTradingConditions(prediction, indicators))
                    return false;

                // Execute trade through trading service
                bool success = await TradingService.ExecuteTradeAsync(
                    prediction.Symbol,
                    prediction.PredictedAction,
                    prediction.CurrentPrice,
                    prediction.TargetPrice);

                if (success)
                {
                    // Update prediction status
                    prediction.LastVerifiedDate = DateTime.Now;

                    // Save the updated prediction as a PredictionAnalysisResult
                    var result = new PredictionAnalysisResult
                    {
                        Symbol = prediction.Symbol,
                        PredictedAction = prediction.PredictedAction,
                        Confidence = prediction.Confidence,
                        CurrentPrice = prediction.CurrentPrice,
                        TargetPrice = prediction.TargetPrice,
                        PotentialReturn = prediction.PotentialReturn,
                        TradingRule = prediction.TradingRule,
                        AnalysisTime = prediction.PredictionDate != default ? prediction.PredictionDate : DateTime.Now,
                        Indicators = prediction.Indicators
                    };
                    _analysisRepository.SaveAnalysisResults(new[] { result });
                }

                return success;
            }
            catch (Exception ex)
            {
                StatusText = $"Error executing trade: {ex.Message}";
                return false;
            }
        }

        private bool ValidateTradingConditions(PredictionModel prediction, Dictionary<string, double> indicators)
        {
            if (prediction.Confidence < MinConfidence)
                return false;

            // Check if indicators support the trade direction
            if (indicators.TryGetValue("TradingSignal", out double signal))
            {
                if (prediction.PredictedAction == "BUY" && signal < -30)
                    return false;
                if (prediction.PredictedAction == "SELL" && signal > 30)
                    return false;
            }

            // Check market conditions
            if (prediction.MarketContext != null)
            {
                if (prediction.MarketContext.VolatilityIndex > 35) // High VIX
                    return false;
            }

            return true;
        }

        private async Task LoadModels()
        {
            try
            {
                // Get all symbols from the repository
                var symbols = await Task.Run(() => _analysisRepository.GetSymbols());

                // Cache the models for faster retrieval
                var modelTasks = symbols.Select(async symbol =>
                {
                    // Get latest indicators for the symbol
                    var indicators = await _indicatorService.GetIndicatorsForPrediction(symbol, "1day");
                    var signals = await _indicatorService.GetAlgorithmicTradingSignals(symbol);

                    var model = new PredictionModel
                    {
                        Symbol = symbol,
                        PredictedAction = signals["TradingSignal"] > 0 ? "Buy" : "Sell",
                        Confidence = signals["MomentumScore"] / 100, // Convert to 0-1 range
                        PredictionDate = DateTime.Now,
                        Indicators = new Dictionary<string, double>(indicators)
                    };

                    // Get additional technical data from Alpha Vantage
                    var technicalData = await _alphaVantageService.GetAllTechnicalIndicatorsAsync(symbol);
                    foreach (var indicator in technicalData)
                    {
                        if (!model.Indicators.ContainsKey(indicator.Key))
                        {
                            model.Indicators[indicator.Key] = indicator.Value;
                        }
                    }

                    return model;
                });

                Models = new ObservableCollection<PredictionModel>(await Task.WhenAll(modelTasks));
                //DatabaseMonolith.Log("Info", $"Loaded {Models.Count} prediction models");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to load prediction models", ex.ToString());
                throw;
            }
        }

        private async Task SavePatterns(List<PatternModel> patterns, string symbol)
        {
            try
            {
                // Save patterns to database 
                //DatabaseMonolith.Log("Info", $"Saving {patterns.Count} patterns for {symbol}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to save patterns for {symbol}", ex.ToString());
            }
        }

        public async Task<PredictionModel> PredictForSymbolAsync(string symbol)
        {
            try
            {
                if (string.IsNullOrEmpty(symbol))
                    return null;

                // Use the selected strategy profile if set, otherwise default to SmaCrossover
                var strategy = SelectedStrategyProfile ?? new Models.SmaCrossoverStrategy();

                // Analyze the symbol using the repository
                var predictionResult = await Task.Run(() => _analysisRepository.AnalyzeSymbol(symbol, strategy));

                if (predictionResult != null)
                {
                    return predictionResult.ToPredictionModel();
                }

                return null;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error in PredictForSymbolAsync for {symbol}: {ex.Message}", ex.ToString());
                return null;
            }
        }

        public async Task<List<PredictionModel>> PredictForMultipleSymbolsAsync()
        {
            try
            {
                var predictions = new List<PredictionModel>();
                var symbols = await Task.Run(() => _analysisRepository.GetSymbols());

                foreach (var symbol in symbols)
                {
                    // Use the selected strategy profile if set, otherwise default to SmaCrossover
                    var strategy = SelectedStrategyProfile ?? new Models.SmaCrossoverStrategy();

                    // Analyze each symbol using the repository
                    var predictionResult = await Task.Run(() => _analysisRepository.AnalyzeSymbol(symbol, strategy));

                    if (predictionResult != null)
                    {
                        var prediction = predictionResult.ToPredictionModel();
                        predictions.Add(prediction);
                    }
                }

                return predictions;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error in PredictForMultipleSymbolsAsync: {ex.Message}", ex.ToString());
                return new List<PredictionModel>();
            }
        }

        /// <summary>
        /// Populates the multi-horizon visualization charts from a TFT prediction result.
        /// </summary>
        /// <param name="tftResult">The TFT prediction result containing multi-horizon data.</param>
        /// <param name="historicalPrices">Optional list of historical prices for context (past 30 days).</param>
        public void UpdateTFTVisualization(Quantra.DAL.Models.TFTPredictionResult tftResult, List<double> historicalPrices = null)
        {
            if (tftResult == null) return;

            // Multi-horizon predictions
            PredictedPrices.Clear();
            UpperBandPrices.Clear();
            LowerBandPrices.Clear();
            DateLabels.Clear();

            // Add historical prices if provided
            HistoricalPrices.Clear();
            if (historicalPrices != null && historicalPrices.Count > 0)
            {
                for (int i = 0; i < historicalPrices.Count; i++)
                {
                    // Historical days are negative (past)
                    int dayOffset = i - historicalPrices.Count;
                    HistoricalPrices.Add(new ObservablePoint(dayOffset, historicalPrices[i]));
                    DateLabels.Add($"{dayOffset}d");
                }
            }

            // Add current price at day 0
            if (tftResult.CurrentPrice > 0)
            {
                HistoricalPrices.Add(new ObservablePoint(0, tftResult.CurrentPrice));
                PredictedPrices.Add(new ObservablePoint(0, tftResult.CurrentPrice));
                UpperBandPrices.Add(new ObservablePoint(0, tftResult.CurrentPrice));
                LowerBandPrices.Add(new ObservablePoint(0, tftResult.CurrentPrice));
                DateLabels.Add("Today");
            }

            // Add multi-horizon predictions
            if (tftResult.Horizons != null && tftResult.Horizons.Count > 0)
            {
                // Parse horizon keys and sort by days
                var sortedHorizons = tftResult.Horizons
                    .Select(h => new { 
                        Key = h.Key, 
                        Days = int.TryParse(h.Key.Replace("d", "").Replace("D", ""), out int d) ? d : 0,
                        Data = h.Value 
                    })
                    .Where(h => h.Days > 0)
                    .OrderBy(h => h.Days);

                foreach (var horizon in sortedHorizons)
                {
                    PredictedPrices.Add(new ObservablePoint(horizon.Days, horizon.Data.MedianPrice));
                    UpperBandPrices.Add(new ObservablePoint(horizon.Days, horizon.Data.UpperBound));
                    LowerBandPrices.Add(new ObservablePoint(horizon.Days, horizon.Data.LowerBound));
                    DateLabels.Add($"+{horizon.Days}d");
                }
            }

            // Attention weights
            AttentionWeights.Clear();
            AttentionLabels.Clear();

            if (tftResult.TemporalAttention != null && tftResult.TemporalAttention.Count > 0)
            {
                foreach (var kvp in tftResult.TemporalAttention.OrderBy(k => k.Key))
                {
                    AttentionWeights.Add(kvp.Value);
                    AttentionLabels.Add($"{Math.Abs(kvp.Key)}d ago");
                }
            }

            // Feature importance
            FeatureImportances.Clear();
            FeatureNames.Clear();

            if (tftResult.FeatureWeights != null && tftResult.FeatureWeights.Count > 0)
            {
                foreach (var kvp in tftResult.FeatureWeights.OrderByDescending(k => k.Value).Take(10))
                {
                    FeatureImportances.Add(kvp.Value);
                    FeatureNames.Add(kvp.Key);
                }
            }

            // Notify UI of changes
            OnPropertyChanged(nameof(HistoricalPrices));
            OnPropertyChanged(nameof(PredictedPrices));
            OnPropertyChanged(nameof(UpperBandPrices));
            OnPropertyChanged(nameof(LowerBandPrices));
            OnPropertyChanged(nameof(DateLabels));
            OnPropertyChanged(nameof(AttentionWeights));
            OnPropertyChanged(nameof(AttentionLabels));
            OnPropertyChanged(nameof(FeatureImportances));
            OnPropertyChanged(nameof(FeatureNames));
        }

        /// <summary>
        /// Gets the list of selected prediction horizons based on checkbox states.
        /// </summary>
        public List<int> GetSelectedHorizons()
        {
            var horizons = new List<int>();
            if (Horizon1DayChecked) horizons.Add(1);
            if (Horizon3DayChecked) horizons.Add(3);
            if (Horizon5DayChecked) horizons.Add(5);
            if (Horizon10DayChecked) horizons.Add(10);
            return horizons;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged(string propertyName) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    // Extension to convert PredictionAnalysisResult to PredictionModel
    public static class PredictionAnalysisResultExtensions
    {
        public static PredictionModel ToPredictionModel(this PredictionAnalysisResult result)
        {
            if (result == null) return null;

            // Copy all indicators directly (no HasValue check needed for double)
            var indicators = result.Indicators != null
                ? new Dictionary<string, double>(result.Indicators)
                : new Dictionary<string, double>();

            return new PredictionModel
            {
                Symbol = result.Symbol,
                PredictedAction = result.PredictedAction,
                Confidence = result.Confidence,
                CurrentPrice = result.CurrentPrice,
                TargetPrice = result.TargetPrice,
                PotentialReturn = result.PotentialReturn,
                TradingRule = result.TradingRule,
                Indicators = indicators,
                PredictionDate = result.AnalysisTime
            };
        }
    }
}
