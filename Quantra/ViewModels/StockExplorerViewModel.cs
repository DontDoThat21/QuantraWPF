using Quantra;
using Quantra.Commands;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Utilities;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System;
using System.Windows;
using Quantra.DAL.Services;

namespace Quantra.ViewModels
{
    public class StockExplorerViewModel : INotifyPropertyChanged, IDisposable
    {
        // Cache management constants
        private const int MAX_CACHED_STOCKS = 50; // Limit cache size to prevent memory issues
        private const int CACHE_CLEANUP_THRESHOLD = 40; // Start cleanup when we reach this many items
        
        // Observable collection of all cached stocks
        public ObservableCollection<QuoteData> CachedStocks { get; } = new();
        
        // Track access order for LRU cache management
        private readonly Dictionary<string, DateTime> _accessTimes = new();
        private readonly object _cacheLock = new object();

        // Pass-through properties for chart bindings (for XAML compatibility)
        public LiveCharts.ChartValues<double> StockPriceValues => SelectedStock?.StockPriceValues ?? new LiveCharts.ChartValues<double>();
        public LiveCharts.ChartValues<double> UpperBandValues => SelectedStock?.UpperBandValues ?? new LiveCharts.ChartValues<double>();
        public LiveCharts.ChartValues<double> MiddleBandValues => SelectedStock?.MiddleBandValues ?? new LiveCharts.ChartValues<double>();
        public LiveCharts.ChartValues<double> LowerBandValues => SelectedStock?.LowerBandValues ?? new LiveCharts.ChartValues<double>();
        public LiveCharts.ChartValues<double> RSIValues => SelectedStock?.RSIValues ?? new LiveCharts.ChartValues<double>();
        public LiveCharts.ChartValues<LiveCharts.Defaults.OhlcPoint> PatternCandles => SelectedStock?.PatternCandles ?? new LiveCharts.ChartValues<LiveCharts.Defaults.OhlcPoint>();
        // For prediction chart (right side)
        public LiveCharts.ChartValues<double> PriceValues => SelectedStock?.StockPriceValues ?? new LiveCharts.ChartValues<double>();

        // Notify chart property changes when SelectedStock changes
        private QuoteData _selectedStock;
        public QuoteData SelectedStock
        {
            get => _selectedStock;
            set
            {
                if (_selectedStock != value)
                {
                    // Clear previous stock's chart data to free memory
                    _selectedStock?.ClearChartData();
                    
                    _selectedStock = value;
                    
                    // Update access time for cache management
                    if (value != null)
                    {
                        lock (_cacheLock)
                        {
                            _accessTimes[value.Symbol] = DateTime.Now;
                        }
                    }
                    
                    OnPropertyChanged(nameof(SelectedStock));
                    // Notify chart property changes
                    OnPropertyChanged(nameof(StockPriceValues));
                    OnPropertyChanged(nameof(UpperBandValues));
                    OnPropertyChanged(nameof(MiddleBandValues));
                    OnPropertyChanged(nameof(LowerBandValues));
                    OnPropertyChanged(nameof(RSIValues));
                    OnPropertyChanged(nameof(PatternCandles));
                    OnPropertyChanged(nameof(PriceValues));
                }
            }
        }

        private string _selectedSymbol;
        public string SelectedSymbol
        {
            get => _selectedSymbol;
            set
            {
                if (_selectedSymbol != value)
                {
                    _selectedSymbol = value;
                    OnPropertyChanged(nameof(SelectedSymbol));
                    // Load and cache data when symbol changes
                    _ = LoadAndCacheStockDataAsync(_selectedSymbol);
                }
            }
        }

        // Symbol search box selection command or handler
        public ICommand SymbolSelectedCommand { get; }
        public ICommand RunPredictionsCommand { get; }

        private readonly StockDataCacheService _cacheService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly StockDataCacheService _stockDataCacheService;
        private readonly RealTimeInferenceService _inferenceService;
        private readonly PredictionCacheService _predictionCacheService;

        public event PropertyChangedEventHandler? PropertyChanged;

        public StockExplorerViewModel(UserSettingsService userSettingsService)
        {
            _cacheService = new StockDataCacheService(userSettingsService);
            _alphaVantageService = new AlphaVantageService(userSettingsService); // Use parameterless constructor
            _inferenceService = new RealTimeInferenceService();
            _predictionCacheService = new PredictionCacheService();
            
            SymbolSelectedCommand = new RelayCommand<string>(OnSymbolSelected);
            RunPredictionsCommand = new RelayCommand(async _ => await RunPredictionsAsync(), _ => CanRunPredictions);
            
            // Load all cached stocks at startup
            LoadCachedStocks();
            LoadSymbolsAsync();
            
            // Start background preloading for frequently accessed symbols
            _ = Task.Run(async () => await StartBackgroundPreloadingAsync());
        }

        private void LoadCachedStocks()
        {
            // Load from cache so table/grid is populated on app start
            var cached = _cacheService.GetAllCachedStocks();
            CachedStocks.Clear();
            foreach (var stock in cached)
                CachedStocks.Add(stock);
        }

        private async void OnSymbolSelected(string symbol)
        {
            // Try to get from cache first using async version to avoid blocking UI thread
            var cached = await Task.Run(async () => await _cacheService.GetCachedStockAsync(symbol)).ConfigureAwait(false);
            
            if (cached != null)
            {
                // Update UI on UI thread
                await System.Windows.Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    AddOrUpdateCachedStock(cached);
                    SelectedStock = cached;
                });
            }
            else
            {
                // Not in cache: fetch from AlphaVantage API in background thread to avoid UI blocking
                _ = GlobalLoadingStateService.WithLoadingState(Task.Run(async () =>
                {
                    try
                    {
                        var prices = await _alphaVantageService.GetHistoricalClosingPricesAsync(symbol, 1).ConfigureAwait(false);
                        if (prices != null && prices.Count > 0)
                        {
                            var quoteData = new QuoteData
                            {
                                Symbol = symbol,
                                Price = prices.Last(),
                                LastAccessed = DateTime.Now,
                                Timestamp = DateTime.Now // Use Timestamp instead of Date
                            };
                            
                            // Cache the new data (database operation in background)
                            await _cacheService.CacheQuoteDataAsync(quoteData).ConfigureAwait(false);
                            
                            // Update UI collections on UI thread
                            await System.Windows.Application.Current.Dispatcher.InvokeAsync(() =>
                            {
                                AddOrUpdateCachedStock(quoteData);
                                SelectedStock = quoteData;
                            });
                        }
                    }
                    catch (Exception ex)
                    {
                        // Log error but don't block UI thread
                        await System.Windows.Application.Current.Dispatcher.InvokeAsync(() =>
                        {
                            //DatabaseMonolith.Log("Error", $"Failed to load symbol data for {symbol} in background", ex.ToString());
                        });
                    }
                }));
            }
            // Optionally, refresh the grid or notify UI
        }

        private async Task LoadAndCacheStockDataAsync(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            IsLoading = true;
            try
            {
                // Use smart caching - only force refresh if cache is stale or missing
                var data = await _stockDataCacheService.GetStockData(symbol, CurrentTimeRange ?? "1day", "1d", forceRefresh: false);
                // Update chart data for StockPriceChart on UI thread
                await LoadChartDataAsync(data);
                //DatabaseMonolith.Log("Debug", $"Loaded and cached data for {symbol}: {data?.Count ?? 0} records");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error loading/caching data for {symbol}", ex.ToString());
            }
            finally
            {
                IsLoading = false;
            }
        }

        // Add IsLoading property for UI loading state
        private bool _isLoading;
        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                if (_isLoading != value)
                {
                    _isLoading = value;
                    OnPropertyChanged(nameof(IsLoading));
                }
            }
        }

        // Prediction loading state
        private bool _isPredictionLoading;
        public bool IsPredictionLoading
        {
            get => _isPredictionLoading;
            set
            {
                if (_isPredictionLoading != value)
                {
                    _isPredictionLoading = value;
                    OnPropertyChanged(nameof(IsPredictionLoading));
                    OnPropertyChanged(nameof(CanRunPredictions));
                }
            }
        }

        // Prediction error handling
        private string _predictionError;
        public string PredictionError
        {
            get => _predictionError;
            set
            {
                if (_predictionError != value)
                {
                    _predictionError = value;
                    OnPropertyChanged(nameof(PredictionError));
                    OnPropertyChanged(nameof(HasPredictionError));
                }
            }
        }

        public bool HasPredictionError => !string.IsNullOrEmpty(PredictionError);

        // Prediction summary for UI display
        private string _predictionSummary = "Ready to run predictions...";
        public string PredictionSummary
        {
            get => _predictionSummary;
            set
            {
                if (_predictionSummary != value)
                {
                    _predictionSummary = value;
                    OnPropertyChanged(nameof(PredictionSummary));
                }
            }
        }

        // Can run predictions
        public bool CanRunPredictions => !IsPredictionLoading && CachedStocks.Any();

        // Add this method to support INotifyPropertyChanged
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public async Task<List<string>> LoadAllStockSymbolsAsync()
        {
            return await _alphaVantageService.GetAllStockSymbols();
        }

        // Add properties for chart data
        public ObservableCollection<string> DateLabels { get; set; }
        public Func<double, string> DateLabelFormatter { get; set; }
        // Alias for XAML compatibility
        public Func<double, string> DateFormatter => DateLabelFormatter;

        // Add this property if 'Quotes' is meant to be in the ViewModel
        public ObservableCollection<QuoteData> Quotes { get; set; } = new ObservableCollection<QuoteData>();

        // --- Add missing properties for SelectedQuoteSeries, SelectedQuote, SelectedQuoteDates ---

        // Represents the series of quotes (e.g., historical prices) for the selected stock
        private ObservableCollection<HistoricalPrice> _selectedQuoteSeries = new();
        public ObservableCollection<HistoricalPrice> SelectedQuoteSeries
        {
            get => _selectedQuoteSeries;
            set
            {
                if (_selectedQuoteSeries != value)
                {
                    _selectedQuoteSeries = value;
                    OnPropertyChanged(nameof(SelectedQuoteSeries));
                }
            }
        }

        // Represents the currently selected quote (could be the same as SelectedStock or a specific data point)
        private QuoteData _selectedQuote;
        public QuoteData SelectedQuote
        {
            get => _selectedQuote;
            set
            {
                if (_selectedQuote != value)
                {
                    _selectedQuote = value;
                    OnPropertyChanged(nameof(SelectedQuote));
                }
            }
        }

        // Represents the dates for the selected quote series (for chart X axis, etc.)
        private ObservableCollection<string> _selectedQuoteDates = new();
        public ObservableCollection<string> SelectedQuoteDates
        {
            get => _selectedQuoteDates;
            set
            {
                if (_selectedQuoteDates != value)
                {
                    _selectedQuoteDates = value;
                    OnPropertyChanged(nameof(SelectedQuoteDates));
                }
            }
        }

        // Load chart data method - made public to allow time range changes
        public async Task LoadChartDataAsync(List<HistoricalPrice> historicalData)
        {
            if (historicalData == null || historicalData.Count == 0)
                return;

            // Populate chart values for the selected stock
            if (SelectedStock != null)
            {
                SharedTitleBar.UpdateDispatcherMonitoring("LoadChartDataAsync");
                await SelectedStock.PopulateChartValuesFromHistorical(historicalData);
            }

            // Update chart-related properties for binding
            DateLabels = new ObservableCollection<string>(
                historicalData.Select(q => q.Date.ToString("MM/dd/yyyy", CultureInfo.InvariantCulture))
            );

            DateLabelFormatter = value =>
            {
                int index = (int)value;
                if (index >= 0 && index < historicalData.Count)
                    return historicalData[index].Date.ToString("MM/dd");
                return string.Empty;
            };

            OnPropertyChanged(nameof(DateLabels));
            OnPropertyChanged(nameof(DateLabelFormatter));
            OnPropertyChanged(nameof(DateFormatter)); // Notify that DateFormatter has also changed
        }

        // Load chart data for specific time range
        public async Task LoadChartDataForTimeRangeAsync(string symbol, string timeRange)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            try
            {
                // Check if we already have cached data for this time range
                var data = await Task.Run(() => _stockDataCacheService.GetStockData(symbol, timeRange, "1d", forceRefresh: false));
                
                if (data == null || data.Count == 0)
                {
                    // No cached data, fetch from API
                    data = await Task.Run(() => _stockDataCacheService.GetStockData(symbol, timeRange, "1d", forceRefresh: true));
                }
                
                // Update chart data on UI thread
                await LoadChartDataAsync(data);
                
                //DatabaseMonolith.Log("Info", $"Loaded chart data for {symbol} with time range {timeRange}: {data?.Count ?? 0} records");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error loading chart data for {symbol} with time range {timeRange}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Refreshes the selected stock by fetching the latest data from the API and updating the cache and UI.
        /// </summary>
        public async Task RefreshSelectedStockAsync()
        {
            if (SelectedStock == null)
                return;

            var symbol = SelectedStock.Symbol;
            IsLoading = true;
            try
            {
                // Force refresh from API using current time range for manual refresh
                var data = await _stockDataCacheService.GetStockData(symbol, CurrentTimeRange, "1d", forceRefresh: true);
                // Optionally update chart data
                await LoadChartDataAsync(data);

                // Get the latest quote data (price, etc.)
                var latestQuote = await _alphaVantageService.GetQuoteDataAsync(symbol);
                if (latestQuote != null)
                {
                    // Update cache and UI collection
                    _cacheService.CacheQuoteData(latestQuote);

                    // Update or add in CachedStocks
                    var existing = CachedStocks.FirstOrDefault(s => s.Symbol == symbol);
                    if (existing != null)
                    {
                        var idx = CachedStocks.IndexOf(existing);
                        CachedStocks[idx] = latestQuote;
                    }
                    else
                    {
                        CachedStocks.Add(latestQuote);
                    }
                    SelectedStock = latestQuote;
                }
            }
            finally
            {
                IsLoading = false;
            }
        }

        // --- Symbol Search/Dropdown Support ---
        private ObservableCollection<string> _filteredSymbols = new();
        public ObservableCollection<string> FilteredSymbols
        {
            get => _filteredSymbols;
            set
            {
                if (_filteredSymbols != value)
                {
                    _filteredSymbols = value;
                    OnPropertyChanged(nameof(FilteredSymbols));
                }
            }
        }

        private string _symbolSearchText;
        public string SymbolSearchText
        {
            get => _symbolSearchText;
            set
            {
                if (_symbolSearchText != value)
                {
                    _symbolSearchText = value;
                    OnPropertyChanged(nameof(SymbolSearchText));
                    UpdateFilteredSymbols();
                }
            }
        }

        // Expose grid items for DataGrid binding
        public ObservableCollection<QuoteData> StockGridItems => CachedStocks;

        private List<string> _allSymbols = new();

        public async void LoadSymbolsAsync()
        {
            // Ensure VIX is available in the database cache for searching
            await Task.Run(() => SymbolCacheUtility.EnsureVixInCache());
            
            _allSymbols = await _alphaVantageService.GetAllStockSymbols();
            UpdateFilteredSymbols();
        }

        private void UpdateFilteredSymbols()
        {
            if (_allSymbols == null || _allSymbols.Count == 0)
            {
                FilteredSymbols = new ObservableCollection<string>();
                return;
            }
            if (string.IsNullOrWhiteSpace(SymbolSearchText))
            {
                FilteredSymbols = new ObservableCollection<string>(_allSymbols.Take(20));
            }
            else
            {
                var filtered = _allSymbols
                    .Where(s => s.Contains(SymbolSearchText, StringComparison.OrdinalIgnoreCase))
                    .Take(20)
                    .ToList();
                FilteredSymbols = new ObservableCollection<string>(filtered);
            }
        }

        // List of indicators to load for the selected symbol
        public List<string> IndicatorsToLoad { get; set; } = new List<string> { "RSI", "ADX", "CCI", "STOCH", "MACD", "BOLLINGER_BANDS" };

        // Current time range property for chart data
        private string _currentTimeRange = "1day"; // Default to 1 day
        public string CurrentTimeRange
        {
            get => _currentTimeRange;
            set
            {
                if (_currentTimeRange != value)
                {
                    _currentTimeRange = value;
                    OnPropertyChanged(nameof(CurrentTimeRange));
                }
            }
        }

        /// <summary>
        /// Runs ML predictions for all cached stocks
        /// </summary>
        private async Task RunPredictionsAsync()
        {
            if (!CachedStocks.Any())
            {
                PredictionError = "No stocks available for prediction. Please load some stock data first.";
                PredictionSummary = "No stocks available for prediction.";
                return;
            }

            // Set busy cursor and update monitoring
            await App.Current.Dispatcher.InvokeAsync(() =>
            {
                System.Windows.Input.Mouse.OverrideCursor = System.Windows.Input.Cursors.Wait;
            });

            IsPredictionLoading = true;
            PredictionError = null;
            PredictionSummary = "Initializing prediction models...";

            try
            {
                // Initialize inference service if needed
                await _inferenceService.InitializeAsync();
                PredictionSummary = "Running predictions for stocks...";

                int processedCount = 0;
                int totalStocks = CachedStocks.Count;
                var tasks = new List<Task>();
                var predictionResults = new List<string>();

                // Run predictions for each stock in parallel (but limited concurrency)
                foreach (var stock in CachedStocks.ToList())
                {
                    tasks.Add(RunPredictionForStockAsync(stock));
                    
                    // Process in batches of 5 to avoid overwhelming the service
                    if (tasks.Count >= 5)
                    {
                        await Task.WhenAll(tasks);
                        tasks.Clear();
                        processedCount += 5;
                        
                        // Update progress
                        await App.Current.Dispatcher.InvokeAsync(() =>
                        {
                            PredictionSummary = $"Processing predictions... {processedCount}/{totalStocks} completed";
                        });
                        //await App.Current.Dispatcher.InvokeAsync(() =>
                        //{
                        //    Quantra.Views.Shared.SharedTitleBar.UpdateDispatcherMonitoring($"ProcessingBatch_{processedCount}", "StockExplorerViewModel");
                        //});
                    }
                }

                // Process remaining stocks
                if (tasks.Any())
                {
                    await Task.WhenAll(tasks);
                    processedCount += tasks.Count;
                }

                // Generate prediction summary
                var buyCount = CachedStocks.Count(s => s.PredictedAction == "BUY");
                var sellCount = CachedStocks.Count(s => s.PredictedAction == "SELL");
                var holdCount = CachedStocks.Count(s => s.PredictedAction == "HOLD");
                var errorCount = CachedStocks.Count(s => s.PredictedAction == "ERROR");
                
                var validPredictions = CachedStocks.Where(s => s.PredictionConfidence > 0).ToList();
                var avgConfidence = validPredictions.Any() ? validPredictions.Average(s => s.PredictionConfidence) : 0.0;

                PredictionSummary = $"Predictions completed for {processedCount} stocks. " +
                                  $"BUY: {buyCount}, SELL: {sellCount}, HOLD: {holdCount}, ERRORS: {errorCount}. " +
                                  $"Average confidence: {avgConfidence:P1}";

                //DatabaseMonolith.Log("Info", $"Completed predictions for {processedCount} stocks");
            }
            catch (Exception ex)
            {
                PredictionError = $"Prediction failed: {ex.Message}";
                PredictionSummary = $"Prediction failed: {ex.Message}";
                //DatabaseMonolith.Log("Error", "Error running predictions", ex.ToString());
            }
            finally
            {
                IsPredictionLoading = false;
                
                // Reset cursor and clear global loading state
                await App.Current.Dispatcher.InvokeAsync(() =>
                {
                    System.Windows.Input.Mouse.OverrideCursor = null;
                });
            }
        }

        /// <summary>
        /// Runs prediction for a single stock
        /// </summary>
        private async Task RunPredictionForStockAsync(QuoteData stock)
        {
            try
            {
                //// Update current call monitoring for this specific stock
                //await App.Current.Dispatcher.InvokeAsync(() =>
                //{
                //    Quantra.Views.Shared.SharedTitleBar.UpdateDispatcherMonitoring($"PredictingStock_{stock.Symbol}", "RunPredictionForStockAsync");
                //});

                // Prepare market data for prediction
                var marketData = new Dictionary<string, double>
                {
                    ["symbol"] = stock.Symbol.GetHashCode(), // Convert symbol to numeric
                    ["close"] = stock.Price,
                    ["volume"] = stock.Volume,
                    ["rsi"] = stock.RSI,
                    ["pe_ratio"] = stock.PERatio,
                    ["day_high"] = stock.DayHigh,
                    ["day_low"] = stock.DayLow,
                    ["market_cap"] = stock.MarketCap
                };

                // Generate cache key
                var modelVersion = "v1.0"; // You might want to make this configurable
                var inputHash = _predictionCacheService.GenerateInputDataHash(marketData);

                // Check cache first
                var cachedResult = _predictionCacheService.GetCachedPrediction(stock.Symbol, modelVersion, inputHash);
                PredictionResult prediction;

                if (cachedResult != null)
                {
                    prediction = cachedResult;
                    prediction.CurrentPrice = stock.Price; // Update current price
                    
                    await App.Current.Dispatcher.InvokeAsync(() =>
                    {
                        //Quantra.Views.Shared.SharedTitleBar.UpdateDispatcherMonitoring($"UsingCachedPrediction_{stock.Symbol}", "RunPredictionForStockAsync");
                    });
                }
                else
                {
                    // Run actual prediction with fallback
                    //await App.Current.Dispatcher.InvokeAsync(() =>
                    //{
                    //    Quantra.Views.Shared.SharedTitleBar.UpdateDispatcherMonitoring($"RunningMLPrediction_{stock.Symbol}", "RunPredictionForStockAsync");
                    //});
                    
                    prediction = await GetPredictionWithFallback(marketData, stock);

                    // Cache the result if successful
                    if (!prediction.HasError)
                    {
                        _predictionCacheService.CachePrediction(stock.Symbol, modelVersion, inputHash, prediction);
                    }
                }

                // Update the stock with prediction results
                // Perform the asynchronous operation outside the UI thread
                await _cacheService.CacheQuoteDataAsync(stock);

                // Update the UI on the UI thread
                await App.Current.Dispatcher.InvokeAsync(() =>
                {
                    //Quantra.Views.Shared.SharedTitleBar.UpdateDispatcherMonitoring($"UpdatingStockData_{stock.Symbol}", "RunPredictionForStockAsync");
                    
                    stock.PredictedPrice = prediction.TargetPrice;
                    stock.PredictedAction = prediction.Action;
                    stock.PredictionConfidence = prediction.Confidence;
                    stock.PredictionTimestamp = prediction.PredictionDate;
                    stock.PredictionModelVersion = modelVersion;
                    
                    // Cache the updated stock data with predictions to database asynchronously
                    //await _cacheService.CacheQuoteDataAsync(stock);
                });
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error predicting for {stock.Symbol}", ex.ToString());
                
                // Set error state on UI thread
                await App.Current.Dispatcher.InvokeAsync(async () =>
                {
                    stock.PredictedAction = "ERROR";
                    stock.PredictionConfidence = 0;
                    
                    // Cache the updated stock data with error state to database asynchronously
                    await _cacheService.CacheQuoteDataAsync(stock);
                });
            }
        }

        /// <summary>
        /// Gets prediction with fallback to simple rule-based approach if ML service fails
        /// </summary>
        private async Task<PredictionResult> GetPredictionWithFallback(Dictionary<string, double> marketData, QuoteData stock)
        {
            try
            {
                // Try ML prediction first
                var prediction = await _inferenceService.GetPredictionAsync(marketData, "auto");
                prediction.Symbol = stock.Symbol;
                prediction.CurrentPrice = stock.Price;
                return prediction;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"ML prediction failed for {stock.Symbol}, using fallback: {ex.Message}");
                
                // Fallback to simple rule-based prediction
                return GenerateFallbackPrediction(stock);
            }
        }

        /// <summary>
        /// Generates a simple rule-based prediction when ML service is unavailable
        /// </summary>
        private PredictionResult GenerateFallbackPrediction(QuoteData stock)
        {
            var action = "HOLD";
            var confidence = 0.5;
            var targetPrice = stock.Price;

            // Simple rule-based logic
            if (stock.RSI < 30 && stock.Price < stock.DayLow * 1.05)
            {
                action = "BUY";
                confidence = 0.6;
                targetPrice = stock.Price * 1.05; // 5% upside target
            }
            else if (stock.RSI > 70 && stock.Price > stock.DayHigh * 0.95)
            {
                action = "SELL";
                confidence = 0.6;
                targetPrice = stock.Price * 0.95; // 5% downside target
            }

            return new PredictionResult
            {
                Symbol = stock.Symbol,
                Action = action,
                Confidence = confidence,
                CurrentPrice = stock.Price,
                TargetPrice = targetPrice,
                PredictionDate = DateTime.Now,
                ModelType = "fallback_rules"
            };
        }

        /// <summary>
        /// Starts background preloading for frequently accessed symbols
        /// </summary>
        private async Task StartBackgroundPreloadingAsync()
        {
            try
            {
                // Wait a bit after startup to avoid interfering with initial UI load
                await Task.Delay(5000);
                
                // Get frequently accessed symbols from cache
                var frequentSymbols = _stockDataCacheService.GetFrequentlyAccessedSymbols(5);
                
                if (frequentSymbols.Count > 0)
                {
                    //DatabaseMonolith.Log("Info", $"Starting background preload for {frequentSymbols.Count} frequently accessed symbols");
                    await _stockDataCacheService.PreloadSymbolsAsync(frequentSymbols, CurrentTimeRange ?? "1day");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Background preloading failed", ex.ToString());
            }
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            // Dispose all cached stock data
            foreach (var stock in CachedStocks)
            {
                stock?.Dispose();
            }
            CachedStocks.Clear();
            
            lock (_cacheLock)
            {
                _accessTimes.Clear();
            }
            
            _inferenceService?.Dispose();
        }

        /// <summary>
        /// Manages cache size using LRU eviction policy
        /// </summary>
        public void ManageCacheSize()
        {
            lock (_cacheLock)
            {
                if (CachedStocks.Count <= MAX_CACHED_STOCKS)
                    return;

                // Get stocks to remove based on LRU policy
                var stocksToRemove = _accessTimes
                    .OrderBy(kvp => kvp.Value)
                    .Take(CachedStocks.Count - CACHE_CLEANUP_THRESHOLD)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var symbol in stocksToRemove)
                {
                    var stockToRemove = CachedStocks.FirstOrDefault(s => s.Symbol == symbol);
                    if (stockToRemove != null)
                    {
                        // Dispose the stock data to free memory
                        stockToRemove.Dispose();
                        CachedStocks.Remove(stockToRemove);
                        _accessTimes.Remove(symbol);
                    }
                }

                // Force garbage collection if we removed items
                if (stocksToRemove.Any())
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
            }
        }

        /// <summary>
        /// Adds or updates a stock in the cache with size management
        /// </summary>
        private void AddOrUpdateCachedStock(QuoteData stock)
        {
            if (stock == null) return;

            lock (_cacheLock)
            {
                // Check if stock already exists
                var existing = CachedStocks.FirstOrDefault(s => s.Symbol == stock.Symbol);
                if (existing != null)
                {
                    // Update existing stock's data
                    var index = CachedStocks.IndexOf(existing);
                    existing.Dispose(); // Dispose old data
                    CachedStocks[index] = stock;
                }
                else
                {
                    // Add new stock
                    CachedStocks.Add(stock);
                }

                // Update access time
                _accessTimes[stock.Symbol] = DateTime.Now;

                // Manage cache size
                ManageCacheSize();
            }
        }
    }
}

