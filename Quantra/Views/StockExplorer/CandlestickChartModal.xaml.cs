using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Defaults;
using Quantra.DAL.Services;
using Quantra.Models;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Polly;
using Polly.Retry;
using System.Text;
using Quantra.Views.StockExplorer;

namespace Quantra.Controls
{
    /// <summary>
    /// Candlestick Chart Modal Window for Real-Time Stock Data Visualization
    /// </summary>
    public partial class CandlestickChartModal : Window, INotifyPropertyChanged
    {
        #region Fields

        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private readonly StockDataCacheService _stockDataCacheService;
        private readonly UserSettingsService _userSettingsService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private readonly string _symbol;
        private DispatcherTimer _refreshTimer;
        private CancellationTokenSource _cancellationTokenSource;
        private CancellationTokenSource _requestCancellationTokenSource;
        private Task _currentLoadTask;
        private readonly SemaphoreSlim _requestSemaphore = new SemaphoreSlim(1, 1);
        private readonly AsyncRetryPolicy _retryPolicy;
        
        // Configurable settings - loaded from user preferences
        private int _refreshIntervalSeconds = 15; // Default: 15 seconds
        private const int API_RATE_LIMIT_CALLS = 5; // 5 calls per minute for free tier
        private const int CACHE_DURATION_SECONDS = 300; // Use StockDataCacheService cache (5 minutes)
        private const int MAX_RETRY_ATTEMPTS = 3;
        private const int INITIAL_RETRY_DELAY_MS = 1000; // 1 second
        
        // API usage tracking
        private int _apiCallsToday = 0;
        private DateTime _lastApiCallDate = DateTime.MinValue;
        
        private SeriesCollection _candlestickSeries;
        private SeriesCollection _volumeSeries;
        private List<string> _timeLabels;
        private bool _isAutoRefreshEnabled = true;
        private bool _isLoading = false;
        private bool _isDataLoaded = false;
        private bool _isNoData = false;
        private string _currentInterval = "5min";
        private DateTime _lastUpdateTime = DateTime.MinValue;
        private double _lastPrice = 0;
        private double _priceChange = 0;
        private double _priceChangePercent = 0;
        private int _maxCandles = 100;
        
        // Caching
        private List<HistoricalPrice> _cachedData;
        private DateTime _cacheTimestamp = DateTime.MinValue;
        private string _cachedInterval;
        
        // Zoom/Pan
        private double? _xAxisMin;
        private double? _xAxisMax;
        private double? _yAxisMin;
        private double? _yAxisMax;
        private double _zoomLevel = 1.0;
        private bool _isPaused = false;
        
        // Technical Indicators
        private bool _showSMA = false;
        private bool _showEMA = false;
        private bool _showRSI = false;
        private bool _showMACD = false;
        private bool _showBollingerBands = false;
        private bool _showVWAP = false;
        
        // Drawing Tools - stores series representing drawn lines
        private List<LineSeries> _drawnLines = new List<LineSeries>();
        
        // Window size/position for persistence
        private double _windowWidth = 1000;
        private double _windowHeight = 700;
        private double _windowLeft = double.NaN;
        private double _windowTop = double.NaN;
        
        // Loading progress
        private double _loadingProgress = 0;
        private bool _isProgressIndeterminate = true;
        private string _loadingProgressText = "";
        
        // Crosshair tracking
        private string _crosshairPriceText = "";

        #endregion

        #region Properties

        public event PropertyChangedEventHandler PropertyChanged;

        public string Symbol => _symbol;

        public string WindowTitle => $"Real-Time Candlestick Chart - {_symbol}";

        public SeriesCollection CandlestickSeries
        {
            get => _candlestickSeries;
            set
            {
                _candlestickSeries = value;
                OnPropertyChanged(nameof(CandlestickSeries));
            }
        }

        public SeriesCollection VolumeSeries
        {
            get => _volumeSeries;
            set
            {
                _volumeSeries = value;
                OnPropertyChanged(nameof(VolumeSeries));
            }
        }

        public List<string> TimeLabels
        {
            get => _timeLabels;
            set
            {
                _timeLabels = value;
                OnPropertyChanged(nameof(TimeLabels));
            }
        }

        public bool IsAutoRefreshEnabled
        {
            get => _isAutoRefreshEnabled;
            set
            {
                _isAutoRefreshEnabled = value;
                OnPropertyChanged(nameof(IsAutoRefreshEnabled));
                OnPropertyChanged(nameof(AutoRefreshText));
                
                if (value)
                    StartAutoRefresh();
                else
                    StopAutoRefresh();
            }
        }

        public string AutoRefreshText => IsAutoRefreshEnabled ? "ON" : "OFF";

        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                _isLoading = value;
                OnPropertyChanged(nameof(IsLoading));
            }
        }

        public bool IsDataLoaded
        {
            get => _isDataLoaded;
            set
            {
                _isDataLoaded = value;
                OnPropertyChanged(nameof(IsDataLoaded));
            }
        }

        public bool IsNoData
        {
            get => _isNoData;
            set
            {
                _isNoData = value;
                OnPropertyChanged(nameof(IsNoData));
            }
        }

        public double LastPrice
        {
            get => _lastPrice;
            set
            {
                _lastPrice = value;
                OnPropertyChanged(nameof(LastPrice));
            }
        }

        public double PriceChange
        {
            get => _priceChange;
            set
            {
                _priceChange = value;
                OnPropertyChanged(nameof(PriceChange));
            }
        }

        public double PriceChangePercent
        {
            get => _priceChangePercent;
            set
            {
                _priceChangePercent = value;
                OnPropertyChanged(nameof(PriceChangePercent));
            }
        }

        public Brush PriceChangeColor => PriceChange >= 0 
            ? new SolidColorBrush(Color.FromRgb(0x20, 0xC0, 0x40)) // Green
            : new SolidColorBrush(Color.FromRgb(0xC0, 0x20, 0x20)); // Red

        public string LastUpdateText => _lastUpdateTime == DateTime.MinValue 
            ? "Never updated" 
            : $"Last update: {_lastUpdateTime:HH:mm:ss}";

        public string StatusText => IsLoading 
            ? "Loading data..." 
            : $"Showing {_timeLabels?.Count ?? 0} candles";

        public string ApiUsageText => $"API Calls Today: {_apiCallsToday} | Refresh: {_refreshIntervalSeconds}s";

        public bool IsPaused
        {
            get => _isPaused;
            set
            {
                _isPaused = value;
                OnPropertyChanged(nameof(IsPaused));
                OnPropertyChanged(nameof(PauseButtonText));
                
                if (value)
                    StopAutoRefresh();
                else
                    StartAutoRefresh();
            }
        }

        public string PauseButtonText => IsPaused ? "? Resume" : "? Pause";
        
        public bool ShowSMA
        {
            get => _showSMA;
            set
            {
                _showSMA = value;
                OnPropertyChanged(nameof(ShowSMA));
                UpdateIndicators();
            }
        }
        
        public bool ShowEMA
        {
            get => _showEMA;
            set
            {
                _showEMA = value;
                OnPropertyChanged(nameof(ShowEMA));
                UpdateIndicators();
            }
        }
        
        public bool ShowRSI
        {
            get => _showRSI;
            set
            {
                _showRSI = value;
                OnPropertyChanged(nameof(ShowRSI));
                UpdateIndicators();
            }
        }
        
        public bool ShowMACD
        {
            get => _showMACD;
            set
            {
                _showMACD = value;
                OnPropertyChanged(nameof(ShowMACD));
                UpdateIndicators();
            }
        }
        
        public bool ShowBollingerBands
        {
            get => _showBollingerBands;
            set
            {
                _showBollingerBands = value;
                OnPropertyChanged(nameof(ShowBollingerBands));
                UpdateIndicators();
            }
        }
        
        public bool ShowVWAP
        {
            get => _showVWAP;
            set
            {
                _showVWAP = value;
                OnPropertyChanged(nameof(ShowVWAP));
                UpdateIndicators();
            }
        }

        public Func<double, string> PriceFormatter => value => $"${value:F2}";

        public Func<double, string> VolumeFormatter => value =>
        {
            if (value >= 1_000_000_000)
                return $"{value / 1_000_000_000:F1}B";
            if (value >= 1_000_000)
                return $"{value / 1_000_000:F1}M";
            if (value >= 1_000)
                return $"{value / 1_000:F1}K";
            return value.ToString("F0");
        };

        public bool IsCacheValid => _cachedData != null && 
                                    _cachedInterval == _currentInterval && 
                                    (DateTime.Now - _cacheTimestamp).TotalSeconds < CACHE_DURATION_SECONDS;

        public string CacheStatusText 
        {
            get
            {
                if (_cachedData != null && _cachedInterval == _currentInterval)
                {
                    var age = (DateTime.Now - _cacheTimestamp).TotalSeconds;
                    if (age < CACHE_DURATION_SECONDS)
                    {
                        return $"DB Cached ({(int)(CACHE_DURATION_SECONDS - age)}s)";
                    }
                }
                return "Live Data";
            }
        }

        public double? XAxisMin
        {
            get => _xAxisMin;
            set
            {
                _xAxisMin = value;
                OnPropertyChanged(nameof(XAxisMin));
            }
        }

        public double? XAxisMax
        {
            get => _xAxisMax;
            set
            {
                _xAxisMax = value;
                OnPropertyChanged(nameof(XAxisMax));
            }
        }

        public double? YAxisMin
        {
            get => _yAxisMin;
            set
            {
                _yAxisMin = value;
                OnPropertyChanged(nameof(YAxisMin));
            }
        }

        public double? YAxisMax
        {
            get => _yAxisMax;
            set
            {
                _yAxisMax = value;
                OnPropertyChanged(nameof(YAxisMax));
            }
        }
        
        public double WindowWidth
        {
            get => _windowWidth;
            set
            {
                _windowWidth = value;
                OnPropertyChanged(nameof(WindowWidth));
            }
        }
        
        public double WindowHeight
        {
            get => _windowHeight;
            set
            {
                _windowHeight = value;
                OnPropertyChanged(nameof(WindowHeight));
            }
        }
        
        public double WindowLeft
        {
            get => _windowLeft;
            set
            {
                _windowLeft = value;
                OnPropertyChanged(nameof(WindowLeft));
            }
        }
        
        public double WindowTop
        {
            get => _windowTop;
            set
            {
                _windowTop = value;
                OnPropertyChanged(nameof(WindowTop));
            }
        }
        
        public double LoadingProgress
        {
            get => _loadingProgress;
            set
            {
                _loadingProgress = value;
                OnPropertyChanged(nameof(LoadingProgress));
            }
        }
        
        public bool IsProgressIndeterminate
        {
            get => _isProgressIndeterminate;
            set
            {
                _isProgressIndeterminate = value;
                OnPropertyChanged(nameof(IsProgressIndeterminate));
            }
        }
        
        public string LoadingProgressText
        {
            get => _loadingProgressText;
            set
            {
                _loadingProgressText = value;
                OnPropertyChanged(nameof(LoadingProgressText));
            }
        }
        
        public string CrosshairPriceText
        {
            get => _crosshairPriceText;
            set
            {
                _crosshairPriceText = value;
                OnPropertyChanged(nameof(CrosshairPriceText));
            }
        }

        #endregion

        #region Constructor

        public CandlestickChartModal(string symbol, AlphaVantageService alphaVantageService, LoggingService loggingService, StockDataCacheService stockDataCacheService = null, UserSettingsService userSettingsService = null, TechnicalIndicatorService technicalIndicatorService = null)
        {
            // CRITICAL: Assign services BEFORE InitializeComponent to prevent null reference
            // InitializeComponent triggers XAML binding events (like IntervalComboBox_SelectionChanged)
            _symbol = symbol;
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _loggingService = loggingService;
            
            // Initialize UserSettingsService
            _userSettingsService = userSettingsService ?? new UserSettingsService(
                new QuantraDbContext(new DbContextOptionsBuilder<QuantraDbContext>()
                    .UseSqlServer(ConnectionHelper.ConnectionString).Options),
                loggingService);
            
            // Initialize StockDataCacheService - create if not provided
            _stockDataCacheService = stockDataCacheService ?? new StockDataCacheService(_userSettingsService, loggingService);
            
            // Initialize TechnicalIndicatorService - create if not provided
            _technicalIndicatorService = technicalIndicatorService ?? new TechnicalIndicatorService(alphaVantageService, _userSettingsService, loggingService);
            
            _cancellationTokenSource = new CancellationTokenSource();
            _requestCancellationTokenSource = new CancellationTokenSource();
            
            // Initialize Polly retry policy with exponential backoff
            _retryPolicy = Policy
                .Handle<Exception>(ex => !(ex is OperationCanceledException))
                .WaitAndRetryAsync(
                    MAX_RETRY_ATTEMPTS,
                    retryAttempt => TimeSpan.FromMilliseconds(INITIAL_RETRY_DELAY_MS * Math.Pow(2, retryAttempt - 1)),
                    onRetry: (exception, timeSpan, retryCount, context) =>
                    {
                        _loggingService?.Log("Warning", $"Retry {retryCount}/{MAX_RETRY_ATTEMPTS} for {_symbol} after {timeSpan.TotalSeconds:F1}s delay. Error: {exception.Message}");
                    });
            
            // Load user preferences for refresh interval
            LoadUserPreferences();
            
            // Now initialize XAML (this may trigger event handlers that need _alphaVantageService)
            InitializeComponent();
            
            DataContext = this;
            
            InitializeChart();
            
            // Load window size/position preferences
            LoadWindowSettings();
            
            // Start with initial data load
            Loaded += async (s, e) => await LoadCandlestickDataAsync();
        }

        #endregion

        #region Initialization

        private void InitializeChart()
        {
            _candlestickSeries = new SeriesCollection();
            _volumeSeries = new SeriesCollection();
            _timeLabels = new List<string>();
            
            IsLoading = false;
            IsDataLoaded = false;
            IsNoData = false;
            
            // Load API usage from today
            LoadApiUsageForToday();
            
            _loggingService?.Log("Info", $"Initialized candlestick chart modal for {_symbol} (Refresh: {_refreshIntervalSeconds}s)");
        }
        
        /// <summary>
        /// Loads window size and position from user settings
        /// </summary>
        private void LoadWindowSettings()
        {
            try
            {
                var settings = _userSettingsService?.GetUserSettings();
                if (settings != null)
                {
                    WindowWidth = settings.CandlestickWindowWidth > 0 ? settings.CandlestickWindowWidth : 1000;
                    WindowHeight = settings.CandlestickWindowHeight > 0 ? settings.CandlestickWindowHeight : 700;
                    WindowLeft = !double.IsNaN(settings.CandlestickWindowLeft) && settings.CandlestickWindowLeft >= 0 
                        ? settings.CandlestickWindowLeft 
                        : (SystemParameters.PrimaryScreenWidth - WindowWidth) / 2;
                    WindowTop = !double.IsNaN(settings.CandlestickWindowTop) && settings.CandlestickWindowTop >= 0 
                        ? settings.CandlestickWindowTop 
                        : (SystemParameters.PrimaryScreenHeight - WindowHeight) / 2;
                    
                    _loggingService?.Log("Info", $"Loaded window settings: {WindowWidth}x{WindowHeight} at ({WindowLeft},{WindowTop})");
                }
                else
                {
                    // Default to center screen
                    WindowLeft = (SystemParameters.PrimaryScreenWidth - WindowWidth) / 2;
                    WindowTop = (SystemParameters.PrimaryScreenHeight - WindowHeight) / 2;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load window settings, using defaults");
                WindowLeft = (SystemParameters.PrimaryScreenWidth - WindowWidth) / 2;
                WindowTop = (SystemParameters.PrimaryScreenHeight - WindowHeight) / 2;
            }
        }
        
        /// <summary>
        /// Saves window size and position to user settings
        /// </summary>
        private void SaveWindowSettings()
        {
            try
            {
                var settings = _userSettingsService?.GetUserSettings();
                if (settings != null)
                {
                    settings.CandlestickWindowWidth = WindowWidth;
                    settings.CandlestickWindowHeight = WindowHeight;
                    settings.CandlestickWindowLeft = WindowLeft;
                    settings.CandlestickWindowTop = WindowTop;
                    _userSettingsService?.SaveUserSettings(settings);
                    
                    _loggingService?.Log("Info", $"Saved window settings: {WindowWidth}x{WindowHeight} at ({WindowLeft},{WindowTop})");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to save window settings");
            }
        }
        
        /// <summary>
        /// Loads user preferences for refresh interval and other settings
        /// </summary>
        private void LoadUserPreferences()
        {
            try
            {
                var settings = _userSettingsService?.GetUserSettings();
                if (settings != null)
                {
                    // Use user's preferred refresh interval (default to 15 seconds if not set)
                    _refreshIntervalSeconds = settings.ChartRefreshIntervalSeconds > 0 
                        ? settings.ChartRefreshIntervalSeconds 
                        : 15;
                    
                    _loggingService?.Log("Info", $"Loaded user preference: Refresh interval = {_refreshIntervalSeconds}s");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load user preferences, using defaults");
                _refreshIntervalSeconds = 15; // Default fallback
            }
        }
        
        /// <summary>
        /// Loads API usage count for today
        /// </summary>
        private void LoadApiUsageForToday()
        {
            try
            {
                var today = DateTime.Today;
                if (_lastApiCallDate.Date != today)
                {
                    // Reset counter for new day
                    _apiCallsToday = 0;
                    _lastApiCallDate = today;
                }
                
                // Get today's API call count from Alpha Vantage service
                _apiCallsToday = _alphaVantageService?.GetAlphaVantageApiUsageCount(DateTime.UtcNow) ?? 0;
                
                OnPropertyChanged(nameof(ApiUsageText));
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load API usage count");
            }
        }

        #endregion

        #region Data Loading

        private async Task LoadCandlestickDataAsync(bool forceRefresh = false)
        {
            // Request deduplication: Cancel any pending request and wait for completion
            if (_currentLoadTask != null && !_currentLoadTask.IsCompleted)
            {
                _loggingService?.Log("Info", $"Cancelling previous request for {_symbol}");
                _requestCancellationTokenSource?.Cancel();
                
                try
                {
                    await _currentLoadTask.ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancelling
                }
                catch (Exception ex)
                {
                    _loggingService?.LogErrorWithContext(ex, "Error waiting for previous request to complete");
                }
            }
            
            // Create new cancellation token for this request
            _requestCancellationTokenSource = new CancellationTokenSource();
            var cancellationToken = _requestCancellationTokenSource.Token;
            
            // Use semaphore to ensure only one request at a time
            await _requestSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
            
            try
            {
                // Store current task for deduplication
                _currentLoadTask = LoadCandlestickDataInternalAsync(forceRefresh, cancellationToken);
                await _currentLoadTask.ConfigureAwait(false);
            }
            finally
            {
                _requestSemaphore.Release();
            }
        }
        
        private async Task LoadCandlestickDataInternalAsync(bool forceRefresh, CancellationToken cancellationToken)
        {
            await Dispatcher.InvokeAsync(() =>
            {
                IsLoading = true;
                IsDataLoaded = false;
                IsNoData = false;
                LoadingProgress = 0;
                IsProgressIndeterminate = false;
                LoadingProgressText = "Preparing request...";
            });
            
            try
            {
                _loggingService?.Log("Info", $"Loading candlestick data for {_symbol} with interval {_currentInterval}");
                
                await Dispatcher.InvokeAsync(() =>
                {
                    LoadingProgress = 10;
                    LoadingProgressText = "Checking cache...";
                });
                
                // Convert interval format for StockDataCacheService (1min, 5min, etc.)
                string timeRange = ConvertIntervalToTimeRange(_currentInterval);
                
                await Dispatcher.InvokeAsync(() =>
                {
                    LoadingProgress = 30;
                    LoadingProgressText = "Fetching data...";
                });
                
                // Execute with retry policy and cancellation support
                var historicalData = await _retryPolicy.ExecuteAsync(async (ct) =>
                {
                    ct.ThrowIfCancellationRequested();
                    
                    // Get data from StockDataCacheService - it handles caching automatically
                    var data = await Task.Run(async () => 
                        await _stockDataCacheService.GetStockData(
                            _symbol, 
                            timeRange, 
                            _currentInterval, 
                            forceRefresh).ConfigureAwait(false),
                        ct
                    ).ConfigureAwait(false);
                    
                    // Track API call if it wasn't from cache
                    if (forceRefresh || data == null)
                    {
                        IncrementApiCallCount();
                    }
                    
                    return data;
                }, cancellationToken).ConfigureAwait(false);
                
                await Dispatcher.InvokeAsync(() =>
                {
                    LoadingProgress = 60;
                    LoadingProgressText = "Processing data...";
                });
                
                cancellationToken.ThrowIfCancellationRequested();
                
                if (historicalData == null || historicalData.Count == 0)
                {
                    _loggingService?.Log("Warning", $"No candlestick data available for {_symbol}");
                    await Dispatcher.InvokeAsync(() =>
                    {
                        IsNoData = true;
                        IsLoading = false;
                    });
                    return;
                }
                
                await Dispatcher.InvokeAsync(() =>
                {
                    LoadingProgress = 80;
                    LoadingProgressText = "Rendering chart...";
                });
                
                // Store in local cache for UI updates
                _cachedData = historicalData;
                _cacheTimestamp = DateTime.Now;
                _cachedInterval = _currentInterval;
                
                // Update chart on UI thread
                await UpdateChartAsync(historicalData).ConfigureAwait(false);
                
                // Show "Complete!" message briefly before hiding loading indicator
                await Dispatcher.InvokeAsync(() =>
                {
                    LoadingProgress = 100;
                    LoadingProgressText = "Complete!";
                    _lastUpdateTime = DateTime.Now;
                    IsDataLoaded = true;
                    IsNoData = false;
                    
                    OnPropertyChanged(nameof(LastUpdateText));
                    OnPropertyChanged(nameof(StatusText));
                    OnPropertyChanged(nameof(PriceChangeColor));
                    OnPropertyChanged(nameof(IsCacheValid));
                    OnPropertyChanged(nameof(CacheStatusText));
                });
                
                // Brief delay to show "Complete!" message (500ms)
                await Task.Delay(500, cancellationToken);
                
                // Hide loading indicator
                await Dispatcher.InvokeAsync(() =>
                {
                    IsLoading = false;
                });
                
                _loggingService?.Log("Info", $"Successfully loaded {historicalData.Count} candles for {_symbol} (API calls today: {_apiCallsToday})");
            }
            catch (OperationCanceledException)
            {
                _loggingService?.Log("Info", $"Request cancelled for {_symbol}");
                await Dispatcher.InvokeAsync(() => IsLoading = false);
                throw;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to load candlestick data for {_symbol} after {MAX_RETRY_ATTEMPTS} retries");
                await Dispatcher.InvokeAsync(() =>
                {
                    IsNoData = true;
                    IsLoading = false;
                });
            }
        }
        
        /// <summary>
        /// Increments API call counter and updates UI
        /// </summary>
        private void IncrementApiCallCount()
        {
            try
            {
                // Reset counter if new day
                if (_lastApiCallDate.Date != DateTime.Today)
                {
                    _apiCallsToday = 0;
                    _lastApiCallDate = DateTime.Today;
                }
                
                _apiCallsToday++;
                _alphaVantageService?.LogApiUsage();
                
                Dispatcher.InvokeAsync(() => OnPropertyChanged(nameof(ApiUsageText)));
                
                _loggingService?.Log("Info", $"API call #{_apiCallsToday} today for {_symbol}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to increment API call count");
            }
        }
        
        /// <summary>
        /// Converts interval format to time range for StockDataCacheService
        /// </summary>
        private string ConvertIntervalToTimeRange(string interval)
        {
            // Map interval to appropriate time range
            return interval switch
            {
                "1min" => "1d",      // 1-minute data: last 1 day
                "5min" => "5d",      // 5-minute data: last 5 days
                "15min" => "1mo",    // 15-minute data: last month
                "30min" => "1mo",    // 30-minute data: last month
                "60min" => "2mo",    // 60-minute data: last 2 months
                _ => "1mo"           // Default: 1 month
            };
        }

        private async Task UpdateChartAsync(List<HistoricalPrice> historicalData)
        {
            // Process data on background thread
            var (sortedData, candleValues, volumeValues, labels, volumeColors, gaps) = await Task.Run(() =>
            {
                // Sort by date ascending
                var sorted = historicalData.OrderBy(h => h.Date).ToList();
                
                // Limit to configured max candles for performance
                if (_maxCandles > 0 && sorted.Count > _maxCandles)
                {
                    sorted = sorted.Skip(sorted.Count - _maxCandles).ToList();
                }
                
                // Create chart data with enhanced features
                var candles = new ChartValues<OhlcPoint>();
                var volumes = new ChartValues<double>();
                var timeLabels = new List<string>();
                var volColors = new List<Brush>();
                var gapIndices = new List<int>();
                
                DateTime? previousDate = null;
                
                for (int i = 0; i < sorted.Count; i++)
                {
                    var candle = sorted[i];
                    
                    // Add OHLCV data
                    candles.Add(new OhlcPoint(candle.Open, candle.High, candle.Low, candle.Close));
                    volumes.Add(candle.Volume);
                    
                    // Enhanced time labels with date when day changes
                    string timeLabel;
                    if (previousDate == null || previousDate.Value.Date != candle.Date.Date)
                    {
                        // Show date and time when day changes
                        timeLabel = candle.Date.ToString("MM/dd\nHH:mm");
                    }
                    else
                    {
                        // Just show time for same day
                        timeLabel = candle.Date.ToString("HH:mm");
                    }
                    timeLabels.Add(timeLabel);
                    
                    // Dynamic volume coloring based on price action
                    bool isPriceUp = candle.Close >= candle.Open;
                    var volumeColor = isPriceUp 
                        ? new SolidColorBrush(Color.FromArgb(128, 0x20, 0xC0, 0x40)) // Green (buying pressure)
                        : new SolidColorBrush(Color.FromArgb(128, 0xC0, 0x20, 0x20)); // Red (selling pressure)
                    volColors.Add(volumeColor);
                    
                    // Detect gaps (significant time jumps between candles)
                    if (previousDate != null)
                    {
                        var expectedInterval = GetExpectedIntervalMinutes(_currentInterval);
                        var actualInterval = (candle.Date - previousDate.Value).TotalMinutes;
                        
                        // If gap is more than 2x the expected interval, mark it
                        if (actualInterval > expectedInterval * 2)
                        {
                            gapIndices.Add(i);
                        }
                    }
                    
                    previousDate = candle.Date;
                }
                
                return (sorted, candles, volumes, timeLabels, volColors, gapIndices);
            }).ConfigureAwait(false);
            
            // Update UI on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                UpdateChartWithData(sortedData, candleValues, volumeValues, labels, volumeColors, gaps);
                UpdatePriceInfo(sortedData);
                UpdateIndicators(); // Refresh technical indicators
            });
        }
        
        private int GetExpectedIntervalMinutes(string interval)
        {
            return interval switch
            {
                "1min" => 1,
                "5min" => 5,
                "15min" => 15,
                "30min" => 30,
                "60min" => 60,
                _ => 5
            };
        }

        private void UpdateChartWithData(List<HistoricalPrice> data, ChartValues<OhlcPoint> candleValues, ChartValues<double> volumeValues, List<string> labels, List<Brush> volumeColors, List<int> gapIndices)
        {
            // Create candlestick series with tooltips
            var candleSeries = new CandleSeries
            {
                Title = _symbol,
                Values = candleValues,
                MaxColumnWidth = 20,
                IncreaseBrush = new SolidColorBrush(Color.FromRgb(0x20, 0xC0, 0x40)), // Green
                DecreaseBrush = new SolidColorBrush(Color.FromRgb(0xC0, 0x20, 0x20)),  // Red
                LabelPoint = point =>
                {
                    // Enhanced tooltip with OHLCV details
                    var ohlc = (OhlcPoint)point.Instance;
                    var index = (int)point.X;
                    if (index >= 0 && index < data.Count)
                    {
                        var candle = data[index];
                        var change = candle.Close - candle.Open;
                        var changePercent = candle.Open != 0 ? (change / candle.Open) * 100 : 0;
                        var direction = change >= 0 ? "?" : "?";
                        
                        // Check if this is a gap candle
                        var gapIndicator = gapIndices.Contains(index) ? " [GAP]" : "";
                        
                        // Check if after-hours (before 9:30 AM or after 4:00 PM ET)
                        var hour = candle.Date.Hour;
                        var afterHoursIndicator = (hour < 9 || (hour == 9 && candle.Date.Minute < 30) || hour >= 16) 
                            ? " [AH]" : "";
                        
                        return $"{candle.Date:MM/dd HH:mm}{gapIndicator}{afterHoursIndicator}\n" +
                               $"Open:  ${ohlc.Open:F2}\n" +
                               $"High:  ${ohlc.High:F2}\n" +
                               $"Low:   ${ohlc.Low:F2}\n" +
                               $"Close: ${ohlc.Close:F2}\n" +
                               $"Volume: {VolumeFormatter(candle.Volume)}\n" +
                               $"Change: {direction} ${Math.Abs(change):F2} ({changePercent:+0.00;-0.00;0.00}%)";
                    }
                    return string.Empty;
                }
            };
            
            CandlestickSeries = new SeriesCollection { candleSeries };
            
            // Create volume series with dynamic coloring
            var volumeSeries = new SeriesCollection();
            
            // If we have individual colors for each volume bar, create separate series
            // (LiveCharts limitation: can't color individual bars in one series easily)
            // For simplicity, we'll use two series: one for up volumes, one for down volumes
            var upVolumes = new ChartValues<double>();
            var downVolumes = new ChartValues<double>();
            
            for (int i = 0; i < volumeValues.Count; i++)
            {
                bool isUp = i < data.Count && data[i].Close >= data[i].Open;
                
                if (isUp)
                {
                    upVolumes.Add(volumeValues[i]);
                    downVolumes.Add(0);
                }
                else
                {
                    upVolumes.Add(0);
                    downVolumes.Add(volumeValues[i]);
                }
            }
            
            volumeSeries.Add(new ColumnSeries
            {
                Title = "Buy Volume",
                Values = upVolumes,
                Fill = new SolidColorBrush(Color.FromArgb(128, 0x20, 0xC0, 0x40)), // Green
                MaxColumnWidth = 20,
                LabelPoint = point => 
                {
                    if (point.Y > 0)
                    {
                        var index = (int)point.X;
                        if (index >= 0 && index < data.Count)
                        {
                            return $"{data[index].Date:HH:mm}\nVolume: {VolumeFormatter(point.Y)}\n(Buying Pressure)";
                        }
                    }
                    return string.Empty;
                }
            });
            
            volumeSeries.Add(new ColumnSeries
            {
                Title = "Sell Volume",
                Values = downVolumes,
                Fill = new SolidColorBrush(Color.FromArgb(128, 0xC0, 0x20, 0x20)), // Red
                MaxColumnWidth = 20,
                LabelPoint = point => 
                {
                    if (point.Y > 0)
                    {
                        var index = (int)point.X;
                        if (index >= 0 && index < data.Count)
                        {
                            return $"{data[index].Date:HH:mm}\nVolume: {VolumeFormatter(point.Y)}\n(Selling Pressure)";
                        }
                    }
                    return string.Empty;
                }
            });
            
            VolumeSeries = volumeSeries;
            
            TimeLabels = labels;
            
            // Reset zoom when new data is loaded
            ResetZoom();
        }

        private void UpdatePriceInfo(List<HistoricalPrice> data)
        {
            if (data.Count < 2) return;
            
            var latest = data.Last();
            var previous = data[data.Count - 2];
            
            LastPrice = latest.Close;
            PriceChange = latest.Close - previous.Close;
            PriceChangePercent = previous.Close != 0 
                ? ((latest.Close - previous.Close) / previous.Close) * 100 
                : 0;
        }

        #endregion

        #region Auto-Refresh

        private void StartAutoRefresh()
        {
            if (_isPaused)
            {
                _loggingService?.Log("Info", $"Auto-refresh paused for {_symbol}");
                return;
            }
            
            if (_refreshTimer != null)
            {
                _refreshTimer.Stop();
                _refreshTimer = null;
            }
            
            _refreshTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(_refreshIntervalSeconds)
            };
            _refreshTimer.Tick += async (s, e) =>
            {
                if (!_isPaused)
                {
                    await LoadCandlestickDataAsync();
                }
            };
            _refreshTimer.Start();
            
            _loggingService?.Log("Info", $"Started auto-refresh for {_symbol} with {_refreshIntervalSeconds}s interval");
        }

        private void StopAutoRefresh()
        {
            if (_refreshTimer != null)
            {
                _refreshTimer.Stop();
                _refreshTimer = null;
                _loggingService?.Log("Info", $"Stopped auto-refresh for {_symbol}");
            }
        }

        #endregion

        #region Keyboard Shortcuts
        
        /// <summary>
        /// Handles keyboard shortcuts for the window
        /// </summary>
        private void Window_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            try
            {
                switch (e.Key)
                {
                    case System.Windows.Input.Key.Escape:
                        // ESC to close window
                        Close();
                        break;
                        
                    case System.Windows.Input.Key.F5:
                        // F5 to refresh data
                        if (!IsLoading)
                        {
                            _ = LoadCandlestickDataAsync(forceRefresh: true);
                        }
                        break;
                        
                    case System.Windows.Input.Key.R:
                        // Ctrl+R to toggle auto-refresh
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            IsAutoRefreshEnabled = !IsAutoRefreshEnabled;
                        }
                        break;
                        
                    case System.Windows.Input.Key.P:
                        // Ctrl+P to pause/resume
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            IsPaused = !IsPaused;
                        }
                        break;
                        
                    case System.Windows.Input.Key.OemPlus:
                    case System.Windows.Input.Key.Add:
                        // +/= to zoom in
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            ZoomInButton_Click(null, null);
                        }
                        break;
                        
                    case System.Windows.Input.Key.OemMinus:
                    case System.Windows.Input.Key.Subtract:
                        // -/_ to zoom out
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            ZoomOutButton_Click(null, null);
                        }
                        break;
                        
                    case System.Windows.Input.Key.D0:
                    case System.Windows.Input.Key.NumPad0:
                        // Ctrl+0 to reset zoom
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            ResetZoomButton_Click(null, null);
                        }
                        break;
                        
                    case System.Windows.Input.Key.I:
                        // Ctrl+I to open interval dialog
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            IntervalComboBox.Focus();
                            IntervalComboBox.IsDropDownOpen = true;
                        }
                        break;
                        
                    case System.Windows.Input.Key.H:
                        // Ctrl+H to add horizontal line
                        if (e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control)
                        {
                            AddHorizontalLineButton_Click(null, null);
                        }
                        break;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error handling keyboard shortcut");
            }
        }
        
        #endregion

        #region Event Handlers

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadCandlestickDataAsync(forceRefresh: true);
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private async void IntervalComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            // Prevent loading data during initialization (before window is fully loaded)
            if (!IsLoaded || _alphaVantageService == null)
                return;
                
            if (IntervalComboBox.SelectedItem is System.Windows.Controls.ComboBoxItem selected)
            {
                var content = selected.Content.ToString();
                var newInterval = content.Replace(" ", "").ToLower();
                
                // Only reload if interval actually changed
                if (newInterval != _currentInterval)
                {
                    _currentInterval = newInterval;
                    _loggingService?.Log("Info", $"Changed interval to {_currentInterval} for {_symbol}");
                    
                    // Invalidate cache and reload data with new interval
                    _cachedInterval = null;
                    await LoadCandlestickDataAsync(forceRefresh: true);
                }
            }
        }

        private async void CandleLimitComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (!IsLoaded || _alphaVantageService == null)
                return;
                
            if (CandleLimitComboBox.SelectedItem is System.Windows.Controls.ComboBoxItem selected)
            {
                var newLimit = int.Parse(selected.Tag.ToString());
                
                if (newLimit != _maxCandles)
                {
                    _maxCandles = newLimit;
                    _loggingService?.Log("Info", $"Changed candle limit to {(_maxCandles == 0 ? "All" : _maxCandles.ToString())} for {_symbol}");
                    
                    // Reload chart with cached data if available
                    if (_cachedData != null)
                    {
                        await UpdateChartAsync(_cachedData).ConfigureAwait(false);
                    }
                }
            }
        }

        private void ZoomInButton_Click(object sender, RoutedEventArgs e)
        {
            _zoomLevel *= 0.8; // Zoom in by 20%
            ApplyZoom();
        }

        private void ZoomOutButton_Click(object sender, RoutedEventArgs e)
        {
            _zoomLevel *= 1.2; // Zoom out by 20%
            ApplyZoom();
        }

        private void ResetZoomButton_Click(object sender, RoutedEventArgs e)
        {
            ResetZoom();
        }

        private void ApplyZoom()
        {
            if (_timeLabels == null || _timeLabels.Count == 0)
                return;

            int visibleCandles = (int)(_timeLabels.Count * _zoomLevel);
            if (visibleCandles < 10) visibleCandles = 10;
            if (visibleCandles > _timeLabels.Count) visibleCandles = _timeLabels.Count;

            int startIndex = Math.Max(0, _timeLabels.Count - visibleCandles);
            int endIndex = _timeLabels.Count - 1;

            XAxisMin = startIndex;
            XAxisMax = endIndex;

            _loggingService?.Log("Info", $"Applied zoom level {_zoomLevel:F2} (showing {visibleCandles} candles)");
        }

        private void ResetZoom()
        {
            _zoomLevel = 1.0;
            XAxisMin = null;
            XAxisMax = null;
            YAxisMin = null;
            YAxisMax = null;
        }
        
        /// <summary>
        /// Tracks mouse position on chart to display crosshair price
        /// </summary>
        private void CandlestickChart_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
        {
            try
            {
                if (CandlestickChart?.Model == null || !IsDataLoaded)
                {
                    CrosshairPriceText = "";
                    return;
                }
                
                var position = e.GetPosition(CandlestickChart);
                
                    // Get the price axis (Y axis)
                var priceAxis = CandlestickChart.AxisY.FirstOrDefault();
                if (priceAxis != null)
                {
                    // Convert mouse Y position to price value
                    var chartHeight = CandlestickChart.ActualHeight;
                    
                    // Use the axis limits to convert position to price
                    var minValue = priceAxis.MinValue;
                    var maxValue = priceAxis.MaxValue;
                    
                    if (minValue != 0 || maxValue != 0)
                    {
                        var range = maxValue - minValue;
                        
                        // Estimate margins (LiveCharts doesn't expose DrawMargin directly)
                        var estimatedTopMargin = 30.0;
                        var estimatedBottomMargin = 50.0;
                        var chartAreaHeight = chartHeight - estimatedTopMargin - estimatedBottomMargin;
                        
                        var relativeY = position.Y - estimatedTopMargin;
                        
                        if (relativeY >= 0 && relativeY <= chartAreaHeight)
                        {
                            // Invert Y axis (0 is at top)
                            var priceValue = maxValue - (relativeY / chartAreaHeight * range);
                            
                            CrosshairPriceText = $"? Price: ${priceValue:F2}";
                        }
                        else
                        {
                            CrosshairPriceText = "";
                        }
                    }
                    else
                    {
                        CrosshairPriceText = "";
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error updating crosshair price");
                CrosshairPriceText = "";
            }
        }

        #endregion
        
        #region Technical Indicators
        
        private void UpdateIndicators()
        {
            if (_cachedData == null || _cachedData.Count == 0 || CandlestickSeries == null)
                return;
                
            try
            {
                // Remove old indicator series
                var indicatorsToRemove = CandlestickSeries.Where(s => 
                    s.Title == "SMA" || s.Title == "EMA" || s.Title == "VWAP" || 
                    s.Title.StartsWith("BB")).ToList();
                    
                foreach (var indicator in indicatorsToRemove)
                {
                    CandlestickSeries.Remove(indicator);
                }
                
                var sortedData = _cachedData.OrderBy(h => h.Date).ToList();
                var closePrices = sortedData.Select(p => p.Close).ToList();
                
                // Add SMA if enabled
                if (ShowSMA)
                {
                    var sma = _technicalIndicatorService.CalculateSMA(closePrices, 20);
                    if (sma != null && sma.Count > 0)
                    {
                        var smaValues = new ChartValues<double>();
                        for (int i = 0; i < sma.Count; i++)
                        {
                            smaValues.Add(double.IsNaN(sma[i]) ? 0 : sma[i]);
                        }
                        
                        CandlestickSeries.Add(new LineSeries
                        {
                            Title = "SMA",
                            Values = smaValues,
                            Stroke = System.Windows.Media.Brushes.Orange,
                            Fill = System.Windows.Media.Brushes.Transparent,
                            StrokeThickness = 2,
                            PointGeometry = null
                        });
                    }
                }
                
                // Add EMA if enabled
                if (ShowEMA)
                {
                    var ema = _technicalIndicatorService.CalculateEMA(closePrices, 20);
                    if (ema != null && ema.Count > 0)
                    {
                        var emaValues = new ChartValues<double>();
                        for (int i = 0; i < ema.Count; i++)
                        {
                            emaValues.Add(double.IsNaN(ema[i]) ? 0 : ema[i]);
                        }
                        
                        CandlestickSeries.Add(new LineSeries
                        {
                            Title = "EMA",
                            Values = emaValues,
                            Stroke = System.Windows.Media.Brushes.Cyan,
                            Fill = System.Windows.Media.Brushes.Transparent,
                            StrokeThickness = 2,
                            PointGeometry = null
                        });
                    }
                }
                
                // Add VWAP if enabled
                if (ShowVWAP)
                {
                    var highPrices = sortedData.Select(p => p.High).ToList();
                    var lowPrices = sortedData.Select(p => p.Low).ToList();
                    var volumes = sortedData.Select(p => (double)p.Volume).ToList();
                    
                    var vwap = _technicalIndicatorService.CalculateVWAP(highPrices, lowPrices, closePrices, volumes);
                    if (vwap != null && vwap.Count > 0)
                    {
                        var vwapValues = new ChartValues<double>();
                        for (int i = 0; i < vwap.Count; i++)
                        {
                            vwapValues.Add(vwap[i]);
                        }
                        
                        CandlestickSeries.Add(new LineSeries
                        {
                            Title = "VWAP",
                            Values = vwapValues,
                            Stroke = System.Windows.Media.Brushes.Yellow,
                            Fill = System.Windows.Media.Brushes.Transparent,
                            StrokeThickness = 2,
                            PointGeometry = null
                        });
                    }
                }
                
                // Add Bollinger Bands if enabled
                if (ShowBollingerBands)
                {
                    var bb = _technicalIndicatorService.CalculateBollingerBands(closePrices, 20, 2.0);
                    if (bb.Upper != null && bb.Middle != null && bb.Lower != null)
                    {
                        var upperValues = new ChartValues<double>();
                        var middleValues = new ChartValues<double>();
                        var lowerValues = new ChartValues<double>();
                        
                        for (int i = 0; i < bb.Upper.Count; i++)
                        {
                            upperValues.Add(double.IsNaN(bb.Upper[i]) ? 0 : bb.Upper[i]);
                            middleValues.Add(double.IsNaN(bb.Middle[i]) ? 0 : bb.Middle[i]);
                            lowerValues.Add(double.IsNaN(bb.Lower[i]) ? 0 : bb.Lower[i]);
                        }
                        
                        CandlestickSeries.Add(new LineSeries
                        {
                            Title = "BB Upper",
                            Values = upperValues,
                            Stroke = new SolidColorBrush(Color.FromArgb(128, 128, 128, 255)),
                            Fill = System.Windows.Media.Brushes.Transparent,
                            StrokeThickness = 1,
                            StrokeDashArray = new System.Windows.Media.DoubleCollection(new[] { 2.0, 2.0 }),
                            PointGeometry = null
                        });
                        
                        CandlestickSeries.Add(new LineSeries
                        {
                            Title = "BB Middle",
                            Values = middleValues,
                            Stroke = new SolidColorBrush(Color.FromArgb(128, 128, 128, 200)),
                            Fill = System.Windows.Media.Brushes.Transparent,
                            StrokeThickness = 1,
                            PointGeometry = null
                        });
                        
                        CandlestickSeries.Add(new LineSeries
                        {
                            Title = "BB Lower",
                            Values = lowerValues,
                            Stroke = new SolidColorBrush(Color.FromArgb(128, 128, 128, 255)),
                            Fill = System.Windows.Media.Brushes.Transparent,
                            StrokeThickness = 1,
                            StrokeDashArray = new System.Windows.Media.DoubleCollection(new[] { 2.0, 2.0 }),
                            PointGeometry = null
                        });
                    }
                }
                
                // Note: RSI and MACD would typically be displayed in separate panels below the main chart
                // This requires more complex layout changes that are beyond the scope of this basic implementation
                
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to update technical indicators");
            }
        }
        
        #endregion
        
        #region Drawing Tools
        
        private void AddHorizontalLineButton_Click(object sender, RoutedEventArgs e)
        {
            // Prompt user for price level
            var dialog = new HorizontalLineDialog();
            dialog.Owner = this;
            
            if (dialog.ShowDialog() == true)
            {
                double priceLevel = dialog.PriceLevel;
                string label = dialog.Label;
                
                AddHorizontalLine(priceLevel, label);
            }
        }
        
        private void AddHorizontalLine(double priceLevel, string label)
        {
            try
            {
                // Create a horizontal line series
                var lineValues = new ChartValues<double>();
                
                // Add the same price level for all x-axis points
                int pointCount = _timeLabels?.Count ?? 100;
                for (int i = 0; i < pointCount; i++)
                {
                    lineValues.Add(priceLevel);
                }
                
                var lineSeries = new LineSeries
                {
                    Title = label ?? $"Level {priceLevel:F2}",
                    Values = lineValues,
                    Stroke = System.Windows.Media.Brushes.Red,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    StrokeThickness = 2,
                    StrokeDashArray = new System.Windows.Media.DoubleCollection(new[] { 4.0, 2.0 }),
                    PointGeometry = null
                };
                
                CandlestickSeries.Add(lineSeries);
                
                _loggingService?.Log("Info", $"Added horizontal line at {priceLevel:F2}: {label}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to add horizontal line");
            }
        }
        
        private void ClearDrawingsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Remove all drawn lines (series with "Level" in the title)
                var linesToRemove = CandlestickSeries
                    .Where(s => s.Title.StartsWith("Level ") || s.Title.Contains("Support") || s.Title.Contains("Resistance"))
                    .ToList();
                    
                foreach (var line in linesToRemove)
                {
                    CandlestickSeries.Remove(line);
                }
                
                _drawnLines.Clear();
                
                _loggingService?.Log("Info", "Cleared all drawn lines");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to clear drawn lines");
            }
        }
        
        #endregion

        #region Cleanup

        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            IsPaused = !IsPaused;
            _loggingService?.Log("Info", $"Chart {(IsPaused ? "paused" : "resumed")} for {_symbol}");
        }
        
        private async void ChangeRefreshIntervalButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new RefreshIntervalDialog(_refreshIntervalSeconds);
            dialog.Owner = this;
            
            if (dialog.ShowDialog() == true)
            {
                var newInterval = dialog.SelectedInterval;
                if (newInterval != _refreshIntervalSeconds)
                {
                    _refreshIntervalSeconds = newInterval;
                    
                    // Save to user preferences
                    try
                    {
                        var settings = _userSettingsService?.GetUserSettings();
                        if (settings != null)
                        {
                            settings.ChartRefreshIntervalSeconds = newInterval;
                            _userSettingsService?.SaveUserSettings(settings);
                        }
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.LogErrorWithContext(ex, "Failed to save refresh interval preference");
                    }
                    
                    // Restart timer with new interval
                    if (IsAutoRefreshEnabled && !IsPaused)
                    {
                        StopAutoRefresh();
                        StartAutoRefresh();
                    }
                    
                    OnPropertyChanged(nameof(ApiUsageText));
                    _loggingService?.Log("Info", $"Changed refresh interval to {newInterval}s for {_symbol}");
                }
            }
        }

        #endregion

        #region Cleanup

        protected override void OnClosing(CancelEventArgs e)
        {
            // Save window size/position
            SaveWindowSettings();
            
            StopAutoRefresh();
            
            // Cancel any pending requests
            _requestCancellationTokenSource?.Cancel();
            _cancellationTokenSource?.Cancel();
            
            // Wait for current request to complete (with timeout)
            if (_currentLoadTask != null && !_currentLoadTask.IsCompleted)
            {
                try
                {
                    Task.WaitAny(_currentLoadTask, Task.Delay(2000));
                }
                catch
                {
                    // Ignore exceptions during shutdown
                }
            }
            
            // Dispose resources
            _requestCancellationTokenSource?.Dispose();
            _cancellationTokenSource?.Dispose();
            _requestSemaphore?.Dispose();
            
            base.OnClosing(e);
        }

        #endregion

        #region INotifyPropertyChanged

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
