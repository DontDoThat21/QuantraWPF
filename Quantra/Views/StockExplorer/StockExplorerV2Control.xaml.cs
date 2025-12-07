using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.DAL.Services;
using Quantra.DAL.Data.Entities;
using Quantra.Models;
using Newtonsoft.Json;

namespace Quantra.Views.StockExplorer
{
    /// <summary>
    /// Stock Explorer V2 - Continuously updating grid and chart visualization for multiple stocks
    /// Integrates with Stock Configuration Manager and PredictionAnalysis features
    /// </summary>
    public partial class StockExplorerV2Control : UserControl, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        // Services
        private readonly AlphaVantageService _alphaVantageService;
        private readonly StockConfigurationService _stockConfigurationService;
        private readonly LoggingService _loggingService;
        private readonly DAL.Services.TFTPredictionService _tftPredictionService;

        // State management
        private DispatcherTimer _updateTimer;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isUpdating = false;
        private List<string> _symbols = new List<string>();

        // Data collections
        private ObservableCollection<StockAnalysisItem> _stockAnalysisData;
        public ObservableCollection<StockAnalysisItem> StockAnalysisData
        {
            get => _stockAnalysisData;
            set
            {
                _stockAnalysisData = value;
                OnPropertyChanged(nameof(StockAnalysisData));
            }
        }

        // Chart properties
        private SeriesCollection _candlestickSeries;
        public SeriesCollection CandlestickSeries
        {
            get => _candlestickSeries;
            set
            {
                _candlestickSeries = value;
                OnPropertyChanged(nameof(CandlestickSeries));
            }
        }

        private SeriesCollection _volumeSeries;
        public SeriesCollection VolumeSeries
        {
            get => _volumeSeries;
            set
            {
                _volumeSeries = value;
                OnPropertyChanged(nameof(VolumeSeries));
            }
        }

        private List<string> _timeLabels;
        public List<string> TimeLabels
        {
            get => _timeLabels;
            set
            {
                _timeLabels = value;
                OnPropertyChanged(nameof(TimeLabels));
            }
        }

        private double _chartMinPrice;
        public double ChartMinPrice
        {
            get => _chartMinPrice;
            set
            {
                _chartMinPrice = value;
                OnPropertyChanged(nameof(ChartMinPrice));
            }
        }

        private double _chartMaxPrice;
        public double ChartMaxPrice
        {
            get => _chartMaxPrice;
            set
            {
                _chartMaxPrice = value;
                OnPropertyChanged(nameof(ChartMaxPrice));
            }
        }

        private ObservableCollection<FeatureAttentionItem> _featureAttentionData;
        public ObservableCollection<FeatureAttentionItem> FeatureAttentionData
        {
            get => _featureAttentionData;
            set
            {
                _featureAttentionData = value;
                OnPropertyChanged(nameof(FeatureAttentionData));
            }
        }

        public Func<double, string> PriceFormatter { get; set; }
        public Func<double, string> VolumeFormatter { get; set; }

        private bool _isChartVisible = false;
        public bool IsChartVisible
        {
            get => _isChartVisible;
            set
            {
                _isChartVisible = value;
                OnPropertyChanged(nameof(IsChartVisible));
            }
        }

        public StockExplorerV2Control()
        {
            InitializeComponent();
            DataContext = this;

            // Initialize services from DI container
            _alphaVantageService = App.ServiceProvider?.GetService(typeof(AlphaVantageService)) as AlphaVantageService;
            _stockConfigurationService = App.ServiceProvider?.GetService(typeof(StockConfigurationService)) as StockConfigurationService;
            _loggingService = App.ServiceProvider?.GetService(typeof(LoggingService)) as LoggingService;

            // Initialize TFT prediction service
            _tftPredictionService = new DAL.Services.TFTPredictionService(_loggingService, _alphaVantageService);

            // Initialize collections
            StockAnalysisData = new ObservableCollection<StockAnalysisItem>();
            CandlestickSeries = new SeriesCollection();
            VolumeSeries = new SeriesCollection();
            TimeLabels = new List<string>();
            FeatureAttentionData = new ObservableCollection<FeatureAttentionItem>();

            // Initialize chart formatters
            PriceFormatter = value => value.ToString("C2");
            VolumeFormatter = value => value >= 1000000
                ? $"{value / 1000000:F1}M"
                : value >= 1000
                    ? $"{value / 1000:F1}K"
                    : value.ToString("N0");

            // Initialize timer
            _updateTimer = new DispatcherTimer();
            _updateTimer.Tick += UpdateTimer_Tick;

            // Check service availability
            if (_alphaVantageService == null)
            {
                StatusText.Text = "Error: AlphaVantageService not available";
                StatusText.Foreground = System.Windows.Media.Brushes.Red;
                StartButton.IsEnabled = false;
            }

            if (_stockConfigurationService == null)
            {
                StatusText.Text = "Error: StockConfigurationService not available";
                StatusText.Foreground = System.Windows.Media.Brushes.Red;
                LoadConfigurationButton.IsEnabled = false;
            }

            _loggingService?.Log("Info", "StockExplorerV2Control initialized");
        }

        private void LoadConfigurationButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_stockConfigurationService == null)
                {
                    StatusText.Text = "Error: StockConfigurationService not available";
                    StatusText.Foreground = System.Windows.Media.Brushes.Red;
                    return;
                }

                var symbols = StockConfigurationManagerWindow.ShowAndGetSymbols(_stockConfigurationService, Window.GetWindow(this));
                
                if (symbols != null && symbols.Count > 0)
                {
                    _symbols = symbols;
                    
                    ConfigurationNameText.Text = $"Configuration loaded ({_symbols.Count} symbols)";
                    ConfigurationNameText.Foreground = System.Windows.Media.Brushes.LimeGreen;
                    ConfigurationNameText.FontStyle = FontStyles.Normal;
                    
                    SymbolCountText.Text = $"Symbols: {_symbols.Count}";
                    
                    StatusText.Text = $"Loaded {_symbols.Count} symbols from configuration";
                    StatusText.Foreground = System.Windows.Media.Brushes.Cyan;
                    
                    StartButton.IsEnabled = true;
                    
                    _loggingService?.Log("Info", $"Loaded stock configuration with {_symbols.Count} symbols");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error loading stock configuration");
                StatusText.Text = $"Error: {ex.Message}";
                StatusText.Foreground = System.Windows.Media.Brushes.Red;
            }
        }

        private void UpdateIntervalComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (UpdateIntervalComboBox.SelectedItem is ComboBoxItem item && item.Tag != null)
            {
                if (int.TryParse(item.Tag.ToString(), out int seconds))
                {
                    if (_updateTimer != null)
                    {
                        _updateTimer.Interval = TimeSpan.FromSeconds(seconds);
                        _loggingService?.Log("Info", $"Update interval changed to {seconds} seconds");
                    }
                }
            }
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (_symbols == null || _symbols.Count == 0)
            {
                StatusText.Text = "Please load a configuration first";
                StatusText.Foreground = System.Windows.Media.Brushes.Orange;
                return;
            }

            StartUpdates();
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            StopUpdates();
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            if (_symbols == null || _symbols.Count == 0)
            {
                StatusText.Text = "Please load a configuration first";
                StatusText.Foreground = System.Windows.Media.Brushes.Orange;
                return;
            }

            await UpdateAllSymbols();
        }

        private async void StockDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (StockDataGrid.SelectedItem is StockAnalysisItem selectedItem)
            {
                // Show chart
                IsChartVisible = true;

                // Update selected symbol text
                SelectedSymbolText.Text = $"- {selectedItem.Symbol}";

                // Load chart data for selected symbol
                await LoadChartDataForSymbol(selectedItem.Symbol);
            }
        }

        private async Task LoadChartDataForSymbol(string symbol)
        {
            try
            {
                StatusText.Text = $"Loading chart data for {symbol}...";
                StatusText.Foreground = System.Windows.Media.Brushes.Yellow;

                // Fetch historical OHLC data (last 60 days for candlestick chart)
                var historicalData = await FetchOHLCDataAsync(symbol, days: 60);

                if (historicalData == null || historicalData.Count == 0)
                {
                    StatusText.Text = $"No historical data available for {symbol}";
                    StatusText.Foreground = System.Windows.Media.Brushes.Orange;
                    return;
                }

                // Get TFT predictions with confidence bands
                var tftPredictions = await GetTFTPredictionsAsync(symbol, historicalData);

                // Update charts
                UpdateCandlestickChart(historicalData, tftPredictions);
                UpdateVolumeChart(historicalData);

                // Update feature attention if TFT predictions include attention weights
                if (tftPredictions?.FeatureAttention != null)
                {
                    UpdateFeatureAttention(tftPredictions.FeatureAttention);
                }

                StatusText.Text = $"Chart loaded for {symbol}";
                StatusText.Foreground = System.Windows.Media.Brushes.LimeGreen;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error loading chart for {symbol}");
                StatusText.Text = $"Error loading chart: {ex.Message}";
                StatusText.Foreground = System.Windows.Media.Brushes.Red;
            }
        }

        private void StartUpdates()
        {
            try
            {
                _cancellationTokenSource = new CancellationTokenSource();
                
                // Get update interval
                int intervalSeconds = 60; // Default
                if (UpdateIntervalComboBox.SelectedItem is ComboBoxItem item && item.Tag != null)
                {
                    int.TryParse(item.Tag.ToString(), out intervalSeconds);
                }
                
                _updateTimer.Interval = TimeSpan.FromSeconds(intervalSeconds);
                _updateTimer.Start();
                
                StartButton.IsEnabled = false;
                StopButton.IsEnabled = true;
                LoadConfigurationButton.IsEnabled = false;
                
                StatusText.Text = $"Updates started - refreshing every {intervalSeconds} seconds";
                StatusText.Foreground = System.Windows.Media.Brushes.LimeGreen;
                
                _loggingService?.Log("Info", $"Started automatic updates for {_symbols.Count} symbols");
                
                // Trigger immediate update
                Task.Run(() => UpdateAllSymbols());
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error starting updates");
                StatusText.Text = $"Error: {ex.Message}";
                StatusText.Foreground = System.Windows.Media.Brushes.Red;
            }
        }

        private void StopUpdates()
        {
            try
            {
                _updateTimer.Stop();
                _cancellationTokenSource?.Cancel();
                
                StartButton.IsEnabled = true;
                StopButton.IsEnabled = false;
                LoadConfigurationButton.IsEnabled = true;
                
                StatusText.Text = "Updates stopped";
                StatusText.Foreground = System.Windows.Media.Brushes.Orange;
                
                _loggingService?.Log("Info", "Stopped automatic updates");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error stopping updates");
            }
        }

        private async void UpdateTimer_Tick(object sender, EventArgs e)
        {
            if (!_isUpdating)
            {
                await UpdateAllSymbols();
            }
        }

        private async Task<List<OHLCDataPoint>> FetchOHLCDataAsync(string symbol, int days = 60)
        {
            try
            {
                var dataPoints = new List<OHLCDataPoint>();

                // Fetch historical daily data from Alpha Vantage
                var historicalData = await _alphaVantageService.GetDailyData(
                    symbol,
                    outputSize: "compact" // Last 100 data points
                );

                if (historicalData == null || historicalData.Count == 0)
                    return dataPoints;

                // Take the last N days and reverse to chronological order
                var recentData = historicalData
                    .OrderByDescending(d => d.Date)
                    .Take(days)
                    .OrderBy(d => d.Date)
                    .ToList();

                foreach (var data in recentData)
                {
                    dataPoints.Add(new OHLCDataPoint
                    {
                        Timestamp = data.Date,
                        Open = data.Open,
                        High = data.High,
                        Low = data.Low,
                        Close = data.Close,
                        Volume = data.Volume
                    });
                }

                return dataPoints;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Error fetching OHLC data for {symbol}: {ex.Message}");
                return new List<OHLCDataPoint>();
            }
        }

        private async Task<TFTPredictionResult> GetTFTPredictionsAsync(string symbol, List<OHLCDataPoint> historicalData)
        {
            try
            {
                if (_tftPredictionService == null)
                {
                    _loggingService?.Log("Warning", "TFT prediction service not available");
                    return null;
                }

                // Convert our OHLCDataPoint to HistoricalPrice
                var historicalPrices = historicalData.Select(d => new Quantra.Models.HistoricalPrice
                {
                    Date = d.Timestamp,
                    Open = d.Open,
                    High = d.High,
                    Low = d.Low,
                    Close = d.Close,
                    Volume = d.Volume,
                    AdjClose = d.Close // Use close as adj close if not available
                }).ToList();

                // Call TFT prediction service
                var tftResult = await _tftPredictionService.GetTFTPredictionsAsync(
                    symbol,
                    historicalPrices,
                    lookbackDays: 60,
                    futureHorizon: 30,
                    forecastHorizons: new List<int> { 5, 10, 20, 30 }
                );

                if (tftResult == null)
                    return null;

                // Convert service result to our UI model
                return new TFTPredictionResult
                {
                    Predictions = tftResult.Predictions?.Select(p => new PredictionPoint
                    {
                        Timestamp = p.Timestamp,
                        PredictedPrice = p.PredictedPrice,
                        UpperConfidence = p.UpperConfidence,
                        LowerConfidence = p.LowerConfidence
                    }).ToList() ?? new List<PredictionPoint>(),
                    FeatureAttention = tftResult.FeatureAttention ?? new Dictionary<string, double>()
                };
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Error getting TFT predictions for {symbol}: {ex.Message}");
                return null;
            }
        }

        private void UpdateCandlestickChart(List<OHLCDataPoint> historicalData, TFTPredictionResult tftPredictions)
        {
            try
            {
                CandlestickSeries.Clear();
                TimeLabels = new List<string>();

                if (historicalData == null || historicalData.Count == 0)
                {
                    // Set default chart range to prevent errors and hide chart
                    ChartMinPrice = 0;
                    ChartMaxPrice = 100;
                    IsChartVisible = false;
                    _loggingService?.Log("Warning", "No historical data available for chart");
                    return;
                }

                // Ensure we have at least 2 data points for a valid chart
                if (historicalData.Count < 2)
                {
                    _loggingService?.Log("Warning", $"Insufficient data points ({historicalData.Count}) for chart");
                    IsChartVisible = false;
                    return;
                }

            // Create OHLC series using LineSeries for high-low and custom rendering
            var closeValues = new ChartValues<double>();
            var highValues = new ChartValues<double>();
            var lowValues = new ChartValues<double>();

            double minPrice = double.MaxValue;
            double maxPrice = double.MinValue;

            int historicalCount = historicalData.Count;

            // Add historical data
            foreach (var dataPoint in historicalData)
            {
                closeValues.Add(dataPoint.Close);
                highValues.Add(dataPoint.High);
                lowValues.Add(dataPoint.Low);

                TimeLabels.Add(dataPoint.Timestamp.ToString("MM/dd"));

                minPrice = Math.Min(minPrice, dataPoint.Low);
                maxPrice = Math.Max(maxPrice, dataPoint.High);
            }

            // Validate that we have actual data before adding series
            if (closeValues.Count == 0 || highValues.Count == 0 || lowValues.Count == 0)
            {
                _loggingService?.Log("Warning", "Empty values collection, cannot create chart");
                IsChartVisible = false;
                return;
            }

            // Add Close price line (main candlestick representation)
            CandlestickSeries.Add(new LineSeries
            {
                Title = "Close",
                Values = closeValues,
                Stroke = System.Windows.Media.Brushes.Cyan,
                Fill = System.Windows.Media.Brushes.Transparent,
                StrokeThickness = 2,
                PointGeometry = null
            });

            // Add High-Low range
            CandlestickSeries.Add(new LineSeries
            {
                Title = "High",
                Values = highValues,
                Stroke = System.Windows.Media.Brushes.LimeGreen,
                Fill = System.Windows.Media.Brushes.Transparent,
                StrokeThickness = 1,
                PointGeometry = null,
                LineSmoothness = 0
            });

            CandlestickSeries.Add(new LineSeries
            {
                Title = "Low",
                Values = lowValues,
                Stroke = System.Windows.Media.Brushes.OrangeRed,
                Fill = System.Windows.Media.Brushes.Transparent,
                StrokeThickness = 1,
                PointGeometry = null,
                LineSmoothness = 0
            });

            // Add TFT predictions if available
            if (tftPredictions?.Predictions != null && tftPredictions.Predictions.Count > 0)
            {
                // Create separate series for predictions that start from the last historical point
                var predictionValues = new ChartValues<double>();
                var upperBandValues = new ChartValues<double>();
                var lowerBandValues = new ChartValues<double>();

                // Add padding to align with historical data (all null values up to last historical point)
                for (int i = 0; i < historicalCount - 1; i++)
                {
                    predictionValues.Add(double.NaN);
                    upperBandValues.Add(double.NaN);
                    lowerBandValues.Add(double.NaN);
                }

                // Add the last historical close price as the starting point for predictions
                predictionValues.Add(historicalData.Last().Close);
                upperBandValues.Add(historicalData.Last().Close);
                lowerBandValues.Add(historicalData.Last().Close);

                // Add prediction points
                foreach (var pred in tftPredictions.Predictions)
                {
                    predictionValues.Add(pred.PredictedPrice);
                    upperBandValues.Add(pred.UpperConfidence);
                    lowerBandValues.Add(pred.LowerConfidence);

                    // Add labels for prediction dates
                    TimeLabels.Add(pred.Timestamp.ToString("MM/dd"));

                    maxPrice = Math.Max(maxPrice, pred.UpperConfidence);
                    minPrice = Math.Min(minPrice, pred.LowerConfidence);
                }

                // Add prediction line
                CandlestickSeries.Add(new LineSeries
                {
                    Title = "TFT Prediction",
                    Values = predictionValues,
                    Stroke = System.Windows.Media.Brushes.Yellow,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    StrokeThickness = 2,
                    StrokeDashArray = new System.Windows.Media.DoubleCollection { 4, 2 },
                    PointGeometry = DefaultGeometries.Circle,
                    PointGeometrySize = 8
                });

                // Add confidence bands
                CandlestickSeries.Add(new LineSeries
                {
                    Title = "Upper Confidence",
                    Values = upperBandValues,
                    Stroke = System.Windows.Media.Brushes.Yellow,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    StrokeThickness = 1,
                    StrokeDashArray = new System.Windows.Media.DoubleCollection { 2, 2 },
                    PointGeometry = null,
                    Opacity = 0.5
                });

                CandlestickSeries.Add(new LineSeries
                {
                    Title = "Lower Confidence",
                    Values = lowerBandValues,
                    Stroke = System.Windows.Media.Brushes.Yellow,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    StrokeThickness = 1,
                    StrokeDashArray = new System.Windows.Media.DoubleCollection { 2, 2 },
                    PointGeometry = null,
                    Opacity = 0.5
                });
            }

            // Set chart bounds with some padding
            double padding = (maxPrice - minPrice) * 0.1;
            
            // Ensure minimum range to prevent LiveCharts exception
            if (Math.Abs(maxPrice - minPrice) < 0.01)
            {
                // If range is too small, use a default range of 5% around the price
                double centerPrice = (maxPrice + minPrice) / 2.0;
                if (centerPrice > 0)
                {
                    padding = centerPrice * 0.05;
                }
                else
                {
                    // Fallback if price is 0 or negative
                    padding = 1.0;
                }
            }
            
            ChartMinPrice = minPrice - padding;
            ChartMaxPrice = maxPrice + padding;
            
            // Final safety check to ensure valid range
            if (Math.Abs(ChartMaxPrice - ChartMinPrice) < 0.01)
            {
                ChartMaxPrice = ChartMinPrice + 1.0;
            }

                OnPropertyChanged(nameof(CandlestickSeries));
                OnPropertyChanged(nameof(TimeLabels));
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error updating candlestick chart");
                // Hide chart on error to prevent LiveCharts exceptions
                IsChartVisible = false;
                // Reset to safe defaults
                ChartMinPrice = 0;
                ChartMaxPrice = 100;
                CandlestickSeries.Clear();
                TimeLabels = new List<string>();
            }
        }

        private void UpdateVolumeChart(List<OHLCDataPoint> historicalData)
        {
            VolumeSeries.Clear();

            if (historicalData == null || historicalData.Count == 0)
                return;

            var volumeValues = new ChartValues<double>();

            foreach (var dataPoint in historicalData)
            {
                volumeValues.Add(dataPoint.Volume);
            }

            VolumeSeries.Add(new ColumnSeries
            {
                Title = "Volume",
                Values = volumeValues,
                Fill = System.Windows.Media.Brushes.SteelBlue,
                DataLabels = false
            });

            OnPropertyChanged(nameof(VolumeSeries));
        }

        private void UpdateFeatureAttention(Dictionary<string, double> attentionWeights)
        {
            FeatureAttentionData.Clear();

            if (attentionWeights == null || attentionWeights.Count == 0)
                return;

            // Calculate total weight for percentage
            double totalWeight = attentionWeights.Values.Sum();

            // Sort by weight descending and take top features
            var sortedFeatures = attentionWeights
                .OrderByDescending(kv => kv.Value)
                .Take(15) // Top 15 features
                .ToList();

            foreach (var feature in sortedFeatures)
            {
                FeatureAttentionData.Add(new FeatureAttentionItem
                {
                    FeatureName = feature.Key,
                    AttentionWeight = feature.Value,
                    ImportancePercent = totalWeight > 0 ? (feature.Value / totalWeight) * 100 : 0
                });
            }
        }

        private async Task UpdateAllSymbols()
        {
            if (_isUpdating) return;

            _isUpdating = true;

            try
            {
                // Capture UI state on the UI thread before async operations
                bool autoAnalyzeEnabled = false;
                await Dispatcher.InvokeAsync(() =>
                {
                    autoAnalyzeEnabled = AutoAnalyzeCheckBox.IsChecked == true;
                    StatusText.Text = $"Updating {_symbols.Count} symbols...";
                    StatusText.Foreground = System.Windows.Media.Brushes.Yellow;
                });

                var newData = new List<StockAnalysisItem>();

                foreach (var symbol in _symbols)
                {
                    try
                    {
                        var item = new StockAnalysisItem { Symbol = symbol };

                        // Fetch quote data
                        var quote = await _alphaVantageService.GetQuoteDataAsync(symbol);
                        if (quote != null)
                        {
                            item.CurrentPrice = quote.Price;
                            item.PercentChange = quote.ChangePercent;
                            item.Volume = (long)quote.Volume;
                        }

                        // Fetch technical indicators
                        var indicators = new Dictionary<string, double>();
                        
                        try
                        {
                            var rsi = await _alphaVantageService.GetRSI(symbol, "daily");
                            if (!double.IsNaN(rsi) && rsi > 0)
                            {
                                item.RSI = rsi;
                                indicators["RSI"] = rsi;
                            }
                        }
                        catch (Exception rsiEx)
                        {
                            _loggingService?.Log("Warning", $"Failed to get RSI for {symbol}: {rsiEx.Message}");
                        }

                        try
                        {
                            var (macd, macdSignal, macdHist) = await _alphaVantageService.GetMACD(symbol, "daily");
                            if (!double.IsNaN(macd))
                            {
                                item.MACD = macd;
                                indicators["MACD"] = macd;
                                indicators["MACD_Signal"] = macdSignal;
                                indicators["MACD_Hist"] = macdHist;
                            }
                        }
                        catch (Exception macdEx)
                        {
                            _loggingService?.Log("Warning", $"Failed to get MACD for {symbol}: {macdEx.Message}");
                        }

                        try
                        {
                            var vwap = await _alphaVantageService.GetVWAP(symbol, "daily");
                            if (!double.IsNaN(vwap) && vwap > 0)
                            {
                                item.VWAP = vwap;
                                indicators["VWAP"] = vwap;
                            }
                        }
                        catch (Exception vwapEx)
                        {
                            _loggingService?.Log("Warning", $"Failed to get VWAP for {symbol}: {vwapEx.Message}");
                        }

                        // Run ML prediction if enabled
                        if (autoAnalyzeEnabled && indicators.Count > 0)
                        {
                            try
                            {
                                // Add current price to indicators
                                indicators["Close"] = item.CurrentPrice;
                                indicators["Volume"] = item.Volume;

                                // Use PythonStockPredictor directly for ML predictions
                                var prediction = await Models.PythonStockPredictor.PredictAsync(indicators);
                                
                                if (prediction != null && string.IsNullOrEmpty(prediction.Error))
                                {
                                    item.PredictedAction = prediction.Action;
                                    item.Confidence = prediction.Confidence;
                                    item.TargetPrice = prediction.TargetPrice;
                                }
                            }
                            catch (Exception predEx)
                            {
                                _loggingService?.Log("Warning", $"Prediction failed for {symbol}: {predEx.Message}");
                            }
                        }

                        item.LastUpdated = DateTime.Now;
                        newData.Add(item);
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.Log("Warning", $"Error updating {symbol}: {ex.Message}");
                    }
                }

                // Update UI on dispatcher thread
                await Dispatcher.InvokeAsync(() =>
                {
                    StockAnalysisData.Clear();
                    foreach (var item in newData)
                    {
                        StockAnalysisData.Add(item);
                    }

                    StatusText.Text = $"Updated {newData.Count} symbols successfully";
                    StatusText.Foreground = System.Windows.Media.Brushes.LimeGreen;
                    LastUpdateText.Text = $"Last update: {DateTime.Now:HH:mm:ss}";
                });
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error updating symbols");
                await Dispatcher.InvokeAsync(() =>
                {
                    StatusText.Text = $"Error: {ex.Message}";
                    StatusText.Foreground = System.Windows.Media.Brushes.Red;
                });
            }
            finally
            {
                _isUpdating = false;
            }
        }
    }

    /// <summary>
    /// Data model for stock analysis grid row
    /// </summary>
    public class StockAnalysisItem : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        private string _symbol;
        public string Symbol
        {
            get => _symbol;
            set { _symbol = value; OnPropertyChanged(nameof(Symbol)); }
        }

        private double _currentPrice;
        public double CurrentPrice
        {
            get => _currentPrice;
            set { _currentPrice = value; OnPropertyChanged(nameof(CurrentPrice)); }
        }

        private double _percentChange;
        public double PercentChange
        {
            get => _percentChange;
            set { _percentChange = value; OnPropertyChanged(nameof(PercentChange)); }
        }

        private double _rsi;
        public double RSI
        {
            get => _rsi;
            set { _rsi = value; OnPropertyChanged(nameof(RSI)); }
        }

        private double _macd;
        public double MACD
        {
            get => _macd;
            set { _macd = value; OnPropertyChanged(nameof(MACD)); }
        }

        private double _vwap;
        public double VWAP
        {
            get => _vwap;
            set { _vwap = value; OnPropertyChanged(nameof(VWAP)); }
        }

        private long _volume;
        public long Volume
        {
            get => _volume;
            set { _volume = value; OnPropertyChanged(nameof(Volume)); }
        }

        private string _predictedAction = "HOLD";
        public string PredictedAction
        {
            get => _predictedAction;
            set { _predictedAction = value; OnPropertyChanged(nameof(PredictedAction)); }
        }

        private double _confidence;
        public double Confidence
        {
            get => _confidence;
            set { _confidence = value; OnPropertyChanged(nameof(Confidence)); }
        }

        private double _targetPrice;
        public double TargetPrice
        {
            get => _targetPrice;
            set { _targetPrice = value; OnPropertyChanged(nameof(TargetPrice)); }
        }

        private DateTime _lastUpdated;
        public DateTime LastUpdated
        {
            get => _lastUpdated;
            set { _lastUpdated = value; OnPropertyChanged(nameof(LastUpdated)); }
        }
    }

    /// <summary>
    /// OHLC data point for candlestick chart
    /// </summary>
    public class OHLCDataPoint
    {
        public DateTime Timestamp { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public long Volume { get; set; }
    }

    /// <summary>
    /// TFT prediction result with confidence bands
    /// </summary>
    public class TFTPredictionResult
    {
        public List<PredictionPoint> Predictions { get; set; }
        public Dictionary<string, double> FeatureAttention { get; set; }
    }

    /// <summary>
    /// Single prediction point with confidence interval
    /// </summary>
    public class PredictionPoint
    {
        public DateTime Timestamp { get; set; }
        public double PredictedPrice { get; set; }
        public double UpperConfidence { get; set; }
        public double LowerConfidence { get; set; }
    }

    /// <summary>
    /// Feature attention weight for TFT model
    /// </summary>
    public class FeatureAttentionItem : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        private string _featureName;
        public string FeatureName
        {
            get => _featureName;
            set { _featureName = value; OnPropertyChanged(nameof(FeatureName)); }
        }

        private double _attentionWeight;
        public double AttentionWeight
        {
            get => _attentionWeight;
            set { _attentionWeight = value; OnPropertyChanged(nameof(AttentionWeight)); }
        }

        private double _importancePercent;
        public double ImportancePercent
        {
            get => _importancePercent;
            set { _importancePercent = value; OnPropertyChanged(nameof(ImportancePercent)); }
        }
    }
}
