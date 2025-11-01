using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System.ComponentModel;
using System.Linq;
using LiveCharts;
using LiveCharts.Wpf;
using System.Threading.Tasks;
using Quantra.DAL.Services;

namespace Quantra.Views.PredictionAnalysis.Components
{
    public partial class PredictionChartModule : UserControl, INotifyPropertyChanged, IDisposable
    {
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly INotificationService _notificationService;
        private readonly IStockDataCacheService _stockDataCacheService;
        private bool _disposed;

        private SeriesCollection _series;
        public SeriesCollection ChartSeries
        {
            get => _series;
            set
            {
                _series = value;
                OnPropertyChanged(nameof(ChartSeries));
            }
        }

        private bool _isLoading;
        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                _isLoading = value;
                OnPropertyChanged(nameof(IsLoading));
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
                    OnPropertyChanged(nameof(Symbol));
                    UpdateChartForSymbol();
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
                    OnPropertyChanged(nameof(Timeframe));
                    UpdateChartForSymbol();
                }
            }
        }

        private IList<string> _labels;
        public IList<string> Labels
        {
            get => _labels;
            set
            {
                _labels = value;
                OnPropertyChanged(nameof(Labels));
            }
        }

        private Func<double, string> _yFormatter;
        public Func<double, string> YFormatter
        {
            get => _yFormatter;
            set
            {
                _yFormatter = value;
                OnPropertyChanged(nameof(YFormatter));
            }
        }

        private IList<string> _timeframes;
        public IList<string> Timeframes
        {
            get => _timeframes;
            set
            {
                _timeframes = value;
                OnPropertyChanged(nameof(Timeframes));
            }
        }

        public PredictionChartModule(
            ITechnicalIndicatorService indicatorService,
            INotificationService notificationService,
            IStockDataCacheService stockDataCacheService)
        {
            // Initialize properties before setting DataContext to avoid binding errors
            InitializeChart();
            InitializeComponent();
            DataContext = this;
            _indicatorService = indicatorService;
            _notificationService = notificationService;
            _stockDataCacheService = stockDataCacheService;
            
            // Subscribe to symbol update events
            SymbolUpdateService.SymbolUpdated += OnSymbolUpdated;
            SymbolUpdateService.CachedSymbolUpdated += OnCachedSymbolUpdated;
        }

        private void InitializeChart()
        {
            ChartSeries = new SeriesCollection();
            Labels = new List<string>();
            YFormatter = value => value.ToString("F2");
            Timeframes = new List<string> { "1min", "5min", "15min", "30min", "1hour", "4hour", "1day", "1week", "1month" };
            Timeframe = "1day"; // Set default timeframe
        }

        private async void UpdateChartForSymbol()
        {
            try
            {
                if (string.IsNullOrEmpty(Symbol) || _disposed)
                    return;
                IsLoading = true;
                
                // Get historical price data
                var historicalData = await _stockDataCacheService.GetStockDataAsync(Symbol, GetRangeFromTimeframe(Timeframe), GetIntervalFromTimeframe(Timeframe));
                
                if (historicalData != null && historicalData.Count > 0)
                {
                    ChartSeries.Clear();
                    var dateLabels = new List<string>();
                    
                    // Sort historical data by date (ascending for proper timeline)
                    var sortedData = historicalData.OrderBy(h => h.Date).Take(50).ToList();
                    
                    // Create price line series (Close prices)
                    var priceValues = new ChartValues<double>();
                    foreach (var price in sortedData)
                    {
                        priceValues.Add(price.Close);
                        dateLabels.Add(price.Date.ToString("MM/dd"));
                    }
                    
                    var priceSeries = new LineSeries
                    {
                        Title = $"{Symbol} Price",
                        Values = priceValues,
                        LineSmoothness = 0.2,
                        StrokeThickness = 3,
                        Stroke = Brushes.Cyan,
                        Fill = Brushes.Transparent,
                        PointGeometry = null // Remove data point markers for cleaner look
                    };
                    ChartSeries.Add(priceSeries);
                    
                    // Add volume as a secondary series (scaled down for visualization)
                    if (sortedData.Any(h => h.Volume > 0))
                    {
                        var volumeValues = new ChartValues<double>();
                        var maxVolume = sortedData.Max(h => h.Volume);
                        var maxPrice = sortedData.Max(h => h.Close);
                        
                        // Prevent division by zero
                        var scaleFactor = (maxVolume > 0 && maxPrice > 0) ? maxPrice / maxVolume * 0.1 : 0.1;
                        
                        foreach (var price in sortedData)
                        {
                            volumeValues.Add(price.Volume * scaleFactor);
                        }
                        
                        var volumeSeries = new LineSeries
                        {
                            Title = "Volume (scaled)",
                            Values = volumeValues,
                            LineSmoothness = 0,
                            StrokeThickness = 1,
                            Stroke = Brushes.Gray,
                            Fill = Brushes.Transparent,
                            PointGeometry = null
                        };
                        ChartSeries.Add(volumeSeries);
                    }
                    
                    Labels = dateLabels;
                }
                else
                {
                    // Fallback to indicator data if no historical data available
                    await LoadIndicatorDataAsFallback();
                }
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error updating chart: {ex.Message}");
                //DatabaseMonolith.Log("Error", $"Failed to update chart for {Symbol}", ex.ToString());
            }
            finally
            {
                IsLoading = false;
            }
        }

        private Brush GetIndicatorBrush(string indicatorName)
        {
            return indicatorName switch
            {
                "RSI" => Brushes.Orange,
                "ADX" => Brushes.Blue,
                "ATR" => Brushes.Red,
                "MACD" => Brushes.Green,
                "MACDSignal" => Brushes.Purple,
                "MACDHistogram" => Brushes.Gray,
                "StochK" => Brushes.Yellow,
                "StochD" => Brushes.Brown,
                "MomentumScore" => Brushes.Pink,
                "TradingSignal" => Brushes.Cyan,
                _ => Brushes.Black
            };
        }

        private string GetRangeFromTimeframe(string timeframe)
        {
            return timeframe switch
            {
                "1min" or "5min" or "15min" or "30min" or "1hour" => "1d",
                "4hour" => "5d",
                "1day" => "1mo",
                "1week" => "3mo",
                "1month" => "1y",
                _ => "1mo"
            };
        }

        private string GetIntervalFromTimeframe(string timeframe)
        {
            return timeframe switch
            {
                "1min" => "1m",
                "5min" => "5m",
                "15min" => "15m",
                "30min" => "30m",
                "1hour" => "1h",
                "4hour" => "4h",
                "1day" => "1d",
                "1week" => "1wk",
                "1month" => "1mo",
                _ => "1d"
            };
        }

        private async Task LoadIndicatorDataAsFallback()
        {
            try
            {
                var indicators = await _indicatorService.GetIndicatorsForPrediction(Symbol, Timeframe ?? "1day");
                if (indicators != null && indicators.Count > 0)
                {
                    ChartSeries.Clear();
                    var labelsList = new List<string>();
                    
                    foreach (var indicator in indicators)
                    {
                        var series = new LineSeries
                        {
                            Title = indicator.Key,
                            Values = new ChartValues<double> { indicator.Value },
                            LineSmoothness = 0,
                            StrokeThickness = 2,
                            Stroke = GetIndicatorBrush(indicator.Key),
                            Fill = Brushes.Transparent
                        };
                        ChartSeries.Add(series);
                        labelsList.Add(indicator.Key);
                    }
                    
                    Labels = labelsList;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to load indicator data as fallback for {Symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Updates the chart with new symbol data from StockExplorer's GetSymbolData
        /// </summary>
        /// <param name="symbol">The symbol to update</param>
        /// <param name="quoteData">Optional quote data if available</param>
        public async Task UpdateFromSymbolData(string symbol, QuoteData quoteData = null)
        {
            try
            {
                if (!string.IsNullOrEmpty(symbol))
                {
                    Symbol = symbol;
                    // UpdateChartForSymbol will be called automatically due to property change
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to update chart from symbol data for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Updates the chart with cached symbol data
        /// </summary>
        /// <param name="symbol">The symbol to update</param>
        public async Task UpdateFromCachedSymbolData(string symbol)
        {
            try
            {
                if (!string.IsNullOrEmpty(symbol))
                {
                    Symbol = symbol;
                    // This will trigger UpdateChartForSymbol which will check cache first
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to update chart from cached symbol data for {symbol}", ex.ToString());
            }
        }

        private async void OnSymbolUpdated(object sender, SymbolUpdatedEventArgs e)
        {
            try
            {
                // Event is now guaranteed to be called on UI thread by SymbolUpdateService
                await UpdateFromSymbolData(e.Symbol);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to handle symbol updated event for {e.Symbol}", ex.ToString());
            }
        }

        private async void OnCachedSymbolUpdated(object sender, SymbolUpdatedEventArgs e)
        {
            try
            {
                // Event is now guaranteed to be called on UI thread by SymbolUpdateService
                await UpdateFromCachedSymbolData(e.Symbol);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to handle cached symbol updated event for {e.Symbol}", ex.ToString());
            }
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    ChartSeries?.Clear();
                    // Unsubscribe from events
                    SymbolUpdateService.SymbolUpdated -= OnSymbolUpdated;
                    SymbolUpdateService.CachedSymbolUpdated -= OnCachedSymbolUpdated;
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~PredictionChartModule()
        {
            Dispose(false);
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}