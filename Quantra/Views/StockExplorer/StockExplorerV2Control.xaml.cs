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
using Quantra.Views.StockExplorer;
using Newtonsoft.Json;

namespace Quantra.Controls
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
        private readonly PythonStockPredictionService _pythonPredictionService;

        // State management
        private DispatcherTimer _updateTimer;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isUpdating = false;
        private List<string> _symbols = new List<string>();
        private StockConfigurationEntity _currentConfiguration;

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
        private SeriesCollection _chartSeries;
        public SeriesCollection ChartSeries
        {
            get => _chartSeries;
            set
            {
                _chartSeries = value;
                OnPropertyChanged(nameof(ChartSeries));
            }
        }

        private List<string> _chartLabels;
        public List<string> ChartLabels
        {
            get => _chartLabels;
            set
            {
                _chartLabels = value;
                OnPropertyChanged(nameof(ChartLabels));
            }
        }

        public Func<double, string> YFormatter { get; set; }

        public StockExplorerV2Control()
        {
            InitializeComponent();
            DataContext = this;

            // Initialize services from DI container
            _alphaVantageService = App.ServiceProvider?.GetService(typeof(AlphaVantageService)) as AlphaVantageService;
            _stockConfigurationService = App.ServiceProvider?.GetService(typeof(StockConfigurationService)) as StockConfigurationService;
            _loggingService = App.ServiceProvider?.GetService(typeof(LoggingService)) as LoggingService;
            _pythonPredictionService = App.ServiceProvider?.GetService(typeof(PythonStockPredictionService)) as PythonStockPredictionService;

            // Initialize collections
            StockAnalysisData = new ObservableCollection<StockAnalysisItem>();
            ChartSeries = new SeriesCollection();
            ChartLabels = new List<string>();

            // Initialize chart formatter
            YFormatter = value => value.ToString("C2");

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
                var configWindow = new StockConfigurationManagerWindow();
                configWindow.Owner = Window.GetWindow(this);
                
                if (configWindow.ShowDialog() == true)
                {
                    _currentConfiguration = configWindow.SelectedConfiguration;
                    
                    if (_currentConfiguration != null)
                    {
                        // Parse symbols from JSON
                        _symbols = JsonConvert.DeserializeObject<List<string>>(_currentConfiguration.Symbols) ?? new List<string>();
                        
                        ConfigurationNameText.Text = $"{_currentConfiguration.Name} ({_symbols.Count} symbols)";
                        ConfigurationNameText.Foreground = System.Windows.Media.Brushes.LimeGreen;
                        ConfigurationNameText.FontStyle = FontStyles.Normal;
                        
                        SymbolCountText.Text = $"Symbols: {_symbols.Count}";
                        
                        StatusText.Text = $"Loaded configuration: {_currentConfiguration.Name}";
                        StatusText.Foreground = System.Windows.Media.Brushes.Cyan;
                        
                        StartButton.IsEnabled = true;
                        
                        _loggingService?.Log("Info", $"Loaded stock configuration: {_currentConfiguration.Name} with {_symbols.Count} symbols");
                    }
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

        private async Task UpdateAllSymbols()
        {
            if (_isUpdating) return;
            
            _isUpdating = true;
            
            try
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    StatusText.Text = $"Updating {_symbols.Count} symbols...";
                    StatusText.Foreground = System.Windows.Media.Brushes.Yellow;
                });

                var newData = new List<StockAnalysisItem>();
                var chartPrices = new List<double>();
                var chartLabels = new List<string>();

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
                        catch { }

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
                        catch { }

                        try
                        {
                            var vwap = await _alphaVantageService.GetVWAP(symbol, "daily");
                            if (!double.IsNaN(vwap) && vwap > 0)
                            {
                                item.VWAP = vwap;
                                indicators["VWAP"] = vwap;
                            }
                        }
                        catch { }

                        // Run ML prediction if enabled
                        if (AutoAnalyzeCheckBox.IsChecked == true && indicators.Count > 0 && _pythonPredictionService != null)
                        {
                            try
                            {
                                // Add current price to indicators
                                indicators["Close"] = item.CurrentPrice;
                                indicators["Volume"] = item.Volume;

                                var prediction = await PythonStockPredictor.PredictAsync(indicators);
                                
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

                        // Add to chart data
                        chartPrices.Add(item.CurrentPrice);
                        chartLabels.Add(symbol);
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

                    // Update chart
                    ChartSeries.Clear();
                    ChartSeries.Add(new ColumnSeries
                    {
                        Title = "Current Price",
                        Values = new ChartValues<double>(chartPrices),
                        Fill = System.Windows.Media.Brushes.Cyan,
                        DataLabels = false
                    });
                    ChartLabels = chartLabels;
                    OnPropertyChanged(nameof(ChartLabels));

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
}
