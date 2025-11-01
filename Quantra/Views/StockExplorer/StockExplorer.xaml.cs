using LiveCharts;
using LiveCharts.Wpf;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data.SQLite;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Reflection;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra;
using LiveCharts.Defaults;
using Microsoft.Extensions.Configuration;
using Quantra.ViewModels;
using Quantra.Enums;
using Quantra.Views.Shared;
using Quantra.Utilities;
using Quantra.DAL.Services;

namespace Quantra.Controls 
{
    /// <summary>
    /// Interaction logic for StockExplorer.xaml
    /// </summary>
    public partial class StockExplorer : UserControl, INotifyPropertyChanged, IDisposable
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        private readonly StockExplorerViewModel _viewModel;
        private readonly StockDataCacheService _cacheService;
        
        // Sentiment Analysis Services
        private readonly OpenAISentimentService _openAISentimentService;
        private readonly FinancialNewsSentimentService _newsSentimentService;
        private readonly AnalystRatingService _analystRatingService;
        private readonly SectorSentimentAnalysisService _sectorSentimentService;
        private readonly TwitterSentimentService _twitterSentimentService;
        
        private bool _isLoaded = false;
        private bool _isHandlingSelectionChanged = false;
        private UIBatchUpdater _uiBatchUpdater;
        
        // Collection to track background tasks for proper cleanup and coordination
        private readonly List<Task> _backgroundTasks = new List<Task>();
        private readonly object _backgroundTasksLock = new object();
        
        // Cancellation support for symbol operations to prevent UI blocking on rapid selections
        private System.Threading.CancellationTokenSource _symbolOperationCancellation = new System.Threading.CancellationTokenSource();
        
        // Memory pressure monitoring
        private readonly System.Windows.Threading.DispatcherTimer _memoryMonitorTimer;
        private const long MEMORY_THRESHOLD_BYTES = 500 * 1024 * 1024; // 500MB threshold
        
        // Stable instance identifier for persisting DataGrid settings across application runs
        private readonly string _instanceId;
        
        // Timer fields for symbol search functionality
        private System.Windows.Threading.DispatcherTimer _symbolSearchTimer;

        // Collection for current indicator values
        private ObservableCollection<TechnicalIndicator> _currentIndicators = new ObservableCollection<TechnicalIndicator>();
        public ObservableCollection<TechnicalIndicator> CurrentIndicators
        {
            get => _currentIndicators;
            set
            {
                if (_currentIndicators != value)
                {
                    _currentIndicators = value;
                    OnPropertyChanged(nameof(CurrentIndicators));
                }
            }
        }

        // Properties for symbol and price display
        private string _symbolText = "";
        public string SymbolText
        {
            get => _symbolText;
            set
            {
                if (_symbolText != value)
                {
                    _symbolText = value;
                    OnPropertyChanged(nameof(SymbolText));
                }
            }
        }

        private string _priceText = "";
        public string PriceText
        {
            get => _priceText;
            set
            {
                if (_priceText != value)
                {
                    _priceText = value;
                    OnPropertyChanged(nameof(PriceText));
                }
            }
        }

        private string _updatedTimestampText = "";
        public string UpdatedTimestampText
        {
            get => _updatedTimestampText;
            set
            {
                if (_updatedTimestampText != value)
                {
                    _updatedTimestampText = value;
                    OnPropertyChanged(nameof(UpdatedTimestampText));
                }
            }
        }

        // Individual indicator properties for consistent UI updates
        private string _rsiValue = "--";
        public string RsiValue
        {
            get => _rsiValue;
            set
            {
                if (_rsiValue != value)
                {
                    _rsiValue = value;
                    OnPropertyChanged(nameof(RsiValue));
                }
            }
        }

        private string _peRatioValue = "--";
        public string PeRatioValue
        {
            get => _peRatioValue;
            set
            {
                if (_peRatioValue != value)
                {
                    _peRatioValue = value;
                    OnPropertyChanged(nameof(PeRatioValue));
                }
            }
        }

        private string _macdValue = "--";
        public string MacdValue
        {
            get => _macdValue;
            set
            {
                if (_macdValue != value)
                {
                    _macdValue = value;
                    OnPropertyChanged(nameof(MacdValue));
                }
            }
        }

        private string _macdSignalValue = "--";
        public string MacdSignalValue
        {
            get => _macdSignalValue;
            set
            {
                if (_macdSignalValue != value)
                {
                    _macdSignalValue = value;
                    OnPropertyChanged(nameof(MacdSignalValue));
                }
            }
        }

        private string _macdHistValue = "--";
        public string MacdHistValue
        {
            get => _macdHistValue;
            set
            {
                if (_macdHistValue != value)
                {
                    _macdHistValue = value;
                    OnPropertyChanged(nameof(MacdHistValue));
                }
            }
        }

        private string _vwapValue = "--";
        public string VwapValue
        {
            get => _vwapValue;
            set
            {
                if (_vwapValue != value)
                {
                    _vwapValue = value;
                    OnPropertyChanged(nameof(VwapValue));
                }
            }
        }

        private string _adxValue = "--";
        public string AdxValue
        {
            get => _adxValue;
            set
            {
                if (_adxValue != value)
                {
                    _adxValue = value;
                    OnPropertyChanged(nameof(AdxValue));
                }
            }
        }

        private string _cciValue = "--";
        public string CciValue
        {
            get => _cciValue;
            set
            {
                if (_cciValue != value)
                {
                    _cciValue = value;
                    OnPropertyChanged(nameof(CciValue));
                }
            }
        }

        private string _atrValue = "--";
        public string AtrValue
        {
            get => _atrValue;
            set
            {
                if (_atrValue != value)
                {
                    _atrValue = value;
                    OnPropertyChanged(nameof(AtrValue));
                }
            }
        }

        private string _mfiValue = "--";
        public string MfiValue
        {
            get => _mfiValue;
            set
            {
                if (_mfiValue != value)
                {
                    _mfiValue = value;
                    OnPropertyChanged(nameof(MfiValue));
                }
            }
        }

        private string _stochKValue = "--";
        public string StochKValue
        {
            get => _stochKValue;
            set
            {
                if (_stochKValue != value)
                {
                    _stochKValue = value;
                    OnPropertyChanged(nameof(StochKValue));
                }
            }
        }

        private string _stochDValue = "--";
        public string StochDValue
        {
            get => _stochDValue;
            set
            {
                if (_stochDValue != value)
                {
                    _stochDValue = value;
                    OnPropertyChanged(nameof(StochDValue));
                }
            }
        }

        // Sentiment Analysis Properties
        private bool _isSentimentLoading = false;
        public bool IsSentimentLoading
        {
            get => _isSentimentLoading;
            set
            {
                if (_isSentimentLoading != value)
                {
                    _isSentimentLoading = value;
                    OnPropertyChanged(nameof(IsSentimentLoading));
                    OnPropertyChanged(nameof(CanRunSentimentAnalysis));
                }
            }
        }

        private bool _canRunSentimentAnalysis = true;
        public bool CanRunSentimentAnalysis
        {
            get => _canRunSentimentAnalysis && !_isSentimentLoading;
            set
            {
                if (_canRunSentimentAnalysis != value)
                {
                    _canRunSentimentAnalysis = value;
                    OnPropertyChanged(nameof(CanRunSentimentAnalysis));
                }
            }
        }

        private string _sentimentError = "";
        public string SentimentError
        {
            get => _sentimentError;
            set
            {
                if (_sentimentError != value)
                {
                    _sentimentError = value;
                    OnPropertyChanged(nameof(SentimentError));
                    OnPropertyChanged(nameof(HasSentimentError));
                }
            }
        }

        public bool HasSentimentError => !string.IsNullOrEmpty(_sentimentError);

        private bool _hasSentimentResults = false;
        public bool HasSentimentResults
        {
            get => _hasSentimentResults;
            set
            {
                if (_hasSentimentResults != value)
                {
                    _hasSentimentResults = value;
                    OnPropertyChanged(nameof(HasSentimentResults));
                }
            }
        }

        private double _overallSentimentScore = 0.0;
        public double OverallSentimentScore
        {
            get => _overallSentimentScore;
            set
            {
                if (Math.Abs(_overallSentimentScore - value) > 0.001)
                {
                    _overallSentimentScore = value;
                    OnPropertyChanged(nameof(OverallSentimentScore));
                }
            }
        }

        private double _newsSentimentScore = 0.0;
        public double NewsSentimentScore
        {
            get => _newsSentimentScore;
            set
            {
                if (Math.Abs(_newsSentimentScore - value) > 0.001)
                {
                    _newsSentimentScore = value;
                    OnPropertyChanged(nameof(NewsSentimentScore));
                }
            }
        }

        private double _socialMediaSentimentScore = 0.0;
        public double SocialMediaSentimentScore
        {
            get => _socialMediaSentimentScore;
            set
            {
                if (Math.Abs(_socialMediaSentimentScore - value) > 0.001)
                {
                    _socialMediaSentimentScore = value;
                    OnPropertyChanged(nameof(SocialMediaSentimentScore));
                }
            }
        }

        private double _analystSentimentScore = 0.0;
        public double AnalystSentimentScore
        {
            get => _analystSentimentScore;
            set
            {
                if (Math.Abs(_analystSentimentScore - value) > 0.001)
                {
                    _analystSentimentScore = value;
                    OnPropertyChanged(nameof(AnalystSentimentScore));
                }
            }
        }

        private string _sentimentSummary = "";
        public string SentimentSummary
        {
            get => _sentimentSummary;
            set
            {
                if (_sentimentSummary != value)
                {
                    _sentimentSummary = value;
                    OnPropertyChanged(nameof(SentimentSummary));
                }
            }
        }

        // Property for the selected symbol title in the indicators chart
        private string _selectedSymbolTitle = "Select a symbol to view indicators";
        public string SelectedSymbolTitle
        {
            get => _selectedSymbolTitle;
            set
            {
                if (_selectedSymbolTitle != value)
                {
                    _selectedSymbolTitle = value;
                    OnPropertyChanged(nameof(SelectedSymbolTitle));
                }
            }
        }

        // Symbol selection mode property
        private SymbolSelectionMode _currentSelectionMode = SymbolSelectionMode.IndividualAsset;
        public SymbolSelectionMode CurrentSelectionMode
        {
            get => _currentSelectionMode;
            set
            {
                if (_currentSelectionMode != value)
                {
                    _currentSelectionMode = value;
                    OnPropertyChanged(nameof(CurrentSelectionMode));
                    OnSelectionModeChanged();
                }
            }
        }

        // Helper method to format indicator values consistently
        private string FormatIndicatorValue(string indicatorName, double value)
        {
            return indicatorName switch
            {
                "RSI" or "StochRSI" or "Stoch K" or "Stoch D" or "MFI" => $"{value:F1}",
                "MACD" or "MACD Signal" or "MACD Hist" => $"{value:F3}",
                "ADX" => $"{value:F1}",
                "ATR" => $"${value:F2}",
                "VWAP" => $"${value:F2}",
                "ROC" => $"{value:F1}%",
                "BullPower" or "BearPower" => $"${value:F2}",
                "Price" => $"${value:F2}",
                "Volume" => $"{value:N0}",
                "MomentumScore" => $"{value:F0}",
                "P/E Ratio" or "PE Ratio" => $"{value:F1}",
                _ => $"{value:F2}"
            };
        }

        public StockExplorer()
        {
            // Generate stable instance identifier for DataGrid settings persistence
            _instanceId = Guid.NewGuid().ToString();
            
            InitializeComponent();
            _viewModel = new StockExplorerViewModel();
            _cacheService = new StockDataCacheService();
            
            // Initialize sentiment analysis services with null checks
            try
            {
                _newsSentimentService = new FinancialNewsSentimentService();
                _analystRatingService = new AnalystRatingService();
                _sectorSentimentService = new SectorSentimentAnalysisService();
                _twitterSentimentService = new TwitterSentimentService();
                // Don't initialize OpenAI service as it requires additional configuration
                _openAISentimentService = null;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Failed to initialize some sentiment services", ex.ToString());
            }
            
            this.DataContext = _viewModel;
            
            // Initialize UI batch updater for performance
            _uiBatchUpdater = new UIBatchUpdater(this.Dispatcher);
            
            // Initialize memory monitoring timer
            _memoryMonitorTimer = new System.Windows.Threading.DispatcherTimer
            {
                Interval = TimeSpan.FromMinutes(2) // Check every 2 minutes
            };
            _memoryMonitorTimer.Tick += OnMemoryMonitorTimer_Tick;
            _memoryMonitorTimer.Start();
            
            // Subscribe to ViewModel property changes
            _viewModel.PropertyChanged += OnViewModelPropertyChanged;
            
            // Initialize labels
            SymbolText = "";
            PriceText = "";
            
            // Initialize indicator values to default placeholders
            RsiValue = "--";
            PeRatioValue = "--";
            MacdValue = "--";
            MacdSignalValue = "--";
            MacdHistValue = "--";
            VwapValue = "--";
            AdxValue = "--";
            CciValue = "--";
            AtrValue = "--";
            MfiValue = "--";
            StochKValue = "--";
            StochDValue = "--";
            
            // Initialize sentiment analysis properties
            IsSentimentLoading = false;
            CanRunSentimentAnalysis = true;
            SentimentError = "";
            HasSentimentResults = false;
            OverallSentimentScore = 0.0;
            NewsSentimentScore = 0.0;
            SocialMediaSentimentScore = 0.0;
            AnalystSentimentScore = 0.0;
            SentimentSummary = "";
            
            // Initialize charts - ensure IndicatorCharts is available before use
            InitializeCharts();
            
            // Initialize time range selection (default to 1 day)
            UpdateTimeRangeButtonStyles("1day");

            // Initialize symbol search timer for automatic loading
            InitializeSymbolSearchTimer();

            // Initialize DataGrid settings and load saved configuration
            InitializeDataGridSettings();
            this.Loaded += StockExplorer_Loaded;
        }

        /// <summary>
        /// Memory monitoring timer to detect memory pressure and trigger cleanup
        /// </summary>
        private void OnMemoryMonitorTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                var currentMemory = GC.GetTotalMemory(false);
                
                if (currentMemory > MEMORY_THRESHOLD_BYTES)
                {
                    //DatabaseMonolith.Log("Info", $"Memory pressure detected: {currentMemory / (1024 * 1024)}MB. Triggering cleanup.");
                    
                    // Force ViewModel cache management
                    _viewModel?.ManageCacheSize();
                    
                    // Clear chart data from non-selected stocks
                    foreach (var stock in _viewModel.CachedStocks.Where(s => s != _viewModel.SelectedStock))
                    {
                        stock.ClearChartData();
                    }
                    
                    // Force garbage collection
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                    
                    var memoryAfterCleanup = GC.GetTotalMemory(false);
                    //DatabaseMonolith.Log("Info", $"Memory after cleanup: {memoryAfterCleanup / (1024 * 1024)}MB. Freed: {(currentMemory - memoryAfterCleanup) / (1024 * 1024)}MB");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Error in memory monitoring", ex.ToString());
            }
        }

        private void StockExplorer_Loaded(object sender, RoutedEventArgs e)
        {
            // Load DataGrid settings after the control is fully loaded
            LoadDataGridSettings();
        }

        // Selection mode changed event handler
        private void SelectionModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (sender is not ComboBox comboBox || comboBox.SelectedIndex < 0)
                return;

            var newMode = (SymbolSelectionMode)comboBox.SelectedIndex;
            CurrentSelectionMode = newMode;
        }

        // Called when selection mode changes to update UI accordingly
        private async void OnSelectionModeChanged()
        {
            try
            {
                // Reset symbol and price fields to their defaults when mode changes
                SymbolText = "";
                PriceText = "";
                UpdatedTimestampText = "";
                
                switch (CurrentSelectionMode)
                {
                    case SymbolSelectionMode.IndividualAsset:
                        // Show the symbol search combo box
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Visible;
                        if (ModeStatusPanel != null)
                            ModeStatusPanel.Visibility = Visibility.Collapsed;
                        // Hide all RSI buttons and Top P/E button
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        SetAllDatabaseButtonVisibility(Visibility.Collapsed);
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Disable search button initially - will be enabled when valid symbol is typed
                        DisableStockSearchButton();
                        break;

                    case SymbolSelectionMode.TopVolumeRsiDiscrepancies:
                        // Hide the individual symbol search for Top Volume RSI mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Top Volume RSI button, hide other buttons
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Top Volume RSI' to search for high volume RSI discrepancies";
                        }
                        break;

                    case SymbolSelectionMode.TopPE:
                        // Hide the individual symbol search for Top P/E mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Top P/E button, hide other buttons
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Top P/E' to search for stocks with highest P/E ratios";
                        }
                        break;

                    case SymbolSelectionMode.HighVolume:
                        // Hide the individual symbol search for High Volume mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Volume button, hide other buttons
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Volume' to search for high volume stocks";
                        }
                        break;

                    case SymbolSelectionMode.LowPE:
                        // Hide the individual symbol search for Low P/E mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Low P/E button, hide other buttons
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Low P/E' to search for stocks with lowest P/E ratios";
                        }
                        break;

                    case SymbolSelectionMode.RsiOversold:
                        // Hide the individual symbol search for RSI Oversold mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show RSI Oversold button, hide other buttons
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Visible;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load RSI Oversold' to search for oversold stocks";
                        }
                        break;

                    case SymbolSelectionMode.RsiOverbought:
                        // Hide the individual symbol search for RSI Overbought mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show RSI Overbought button, hide other buttons
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load RSI Overbought' to search for overbought stocks";
                        }
                        break;

                    case SymbolSelectionMode.AllDatabase:
                        // Hide the individual symbol search for All Database mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show All Database button, hide other buttons
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load All Database' to display all cached stock data";
                        }
                        break;

                    case SymbolSelectionMode.HighTheta:
                        // Hide the individual symbol search for High Theta mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Theta button, hide other buttons
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Theta' to find stocks with high time decay opportunities";
                        }
                        break;

                    case SymbolSelectionMode.HighBeta:
                        // Hide the individual symbol search for High Beta mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Beta button, hide other buttons
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Beta' to find stocks with high market correlation";
                        }
                        break;

                    case SymbolSelectionMode.HighAlpha:
                        // Hide the individual symbol search for High Alpha mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Alpha button, hide other buttons
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Alpha' to find stocks generating excess returns";
                        }
                        break;

                    case SymbolSelectionMode.BullishCupAndHandle:
                        // Hide the individual symbol search for Bullish Cup and Handle mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Bullish Cup and Handle button, hide other buttons
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Visible;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Cup and Handle' to find stocks with bullish cup and handle patterns";
                        }
                        break;

                    case SymbolSelectionMode.BearishCupAndHandle:
                        // Hide the individual symbol search for Bearish Cup and Handle mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Bearish Cup and Handle button, hide other buttons
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Visible;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Bearish Cup and Handle' to find stocks with bearish cup and handle patterns";
                        }
                        break;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error changing selection mode", ex.ToString());
                CustomModal.ShowError($"Error changing selection mode: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = $"Error loading {CurrentSelectionMode} data";
            }
        }

        // Load symbols based on the selected mode
        private async Task LoadSymbolsForMode(SymbolSelectionMode mode)
        {
            // Set busy cursor during symbol mode loading on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                Mouse.OverrideCursor = Cursors.Wait;
            });
            
            try
            {
                var stockList = new List<QuoteData>();
                
                switch (mode)
                {
                    case SymbolSelectionMode.AllDatabase:
                        // Get all cached stocks from database - NO API CALLS
                        stockList = _cacheService.GetAllCachedStocks();
                        break;

                    case SymbolSelectionMode.TopVolumeRsiDiscrepancies:
                        // Get stocks with high volume and RSI discrepancies
                        var alphaVantageService = new AlphaVantageService();
                        stockList = await GetTopVolumeRsiDiscrepancies();
                        break;

                    case SymbolSelectionMode.TopPE:
                        // Get stocks with highest P/E ratios
                        stockList = await GetTopPEStocks();
                        break;

                    case SymbolSelectionMode.HighVolume:
                        // Get stocks with highest volume
                        stockList = await GetHighVolumeStocks();
                        break;

                    case SymbolSelectionMode.LowPE:
                        // Get stocks with lowest P/E ratios
                        stockList = await GetLowPEStocks();
                        break;

                    case SymbolSelectionMode.RsiOversold:
                        // Get stocks with RSI < 30
                        stockList = await GetRsiOversoldStocks();
                        break;

                    case SymbolSelectionMode.RsiOverbought:
                        // Get stocks with RSI > 70
                        stockList = await GetRsiOverboughtStocks();
                        break;

                    case SymbolSelectionMode.HighTheta:
                        // Get stocks with high time decay for theta harvesting
                        stockList = await GetHighThetaStocks();
                        break;

                    case SymbolSelectionMode.HighBeta:
                        // Get stocks with high market correlation (beta > 1.2)
                        stockList = await GetHighBetaStocks();
                        break;

                    case SymbolSelectionMode.HighAlpha:
                        // Get stocks generating consistent excess returns
                        stockList = await GetHighAlphaStocks();
                        break;

                    case SymbolSelectionMode.BullishCupAndHandle:
                        // Get stocks with bullish cup and handle patterns
                        stockList = await GetBullishCupAndHandleStocks();
                        break;

                    case SymbolSelectionMode.BearishCupAndHandle:
                        // Get stocks with bearish cup and handle patterns
                        stockList = await GetBearishCupAndHandleStocks();
                        break;
                }

                // Update the DataGrid with the filtered stocks
                if (_viewModel != null)
                {
                    _viewModel.CachedStocks.Clear();
                    foreach (var stock in stockList)
                    {
                        _viewModel.CachedStocks.Add(stock);
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error loading symbols for mode {mode}", ex.ToString());
                CustomModal.ShowError($"Error loading symbols: {ex.Message}", "Error", Window.GetWindow(this));
            }
            finally
            {
                // Always reset cursor back to normal on UI thread
                await Dispatcher.InvokeAsync(() =>
                {
                    Mouse.OverrideCursor = null;
                });
            }
        }

        // SymbolComboBox selection changed event handler
        private async void SymbolComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHandlingSelectionChanged || sender is not ComboBox comboBox) 
                return;

            var selectedSymbol = comboBox.SelectedItem as string;
            if (!string.IsNullOrEmpty(selectedSymbol))
            {
                EnableStockSearchButton();
                // Update the title when selecting from ComboBox
                await HandleSymbolSelectionAsync(selectedSymbol, "ComboBox");
            }
        }

        private void EnableStockSearchButton()
        {
            RefreshButton.IsEnabled = true;
        }

        private void DisableStockSearchButton()
        {
            RefreshButton.IsEnabled = false;
        }

        private void ValidateSearchButtonState()
        {
            if (CurrentSelectionMode == Quantra.Enums.SymbolSelectionMode.IndividualAsset && _viewModel != null)
            {
                var searchText = _viewModel.SymbolSearchText?.ToUpper().Trim();
                if (!string.IsNullOrEmpty(searchText) && _viewModel.FilteredSymbols.Contains(searchText))
                {
                    EnableStockSearchButton();
                }
                else
                {
                    DisableStockSearchButton();
                }
            }
        }

        private void SetAllDatabaseButtonVisibility(Visibility visibility)
        {
            if (AllDatabaseButton != null)
                AllDatabaseButton.Visibility = visibility;
        }

        private void InitializeSymbolSearchTimer()
        {
            _symbolSearchTimer = new System.Windows.Threading.DispatcherTimer();
            _symbolSearchTimer.Interval = TimeSpan.FromMilliseconds(500); // 500ms delay for search
            _symbolSearchTimer.Tick += SymbolSearchTimer_Tick;
        }



        private void SymbolSearchTimer_Tick(object sender, EventArgs e)
        {
            _symbolSearchTimer?.Stop();
            
            // Handle delayed symbol search logic
            if (CurrentSelectionMode == SymbolSelectionMode.IndividualAsset && _viewModel != null)
            {
                ValidateSearchButtonState();
            }
        }



        // RSI Oversold button click event handler
        private async void RsiOversoldButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Disable button to prevent multiple clicks
                if (RsiOversoldButton != null)
                    RsiOversoldButton.IsEnabled = false;

                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading RSI Oversold data...";
                }

                // Load RSI oversold stocks
                await LoadSymbolsForMode(SymbolSelectionMode.RsiOversold);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for RSI Oversold";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading RSI oversold stocks", ex.ToString());
                CustomModal.ShowError($"Error loading RSI oversold stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading RSI Oversold data";
            }
            finally
            {
                // Always re-enable button
                if (RsiOversoldButton != null)
                    RsiOversoldButton.IsEnabled = true;
            }
        }

        // RSI Overbought button click event handler
        private async void RsiOverboughtButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading RSI Overbought data...";
                }

                // Load RSI overbought stocks
                await LoadSymbolsForMode(SymbolSelectionMode.RsiOverbought);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for RSI Overbought";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading RSI overbought stocks", ex.ToString());
                CustomModal.ShowError($"Error loading RSI overbought stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading RSI Overbought data";
            }
        }

        // Top Volume RSI button click event handler
        private async void TopVolumeRsiButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Top Volume RSI data...";
                }

                // Load top volume RSI discrepancies stocks
                await LoadSymbolsForMode(SymbolSelectionMode.TopVolumeRsiDiscrepancies);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for Top Volume RSI Discrepancies";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading top volume RSI discrepancies", ex.ToString());
                CustomModal.ShowError($"Error loading top volume RSI discrepancies: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Top Volume RSI data";
            }
        }

        // Top P/E button click event handler
        private async void TopPEButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Top P/E data...";
                }

                // Load top P/E stocks
                await LoadSymbolsForMode(SymbolSelectionMode.TopPE);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for Top P/E";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading top P/E stocks", ex.ToString());
                CustomModal.ShowError($"Error loading top P/E stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Top P/E data";
            }
        }

        // High Volume button click event handler
        private async void HighVolumeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Volume data...";
                }

                // Load high volume stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighVolume);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Volume";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high volume stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high volume stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Volume data";
            }
        }

        // Low P/E button click event handler
        private async void LowPEButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Low P/E data...";
                }

                // Load low P/E stocks
                await LoadSymbolsForMode(SymbolSelectionMode.LowPE);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for Low P/E";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading low P/E stocks", ex.ToString());
                CustomModal.ShowError($"Error loading low P/E stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Low P/E data";
            }
        }

        // All Database button click event handler
        private async void AllDatabaseButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading all cached database data...";
                }

                // Load all cached stocks directly from the cache service
                await LoadSymbolsForMode(SymbolSelectionMode.AllDatabase);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks from database cache";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading all database stocks", ex.ToString());
                CustomModal.ShowError($"Error loading all database stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading All Database data";
            }
        }

        private async void HighThetaButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Theta data...";
                }

                // Load high theta stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighTheta);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Theta";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high theta stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high theta stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Theta data";
            }
        }

        private async void HighBetaButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Beta data...";
                }

                // Load high beta stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighBeta);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Beta";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high beta stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high beta stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Beta data";
            }
        }

        private async void HighAlphaButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Alpha data...";
                }

                // Load high alpha stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighAlpha);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Alpha";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high alpha stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high alpha stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Alpha data";
            }
        }

        private async void BullishCupAndHandleButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Bullish Cup and Handle data...";
                }

                // Load stocks with bullish cup and handle patterns
                await LoadSymbolsForMode(SymbolSelectionMode.BullishCupAndHandle);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks with Cup and Handle patterns";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading bullish cup and handle stocks", ex.ToString());
                CustomModal.ShowError($"Error loading bullish cup and handle stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Cup and Handle data";
            }
        }

        private async void BearishCupAndHandleButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Bearish Cup and Handle data...";
                }

                // Load stocks with bearish cup and handle patterns
                await LoadSymbolsForMode(SymbolSelectionMode.BearishCupAndHandle);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks with Bearish Cup and Handle patterns";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading bearish cup and handle stocks", ex.ToString());
                CustomModal.ShowError($"Error loading bearish cup and handle stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Bearish Cup and Handle data";
            }
        }

        private async void OnViewModelPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            // Update labels when ViewModel data changes
            if (e.PropertyName == nameof(_viewModel.SelectedSymbol) || 
                e.PropertyName == nameof(_viewModel.SelectedStock))
            {
                await UpdatePriceAndRsiLabels(_viewModel.SelectedSymbol);
                
                // Update the selected symbol title when ViewModel changes
                SelectedSymbolTitle = !string.IsNullOrEmpty(_viewModel.SelectedSymbol) 
                    ? $"{_viewModel.SelectedSymbol} - Historical Data" 
                    : "Select a symbol to view historical data";
            }
            // Validate search button state when symbol search text changes in Individual Asset mode
            else if (e.PropertyName == nameof(_viewModel.SymbolSearchText) && 
                     CurrentSelectionMode == Quantra.Enums.SymbolSelectionMode.IndividualAsset)
            {
                ValidateSearchButtonState();
            }
        }

        /// <summary>
        /// Centralized method to handle symbol selection from any source (ComboBox, DataGrid, etc.)
        /// This modular approach allows reuse of the same logic across different UI components
        /// </summary>
        /// <param name="selectedSymbol">The symbol to select</param>
        /// <param name="source">The source of the selection (for debugging and sync purposes)</param>
        private async Task HandleSymbolSelectionAsync(string selectedSymbol, string source)
        {
            try
            {
                // Set cursor to waiting state immediately when selection starts
                Mouse.OverrideCursor = Cursors.Wait;

                // Cancel any existing symbol operations to prevent resource conflicts
                _symbolOperationCancellation?.Cancel();
                _symbolOperationCancellation?.Dispose();
                _symbolOperationCancellation = new System.Threading.CancellationTokenSource();

                var cancellationToken = _symbolOperationCancellation.Token;

                // Capture the owner window on the UI thread before entering background tasks
                var ownerWindow = Window.GetWindow(this);

                // Set the ViewModel's selected symbol to trigger historical data loading (UI thread operation)
                _viewModel.SelectedSymbol = selectedSymbol;
                
                // Update the selected symbol title immediately to show the ticker code (UI thread operation)
                SelectedSymbolTitle = !string.IsNullOrEmpty(selectedSymbol) 
                    ? $"{selectedSymbol} - Historical Data" 
                    : "Select a symbol to view historical data";

                // Execute data operations in background to avoid blocking UI
                await Task.Run(async () =>
                {
                    try
                    {
                        // Check for cancellation before starting operations
                        cancellationToken.ThrowIfCancellationRequested();

                        // Get symbol data (will use background threads internally)
                        await GetSymbolDataAsync(selectedSymbol, cancellationToken).ConfigureAwait(false);
                        
                        // Check for cancellation before UI updates
                        cancellationToken.ThrowIfCancellationRequested();
                        
                        // Update price and RSI labels on UI thread
                        await Dispatcher.InvokeAsync(async () =>
                        {
                            await UpdatePriceAndRsiLabels(selectedSymbol);
                        });
                        
                        // Check for cancellation before loading indicators
                        cancellationToken.ThrowIfCancellationRequested();
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Operation was cancelled - this is expected when user selects another symbol quickly
                        //DatabaseMonolith.Log("Info", $"Symbol selection for {selectedSymbol} was cancelled due to new selection");
                    }
                    catch (Exception backgroundEx)
                    {
                        // Handle errors on UI thread with proper owner window context
                        await Dispatcher.InvokeAsync(() =>
                        {
                            //DatabaseMonolith.Log("Error", $"Error in background data loading from {source}", backgroundEx.ToString());
                            CustomModal.ShowError($"Error loading symbol data: {backgroundEx.Message}", "Error", ownerWindow);
                        });
                    }
                }, cancellationToken);
                
                // Load indicator data from UI thread (it handles its own threading internally)
                await LoadIndicatorDataAsync(selectedSymbol, cancellationToken);
            }
            catch (System.OperationCanceledException)
            {
                // Operation was cancelled - this is expected behavior when user rapidly selects symbols
                //DatabaseMonolith.Log("Info", $"Symbol selection for {selectedSymbol} from {source} was cancelled due to new selection");
                // Reset cursor when operation is cancelled
                Mouse.OverrideCursor = null;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error in HandleSymbolSelectionAsync from {source}", ex.ToString());
                CustomModal.ShowError($"Error selecting symbol: {ex.Message}", "Error", Window.GetWindow(this));
                // Reset cursor on error since LoadIndicatorDataAsync won't be reached
                Mouse.OverrideCursor = null;
            }
        }

        private async Task GetSymbolDataAsync(string symbol, System.Threading.CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            // Set busy cursor during data loading on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                SharedTitleBar.UpdateDispatcherMonitoring("GetSymbolDataAsync");
                Mouse.OverrideCursor = Cursors.Wait;
            });
            
            try
            {
                // Check for cancellation before starting
                cancellationToken.ThrowIfCancellationRequested();

                // First, try to get cached data (fast operation)
                var cachedStock = _viewModel.CachedStocks?.FirstOrDefault(s => s.Symbol == symbol);
                if (cachedStock != null)
                {
                    // Use cached data and update the ViewModel's selected stock
                    _viewModel.SelectedStock = cachedStock;
                    
                    // Update the LastUpdated timestamp to reflect when it was selected
                    cachedStock.LastUpdated = DateTime.Now;
                    
                    // Notify that cached symbol data was accessed
                    SymbolUpdateService.NotifyCachedSymbolDataAccessed(symbol, "StockExplorer");
                    
                    //DatabaseMonolith.Log("Info", $"Using cached data for symbol: {symbol}");
                    return;
                }

                // Check for cancellation before API call
                cancellationToken.ThrowIfCancellationRequested();

                // If no cached data available, get from API in background thread to avoid UI blocking
                var quoteData = await Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var alphaVantageService = new AlphaVantageService();
                        return await alphaVantageService.GetQuoteDataAsync(symbol).ConfigureAwait(false);
                    }
                    catch (System.OperationCanceledException)
                    {
                        throw; // Re-throw cancellation
                    }
                    catch (Exception apiEx)
                    {
                        // If API fails, try to use any available cached data from the database
                        //DatabaseMonolith.Log("Warning", $"API call failed for {symbol}, attempting to use database cache", apiEx.ToString());
                        
                        cancellationToken.ThrowIfCancellationRequested();
                        var cacheService = new StockDataCacheService();
                        return await cacheService.GetCachedStockAsync(symbol).ConfigureAwait(false);
                    }
                }, cancellationToken).ConfigureAwait(false);
                
                // Check for cancellation before UI updates
                cancellationToken.ThrowIfCancellationRequested();
                
                if (quoteData != null)
                {
                    // Update UI on UI thread using batch updater for better performance
                    _uiBatchUpdater.QueueUpdate($"symbol_data_{symbol}", () =>
                    {
                        SharedTitleBar.UpdateDispatcherMonitoring("GetSymbolDataAsync");
                        
                        // Clear previous data from ViewModel's cached collection to prevent memory leaks
                        var existingStock = _viewModel.CachedStocks.FirstOrDefault(s => s.Symbol == symbol);
                        if (existingStock != null)
                        {
                            existingStock.Dispose(); // Free memory
                            _viewModel.CachedStocks.Remove(existingStock);
                        }
                        
                        // Update the ViewModel's selected stock
                        _viewModel.SelectedStock = quoteData;
                        
                        // Add to the cached stocks collection
                        _viewModel.CachedStocks.Add(quoteData);
                    });
                    
                    // Cache the complete quote data in background
                    lock (_backgroundTasksLock)
                    {
                        _backgroundTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                cancellationToken.ThrowIfCancellationRequested();
                                var cacheService = new StockDataCacheService();
                                await cacheService.CacheQuoteDataAsync(quoteData).ConfigureAwait(false);
                            }
                            catch (System.OperationCanceledException)
                            {
                                // Caching was cancelled - this is fine
                            }
                            catch (Exception cacheEx)
                            {
                                //DatabaseMonolith.Log("Warning", $"Failed to cache data for {symbol}", cacheEx.ToString());
                            }
                        }, cancellationToken));
                    }
                    
                    // Notify other components that symbol data has been retrieved
                    SymbolUpdateService.NotifySymbolDataRetrieved(symbol, "StockExplorer");
                    
                    //DatabaseMonolith.Log("Info", $"Successfully loaded fresh data for symbol: {symbol}");
                }
                else
                {
                    //DatabaseMonolith.Log("Warning", $"No data available for symbol: {symbol}");
                }
            }
            catch (System.OperationCanceledException)
            {
                //DatabaseMonolith.Log("Info", $"Symbol data loading for {symbol} was cancelled");
                throw; // Re-throw to handle at higher level
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error getting symbol data for {symbol}", ex.ToString());
                throw;
            }
            finally
            {
                // Always reset cursor back to normal on UI thread using batch updater
                _uiBatchUpdater.QueueUpdate("reset_cursor", () =>
                {
                    SharedTitleBar.UpdateDispatcherMonitoring("GetSymbolDataAsync");
                    Mouse.OverrideCursor = null;
                });
            }
        }

        private async Task GetSymbolData(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            // Set busy cursor during data loading on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                Mouse.OverrideCursor = Cursors.Wait;
            });
            
            try
            {
                // First, try to get cached data (fast operation)
                var cachedStock = _viewModel.CachedStocks?.FirstOrDefault(s => s.Symbol == symbol);
                if (cachedStock != null)
                {
                    // Use cached data and update the ViewModel's selected stock
                    _viewModel.SelectedStock = cachedStock;
                    
                    // Update the LastUpdated timestamp to reflect when it was selected
                    cachedStock.LastUpdated = DateTime.Now;
                    
                    // Notify that cached symbol data was accessed
                    SymbolUpdateService.NotifyCachedSymbolDataAccessed(symbol, "StockExplorer");
                    
                    //DatabaseMonolith.Log("Info", $"Using cached data for symbol: {symbol}");
                    return;
                }

                // If no cached data available, get from API in background thread to avoid UI blocking
                var quoteData = await Task.Run(async () =>
                {
                    try
                    {
                        var alphaVantageService = new AlphaVantageService();
                        return await alphaVantageService.GetQuoteDataAsync(symbol).ConfigureAwait(false);
                    }
                    catch (Exception apiEx)
                    {
                        // If API fails, try to use any available cached data from the database
                        //DatabaseMonolith.Log("Warning", $"API call failed for {symbol}, attempting to use database cache", apiEx.ToString());
                        
                        var cacheService = new StockDataCacheService();
                        return await cacheService.GetCachedStockAsync(symbol).ConfigureAwait(false);
                    }
                }).ConfigureAwait(false);
                
                if (quoteData != null)
                {
                    // Update UI on UI thread
                    await Dispatcher.InvokeAsync(() =>
                    {
                        // Update the ViewModel's selected stock
                        _viewModel.SelectedStock = quoteData;
                        
                        // Add to the cached stocks collection
                        _viewModel.CachedStocks.Add(quoteData);
                    });
                    
                    // Cache the complete quote data in background
                    lock (_backgroundTasksLock)
                    {
                        _backgroundTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var cacheService = new StockDataCacheService();
                                await cacheService.CacheQuoteDataAsync(quoteData).ConfigureAwait(false);
                            }
                            catch (Exception cacheEx)
                            {
                                //DatabaseMonolith.Log("Warning", $"Failed to cache data for {symbol}", cacheEx.ToString());
                            }
                        }));
                    }
                    
                    // Notify other components that symbol data has been retrieved
                    SymbolUpdateService.NotifySymbolDataRetrieved(symbol, "StockExplorer");
                    
                    //DatabaseMonolith.Log("Info", $"Successfully loaded fresh data for symbol: {symbol}");
                }
                else
                {
                    //DatabaseMonolith.Log("Warning", $"No data available for symbol: {symbol}");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error getting symbol data for {symbol}", ex.ToString());
                throw;
            }
            finally
            {
                // Always reset cursor back to normal on UI thread
                await Dispatcher.InvokeAsync(() =>
                {
                    Mouse.OverrideCursor = null;
                });
            }
        }

        private async Task RefreshSymbolDataFromAPI(string symbol, System.Threading.CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            // Set busy cursor during data refresh on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                Mouse.OverrideCursor = Cursors.Wait;
            });
            
            try
            {
                // Check for cancellation before starting
                cancellationToken.ThrowIfCancellationRequested();

                // Always get fresh data from the API when explicitly refreshing - run in background
                var quoteData = await Task.Run(async () =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var alphaVantageService = new AlphaVantageService();
                    return await alphaVantageService.GetQuoteDataAsync(symbol).ConfigureAwait(false);
                }, cancellationToken).ConfigureAwait(false);
                
                if (quoteData != null)
                {
                    // Check for cancellation before UI updates
                    cancellationToken.ThrowIfCancellationRequested();

                    // Update UI on UI thread
                    await Dispatcher.InvokeAsync(() =>
                    {
                        // Update the ViewModel's selected stock
                        _viewModel.SelectedStock = quoteData;
                        
                        // Update or add to the cached stocks collection
                        var existing = _viewModel.CachedStocks.FirstOrDefault(s => s.Symbol == symbol);
                        if (existing != null)
                        {
                            var index = _viewModel.CachedStocks.IndexOf(existing);
                            _viewModel.CachedStocks[index] = quoteData;
                        }
                        else
                        {
                            _viewModel.CachedStocks.Add(quoteData);
                        }
                    });
                    
                    // Cache the complete quote data in background
                    lock (_backgroundTasksLock)
                    {
                        _backgroundTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var cacheService = new StockDataCacheService();
                                await cacheService.CacheQuoteDataAsync(quoteData).ConfigureAwait(false);
                            }
                            catch (Exception cacheEx)
                            {
                                //DatabaseMonolith.Log("Warning", $"Failed to cache refreshed data for {symbol}", cacheEx.ToString());
                            }
                        }));
                    }
                    
                    // Notify other components that symbol data has been retrieved
                    SymbolUpdateService.NotifySymbolDataRetrieved(symbol, "StockExplorer");
                    
                    //DatabaseMonolith.Log("Info", $"Successfully refreshed data for symbol: {symbol}");
                }

                // Wait for all background tasks to complete
                Task[] tasksToWait;
                lock (_backgroundTasksLock)
                {
                    tasksToWait = _backgroundTasks.ToArray();
                }
                await Task.WhenAll(tasksToWait);
            }
            catch (System.OperationCanceledException)
            {
                // Operation was cancelled - this is expected, don't log as error
                //DatabaseMonolith.Log("Info", $"Symbol data refresh for {symbol} was cancelled");
                // Don't re-throw cancellation exceptions in refresh operations
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error refreshing symbol data for {symbol}", ex.ToString());
                throw;
            }
            finally
            {
                // Always reset cursor back to normal on UI thread
                await Dispatcher.InvokeAsync(() =>
                {
                    Mouse.OverrideCursor = null;
                });
            }
        }

        // Time Range Button Click Handler
        private async void TimeRangeButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string timeRange)
            {
                try
                {
                    // Update the current time range in the ViewModel
                    _viewModel.CurrentTimeRange = timeRange;
                    
                    // Update button styles
                    UpdateTimeRangeButtonStyles(timeRange);
                    
                    // If we have a selected symbol, reload its chart data with the new time range
                    if (!string.IsNullOrEmpty(_viewModel.SelectedSymbol))
                    {
                        await LoadChartDataForTimeRange(_viewModel.SelectedSymbol, timeRange);
                    }
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Error changing time range to {timeRange}", ex.ToString());
                    CustomModal.ShowError($"Error changing time range: {ex.Message}", "Error", Window.GetWindow(this));
                }
            }
        }

        // Update time range button styles to show active selection
        private void UpdateTimeRangeButtonStyles(string activeTimeRange)
        {
            // Reset all buttons to default style
            var buttons = new[] { TimeRange1D, TimeRange5D, TimeRange1M, TimeRange6M, TimeRange1Y, TimeRange5Y, TimeRangeAll };
            
            foreach (var button in buttons)
            {
                if (button != null)
                {
                    button.Background = new SolidColorBrush(Color.FromRgb(0x3A, 0x6E, 0xA5)); // Default blue
                }
            }
            
            // Highlight the active button
            Button activeButton = activeTimeRange switch
            {
                "1day" => TimeRange1D,
                "5day" => TimeRange5D,
                "1mo" => TimeRange1M,
                "6mo" => TimeRange6M,
                "1y" => TimeRange1Y,
                "5y" => TimeRange5Y,
                "all" => TimeRangeAll,
                _ => TimeRange1D // Default to 1D instead of 1M
            };
            
            if (activeButton != null)
            {
                activeButton.Background = new SolidColorBrush(Color.FromRgb(0x6A, 0x5A, 0xCD)); // Purple for active
            }
        }

        // Load chart data for specific time range
        private async Task LoadChartDataForTimeRange(string symbol, string timeRange)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            // Set busy cursor during chart data loading on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                Mouse.OverrideCursor = Cursors.Wait;
            });
            
            try
            {
                // Use the ViewModel method to load chart data for the time range
                await _viewModel.LoadChartDataForTimeRangeAsync(symbol, timeRange);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error loading chart data for {symbol} with time range {timeRange}", ex.ToString());
                throw;
            }
            finally
            {
                // Always reset cursor back to normal on UI thread
                await Dispatcher.InvokeAsync(() =>
                {
                    Mouse.OverrideCursor = null;
                });
            }
        }

        /// <summary>
        /// Synchronizes UI components to reflect the selected symbol
        /// Ensures ComboBox and DataGrid selections stay in sync
        /// </summary>
        /// <param name="selectedSymbol">The selected symbol</param>
        /// <param name="source">The source component that triggered the selection</param>
        private async Task SynchronizeUIComponents(string selectedSymbol, string source)
        {
            try
            {
                // Update ComboBox selection if source was not ComboBox
                if (source != "ComboBox" && SymbolComboBox != null)
                {
                    SymbolComboBox.SelectedItem = selectedSymbol;
                }
                
                // TODO Future: Add other UI synchronization logic here
                // For example, if there were other selection components like a symbol list, 
                // search results, or chart click handlers, their sync logic would go here
                
                // Note: DataGrid synchronization happens automatically via ViewModel binding
                // since SelectedSymbol change triggers data loading which updates the grid
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error synchronizing UI components", ex.ToString());
            }
        }

        // StockDataGrid selection changed event handler
        private async void StockDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHandlingSelectionChanged || sender is not DataGrid dataGrid) 
                return;

            if (dataGrid.SelectedItem is QuoteData selectedQuote)
            {
                await HandleSymbolSelectionAsync(selectedQuote.Symbol, "DataGrid");
            }
        }

        private async Task UpdatePriceAndRsiLabels(string symbol)
        {
            try
            {
                // Update symbol label
                SymbolText = !string.IsNullOrEmpty(symbol) ? $"Symbol: {symbol}" : "";
                
                // Use data already available in the ViewModel to avoid unnecessary API calls
                if (_viewModel.SelectedStock != null)
                {
                    PriceText = $"Price: ${_viewModel.SelectedStock.Price:F2}";
                    
                    // Set LastAccessed to current time when displaying data
                    _viewModel.SelectedStock.LastAccessed = DateTime.Now;
                    
                    UpdatedTimestampText = _viewModel.SelectedStock.LastAccessed != default(DateTime) 
                        ? $"Updated: {_viewModel.SelectedStock.LastAccessed:MM/dd/yyyy HH:mm}"
                        : "Updated: Never";
                }
                else
                {
                    // Try to get cached data
                    var cachedStock = _viewModel.CachedStocks?.FirstOrDefault(s => s.Symbol == symbol);
                    if (cachedStock != null)
                    {
                        PriceText = $"Price: ${cachedStock.Price:F2}";
                        
                        // Set LastAccessed to current time when accessing cached data
                        cachedStock.LastAccessed = DateTime.Now;
                        
                        UpdatedTimestampText = cachedStock.LastAccessed != default(DateTime) 
                            ? $"Updated: {cachedStock.LastAccessed:MM/dd/yyyy HH:mm}"
                            : "Updated: Never";
                        // Notify that cached symbol data was accessed
                        SymbolUpdateService.NotifyCachedSymbolDataAccessed(symbol, "StockExplorer");
                    }
                    else
                    {
                        PriceText = "Price: --";
                        UpdatedTimestampText = "";
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error updating price and symbol labels", ex.ToString());
                SymbolText = !string.IsNullOrEmpty(symbol) ? $"Symbol: {symbol}" : "";
                PriceText = "Price: --";
                UpdatedTimestampText = "";
            }
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            // In Individual Asset mode, search for the typed symbol
            if (CurrentSelectionMode == Quantra.Enums.SymbolSelectionMode.IndividualAsset)
            {
                var searchText = _viewModel.SymbolSearchText?.ToUpper().Trim();
                if (!string.IsNullOrEmpty(searchText) && _viewModel.FilteredSymbols.Contains(searchText))
                {
                    try
                    {
                        // Load data for the typed symbol
                        await HandleSymbolSelectionAsync(searchText, "SearchButton");
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Operation was cancelled - handle gracefully for search
                        //DatabaseMonolith.Log("Info", "Stock symbol search was cancelled");
                        CustomModal.ShowWarning("Symbol search was cancelled.", "Operation Cancelled", Window.GetWindow(this));
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", "Error searching for stock data", ex.ToString());
                        CustomModal.ShowError($"Error searching for data: {ex.Message}", "Error", Window.GetWindow(this));
                    }
                }
                else
                {
                    CustomModal.ShowWarning("Please enter a valid stock symbol to search.", "Invalid Symbol", Window.GetWindow(this));
                }
            }
            else if (!string.IsNullOrEmpty(_viewModel.SelectedSymbol))
            {
                // Create a new cancellation token source for the refresh operation
                using var refreshCancellation = new System.Threading.CancellationTokenSource();
                var refreshToken = refreshCancellation.Token;
                
                try
                {
                    // Refresh the data for the currently selected symbol from API (for other modes)
                    await RefreshSymbolDataFromAPI(_viewModel.SelectedSymbol, refreshToken);
                    
                    // Update the price and RSI labels
                    await UpdatePriceAndRsiLabels(_viewModel.SelectedSymbol);
                    
                    // Reload indicator data asynchronously to avoid blocking UI
                    await LoadIndicatorDataAsync(_viewModel.SelectedSymbol, refreshToken);
                }
                catch (System.OperationCanceledException)
                {
                    // Operation was cancelled - this shouldn't normally happen for refresh but handle gracefully
                    //DatabaseMonolith.Log("Info", "Stock data refresh was cancelled");
                    CustomModal.ShowWarning("Data refresh was cancelled.", "Operation Cancelled", Window.GetWindow(this));
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", "Error refreshing stock data", ex.ToString());
                    CustomModal.ShowError($"Error refreshing data: {ex.Message}", "Error", Window.GetWindow(this));
                }
            }
            else
            {
                CustomModal.ShowWarning("Please select a stock symbol to refresh.", "No Symbol Selected", Window.GetWindow(this));
            }
        }

        private async Task LoadIndicatorDataAsync(string symbol, System.Threading.CancellationToken cancellationToken = default)
        {
            try
            {
                // Check for cancellation before starting
                cancellationToken.ThrowIfCancellationRequested();

                // Set UI state on UI thread
                await Dispatcher.InvokeAsync(() =>
                {
                    SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                    // Set busy cursor during indicator loading
                    Mouse.OverrideCursor = Cursors.Wait;
                    // Set global loading state for calculations
                    GlobalLoadingStateService.SetLoadingState(true);
                });
                
                var alphaVantageService = new AlphaVantageService();
                
                // Instead of clearing and rebuilding, update individual indicator properties
                // This eliminates the flashing UI by maintaining consistent elements
                // Collect all indicator calculation tasks to await them together
                var indicatorTasks = new List<Task>();

                // RSI, MACD, VWAP and most influential indicators should be first
                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var rsi = await alphaVantageService.GetRSI(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            RsiValue = FormatIndicatorValue("RSI", rsi);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load RSI for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            RsiValue = "N/A";
                        });
                    }
                }, cancellationToken));

                // Get P/E Ratio (fundamental indicator)
                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var peRatio = await alphaVantageService.GetPERatioAsync(symbol).ConfigureAwait(false);
                        if (peRatio.HasValue)
                        {
                            await Dispatcher.InvokeAsync(() => {
                                SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                                PeRatioValue = FormatIndicatorValue("P/E Ratio", peRatio.Value);
                            });
                        }
                        else
                        {
                            await Dispatcher.InvokeAsync(() => {
                                SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                                PeRatioValue = "N/A";
                            });
                        }
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load P/E Ratio for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            PeRatioValue = "N/A";
                        });
                    }
                }, cancellationToken));

                // Get MACD components
                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var macdData = await alphaVantageService.GetMACD(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() =>
                        {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            MacdValue = FormatIndicatorValue("MACD", macdData.Macd);
                            MacdSignalValue = FormatIndicatorValue("MACD Signal", macdData.MacdSignal);
                            MacdHistValue = FormatIndicatorValue("MACD Hist", macdData.MacdHist);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load MACD for {symbol}", ex.ToString());
                        // If MACD fails, set placeholder values
                        await Dispatcher.InvokeAsync(() =>
                        {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            MacdValue = "N/A";
                            MacdSignalValue = "N/A";
                            MacdHistValue = "N/A";
                        });
                    }
                }, cancellationToken));

                // Get VWAP
                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var vwap = await alphaVantageService.GetVWAP(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            VwapValue = FormatIndicatorValue("VWAP", vwap);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load VWAP for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            VwapValue = "N/A";
                        });
                    }
                }, cancellationToken));

                // Add other influential indicators
                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var adx = await alphaVantageService.GetLatestADX(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            AdxValue = FormatIndicatorValue("ADX", adx);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load ADX for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            AdxValue = "N/A";
                        });
                    }
                }, cancellationToken));

                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var cci = await alphaVantageService.GetCCI(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            CciValue = FormatIndicatorValue("CCI", cci);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load CCI for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            CciValue = "N/A";
                        });
                    }
                }, cancellationToken));

                // Add additional indicators
                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var atr = await alphaVantageService.GetATR(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            AtrValue = FormatIndicatorValue("ATR", atr);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load ATR for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            AtrValue = "N/A";
                        });
                    }
                }, cancellationToken));

                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var mfi = await alphaVantageService.GetMFI(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            MfiValue = FormatIndicatorValue("MFI", mfi);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load MFI for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() => {
                            SharedTitleBar.UpdateDispatcherMonitoring("LoadIndicatorDataAsync");
                            MfiValue = "N/A";
                        });
                    }
                }, cancellationToken));

                indicatorTasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var stoch = await alphaVantageService.GetSTOCH(symbol).ConfigureAwait(false);
                        await Dispatcher.InvokeAsync(() =>
                        {
                            StochKValue = FormatIndicatorValue("Stoch K", stoch.StochK);
                            StochDValue = FormatIndicatorValue("Stoch D", stoch.StochD);
                        });
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Indicator loading was cancelled - this is fine
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Failed to load STOCH for {symbol}", ex.ToString());
                        await Dispatcher.InvokeAsync(() =>
                        {
                            StochKValue = "N/A";
                            StochDValue = "N/A";
                        });
                    }
                }, cancellationToken));

                // Wait for all indicator tasks to complete
                await Task.WhenAll(indicatorTasks);

                // Keep the CurrentIndicators collection for backward compatibility
                // Note: We can gradually phase out this collection in favor of the individual properties
                // For now, keep it empty to avoid any flashing while maintaining compatibility
                cancellationToken.ThrowIfCancellationRequested();
                await Dispatcher.InvokeAsync(() =>
                {
                    if (CurrentIndicators.Count > 0)
                    {
                        CurrentIndicators.Clear();
                    }
                });
            }
            catch (System.OperationCanceledException)
            {
                //DatabaseMonolith.Log("Info", $"Indicator loading for {symbol} was cancelled");
                throw; // Re-throw to handle at higher level
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error loading indicator data for {symbol}", ex.ToString());
                // Set all indicator values to error state using batched update
                _uiBatchUpdater.QueueUpdate("error_indicators", () =>
                {
                    RsiValue = "Error";
                    PeRatioValue = "Error";
                    MacdValue = "Error";
                    MacdSignalValue = "Error";
                    MacdHistValue = "Error";
                    VwapValue = "Error";
                    AdxValue = "Error";
                    CciValue = "Error";
                    AtrValue = "Error";
                    MfiValue = "Error";
                    StochKValue = "Error";
                    StochDValue = "Error";
                });
                await _uiBatchUpdater.FlushUpdates();
            }
            finally
            {
                // Clear global loading state and reset cursor using batched update
                _uiBatchUpdater.QueueUpdate("cleanup_ui", () =>
                {
                    GlobalLoadingStateService.SetLoadingState(false);
                    Mouse.OverrideCursor = null;
                });
                await _uiBatchUpdater.FlushUpdates();
            }
        }

        // Helper methods for different selection modes

        private async Task<List<QuoteData>> GetTopVolumeRsiDiscrepancies()
        {
            var result = new List<QuoteData>();
            var startTime = DateTime.Now;
            
            try
            {
                //DatabaseMonolith.Log("Info", "Starting dynamic Top Volume RSI Discrepancies algorithm");
                
                // Phase 1: Build comprehensive stock universe from multiple sources
                var stockUniverse = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Phase 1 Complete: Built stock universe with {stockUniverse.Count} symbols");
                
                // Phase 2: Pre-filter for volume leaders to identify catalytic stocks
                var volumeLeaders = await IdentifyVolumeLeaders(stockUniverse);
                //DatabaseMonolith.Log("Info", $"Phase 2 Complete: Identified {volumeLeaders.Count} volume leaders from stock universe");
                
                // Phase 3: Batch process volume leaders for RSI discrepancies
                var rsiDiscrepancies = await BatchProcessRsiDiscrepancies(volumeLeaders);
                //DatabaseMonolith.Log("Info", $"Phase 3 Complete: Found {rsiDiscrepancies.Count} stocks with RSI discrepancies");
                
                // Phase 4: Calculate catalytic scores and rank results
                var rankedResults = await CalculateCatalyticScores(rsiDiscrepancies);
                //DatabaseMonolith.Log("Info", $"Phase 4 Complete: Calculated catalytic scores for {rankedResults.Count} stocks");
                
                // Phase 5: Return top 100 dynamic results
                result = rankedResults.Take(100).ToList();
                
                var duration = DateTime.Now - startTime;
                //DatabaseMonolith.Log("Info", $"Dynamic algorithm completed in {duration.TotalSeconds:F2} seconds, returning {result.Count} top catalytic stocks");
                
                return result;
            }
            catch (Exception ex)
            {
                var duration = DateTime.Now - startTime;
                //DatabaseMonolith.Log("Error", $"Error in dynamic top volume RSI discrepancies algorithm after {duration.TotalSeconds:F2} seconds", ex.ToString());
                return result;
            }
        }

        /// <summary>
        /// Phase 1: Build a comprehensive and dynamic stock universe from multiple sources
        /// </summary>
        private async Task<List<string>> BuildDynamicStockUniverse()
        {
            var universe = new HashSet<string>();
            
            // Start with core liquid symbols as base universe
            var coreSymbols = StockSymbols.CommonSymbols;
            foreach (var symbol in coreSymbols)
            {
                universe.Add(symbol);
            }
            //DatabaseMonolith.Log("Info", $"Added {coreSymbols.Count} core symbols to universe");
            
            // Add major index components and ETFs for broader market coverage
            var indexComponents = new[]
            {
                // Major tech and growth stocks
                "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "AMD", "NFLX",
                "ADBE", "CRM", "ORCL", "INTC", "CSCO", "PYPL", "SHOP", "SQ", "ROKU", "ZM",
                
                // Financial sector leaders
                "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "BRK.B",
                
                // Healthcare and biotech
                "JNJ", "PFE", "UNH", "MRNA", "BNTX", "GILD", "AMGN", "BMY", "LLY", "ABT",
                
                // Energy and commodities
                "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "MRO", "DVN", "EOG", "PXD",
                
                // Consumer and retail
                "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "DIS", "COST", "AMZN",
                
                // High-beta and momentum stocks
                "GME", "AMC", "BB", "NOK", "PLTR", "SPCE", "WISH", "CLOV", "SOFI", "RIVN",
                "LCID", "F", "COIN", "HOOD", "UPST", "ARKK", "QQQ", "SPY", "IWM", "VTI"
            };
            
            foreach (var symbol in indexComponents)
            {
                universe.Add(symbol);
            }
            //DatabaseMonolith.Log("Info", $"Added {indexComponents.Length} index components, total universe: {universe.Count}");
            
            return universe.ToList();
        }

        /// <summary>
        /// Identify high volume stocks from the stock universe (for High Volume mode)
        /// </summary>
        private async Task<List<(string Symbol, double Volume, double Price, double ChangePercent)>> IdentifyHighVolumeStocks(List<string> stockUniverse)
        {
            var highVolumeStocks = new List<(string Symbol, double Volume, double Price, double ChangePercent)>();
            var alphaVantageService = new AlphaVantageService();
            var batchSize = 20; // Process in smaller batches to avoid API rate limits
            var processedCount = 0;
            
            //DatabaseMonolith.Log("Info", $"Processing {stockUniverse.Count} symbols in batches of {batchSize} to identify high volume stocks");
            
            for (int i = 0; i < stockUniverse.Count; i += batchSize)
            {
                var batch = stockUniverse.Skip(i).Take(batchSize).ToList();
                var batchTasks = new List<Task>();
                
                foreach (var symbol in batch)
                {
                    batchTasks.Add(Task.Run(async () =>
                    {
                        try
                        {
                            var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                            if (quote != null)
                            {
                                // Include all stocks with reasonable volume (no additional filters for pure high volume mode)
                                var volumeThreshold = 100000; // Minimum 100K volume to filter out very low-volume stocks
                                
                                if (quote.Volume >= volumeThreshold)
                                {
                                    lock (highVolumeStocks)
                                    {
                                        highVolumeStocks.Add((symbol, quote.Volume, quote.Price, quote.ChangePercent));
                                    }
                                    //DatabaseMonolith.Log("Debug", $"High volume stock: {symbol} - Volume: {quote.Volume:N0}");
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Error processing volume data for {symbol}", ex.ToString());
                        }
                    }));
                }
                
                await Task.WhenAll(batchTasks);
                processedCount += batch.Count;
                
                // Add delay between batches to respect API limits
                if (i + batchSize < stockUniverse.Count)
                {
                    await Task.Delay(2000); // 2 second delay between batches
                }
                
                //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{stockUniverse.Count}");
            }
            
            // Sort by volume to get the highest volume stocks
            highVolumeStocks = highVolumeStocks
                .OrderByDescending(x => x.Volume)
                .Take(50) // Take top 50 by volume for further processing
                .ToList();
                
            //DatabaseMonolith.Log("Info", $"Selected top {highVolumeStocks.Count} high volume stocks");
            return highVolumeStocks;
        }

        /// <summary>
        /// Convert volume leader tuples to QuoteData objects with additional indicator data
        /// </summary>
        private async Task<List<QuoteData>> ConvertToQuoteDataList(List<(string Symbol, double Volume, double Price, double ChangePercent)> volumeStocks)
        {
            var result = new List<QuoteData>();
            var alphaVantageService = new AlphaVantageService();
            var batchSize = 15; // Smaller batches for indicator calls
            var processedCount = 0;
            
            //DatabaseMonolith.Log("Info", $"Converting {volumeStocks.Count} stocks to QuoteData format in batches of {batchSize}");
            
            for (int i = 0; i < volumeStocks.Count; i += batchSize)
            {
                var batch = volumeStocks.Skip(i).Take(batchSize).ToList();
                var batchTasks = new List<Task>();
                
                foreach (var (symbol, volume, price, changePercent) in batch)
                {
                    batchTasks.Add(Task.Run(async () =>
                    {
                        try
                        {
                            var extendedQuote = new QuoteData
                            {
                                Symbol = symbol,
                                Price = price,
                                Volume = volume,
                                ChangePercent = changePercent,
                                DayHigh = 0, // Will be populated by indicator calls if available
                                DayLow = 0,  // Will be populated by indicator calls if available
                                LastUpdated = DateTime.Now
                            };

                            // Try to get RSI data
                            try
                            {
                                var rsi = await alphaVantageService.GetRSI(symbol);
                                extendedQuote.RSI = rsi;
                            }
                            catch
                            {
                                extendedQuote.RSI = 0;
                            }

                            // Try to get P/E ratio data
                            try
                            {
                                var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                extendedQuote.PERatio = peRatio ?? 0;
                            }
                            catch
                            {
                                extendedQuote.PERatio = 0;
                            }

                            // Try to get VWAP data
                            try
                            {
                                var vwap = await alphaVantageService.GetVWAP(symbol);
                                extendedQuote.VWAP = vwap;
                            }
                            catch
                            {
                                extendedQuote.VWAP = 0;
                            }

                            lock (result)
                            {
                                result.Add(extendedQuote);
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Error converting {symbol} to QuoteData", ex.ToString());
                        }
                    }));
                }
                
                await Task.WhenAll(batchTasks);
                processedCount += batch.Count;
                
                // Add delay between batches to respect API limits
                if (i + batchSize < volumeStocks.Count)
                {
                    await Task.Delay(2000); // 2 second delay between batches
                }
                
                //DatabaseMonolith.Log("Info", $"Converted batch {i / batchSize + 1}, total stocks processed: {processedCount}/{volumeStocks.Count}");
            }
            
            return result;
        }

        /// <summary>
        /// Phase 2: Identify volume leaders to focus on catalytic stocks
        /// </summary>
        private async Task<List<(string Symbol, double Volume, double Price, double ChangePercent)>> IdentifyVolumeLeaders(List<string> stockUniverse)
        {
            var volumeLeaders = new List<(string Symbol, double Volume, double Price, double ChangePercent)>();
            var alphaVantageService = new AlphaVantageService();
            var batchSize = 20; // Process in smaller batches to avoid API rate limits
            var processedCount = 0;
            
            //DatabaseMonolith.Log("Info", $"Processing {stockUniverse.Count} symbols in batches of {batchSize} to identify volume leaders");
            
            for (int i = 0; i < stockUniverse.Count; i += batchSize)
            {
                var batch = stockUniverse.Skip(i).Take(batchSize).ToList();
                var batchTasks = new List<Task>();
                
                foreach (var symbol in batch)
                {
                    batchTasks.Add(Task.Run(async () =>
                    {
                        try
                        {
                            var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                            if (quote != null)
                            {
                                // Criteria for catalytic stocks: High volume AND significant price movement
                                var volumeThreshold = 500000; // Minimum 500K volume
                                var priceMovementThreshold = 2.0; // Minimum 2% price change
                                
                                if (quote.Volume >= volumeThreshold && Math.Abs(quote.ChangePercent) >= priceMovementThreshold)
                                {
                                    lock (volumeLeaders)
                                    {
                                        volumeLeaders.Add((symbol, quote.Volume, quote.Price, quote.ChangePercent));
                                    }
                                    //DatabaseMonolith.Log("Debug", $"Volume leader identified: {symbol} - Volume: {quote.Volume:N0}, Change: {quote.ChangePercent:F2}%");
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Error processing volume data for {symbol}", ex.ToString());
                        }
                    }));
                }
                
                await Task.WhenAll(batchTasks);
                processedCount += batch.Count;
                
                // Add delay between batches to respect API limits
                if (i + batchSize < stockUniverse.Count)
                {
                    await Task.Delay(2000); // 2 second delay between batches
                }
                
                //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{stockUniverse.Count}");
            }
            
            // Sort by volume * price movement score to prioritize most catalytic stocks
            volumeLeaders = volumeLeaders
                .OrderByDescending(x => x.Volume * Math.Abs(x.ChangePercent))
                .Take(150) // Take top 150 volume leaders for RSI analysis
                .ToList();
                
            //DatabaseMonolith.Log("Info", $"Selected top {volumeLeaders.Count} volume leaders for RSI discrepancy analysis");
            return volumeLeaders;
        }

        /// <summary>
        /// Phase 3: Batch process volume leaders for RSI discrepancies
        /// </summary>
        private async Task<List<QuoteData>> BatchProcessRsiDiscrepancies(List<(string Symbol, double Volume, double Price, double ChangePercent)> volumeLeaders)
        {
            var rsiDiscrepancies = new List<QuoteData>();
            var alphaVantageService = new AlphaVantageService();
            var batchSize = 15; // Smaller batches for RSI calls which are more expensive
            var processedCount = 0;
            
            //DatabaseMonolith.Log("Info", $"Analyzing RSI discrepancies for {volumeLeaders.Count} volume leaders in batches of {batchSize}");
            
            for (int i = 0; i < volumeLeaders.Count; i += batchSize)
            {
                var batch = volumeLeaders.Skip(i).Take(batchSize).ToList();
                var batchTasks = new List<Task>();
                
                foreach (var leader in batch)
                {
                    batchTasks.Add(Task.Run(async () =>
                    {
                        try
                        {
                            var rsi = await alphaVantageService.GetRSI(leader.Symbol);
                            
                            // Enhanced RSI discrepancy criteria
                            var isRsiDiscrepancy = false;
                            var discrepancyType = "";
                            var discrepancyScore = 0.0;
                            
                            if (rsi <= 25) // Extremely oversold
                            {
                                isRsiDiscrepancy = true;
                                discrepancyType = "Extremely Oversold";
                                discrepancyScore = (30 - rsi) * 2; // Higher score for more extreme oversold
                            }
                            else if (rsi <= 30) // Standard oversold
                            {
                                isRsiDiscrepancy = true;
                                discrepancyType = "Oversold";
                                discrepancyScore = 30 - rsi;
                            }
                            else if (rsi >= 75) // Extremely overbought
                            {
                                isRsiDiscrepancy = true;
                                discrepancyType = "Extremely Overbought";
                                discrepancyScore = (rsi - 70) * 2; // Higher score for more extreme overbought
                            }
                            else if (rsi >= 70) // Standard overbought
                            {
                                isRsiDiscrepancy = true;
                                discrepancyType = "Overbought";
                                discrepancyScore = rsi - 70;
                            }
                            
                            if (isRsiDiscrepancy)
                            {
                                // Get additional quote data for complete analysis
                                var quote = await alphaVantageService.GetQuoteDataAsync(leader.Symbol);
                                if (quote != null)
                                {
                                    var extendedQuote = new QuoteData
                                    {
                                        Symbol = leader.Symbol,
                                        Price = leader.Price,
                                        Volume = leader.Volume,
                                        ChangePercent = leader.ChangePercent,
                                        DayHigh = quote.DayHigh,
                                        DayLow = quote.DayLow,
                                        RSI = rsi,
                                        LastUpdated = DateTime.Now
                                    };
                                    
                                    // Store discrepancy metadata for scoring
                                    extendedQuote.PredictedAction = discrepancyType; // Temporarily store in this field
                                    extendedQuote.PredictionConfidence = discrepancyScore; // Temporarily store score
                                    
                                    // Try to get P/E ratio for fundamental analysis
                                    try
                                    {
                                        var peRatio = await alphaVantageService.GetPERatioAsync(leader.Symbol);
                                        extendedQuote.PERatio = peRatio ?? 0;
                                    }
                                    catch
                                    {
                                        extendedQuote.PERatio = 0;
                                    }
                                    
                                    // Try to get VWAP data
                                    try
                                    {
                                        var vwap = await alphaVantageService.GetVWAP(leader.Symbol);
                                        extendedQuote.VWAP = vwap;
                                    }
                                    catch
                                    {
                                        extendedQuote.VWAP = 0;
                                    }
                                    
                                    lock (rsiDiscrepancies)
                                    {
                                        rsiDiscrepancies.Add(extendedQuote);
                                    }
                                    
                                    //DatabaseMonolith.Log("Debug", $"RSI discrepancy found: {leader.Symbol} - RSI: {rsi:F1} ({discrepancyType}), Volume: {leader.Volume:N0}, Score: {discrepancyScore:F2}");
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Error analyzing RSI for {leader.Symbol}", ex.ToString());
                        }
                    }));
                }
                
                await Task.WhenAll(batchTasks);
                processedCount += batch.Count;
                
                // Add delay between RSI batches (RSI calls are more expensive)
                if (i + batchSize < volumeLeaders.Count)
                {
                    await Task.Delay(3000); // 3 second delay between RSI batches
                }
                
                //DatabaseMonolith.Log("Info", $"Completed RSI batch {i / batchSize + 1}, analyzed: {processedCount}/{volumeLeaders.Count}, found {rsiDiscrepancies.Count} discrepancies so far");
            }
            
            //DatabaseMonolith.Log("Info", $"RSI discrepancy analysis complete: {rsiDiscrepancies.Count} stocks with RSI discrepancies identified");
            return rsiDiscrepancies;
        }

        /// <summary>
        /// Phase 4: Calculate catalytic scores combining volume, RSI discrepancy, and price momentum
        /// </summary>
        private async Task<List<QuoteData>> CalculateCatalyticScores(List<QuoteData> rsiDiscrepancies)
        {
            //DatabaseMonolith.Log("Info", $"Calculating catalytic scores for {rsiDiscrepancies.Count} stocks with RSI discrepancies");
            
            var scoredStocks = new List<QuoteData>();
            
            foreach (var stock in rsiDiscrepancies)
            {
                try
                {
                    // Multi-factor catalytic score calculation
                    var volumeScore = Math.Log10(stock.Volume) * 10; // Logarithmic volume scaling
                    var rsiDiscrepancyScore = stock.PredictionConfidence; // Retrieved from temporary storage
                    var priceMovementScore = Math.Abs(stock.ChangePercent) * 5; // Price momentum factor
                    var volatilityScore = stock.DayHigh > 0 ? ((stock.DayHigh - stock.DayLow) / stock.Price) * 100 : 0; // Intraday volatility
                    
                    // Bonus for extreme discrepancies
                    var extremeBonus = 0.0;
                    if (stock.RSI <= 20 || stock.RSI >= 80)
                    {
                        extremeBonus = 25.0; // Major bonus for extreme RSI levels
                    }
                    else if (stock.RSI <= 25 || stock.RSI >= 75)
                    {
                        extremeBonus = 15.0; // Moderate bonus for very high/low RSI
                    }
                    
                    // Final catalytic score combining all factors
                    var catalyticScore = volumeScore + rsiDiscrepancyScore + priceMovementScore + volatilityScore + extremeBonus;
                    
                    // Store the catalytic score in PredictionConfidence for sorting
                    stock.PredictionConfidence = catalyticScore;
                    
                    // Clean up temporary data and set proper values
                    var discrepancyType = stock.PredictedAction; // Retrieve from temporary storage
                    stock.PredictedAction = $"{discrepancyType} (Score: {catalyticScore:F1})";
                    
                    scoredStocks.Add(stock);
                    
                    //DatabaseMonolith.Log("Debug", $"Catalytic score calculated for {stock.Symbol}: {catalyticScore:F1} " +
                                               $"(Vol: {volumeScore:F1}, RSI: {rsiDiscrepancyScore:F1}, Price: {priceMovementScore:F1}, " +
                                               $"Volatility: {volatilityScore:F1}, Bonus: {extremeBonus:F1})");
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", $"Error calculating catalytic score for {stock.Symbol}", ex.ToString());
                }
            }
            
            // Sort by catalytic score (highest first) to return most catalytic stocks
            var rankedResults = scoredStocks.OrderByDescending(x => x.PredictionConfidence).ToList();
            
            //DatabaseMonolith.Log("Info", $"Catalytic scoring complete. Top stock: {rankedResults.FirstOrDefault()?.Symbol} " +
                                       $"(Score: {rankedResults.FirstOrDefault()?.PredictionConfidence:F1})");
            
            return rankedResults;
        }

        private async Task<List<QuoteData>> GetTopPEStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                //DatabaseMonolith.Log("Info", "Starting dynamic Top P/E stock analysis...");
                var alphaVantageService = new AlphaVantageService();
                var cacheService = new StockDataCacheService();

                // Get comprehensive list of symbols, prioritizing popular/liquid stocks first
                var allSymbols = await alphaVantageService.GetAllStockSymbols();
                
                // Prioritize symbols: start with popular stocks, then S&P 500, then others
                var prioritizedSymbols = new List<string>();
                
                // Add popular stocks first (most likely to have good P/E data)
                if (StockSymbols.Indices.ContainsKey("Popular"))
                    prioritizedSymbols.AddRange(StockSymbols.Indices["Popular"]);
                
                if (StockSymbols.Indices.ContainsKey("S&P500Top"))
                    prioritizedSymbols.AddRange(StockSymbols.Indices["S&P500Top"].Where(s => !prioritizedSymbols.Contains(s)));
                
                if (StockSymbols.Indices.ContainsKey("NASDAQ100"))
                    prioritizedSymbols.AddRange(StockSymbols.Indices["NASDAQ100"].Where(s => !prioritizedSymbols.Contains(s)));

                // Add remaining symbols from API if available
                if (allSymbols.Any())
                {
                    prioritizedSymbols.AddRange(allSymbols.Where(s => !prioritizedSymbols.Contains(s)));
                }
                else
                {
                    // Fallback to common symbols if API fails
                    prioritizedSymbols.AddRange(StockSymbols.CommonSymbols.Where(s => !prioritizedSymbols.Contains(s)));
                    //DatabaseMonolith.Log("Warning", "API failed, using fallback common symbols for P/E analysis");
                }

                //DatabaseMonolith.Log("Info", $"Analyzing P/E ratios for {prioritizedSymbols.Count} symbols (prioritized order)...");

                // Batch process symbols to respect API rate limits
                const int batchSize = 8; // Process 8 symbols at a time (more conservative)
                const int maxSymbolsToProcess = 150; // Reduced to balance speed with comprehensiveness
                var symbolsToProcess = prioritizedSymbols.Take(maxSymbolsToProcess).ToList();
                
                var stocksWithPE = new List<QuoteData>();

                for (int i = 0; i < symbolsToProcess.Count; i += batchSize)
                {
                    var batch = symbolsToProcess.Skip(i).Take(batchSize);
                    //DatabaseMonolith.Log("Info", $"Processing P/E batch {(i / batchSize) + 1}/{(symbolsToProcess.Count + batchSize - 1) / batchSize}");

                    // Process batch concurrently but with controlled parallelism
                    var batchTasks = batch.Select(async symbol =>
                    {
                        try
                        {
                            // Check if we have recent cached P/E data first
                            var cachedPE = DatabaseMonolith.GetCachedFundamentalData(symbol, "PE_RATIO", 4); // 4 hour cache
                            
                            double? peRatio = null;
                            if (cachedPE.HasValue)
                            {
                                peRatio = cachedPE.Value;
                            }
                            else
                            {
                                // Fetch P/E ratio from API
                                peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                            }

                            if (peRatio.HasValue && peRatio.Value > 0)
                            {
                                // Try to get quote data from cache first, then API if needed
                                QuoteData quote = null;
                                
                                // Check if we have recent cached quote data
                                var cachedStock = cacheService.GetCachedStock(symbol);
                                
                                if (cachedStock != null && (DateTime.Now - cachedStock.LastUpdated).TotalHours < 2)
                                {
                                    quote = cachedStock;
                                }
                                else
                                {
                                    // Fetch from API if not in cache or cache is stale
                                    quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                }

                                if (quote != null)
                                {
                                    var stockData = new QuoteData
                                    {
                                        Symbol = quote.Symbol,
                                        Price = quote.Price,
                                        Volume = quote.Volume,
                                        ChangePercent = quote.ChangePercent,
                                        DayHigh = quote.DayHigh,
                                        DayLow = quote.DayLow,
                                        PERatio = peRatio.Value,
                                        LastUpdated = DateTime.Now
                                    };

                                    // Optionally add RSI if available in cache (to avoid additional API calls)
                                    try
                                    {
                                        var cachedRSI = DatabaseMonolith.GetCachedFundamentalData(symbol, "RSI", 2); // 2 hour cache for RSI
                                        if (cachedRSI.HasValue)
                                        {
                                            stockData.RSI = cachedRSI.Value;
                                        }
                                    }
                                    catch
                                    {
                                        stockData.RSI = 0;
                                    }

                                    // Try to get VWAP data
                                    try
                                    {
                                        var vwap = await alphaVantageService.GetVWAP(symbol);
                                        stockData.VWAP = vwap;
                                    }
                                    catch
                                    {
                                        stockData.VWAP = 0;
                                    }

                                    return stockData;
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for top P/E", ex.ToString());
                        }
                        return null;
                    }).ToArray();

                    // Wait for batch to complete
                    var batchResults = await Task.WhenAll(batchTasks);
                    
                    // Add valid results
                    stocksWithPE.AddRange(batchResults.Where(stock => stock != null));

                    // Small delay between batches to be respectful of API limits
                    if (i + batchSize < symbolsToProcess.Count)
                    {
                        await Task.Delay(500); // 0.5 second delay between batches (reduced for better performance)
                    }
                }

                // Sort by P/E ratio (highest first) and return top 100
                result = stocksWithPE
                    .Where(x => x.PERatio > 0)
                    .OrderByDescending(x => x.PERatio)
                    .Take(100)
                    .ToList();

                //DatabaseMonolith.Log("Info", $"Dynamic P/E analysis complete. Found {result.Count} stocks with valid P/E ratios. " +
                                           $"Top stock: {result.FirstOrDefault()?.Symbol} (P/E: {result.FirstOrDefault()?.PERatio:F2})");

                return result;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting dynamic top P/E stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetHighVolumeStocks()
        {
            var result = new List<QuoteData>();
            var startTime = DateTime.Now;
            
            try
            {
                //DatabaseMonolith.Log("Info", "Starting High Volume stocks analysis");
                
                // Phase 1: Build comprehensive stock universe from multiple sources
                var stockUniverse = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Phase 1 Complete: Built stock universe with {stockUniverse.Count} symbols");
                
                // Phase 2: Identify high volume stocks from the entire universe
                var highVolumeStocks = await IdentifyHighVolumeStocks(stockUniverse);
                //DatabaseMonolith.Log("Info", $"Phase 2 Complete: Identified {highVolumeStocks.Count} high volume stocks");
                
                // Phase 3: Convert to QuoteData format and return top results
                result = await ConvertToQuoteDataList(highVolumeStocks);
                //DatabaseMonolith.Log("Info", $"Phase 3 Complete: Converted {result.Count} stocks to QuoteData format");
                
                // Return top 20 high volume stocks
                result = result.OrderByDescending(x => x.Volume).Take(20).ToList();
                
                var duration = DateTime.Now - startTime;
                //DatabaseMonolith.Log("Info", $"High Volume analysis completed in {duration.TotalSeconds:F2} seconds, returning {result.Count} stocks");
                
                return result;
            }
            catch (Exception ex)
            {
                var duration = DateTime.Now - startTime;
                //DatabaseMonolith.Log("Error", $"Error in High Volume analysis after {duration.TotalSeconds:F2} seconds", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetLowPEStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                //DatabaseMonolith.Log("Info", "Starting dynamic Low P/E stock analysis...");
                var alphaVantageService = new AlphaVantageService();
                var cacheService = new StockDataCacheService();

                // Get comprehensive list of symbols, prioritizing popular/liquid stocks first
                var allSymbols = await alphaVantageService.GetAllStockSymbols();
                
                // Prioritize symbols: start with popular stocks, then S&P 500, then others
                var prioritizedSymbols = new List<string>();
                
                // Add popular stocks first (most likely to have good P/E data)
                if (StockSymbols.Indices.ContainsKey("Popular"))
                    prioritizedSymbols.AddRange(StockSymbols.Indices["Popular"]);
                
                if (StockSymbols.Indices.ContainsKey("S&P500Top"))
                    prioritizedSymbols.AddRange(StockSymbols.Indices["S&P500Top"].Where(s => !prioritizedSymbols.Contains(s)));
                
                if (StockSymbols.Indices.ContainsKey("NASDAQ100"))
                    prioritizedSymbols.AddRange(StockSymbols.Indices["NASDAQ100"].Where(s => !prioritizedSymbols.Contains(s)));

                // Add remaining symbols from API if available
                if (allSymbols.Any())
                {
                    prioritizedSymbols.AddRange(allSymbols.Where(s => !prioritizedSymbols.Contains(s)));
                }
                else
                {
                    // Fallback to common symbols if API fails
                    prioritizedSymbols.AddRange(StockSymbols.CommonSymbols.Where(s => !prioritizedSymbols.Contains(s)));
                    //DatabaseMonolith.Log("Warning", "API failed, using fallback common symbols for Low P/E analysis");
                }

                //DatabaseMonolith.Log("Info", $"Analyzing P/E ratios for {prioritizedSymbols.Count} symbols (prioritized order)...");

                // Batch process symbols to respect API rate limits
                const int batchSize = 8; // Process 8 symbols at a time (more conservative)
                const int maxSymbolsToProcess = 150; // Reduced to balance speed with comprehensiveness
                var symbolsToProcess = prioritizedSymbols.Take(maxSymbolsToProcess).ToList();
                
                var stocksWithPE = new List<QuoteData>();

                for (int i = 0; i < symbolsToProcess.Count; i += batchSize)
                {
                    var batch = symbolsToProcess.Skip(i).Take(batchSize);
                    //DatabaseMonolith.Log("Info", $"Processing Low P/E batch {(i / batchSize) + 1}/{(symbolsToProcess.Count + batchSize - 1) / batchSize}");

                    // Process batch concurrently but with controlled parallelism
                    var batchTasks = batch.Select(async symbol =>
                    {
                        try
                        {
                            // Check if we have recent cached P/E data first
                            var cachedPE = DatabaseMonolith.GetCachedFundamentalData(symbol, "PE_RATIO", 4); // 4 hour cache
                            
                            double? peRatio = null;
                            if (cachedPE.HasValue)
                            {
                                peRatio = cachedPE.Value;
                            }
                            else
                            {
                                // Fetch P/E ratio from API
                                peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                            }

                            if (peRatio.HasValue && peRatio.Value > 0)
                            {
                                // Try to get quote data from cache first, then API if needed
                                QuoteData quote = null;
                                
                                // Check if we have recent cached quote data
                                var cachedStock = cacheService.GetCachedStock(symbol);
                                
                                if (cachedStock != null && (DateTime.Now - cachedStock.LastUpdated).TotalHours < 2)
                                {
                                    quote = cachedStock;
                                }
                                else
                                {
                                    // Fetch from API if not in cache or cache is stale
                                    quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                }

                                if (quote != null)
                                {
                                    var stockData = new QuoteData
                                    {
                                        Symbol = quote.Symbol,
                                        Price = quote.Price,
                                        Volume = quote.Volume,
                                        ChangePercent = quote.ChangePercent,
                                        DayHigh = quote.DayHigh,
                                        DayLow = quote.DayLow,
                                        PERatio = peRatio.Value,
                                        LastUpdated = DateTime.Now
                                    };

                                    // Optionally add RSI if available in cache (to avoid additional API calls)
                                    try
                                    {
                                        var cachedRSI = DatabaseMonolith.GetCachedFundamentalData(symbol, "RSI", 2); // 2 hour cache for RSI
                                        if (cachedRSI.HasValue)
                                        {
                                            stockData.RSI = cachedRSI.Value;
                                        }
                                    }
                                    catch
                                    {
                                        stockData.RSI = 0;
                                    }

                                    // Try to get VWAP data
                                    try
                                    {
                                        var vwap = await alphaVantageService.GetVWAP(symbol);
                                        stockData.VWAP = vwap;
                                    }
                                    catch
                                    {
                                        stockData.VWAP = 0;
                                    }

                                    return stockData;
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for low P/E", ex.ToString());
                        }
                        return null;
                    }).ToArray();

                    // Wait for batch to complete
                    var batchResults = await Task.WhenAll(batchTasks);
                    
                    // Add valid results
                    stocksWithPE.AddRange(batchResults.Where(stock => stock != null));

                    // Small delay between batches to be respectful of API limits
                    if (i + batchSize < symbolsToProcess.Count)
                    {
                        await Task.Delay(500); // 0.5 second delay between batches (reduced for better performance)
                    }
                }

                // Sort by P/E ratio (lowest first) and return top 100
                result = stocksWithPE
                    .Where(x => x.PERatio > 0)
                    .OrderBy(x => x.PERatio)
                    .Take(100)
                    .ToList();

                //DatabaseMonolith.Log("Info", $"Dynamic Low P/E analysis complete. Found {result.Count} stocks with valid P/E ratios. " +
                                           $"Lowest P/E stock: {result.FirstOrDefault()?.Symbol} (P/E: {result.FirstOrDefault()?.PERatio:F2})");

                return result;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting dynamic low P/E stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetRsiOversoldStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for RSI oversold scan: {symbols.Count} symbols");

                var alphaVantageService = new AlphaVantageService();
                var batchSize = 20; // Process in smaller batches to avoid API rate limits
                var processedCount = 0;
                var oversoldThreshold = 35; // More flexible threshold for better results

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify RSI oversold stocks (threshold: < {oversoldThreshold})");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                var rsi = await alphaVantageService.GetRSI(symbol);
                                
                                if (quote != null && rsi < oversoldThreshold) // Oversold condition
                                {
                                    var extendedQuote = new QuoteData
                                    {
                                        Symbol = quote.Symbol,
                                        Price = quote.Price,
                                        Volume = quote.Volume,
                                        ChangePercent = quote.ChangePercent,
                                        DayHigh = quote.DayHigh,
                                        DayLow = quote.DayLow,
                                        RSI = rsi,
                                        LastUpdated = DateTime.Now
                                    };

                                    try
                                    {
                                        var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                        extendedQuote.PERatio = peRatio ?? 0;
                                    }
                                    catch
                                    {
                                        extendedQuote.PERatio = 0;
                                    }

                                    // Try to get VWAP data
                                    try
                                    {
                                        var vwap = await alphaVantageService.GetVWAP(symbol);
                                        extendedQuote.VWAP = vwap;
                                    }
                                    catch
                                    {
                                        extendedQuote.VWAP = 0;
                                    }

                                    lock (result)
                                    {
                                        result.Add(extendedQuote);
                                    }
                                    
                                    //DatabaseMonolith.Log("Debug", $"RSI oversold found: {symbol} - RSI: {rsi:F1}, Price: ${quote.Price:F2}");
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for RSI oversold", ex.ToString());
                            }
                        }));
                    }

                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;

                    // Add delay between batches to respect API limits
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(2000); // 2 second delay between batches
                    }

                    //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{symbols.Count}, oversold found: {result.Count}");
                }

                // Return all oversold stocks, ordered by RSI ascending (no Take(10) limit)
                var sortedResult = result.OrderBy(x => x.RSI).ToList();
                //DatabaseMonolith.Log("Info", $"RSI oversold scan completed: {sortedResult.Count} oversold stocks found from {symbols.Count} symbols");
                
                return sortedResult;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting RSI oversold stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetRsiOverboughtStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for RSI overbought scan: {symbols.Count} symbols");

                var alphaVantageService = new AlphaVantageService();
                var batchSize = 20; // Process in smaller batches to avoid API rate limits
                var processedCount = 0;
                var overboughtThreshold = 65; // More flexible threshold for better results

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify RSI overbought stocks (threshold: > {overboughtThreshold})");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                var rsi = await alphaVantageService.GetRSI(symbol);
                                
                                if (quote != null && rsi > overboughtThreshold) // Overbought condition
                                {
                                    var extendedQuote = new QuoteData
                                    {
                                        Symbol = quote.Symbol,
                                        Price = quote.Price,
                                        Volume = quote.Volume,
                                        ChangePercent = quote.ChangePercent,
                                        DayHigh = quote.DayHigh,
                                        DayLow = quote.DayLow,
                                        RSI = rsi,
                                        LastUpdated = DateTime.Now
                                    };

                                    try
                                    {
                                        var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                        extendedQuote.PERatio = peRatio ?? 0;
                                    }
                                    catch
                                    {
                                        extendedQuote.PERatio = 0;
                                    }

                                    // Try to get VWAP data
                                    try
                                    {
                                        var vwap = await alphaVantageService.GetVWAP(symbol);
                                        extendedQuote.VWAP = vwap;
                                    }
                                    catch
                                    {
                                        extendedQuote.VWAP = 0;
                                    }

                                    lock (result)
                                    {
                                        result.Add(extendedQuote);
                                    }
                                    
                                    //DatabaseMonolith.Log("Debug", $"RSI overbought found: {symbol} - RSI: {rsi:F1}, Price: ${quote.Price:F2}");
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for RSI overbought", ex.ToString());
                            }
                        }));
                    }

                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;

                    // Add delay between batches to respect API limits
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(2000); // 2 second delay between batches
                    }

                    //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{symbols.Count}, overbought found: {result.Count}");
                }

                // Return all overbought stocks, ordered by RSI descending (no Take(10) limit)
                var sortedResult = result.OrderByDescending(x => x.RSI).ToList();
                //DatabaseMonolith.Log("Info", $"RSI overbought scan completed: {sortedResult.Count} overbought stocks found from {symbols.Count} symbols");
                
                return sortedResult;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting RSI overbought stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetHighThetaStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for High Theta scan: {symbols.Count} symbols");

                var alphaVantageService = new AlphaVantageService();
                var batchSize = 15; // Smaller batches due to additional calculations
                var processedCount = 0;
                var volatilityThreshold = 0.25; // High implied volatility threshold

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify High Theta opportunities (volatility > {volatilityThreshold})");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                if (quote != null)
                                {
                                    // Calculate volatility from price history as a proxy for theta opportunities
                                    var priceHistory = await GetRecentPriceHistory(symbol, alphaVantageService);
                                    var volatility = CalculateVolatility(priceHistory);
                                    
                                    // High theta opportunities: high volatility stocks suitable for covered calls, cash-secured puts
                                    if (volatility > volatilityThreshold && quote.Volume > 100000) // High vol + high volume
                                    {
                                        var extendedQuote = new QuoteData
                                        {
                                            Symbol = quote.Symbol,
                                            Price = quote.Price,
                                            Volume = quote.Volume,
                                            ChangePercent = quote.ChangePercent,
                                            DayHigh = quote.DayHigh,
                                            DayLow = quote.DayLow,
                                            LastUpdated = DateTime.Now
                                        };

                                        // Add volatility as a custom property for sorting
                                        extendedQuote.RSI = volatility * 100; // Store volatility in RSI field for display
                                        
                                        try
                                        {
                                            var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                            extendedQuote.PERatio = peRatio ?? 0;
                                        }
                                        catch
                                        {
                                            extendedQuote.PERatio = 0;
                                        }

                                        // Try to get VWAP data
                                        try
                                        {
                                            var vwap = await alphaVantageService.GetVWAP(symbol);
                                            extendedQuote.VWAP = vwap;
                                        }
                                        catch
                                        {
                                            extendedQuote.VWAP = 0;
                                        }

                                        lock (result)
                                        {
                                            result.Add(extendedQuote);
                                        }
                                        
                                        //DatabaseMonolith.Log("Debug", $"High Theta opportunity found: {symbol} - Volatility: {volatility:F3}, Volume: {quote.Volume:N0}");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for High Theta", ex.ToString());
                            }
                        }));
                    }

                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;

                    // Add delay between batches to respect API limits
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(2500); // Longer delay due to additional calculations
                    }

                    //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{symbols.Count}, High Theta found: {result.Count}");
                }

                // Return stocks ordered by volatility descending (highest theta opportunities first)
                var sortedResult = result.OrderByDescending(x => x.RSI).Take(25).ToList(); // Limit to top 25
                //DatabaseMonolith.Log("Info", $"High Theta scan completed: {sortedResult.Count} opportunities found from {symbols.Count} symbols");
                
                return sortedResult;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting High Theta stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetHighBetaStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for High Beta scan: {symbols.Count} symbols");

                var alphaVantageService = new AlphaVantageService();
                var batchSize = 15; // Smaller batches due to beta calculations
                var processedCount = 0;
                var betaThreshold = 1.2; // High beta threshold for momentum trading

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify High Beta stocks (beta > {betaThreshold})");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                if (quote != null)
                                {
                                    // Calculate beta using correlation with market movements
                                    var beta = await CalculateBeta(symbol, alphaVantageService);
                                    
                                    // High beta stocks: strong correlation with market movements
                                    if (beta > betaThreshold && quote.Volume > 500000) // High beta + high volume for liquidity
                                    {
                                        var extendedQuote = new QuoteData
                                        {
                                            Symbol = quote.Symbol,
                                            Price = quote.Price,
                                            Volume = quote.Volume,
                                            ChangePercent = quote.ChangePercent,
                                            DayHigh = quote.DayHigh,
                                            DayLow = quote.DayLow,
                                            LastUpdated = DateTime.Now
                                        };

                                        // Store beta in RSI field for display purposes
                                        extendedQuote.RSI = beta;
                                        
                                        try
                                        {
                                            var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                            extendedQuote.PERatio = peRatio ?? 0;
                                        }
                                        catch
                                        {
                                            extendedQuote.PERatio = 0;
                                        }

                                        // Try to get VWAP data
                                        try
                                        {
                                            var vwap = await alphaVantageService.GetVWAP(symbol);
                                            extendedQuote.VWAP = vwap;
                                        }
                                        catch
                                        {
                                            extendedQuote.VWAP = 0;
                                        }

                                        lock (result)
                                        {
                                            result.Add(extendedQuote);
                                        }
                                        
                                        //DatabaseMonolith.Log("Debug", $"High Beta stock found: {symbol} - Beta: {beta:F2}, Volume: {quote.Volume:N0}");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for High Beta", ex.ToString());
                            }
                        }));
                    }

                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;

                    // Add delay between batches to respect API limits
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(3000); // Longer delay due to beta calculations
                    }

                    //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{symbols.Count}, High Beta found: {result.Count}");
                }

                // Return stocks ordered by beta descending (highest beta first)
                var sortedResult = result.OrderByDescending(x => x.RSI).Take(20).ToList(); // Limit to top 20
                //DatabaseMonolith.Log("Info", $"High Beta scan completed: {sortedResult.Count} high beta stocks found from {symbols.Count} symbols");
                
                return sortedResult;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting High Beta stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetHighAlphaStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for High Alpha scan: {symbols.Count} symbols");

                var alphaVantageService = new AlphaVantageService();
                var batchSize = 12; // Smaller batches due to complex alpha calculations
                var processedCount = 0;
                var alphaThreshold = 0.05; // 5% excess return threshold

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify High Alpha stocks (alpha > {alphaThreshold:P})");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                if (quote != null)
                                {
                                    // Calculate alpha by comparing stock returns vs market returns
                                    var alpha = await CalculateAlpha(symbol, alphaVantageService);
                                    
                                    // High alpha stocks: consistently outperforming the market
                                    if (alpha > alphaThreshold && quote.Volume > 250000) // Good alpha + sufficient volume
                                    {
                                        var extendedQuote = new QuoteData
                                        {
                                            Symbol = quote.Symbol,
                                            Price = quote.Price,
                                            Volume = quote.Volume,
                                            ChangePercent = quote.ChangePercent,
                                            DayHigh = quote.DayHigh,
                                            DayLow = quote.DayLow,
                                            LastUpdated = DateTime.Now
                                        };

                                        // Store alpha percentage in RSI field for display purposes
                                        extendedQuote.RSI = alpha * 100;
                                        
                                        try
                                        {
                                            var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                            extendedQuote.PERatio = peRatio ?? 0;
                                        }
                                        catch
                                        {    
                                            extendedQuote.PERatio = 0;
                                        }

                                        // Try to get VWAP data
                                        try
                                        {
                                            var vwap = await alphaVantageService.GetVWAP(symbol);
                                            extendedQuote.VWAP = vwap;
                                        }
                                        catch
                                        {
                                            extendedQuote.VWAP = 0;
                                        }

                                        lock (result)
                                        {
                                            result.Add(extendedQuote);
                                        }
                                        
                                        //DatabaseMonolith.Log("Debug", $"High Alpha stock found: {symbol} - Alpha: {alpha:P2}, Volume: {quote.Volume:N0}");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for High Alpha", ex.ToString());
                            }
                        }));
                    }

                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;

                    // Add delay between batches to respect API limits  
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(3500); // Longest delay due to most complex calculations
                    }

                    //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{symbols.Count}, High Alpha found: {result.Count}");
                }

                // Return stocks ordered by alpha descending (highest alpha first)
                var sortedResult = result.OrderByDescending(x => x.RSI).Take(15).ToList(); // Limit to top 15
                //DatabaseMonolith.Log("Info", $"High Alpha scan completed: {sortedResult.Count} high alpha stocks found from {symbols.Count} symbols");
                
                return sortedResult;  
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting High Alpha stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetBullishCupAndHandleStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for Cup and Handle scan: {symbols.Count} symbols");

                var patternService = new PricePatternRecognitionService(_cacheService);
                var alphaVantageService = new AlphaVantageService();
                var batchSize = 8; // Smaller batches due to complex pattern analysis
                var processedCount = 0;
                var minConfidence = 70.0; // Minimum confidence for cup and handle patterns

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify Cup and Handle patterns (confidence > {minConfidence}%)");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                if (quote != null && quote.Volume > 100000) // Ensure sufficient liquidity
                                {
                                    // Detect cup and handle patterns for this symbol
                                    var patterns = await patternService.DetectCupAndHandlePatternsAsync(symbol);
                                    
                                    // Look for high-confidence bullish cup and handle patterns
                                    var bullishCupAndHandle = patterns
                                        .Where(p => p.Type == PricePatternRecognitionService.PatternType.CupAndHandle 
                                                   && p.Bias == PricePatternRecognitionService.PatternBias.Bullish 
                                                   && p.Confidence >= minConfidence)
                                        .OrderByDescending(p => p.Confidence)
                                        .FirstOrDefault();
                                    
                                    if (bullishCupAndHandle != null)
                                    {
                                        var extendedQuote = new QuoteData
                                        {
                                            Symbol = quote.Symbol,
                                            Price = quote.Price,
                                            Volume = quote.Volume,
                                            ChangePercent = quote.ChangePercent,
                                            DayHigh = quote.DayHigh,
                                            DayLow = quote.DayLow,
                                            LastUpdated = DateTime.Now
                                        };

                                        // Store pattern confidence in RSI field for display purposes
                                        extendedQuote.RSI = bullishCupAndHandle.Confidence;
                                        
                                        try
                                        {
                                            var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                            extendedQuote.PERatio = peRatio ?? 0;
                                        }
                                        catch
                                        {    
                                            extendedQuote.PERatio = 0;
                                        }

                                        // Try to get VWAP data
                                        try
                                        {
                                            var vwap = await alphaVantageService.GetVWAP(symbol);
                                            extendedQuote.VWAP = vwap;
                                        }
                                        catch
                                        {
                                            extendedQuote.VWAP = 0;
                                        }

                                        lock (result)
                                        {
                                            result.Add(extendedQuote);
                                        }
                                        
                                        //DatabaseMonolith.Log("Debug", $"Cup and Handle pattern found: {symbol} - Confidence: {bullishCupAndHandle.Confidence:F1}%, Volume: {quote.Volume:N0}");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for Cup and Handle pattern", ex.ToString());
                            }
                        }));
                    }

                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;

                    // Add delay between batches to respect API limits  
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(4000); // Longer delay due to pattern analysis complexity
                    }

                    //DatabaseMonolith.Log("Info", $"Processed batch {i / batchSize + 1}, total symbols processed: {processedCount}/{symbols.Count}, Cup and Handle patterns found: {result.Count}");
                }

                // Return stocks ordered by pattern confidence descending (highest confidence first)
                var sortedResult = result.OrderByDescending(x => x.RSI).Take(12).ToList(); // Limit to top 12
                //DatabaseMonolith.Log("Info", $"Cup and Handle scan completed: {sortedResult.Count} patterns found from {symbols.Count} symbols");
                
                return sortedResult;  
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting Bullish Cup and Handle stocks", ex.ToString());
                return result;
            }
        }

        private async Task<List<QuoteData>> GetBearishCupAndHandleStocks()
        {
            var result = new List<QuoteData>();
            try
            {
                // Use dynamic stock universe for broader market coverage
                var symbols = await BuildDynamicStockUniverse();
                //DatabaseMonolith.Log("Info", $"Using dynamic stock universe for Bearish Cup and Handle scan: {symbols.Count} symbols");

                var patternService = new PricePatternRecognitionService(_cacheService);
                var alphaVantageService = new AlphaVantageService();
                var batchSize = 8; // Smaller batches due to complex pattern analysis
                var processedCount = 0;
                var minConfidence = 70.0; // Minimum confidence for bearish cup and handle patterns

                //DatabaseMonolith.Log("Info", $"Processing {symbols.Count} symbols in batches of {batchSize} to identify Bearish Cup and Handle patterns (confidence > {minConfidence}%)");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new List<Task>();

                    foreach (var symbol in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                                if (quote != null && quote.Volume > 100000) // Ensure sufficient liquidity
                                {
                                    // Detect bearish cup and handle patterns for this symbol
                                    var patterns = await patternService.DetectBearishCupAndHandlePatternsAsync(symbol);
                                    
                                    // Look for high-confidence bearish cup and handle patterns
                                    var bearishCupAndHandle = patterns
                                        .Where(p => p.Type == PricePatternRecognitionService.PatternType.CupAndHandle 
                                                   && p.Bias == PricePatternRecognitionService.PatternBias.Bearish 
                                                   && p.Confidence >= minConfidence)
                                        .OrderByDescending(p => p.Confidence)
                                        .FirstOrDefault();
                                    
                                    if (bearishCupAndHandle != null)
                                    {
                                        var extendedQuote = new QuoteData
                                        {
                                            Symbol = quote.Symbol,
                                            Price = quote.Price,
                                            Volume = quote.Volume,
                                            ChangePercent = quote.ChangePercent,
                                            DayHigh = quote.DayHigh,
                                            DayLow = quote.DayLow,
                                            LastUpdated = DateTime.Now
                                        };

                                        // Store pattern confidence in RSI field for display purposes
                                        extendedQuote.RSI = bearishCupAndHandle.Confidence;
                                        
                                        try
                                        {
                                            var peRatio = await alphaVantageService.GetPERatioAsync(symbol);
                                            extendedQuote.PERatio = peRatio ?? 0;
                                        }
                                        catch
                                        {    
                                            extendedQuote.PERatio = 0;
                                        }

                                        // Try to get VWAP data
                                        try
                                        {
                                            var vwap = await alphaVantageService.GetVWAP(symbol);
                                            extendedQuote.VWAP = vwap;
                                        }
                                        catch
                                        {
                                            extendedQuote.VWAP = 0;
                                        }

                                        lock (result)
                                        {
                                            result.Add(extendedQuote);
                                        }
                                        
                                        //DatabaseMonolith.Log("Debug", $"Bearish Cup and Handle pattern found: {symbol} - Confidence: {bearishCupAndHandle.Confidence:F1}%, Volume: {quote.Volume:N0}");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Error processing symbol {symbol} for Bearish Cup and Handle pattern", ex.ToString());
                            }
                        }));
                    }

                    // Wait for this batch to complete before processing the next
                    await Task.WhenAll(batchTasks);
                    processedCount += batch.Count;
                    
                    // Log progress every 100 symbols
                    if (processedCount % 100 == 0 || processedCount == symbols.Count)
                    {
                        //DatabaseMonolith.Log("Info", $"Processed {processedCount}/{symbols.Count} symbols for Bearish Cup and Handle patterns. Found {result.Count} matches so far.");
                    }

                    // Small delay between batches to prevent API rate limiting
                    await Task.Delay(100);
                }

                //DatabaseMonolith.Log("Info", $"Bearish Cup and Handle scan complete. Found {result.Count} stocks with bearish cup and handle patterns.");
                return result.OrderByDescending(s => s.RSI).ToList(); // Order by confidence (stored in RSI field)
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting Bearish Cup and Handle stocks", ex.ToString());
                return result;
            }
        }

        /// <summary>
        /// Helper method to get recent price history for volatility calculation
        /// </summary>
        private async Task<List<double>> GetRecentPriceHistory(string symbol, AlphaVantageService alphaVantageService)
        {
            var prices = new List<double>();
            try
            {
                // Simple implementation using daily high/low/close for basic volatility
                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                if (quote != null)
                {
                    // Add some price points for basic volatility calculation
                    prices.Add(quote.Price);
                    prices.Add(quote.DayHigh);
                    prices.Add(quote.DayLow);
                    
                    // Add some variation based on change percent to simulate historical data
                    var basePrice = quote.Price / (1 + quote.ChangePercent / 100);
                    prices.Add(basePrice);
                    prices.Add(basePrice * 1.02); // Simulate some historical variation
                    prices.Add(basePrice * 0.98);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Error getting price history for {symbol}: {ex.Message}");
            }
            return prices;
        }

        /// <summary>
        /// Calculate volatility from price history
        /// </summary>
        private double CalculateVolatility(List<double> prices)
        {
            if (prices.Count < 2) return 0.0;
            
            try
            {
                var returns = new List<double>();
                for (int i = 1; i < prices.Count; i++)
                {
                    if (prices[i - 1] > 0)
                    {
                        returns.Add(Math.Log(prices[i] / prices[i - 1]));
                    }
                }
                
                if (returns.Count == 0) return 0.0;
                
                var meanReturn = returns.Average();
                var variance = returns.Select(r => Math.Pow(r - meanReturn, 2)).Average();
                return Math.Sqrt(variance * 252); // Annualized volatility
            }
            catch
            {
                return 0.0;
            }
        }

        /// <summary>
        /// Calculate beta by comparing stock movement to market movement
        /// </summary>
        private async Task<double> CalculateBeta(string symbol, AlphaVantageService alphaVantageService)
        {
            try
            {
                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                if (quote == null) return 1.0;
                
                // Simplified beta calculation based on daily change percentage
                // In production, this would use historical data correlation with market index
                var stockChange = Math.Abs(quote.ChangePercent);
                
                // Estimate beta based on volatility of daily moves
                // Stocks with high daily volatility tend to have higher beta
                if (stockChange > 5.0) return 1.8; // Very high beta
                if (stockChange > 3.0) return 1.4; // High beta  
                if (stockChange > 2.0) return 1.2; // Above market beta
                if (stockChange > 1.0) return 1.0; // Market beta
                return 0.8; // Below market beta
            }
            catch
            {
                return 1.0; // Default market beta
            }
        }

        /// <summary>
        /// Calculate alpha by comparing stock performance to expected market performance
        /// </summary>
        private async Task<double> CalculateAlpha(string symbol, AlphaVantageService alphaVantageService)
        {
            try
            {
                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                if (quote == null) return 0.0;
                
                // Simplified alpha calculation - in production would use historical returns vs benchmark
                // Positive change percent suggests positive momentum/alpha generation
                var dailyReturn = quote.ChangePercent / 100.0;
                
                // Estimate alpha based on consistent outperformance patterns
                // This is a simplified model - real alpha calculation requires historical analysis
                if (dailyReturn > 0.03) return 0.15; // Strong positive alpha
                if (dailyReturn > 0.02) return 0.10; // Good positive alpha  
                if (dailyReturn > 0.01) return 0.06; // Moderate positive alpha
                if (dailyReturn > 0) return 0.03; // Slight positive alpha
                return -0.01; // Negative alpha
            }
            catch
            {
                return 0.0; // Neutral alpha
            }
        }

        /// <summary>
        /// Get symbols from database StockSymbols table
        /// </summary>
        private List<string> GetSymbolsFromDatabase()
        {
            List<string> symbols = new List<string>();
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    using (var command = new SQLiteCommand("SELECT Symbol FROM StockSymbols", connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                symbols.Add(reader["Symbol"].ToString());
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to get symbols from database: {ex.Message}");
            }
            
            return symbols;
        }

        // Add additional helper methods and event handlers as needed

        /// <summary>
        /// Waits for all background tasks to complete
        /// </summary>
        private async Task WaitForBackgroundTasksAsync()
        {
            Task[] tasksToWait;
            lock (_backgroundTasksLock)
            {
                tasksToWait = _backgroundTasks.Where(t => !t.IsCompleted).ToArray();
            }
            
            if (tasksToWait.Length > 0)
            {
                await Task.WhenAll(tasksToWait);
            }
        }

        /// <summary>
        /// Cleans up completed background tasks
        /// </summary>
        private void CleanupCompletedBackgroundTasks()
        {
            lock (_backgroundTasksLock)
            {
                _backgroundTasks.RemoveAll(t => t.IsCompleted);
            }
        }

        /// <summary>
        /// Dispose resources including the ViewModel
        /// </summary>
        public void Dispose()
        {
            // Stop memory monitoring timer
            _memoryMonitorTimer?.Stop();
            _memoryMonitorTimer?.Tick -= OnMemoryMonitorTimer_Tick;
            
            // Wait for background tasks to complete before disposing
            try
            {
                WaitForBackgroundTasksAsync().Wait(TimeSpan.FromSeconds(5)); // 5 second timeout
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Error waiting for background tasks during dispose", ex.ToString());
            }

            // Clean up symbol search timer
            if (_symbolSearchTimer != null)
            {
                _symbolSearchTimer.Stop();
                _symbolSearchTimer.Tick -= SymbolSearchTimer_Tick;
                _symbolSearchTimer = null;
            }

            // Clean up cancellation token source
            _symbolOperationCancellation?.Cancel();
            _symbolOperationCancellation?.Dispose();
            _symbolOperationCancellation = null;

            // Dispose UI batch updater
            _uiBatchUpdater?.Dispose();
            _uiBatchUpdater = null;

            _viewModel?.Dispose();
        }

        // Sentiment Analysis Event Handler
        private async void RunSentimentAnalysisButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Check if we have a selected symbol
                var selectedStock = StockDataGrid?.SelectedItem as QuoteData;
                if (selectedStock == null || string.IsNullOrEmpty(selectedStock.Symbol))
                {
                    SentimentError = "Please select a stock symbol first.";
                    return;
                }

                // Clear previous errors and results
                SentimentError = "";
                HasSentimentResults = false;
                IsSentimentLoading = true;

                // Set busy cursor during sentiment analysis
                Mouse.OverrideCursor = Cursors.Wait;

                string symbol = selectedStock.Symbol;

                //DatabaseMonolith.Log("Info", $"Starting sentiment analysis for {symbol}");

                // Run sentiment analysis for different sources in parallel
                var sentimentTasks = new List<Task<double>>();
                
                // News sentiment
                var newsTask = GetNewsSentimentAsync(symbol);
                sentimentTasks.Add(newsTask);
                
                // Social media sentiment (using OpenAI for now)
                var socialMediaTask = GetSocialMediaSentimentAsync(symbol);
                sentimentTasks.Add(socialMediaTask);
                
                // Analyst sentiment
                var analystTask = GetAnalystSentimentAsync(symbol);
                sentimentTasks.Add(analystTask);

                // Wait for all sentiment analysis tasks to complete
                var results = await Task.WhenAll(sentimentTasks);

                // Update UI with results
                await Dispatcher.InvokeAsync(() =>
                {
                    NewsSentimentScore = results[0];
                    SocialMediaSentimentScore = results[1];
                    AnalystSentimentScore = results[2];

                    // Calculate overall sentiment (weighted average)
                    OverallSentimentScore = (NewsSentimentScore + SocialMediaSentimentScore + AnalystSentimentScore) / 3.0;

                    // Generate summary
                    SentimentSummary = GenerateSentimentSummary(symbol);

                    HasSentimentResults = true;
                    IsSentimentLoading = false;

                    // Reset cursor when sentiment analysis completes
                    Mouse.OverrideCursor = null;
                });

                //DatabaseMonolith.Log("Info", $"Sentiment analysis completed for {symbol}. Overall: {OverallSentimentScore:F2}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error during sentiment analysis", ex.ToString());
                
                await Dispatcher.InvokeAsync(() =>
                {
                    SentimentError = $"Error running sentiment analysis: {ex.Message}";
                    IsSentimentLoading = false;
                    
                    // Reset cursor on error
                    Mouse.OverrideCursor = null;
                });
            }
        }

        private async Task<double> GetNewsSentimentAsync(string symbol)
        {
            try
            {
                if (_newsSentimentService != null)
                {
                    return await _newsSentimentService.GetSymbolSentimentAsync(symbol);
                }
                return 0.0;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to get news sentiment for {symbol}", ex.ToString());
                return 0.0;
            }
        }

        private async Task<double> GetSocialMediaSentimentAsync(string symbol)
        {
            try
            {
                if (_twitterSentimentService != null)
                {
                    return await _twitterSentimentService.GetSymbolSentimentAsync(symbol);
                }
                return 0.0;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to get social media sentiment for {symbol}", ex.ToString());
                return 0.0;
            }
        }

        private async Task<double> GetAnalystSentimentAsync(string symbol)
        {
            try
            {
                if (_analystRatingService != null)
                {
                    var analysisResult = await _analystRatingService.GetAggregatedRatingsAsync(symbol);
                    if (analysisResult != null)
                    {
                        // Convert analyst ratings to sentiment score (-1 to 1)
                        // Strong Buy = 1, Buy = 0.5, Hold = 0, Sell = -0.5, Strong Sell = -1
                        var totalRatings = analysisResult.BuyCount + analysisResult.HoldCount + analysisResult.SellCount;
                        if (totalRatings > 0)
                        {
                            var buyWeight = analysisResult.BuyCount * 1.0;
                            var holdWeight = analysisResult.HoldCount * 0.0;
                            var sellWeight = analysisResult.SellCount * -1.0;
                            return (buyWeight + holdWeight + sellWeight) / totalRatings;
                        }
                    }
                }
                return 0.0;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to get analyst sentiment for {symbol}", ex.ToString());
                return 0.0;
            }
        }

        private string GenerateSentimentSummary(string symbol)
        {
            try
            {
                var sentiment = OverallSentimentScore;
                var sentimentText = sentiment switch
                {
                    > 0.5 => "Very Positive",
                    > 0.2 => "Positive", 
                    > -0.2 => "Neutral",
                    > -0.5 => "Negative",
                    _ => "Very Negative"
                };

                return $"Overall sentiment for {symbol} is {sentimentText} ({sentiment:F2})";
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to generate sentiment summary", ex.ToString());
                return "Sentiment analysis completed.";
            }
        }
    }
}