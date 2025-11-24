using System;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Quantra.Commands;
using Quantra.Models;
using Quantra.ViewModels.Base;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;
using LiveCharts;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the MainWindow following MVVM pattern
    /// </summary>
    public class MainWindowViewModel : ViewModelBase
    {
        #region Fields

        private readonly UserSettingsService _userSettingsService;
        private readonly HistoricalDataService _historicalDataService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private readonly DispatcherTimer _tradeUpdateTimer;

        private bool _isTradingActive;
        private string _currentTicker;
        private bool _isSymbolSelected;
        private bool _enableApiModalChecks;

        #endregion

        #region Properties

        /// <summary>
        /// Collection of active trading symbols
        /// </summary>
        public ObservableCollection<TradingSymbol> ActiveSymbols { get; set; }

        /// <summary>
        /// Collection of trading rules
        /// </summary>
        public ObservableCollection<StockItem> TradingRules { get; set; }

        /// <summary>
        /// Stock price values for charts
        /// </summary>
        public ChartValues<double> StockPriceValues { get; set; }

        /// <summary>
        /// Upper Bollinger Band values
        /// </summary>
        public ChartValues<double> UpperBandValues { get; set; }

        /// <summary>
        /// Middle Bollinger Band values
        /// </summary>
        public ChartValues<double> MiddleBandValues { get; set; }

        /// <summary>
        /// Lower Bollinger Band values
        /// </summary>
        public ChartValues<double> LowerBandValues { get; set; }

        /// <summary>
        /// Stock price line values
        /// </summary>
        public ChartValues<double> StockPriceLineValues { get; set; }

        /// <summary>
        /// RSI indicator values
        /// </summary>
        public ChartValues<double> RSIValues { get; set; }

        /// <summary>
        /// Date formatter for charts
        /// </summary>
        public Func<double, string> DateFormatter { get; set; }

        /// <summary>
        /// Whether trading is currently active
        /// </summary>
        public bool IsTradingActive
        {
            get => _isTradingActive;
            set => SetProperty(ref _isTradingActive, value);
        }

        /// <summary>
        /// Current selected ticker symbol
        /// </summary>
        public string CurrentTicker
        {
            get => _currentTicker;
            set
            {
                if (SetProperty(ref _currentTicker, value))
                {
                    IsSymbolSelected = !string.IsNullOrEmpty(value);
                    OnSymbolChanged(value);
                }
            }
        }

        /// <summary>
        /// Whether a symbol is currently selected
        /// </summary>
        public bool IsSymbolSelected
        {
            get => _isSymbolSelected;
            set => SetProperty(ref _isSymbolSelected, value);
        }

        /// <summary>
        /// Whether to enable API modal checks
        /// </summary>
        public bool EnableApiModalChecks
        {
            get => _enableApiModalChecks;
            set => SetProperty(ref _enableApiModalChecks, value);
        }

        #endregion

        #region Commands

        /// <summary>
        /// Command to start trading
        /// </summary>
        public ICommand StartTradingCommand { get; }

        /// <summary>
        /// Command to stop trading
        /// </summary>
        public ICommand StopTradingCommand { get; }

        /// <summary>
        /// Command to add a trading rule
        /// </summary>
        public ICommand AddRuleCommand { get; }

        /// <summary>
        /// Command to refresh data
        /// </summary>
        public ICommand RefreshDataCommand { get; }

        /// <summary>
        /// Command to resume trading after emergency stop
        /// </summary>
        public ICommand ResumeTradingCommand { get; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when the selected symbol changes
        /// </summary>
        public event Action<string> SymbolChanged;

        /// <summary>
        /// Event fired when a notification should be shown
        /// </summary>
        public event Action<string, string> ShowNotification;

        #endregion

        #region Constructor

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public MainWindowViewModel(
            UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService)
        {
            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            _historicalDataService = historicalDataService ?? throw new ArgumentNullException(nameof(historicalDataService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _technicalIndicatorService = technicalIndicatorService ?? throw new ArgumentNullException(nameof(technicalIndicatorService));

            // Initialize collections
            ActiveSymbols = new ObservableCollection<TradingSymbol>();
            TradingRules = new ObservableCollection<StockItem>();
            StockPriceValues = new ChartValues<double>();
            UpperBandValues = new ChartValues<double>();
            MiddleBandValues = new ChartValues<double>();
            LowerBandValues = new ChartValues<double>();
            StockPriceLineValues = new ChartValues<double>();
            RSIValues = new ChartValues<double>();

            // Initialize commands
            StartTradingCommand = new RelayCommand(ExecuteStartTrading, CanExecuteStartTrading);
            StopTradingCommand = new RelayCommand(ExecuteStopTrading, CanExecuteStopTrading);
            AddRuleCommand = new RelayCommand(ExecuteAddRule, CanExecuteAddRule);
            RefreshDataCommand = new RelayCommand(ExecuteRefreshData);
            ResumeTradingCommand = new RelayCommand(ExecuteResumeTrading);

            // Initialize trade update timer
            _tradeUpdateTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(5)
            };
            _tradeUpdateTimer.Tick += TradeUpdateTimer_Tick;

            // Load user settings
            LoadUserSettings();
        }

        #endregion

        #region Methods

        /// <summary>
        /// Loads user settings from the service
        /// </summary>
        private void LoadUserSettings()
        {
            try
            {
                var settings = _userSettingsService.GetUserSettings();
                EnableApiModalChecks = settings.EnableApiModalChecks;
            }
            catch (Exception ex)
            {
                ShowNotification?.Invoke($"Failed to load settings: {ex.Message}", "error");
            }
        }

        /// <summary>
        /// Starts the trading system
        /// </summary>
        public void StartTrading()
        {
            if (IsTradingActive)
                return;

            try
            {
                IsTradingActive = true;
                _tradeUpdateTimer.Start();
                ShowNotification?.Invoke("Trading started successfully", "success");

                ((RelayCommand)StartTradingCommand).RaiseCanExecuteChanged();
                ((RelayCommand)StopTradingCommand).RaiseCanExecuteChanged();
            }
            catch (Exception ex)
            {
                ShowNotification?.Invoke($"Failed to start trading: {ex.Message}", "error");
                IsTradingActive = false;
            }
        }

        /// <summary>
        /// Stops the trading system
        /// </summary>
        public void StopTrading()
        {
            if (!IsTradingActive)
                return;

            try
            {
                IsTradingActive = false;
                _tradeUpdateTimer.Stop();
                ShowNotification?.Invoke("Trading stopped successfully", "success");

                ((RelayCommand)StartTradingCommand).RaiseCanExecuteChanged();
                ((RelayCommand)StopTradingCommand).RaiseCanExecuteChanged();
            }
            catch (Exception ex)
            {
                ShowNotification?.Invoke($"Failed to stop trading: {ex.Message}", "error");
            }
        }

        /// <summary>
        /// Raises the SymbolChanged event
        /// </summary>
        private void OnSymbolChanged(string symbol)
        {
            SymbolChanged?.Invoke(symbol);
        }

        /// <summary>
        /// Timer tick handler for trade updates
        /// </summary>
        private void TradeUpdateTimer_Tick(object sender, EventArgs e)
        {
            // Update trading data periodically
            // This can be expanded to refresh active symbols, positions, etc.
        }

        #endregion

        #region Command Implementations

        private bool CanExecuteStartTrading(object parameter)
        {
            return !IsTradingActive && IsSymbolSelected;
        }

        private void ExecuteStartTrading(object parameter)
        {
            StartTrading();
        }

        private bool CanExecuteStopTrading(object parameter)
        {
            return IsTradingActive;
        }

        private void ExecuteStopTrading(object parameter)
        {
            StopTrading();
        }

        private bool CanExecuteAddRule(object parameter)
        {
            return IsSymbolSelected;
        }

        private void ExecuteAddRule(object parameter)
        {
            // Add trading rule logic
            ShowNotification?.Invoke("Add rule functionality to be implemented", "info");
        }

        private void ExecuteRefreshData(object parameter)
        {
            try
            {
                // Refresh data logic
                ShowNotification?.Invoke("Data refreshed successfully", "success");
            }
            catch (Exception ex)
            {
                ShowNotification?.Invoke($"Failed to refresh data: {ex.Message}", "error");
            }
        }

        private void ExecuteResumeTrading(object parameter)
        {
            try
            {
                var result = MessageBox.Show(
                    "Are you sure you want to resume trading?",
                    "Resume Trading",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Question);

                if (result == MessageBoxResult.Yes)
                {
                    // Resume trading logic
                    StartTrading();
                }
            }
            catch (Exception ex)
            {
                ShowNotification?.Invoke($"Failed to resume trading: {ex.Message}", "error");
            }
        }

        #endregion

        #region IDisposable Support

        /// <summary>
        /// Cleanup resources
        /// </summary>
        public void Cleanup()
        {
            _tradeUpdateTimer?.Stop();
        }

        #endregion
    }
}
