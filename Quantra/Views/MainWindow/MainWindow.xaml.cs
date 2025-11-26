using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;
using CefSharp;
using CefSharp.OffScreen;
using LiveCharts;
using LiveCharts.Wpf;
using System.Windows.Media;
using System.ComponentModel;
using System.Collections.ObjectModel;
using Quantra.Controls;
using System.Windows.Shapes;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Enums;
using Dapper;
using Microsoft.Extensions.DependencyInjection;
using Quantra.DAL.Services;
using Quantra.DAL.Data;
using Quantra.ViewModels;

namespace Quantra
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        // Add the PropertyChanged event
        public event PropertyChangedEventHandler PropertyChanged;

        // Add the SymbolChanged event
        public event Action<string> SymbolChanged;

        #region Core Properties and Fields
        private WebullTradingBot tradingBot;
        private UserSettingsService _userSettingsService;
        private HistoricalDataService _historicalDataService;
        private TechnicalIndicatorService _technicalIndicatorService;
        private DispatcherTimer tradeUpdateTimer;
        private DispatcherTimer debounceTimer;
        private Dictionary<string, Dictionary<string, StockData>> stockDataCache = new Dictionary<string, Dictionary<string, StockData>>();
        private Action confirmApiCallAction;
        private List<string> availableStocks = new List<string>();
        private string currentTicker = "";
        private bool isTradingActive = false;
        
        // ViewModel
        private MainWindowViewModel _viewModel;
        public MainWindowViewModel ViewModel
        {
            get => _viewModel;
            set
            {
                _viewModel = value;
                OnPropertyChanged(nameof(ViewModel));
            }
        }

        // Expose ViewModel properties for backward compatibility
        public ObservableCollection<TradingSymbol> ActiveSymbols => ViewModel?.ActiveSymbols;
        public ObservableCollection<StockItem> TradingRules => ViewModel?.TradingRules;
        public ChartValues<double> StockPriceValues => ViewModel?.StockPriceValues;
        public ChartValues<double> UpperBandValues => ViewModel?.UpperBandValues;
        public ChartValues<double> MiddleBandValues => ViewModel?.MiddleBandValues;
        public ChartValues<double> LowerBandValues => ViewModel?.LowerBandValues;
        public ChartValues<double> StockPriceLineValues => ViewModel?.StockPriceLineValues;
        public ChartValues<double> RSIValues => ViewModel?.RSIValues;
        public Func<double, string> DateFormatter => ViewModel?.DateFormatter;
        
        public bool IsSymbolSelected
        {
            get => ViewModel?.IsSymbolSelected ?? false;
            set
            {
                if (ViewModel != null)
                {
                    ViewModel.IsSymbolSelected = value;
                }
            }
        }

        private bool EnableApiModalChecks => ViewModel?.EnableApiModalChecks ?? false;

        // Add TabManager property
        public Utilities.TabManager TabManager { get; private set; }
        #endregion

        // Parameterless constructor for XAML designer support
        public MainWindow()
        {
            InitializeComponent();
            
            // Initialize services from DI container to avoid null reference
            _userSettingsService = App.ServiceProvider?.GetService<UserSettingsService>();
            var historicalDataService = App.ServiceProvider?.GetService<HistoricalDataService>();
            var alphaVantageService = App.ServiceProvider?.GetService<AlphaVantageService>();
            var technicalIndicatorService = App.ServiceProvider?.GetService<TechnicalIndicatorService>();
            
            if (_userSettingsService == null || historicalDataService == null || 
                alphaVantageService == null || technicalIndicatorService == null)
            {
                // Log warning but don't throw - XAML designer might be using this constructor
                System.Diagnostics.Debug.WriteLine("Warning: Services not available in parameterless constructor");
                return;
            }
            
            // Initialize ViewModel
            ViewModel = new MainWindowViewModel(
                _userSettingsService,
                historicalDataService,
                alphaVantageService,
                technicalIndicatorService);
            
            DataContext = ViewModel;
        }

        public MainWindow(UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService
            )
        {
            InitializeComponent();
            
            // Initialize services
            _userSettingsService = userSettingsService;
            _historicalDataService = historicalDataService;
            _technicalIndicatorService = technicalIndicatorService;
            _alphaVantageService = alphaVantageService;

            // Initialize ViewModel
            ViewModel = new MainWindowViewModel(
                userSettingsService,
                historicalDataService,
                alphaVantageService,
                technicalIndicatorService);
            
            DataContext = ViewModel;
            
            // Subscribe to ViewModel events
            ViewModel.SymbolChanged += OnSymbolChanged;
            ViewModel.ShowNotification += OnShowNotification;
            
            // Initialize trading bot
            tradingBot = new WebullTradingBot(userSettingsService,
                historicalDataService,
                alphaVantageService,
                technicalIndicatorService);

            // Initialize services needed by TabManagement.cs from DI container
            _notificationService = App.ServiceProvider.GetService<NotificationService>() 
                ?? throw new InvalidOperationException("NotificationService not registered in DI container");
            _indicatorService = technicalIndicatorService as TechnicalIndicatorService 
                ?? throw new InvalidOperationException("TechnicalIndicatorService is not of expected type");
            _stockDataCacheService = App.ServiceProvider.GetService<StockDataCacheService>()
                ?? throw new InvalidOperationException("StockDataCacheService not registered in DI container");
            _emailService = App.ServiceProvider.GetService<EmailService>()
                ?? throw new InvalidOperationException("EmailService not registered in DI container");
            _tradingService = App.ServiceProvider.GetService<TradingService>()
                ?? throw new InvalidOperationException("TradingService not registered in DI container");
            _settingsService = App.ServiceProvider.GetService<SettingsService>()
                ?? throw new InvalidOperationException("SettingsService not registered in DI container");
            _loggingService = App.ServiceProvider.GetService<LoggingService>()
                ?? throw new InvalidOperationException("LoggingService not registered in DI container");
            _quantraDbContext = App.ServiceProvider.GetService<QuantraDbContext>()
                ?? throw new InvalidOperationException("QuantraDbContext not registered in DI container");
            _inferenceService = App.ServiceProvider.GetService<RealTimeInferenceService>()
                ?? throw new InvalidOperationException("RealTimeInferenceService not registered in DI container");
            _predictionCacheService = App.ServiceProvider.GetService<PredictionCacheService>()
                ?? throw new InvalidOperationException("PredictionCacheService not registered in DI container");

            // Initialize debounce timer - increased interval to reduce UI load
            debounceTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(1000) };
            debounceTimer.Tick += DebounceTimer_Tick;

            // Initialize tab management
            InitializeTabManagement();

            // Ensure database tables exist using the new service
            var dbInitService = App.ServiceProvider.GetService<IDatabaseInitializationService>();
            if (dbInitService != null)
            {
                dbInitService.EnsureUserAppSettingsTable();
            }

            // Load all tabs from database
            TabManager.LoadCustomTabs();

            // Load card positions
            LoadCardPositions();

            // Set the last non-'+' tab during initialization
            SetInitialLastNonPlusTab();

            // Initialize tradeUpdateTimer - increased interval to reduce UI load
            tradeUpdateTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(5) };
            tradeUpdateTimer.Tick += TradeUpdateTimer_Tick;
            
            // Subscribe to notification service events
            var notificationService = App.ServiceProvider.GetService<INotificationService>() as NotificationService;
            if (notificationService != null)
            {
                notificationService.OnShowNotification += OnNotificationReceived;
            }
            
            // Setup emergency stop banner check timer
            var orderService = App.ServiceProvider.GetService<IOrderService>();
            if (orderService != null)
            {
                // Increased interval to reduce UI load from emergency stop checking
                var emergencyStopTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(5) };
                emergencyStopTimer.Tick += (s, e) => 
                {
                    EmergencyStopBanner.Visibility = orderService.IsEmergencyStopActive() ? Visibility.Visible : Visibility.Collapsed;
                };
                emergencyStopTimer.Start();
                EmergencyStopBanner.Visibility = orderService.IsEmergencyStopActive() ? Visibility.Visible : Visibility.Collapsed;
            }

            this.Loaded += MainWindow_Loaded;
        }

        #region Core functionality methods

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            SaveCardPositions();

            // Cleanup ViewModel
            ViewModel?.Cleanup();

            // Save window state for next startup
            _userSettingsService?.SaveWindowState(this.WindowState);

            // Ensure the LoginWindow is shown when MainWindow is closed
            if (Application.Current.Windows.OfType<LoginWindow>().FirstOrDefault() == null)
            {
                var loginWindow = new LoginWindow(_userSettingsService, _historicalDataService, _alphaVantageService, _technicalIndicatorService);
                loginWindow.Show();
            }
        }
        
        private void ResumeTrading_Click(object sender, RoutedEventArgs e)
        {
            var orderService = App.ServiceProvider.GetService<IOrderService>();
            if (orderService != null)
            {
                var resumeResult = MessageBox.Show(
                    "Are you sure you want to resume trading?", 
                    "Resume Trading", 
                    MessageBoxButton.YesNo, 
                    MessageBoxImage.Question);
                    
                if (resumeResult == MessageBoxResult.Yes)
                {
                    bool success = orderService.DeactivateEmergencyStop();
                    EmergencyStopBanner.Visibility = orderService.IsEmergencyStopActive() ? Visibility.Visible : Visibility.Collapsed;
                    
                    if (success)
                    {
                        MessageBox.Show("Trading has been resumed.", "Trading Resumed", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
            }
        }

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        // Helper method to find a visual child element
        private T FindVisualChild<T>(DependencyObject parent, Func<T, bool> condition) where T : DependencyObject
        {
            if (parent == null) return null;

            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            {
                var child = VisualTreeHelper.GetChild(parent, i);

                // Check if the child is what we're looking for
                if (child is T typedChild && condition(typedChild))
                    return typedChild;

                // Recursively check child elements
                var result = FindVisualChild<T>(child, condition);
                if (result != null)
                    return result;
            }

            return null;
        }

        // Add this method to invoke the event when a symbol changes
        private void OnSymbolChanged(string symbol)
        {
            SymbolChanged?.Invoke(symbol);
        }
        
        // Handle ViewModel show notification event
        private void OnShowNotification(string message, string type)
        {
            // Map to existing notification mechanism if available
            // This can be expanded based on your notification system
            AppendAlert(message, type);
        }

        #endregion
    }

    #region Model Classes
    public class TradingSymbol
    {
        public string Symbol { get; set; }
        public string CurrentPrice { get; set; }
        public string DiffFromPositionAvg { get; set; }
    }

    public class StockItem
    {
        public string Symbol { get; set; }
        public string CurrentPrice { get; set; }
        public string PercentageDiff { get; set; }
    }
    #endregion
}