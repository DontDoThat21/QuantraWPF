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

namespace Quantra
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        // Add the PropertyChanged event
        public event PropertyChangedEventHandler PropertyChanged;

        // Add the SymbolChanged event
        public event Action<string> SymbolChanged;

        #region Core Properties and Fields
        private WebullTradingBot tradingBot;// = new WebullTradingBot();
        private UserSettingsService _userSettingsService;
        private HistoricalDataService _historicalDataService;
        private TechnicalIndicatorService _technicalIndicatorService;
        private DispatcherTimer tradeUpdateTimer;
        public ObservableCollection<TradingSymbol> ActiveSymbols { get; set; }
        public ObservableCollection<StockItem> TradingRules { get; set; }
        private bool isTradingActive = false;
        public ChartValues<double> StockPriceValues { get; set; }
        public ChartValues<double> UpperBandValues { get; set; }
        public ChartValues<double> MiddleBandValues { get; set; }
        public ChartValues<double> LowerBandValues { get; set; }
        public ChartValues<double> StockPriceLineValues { get; set; }
        public ChartValues<double> RSIValues { get; set; }
        private List<string> availableStocks = new List<string>();
        private string currentTicker = "";
        private DispatcherTimer debounceTimer;
        private Dictionary<string, Dictionary<string, StockData>> stockDataCache = new Dictionary<string, Dictionary<string, StockData>>();
        private Action confirmApiCallAction;

        private bool isSymbolSelected;
        public bool IsSymbolSelected
        {
            get => isSymbolSelected;
            set
            {
                if (isSymbolSelected != value)
                {
                    isSymbolSelected = value;
                    OnPropertyChanged(nameof(IsSymbolSelected));
                }
            }
        }

        public Func<double, string> DateFormatter { get; set; }
        private bool EnableApiModalChecks { get; set; }

        // Add TabManager property
        public Utilities.TabManager TabManager { get; private set; }
        #endregion

        public MainWindow(UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService
            )
        {
            InitializeComponent();
            tradingBot = new WebullTradingBot(userSettingsService,
                historicalDataService,
                alphaVantageService,
                technicalIndicatorService);
            ActiveSymbols = [];
            TradingRules = [];

            _userSettingsService = userSettingsService;

            UpperBandValues = [];
            MiddleBandValues = [];
            LowerBandValues = [];
            StockPriceLineValues = [];
            RSIValues = [];

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

            // Load user settings
            var settings = _userSettingsService.GetUserSettings();
            EnableApiModalChecks = settings.EnableApiModalChecks;

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

            // Save window state for next startup
            _userSettingsService.SaveWindowState(this.WindowState);

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