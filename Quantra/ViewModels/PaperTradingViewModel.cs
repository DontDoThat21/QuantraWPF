using MaterialDesignThemes.Wpf;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Services;
using Quantra.DAL.TradingEngine.Core;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;
using Quantra.DAL.TradingEngine.Time;
using Quantra.ViewModels.Base;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Paper Trading control
    /// </summary>
    public class PaperTradingViewModel : ViewModelBase, IDisposable
    {
        private TradingEngine _tradingEngine;
        private PortfolioManager _portfolioManager;
        private RealTimeClock _clock;
        private IDataSource _dataSource;
        private PaperTradingSessionService _sessionService;
        private Guid? _currentSessionId;
        private bool _isDisposed;

        // Portfolio properties
        private decimal _totalValue;
        private decimal _cashBalance;
        private decimal _unrealizedPnL;
        private decimal _realizedPnL;
        private decimal _buyingPower;
        private int _positionCount;

        // Order entry properties
        private string _orderSymbol = string.Empty;
        private int _orderQuantity = 100;
        private decimal _limitPrice;
        private decimal _stopPrice;
        private ComboBoxItem _selectedOrderType;
        private bool _showAllOrders;

        // Engine status
        private bool _isEngineRunning;

        // Notification properties
        private string _notificationText = string.Empty;
        private PackIconKind _notificationIcon;
        private Brush _notificationIconColor;
        private Brush _notificationBorderBrush;

        // Collections
        private ObservableCollection<TradingPosition> _positions;
        private ObservableCollection<Order> _orders;
        private Order _selectedOrder;

        public PaperTradingViewModel(IAlphaVantageService alphaVantageService)
        {
            _positions = new ObservableCollection<TradingPosition>();
            _orders = new ObservableCollection<Order>();

            // Initialize with default values
            _cashBalance = 100000m;
            _totalValue = 100000m;
            _buyingPower = 100000m;

            // Initialize session service with connection string (not DbContext instance)
            // This allows the service to create its own DbContext per operation, avoiding threading issues
            var loggingService = new LoggingService();
            _sessionService = new PaperTradingSessionService(ConnectionHelper.ConnectionString, loggingService);

            // Store AlphaVantage service for initialization
            _alphaVantageService = alphaVantageService;
        }

        private IAlphaVantageService _alphaVantageService;

        #region Portfolio Properties

        public decimal TotalValue
        {
            get => _totalValue;
            set => SetProperty(ref _totalValue, value);
        }

        public decimal CashBalance
        {
            get => _cashBalance;
            set => SetProperty(ref _cashBalance, value);
        }

        public decimal UnrealizedPnL
        {
            get => _unrealizedPnL;
            set => SetProperty(ref _unrealizedPnL, value);
        }

        public decimal RealizedPnL
        {
            get => _realizedPnL;
            set => SetProperty(ref _realizedPnL, value);
        }

        public decimal BuyingPower
        {
            get => _buyingPower;
            set => SetProperty(ref _buyingPower, value);
        }

        public int PositionCount
        {
            get => _positionCount;
            set => SetProperty(ref _positionCount, value);
        }

        #endregion

        #region Order Entry Properties

        public string OrderSymbol
        {
            get => _orderSymbol;
            set => SetProperty(ref _orderSymbol, value?.ToUpperInvariant() ?? string.Empty);
        }

        public int OrderQuantity
        {
            get => _orderQuantity;
            set => SetProperty(ref _orderQuantity, value);
        }

        public decimal LimitPrice
        {
            get => _limitPrice;
            set => SetProperty(ref _limitPrice, value);
        }

        public decimal StopPrice
        {
            get => _stopPrice;
            set => SetProperty(ref _stopPrice, value);
        }

        public ComboBoxItem SelectedOrderType
        {
            get => _selectedOrderType;
            set
            {
                if (SetProperty(ref _selectedOrderType, value))
                {
                    OnPropertyChanged(nameof(ShowLimitPrice));
                    OnPropertyChanged(nameof(ShowStopPrice));
                }
            }
        }

        public Visibility ShowLimitPrice
        {
            get
            {
                var orderType = SelectedOrderType?.Content?.ToString() ?? "Market";
                return orderType.Contains("Limit") ? Visibility.Visible : Visibility.Collapsed;
            }
        }

        public Visibility ShowStopPrice
        {
            get
            {
                var orderType = SelectedOrderType?.Content?.ToString() ?? "Market";
                return orderType.Contains("Stop") ? Visibility.Visible : Visibility.Collapsed;
            }
        }

        public bool ShowAllOrders
        {
            get => _showAllOrders;
            set
            {
                if (SetProperty(ref _showAllOrders, value))
                {
                    RefreshOrders();
                }
            }
        }

        #endregion

        #region Engine Status Properties

        public bool IsEngineRunning
        {
            get => _isEngineRunning;
            set
            {
                if (SetProperty(ref _isEngineRunning, value))
                {
                    OnPropertyChanged(nameof(EngineStatusText));
                    OnPropertyChanged(nameof(EngineStatusColor));
                    OnPropertyChanged(nameof(StartStopButtonText));
                    OnPropertyChanged(nameof(StartStopButtonColor));
                }
            }
        }

        public string EngineStatusText => IsEngineRunning ? "RUNNING" : "STOPPED";

        public Brush EngineStatusColor => IsEngineRunning
            ? new SolidColorBrush(Color.FromRgb(34, 139, 34))  // Forest Green
            : new SolidColorBrush(Color.FromRgb(220, 20, 60)); // Crimson

        public string StartStopButtonText => IsEngineRunning ? "Stop" : "Start";

        public Brush StartStopButtonColor => IsEngineRunning
            ? new SolidColorBrush(Color.FromRgb(220, 20, 60)) // Crimson for Stop
            : new SolidColorBrush(Color.FromRgb(34, 139, 34)); // Forest Green for Start

        #endregion

        #region Notification Properties

        public string NotificationText
        {
            get => _notificationText;
            set => SetProperty(ref _notificationText, value);
        }

        public PackIconKind NotificationIcon
        {
            get => _notificationIcon;
            set => SetProperty(ref _notificationIcon, value);
        }

        public Brush NotificationIconColor
        {
            get => _notificationIconColor;
            set => SetProperty(ref _notificationIconColor, value);
        }

        public Brush NotificationBorderBrush
        {
            get => _notificationBorderBrush;
            set => SetProperty(ref _notificationBorderBrush, value);
        }

        #endregion

        #region Collections

        public ObservableCollection<TradingPosition> Positions
        {
            get => _positions;
            set => SetProperty(ref _positions, value);
        }

        public ObservableCollection<Order> Orders
        {
            get => _orders;
            set => SetProperty(ref _orders, value);
        }

        public Order SelectedOrder
        {
            get => _selectedOrder;
            set => SetProperty(ref _selectedOrder, value);
        }

        #endregion

        #region Methods

        /// <summary>
        /// Initializes the trading engine and components
        /// </summary>
        public void Initialize()
        {
            try
            {
                // Create components with REAL-TIME data source
                _dataSource = new RealTimeDataSource(_alphaVantageService);
                _clock = new RealTimeClock(1000); // 1 second tick interval
                _portfolioManager = new PortfolioManager(100000m);
                _tradingEngine = new TradingEngine();

                // Initialize the engine
                _tradingEngine.Initialize(_dataSource, _clock, _portfolioManager);

                // Subscribe to events
                _tradingEngine.OrderFilled += OnOrderFilled;
                _tradingEngine.OrderStateChanged += OnOrderStateChanged;
                _portfolioManager.PortfolioChanged += OnPortfolioChanged;

                // Update UI
                RefreshPortfolio();
                RefreshPositions();
                RefreshOrders();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error initializing PaperTradingViewModel: {ex.Message}");
            }
        }

        /// <summary>
        /// Toggles the trading engine on/off
        /// </summary>
        public async void ToggleEngine()
        {
            if (_tradingEngine == null) return;

            if (IsEngineRunning)
            {
                // Stop the engine
                _tradingEngine.Stop();
                IsEngineRunning = false;

                // Save session to database
                if (_currentSessionId.HasValue && _portfolioManager != null)
                {
                    await _sessionService.StopSessionAsync(
                        _currentSessionId.Value,
                        _portfolioManager.CashBalance,
                        _portfolioManager.TotalValue,
                        _portfolioManager.RealizedPnL,
                        _portfolioManager.UnrealizedPnL
                    );
                    _currentSessionId = null;
                }
            }
            else
            {
                // Start the engine
                _tradingEngine.Start();
                IsEngineRunning = true;

                // Create new session in database
                if (_portfolioManager != null)
                {
                    _currentSessionId = await _sessionService.StartSessionAsync(_portfolioManager.CashBalance);
                }
            }
        }

        /// <summary>
        /// Places an order with the specified side
        /// </summary>
        public async Task PlaceOrderAsync(OrderSide side)
        {
            if (_tradingEngine == null || string.IsNullOrWhiteSpace(OrderSymbol))
            {
                return;
            }

            Order order;
            var orderTypeString = SelectedOrderType?.Content?.ToString() ?? "Market";

            switch (orderTypeString)
            {
                case "Limit":
                    order = Order.CreateLimitOrder(OrderSymbol, side, OrderQuantity, LimitPrice);
                    break;
                case "Stop":
                    order = Order.CreateStopOrder(OrderSymbol, side, OrderQuantity, StopPrice);
                    break;
                case "Stop Limit":
                    order = Order.CreateStopLimitOrder(OrderSymbol, side, OrderQuantity, StopPrice, LimitPrice);
                    break;
                default: // Market
                    order = Order.CreateMarketOrder(OrderSymbol, side, OrderQuantity);
                    break;
            }

            await _tradingEngine.PlaceOrderAsync(order);
            RefreshOrders();
        }

        /// <summary>
        /// Cancels an order
        /// </summary>
        public bool CancelOrder(Guid orderId)
        {
            if (_tradingEngine == null) return false;

            var result = _tradingEngine.CancelOrder(orderId);
            if (result)
            {
                RefreshOrders();
            }
            return result;
        }

        /// <summary>
        /// Resets the paper trading account
        /// </summary>
        public async void ResetAccount()
        {
            if (_tradingEngine != null)
            {
                _tradingEngine.Stop();
            }

            _portfolioManager?.Reset(100000m);
            _tradingEngine?.Reset();

            IsEngineRunning = false;

            // Reset session in database
            if (_currentSessionId.HasValue)
            {
                _currentSessionId = await _sessionService.ResetSessionAsync(_currentSessionId.Value, 100000m);
            }

            RefreshPortfolio();
            RefreshPositions();
            RefreshOrders();
        }

        /// <summary>
        /// Refreshes all data
        /// </summary>
        public void Refresh()
        {
            RefreshPortfolio();
            RefreshPositions();
            RefreshOrders();
        }

        /// <summary>
        /// Refreshes portfolio summary
        /// </summary>
        private void RefreshPortfolio()
        {
            if (_portfolioManager == null) return;

            Application.Current?.Dispatcher?.Invoke(() =>
            {
                TotalValue = _portfolioManager.TotalValue;
                CashBalance = _portfolioManager.CashBalance;
                UnrealizedPnL = _portfolioManager.UnrealizedPnL;
                RealizedPnL = _portfolioManager.RealizedPnL;
                BuyingPower = _portfolioManager.BuyingPower;
                PositionCount = _portfolioManager.Positions.Count;
            });
        }

        /// <summary>
        /// Refreshes positions list
        /// </summary>
        private void RefreshPositions()
        {
            if (_portfolioManager == null) return;

            Application.Current?.Dispatcher?.Invoke(() =>
            {
                Positions.Clear();
                foreach (var position in _portfolioManager.Positions.Values)
                {
                    Positions.Add(position);
                }
            });
        }

        /// <summary>
        /// Refreshes orders list
        /// </summary>
        public void RefreshOrders()
        {
            if (_tradingEngine == null) return;

            Application.Current?.Dispatcher?.Invoke(() =>
            {
                Orders.Clear();
                var ordersToShow = ShowAllOrders
                    ? _tradingEngine.GetAllOrders()
                    : _tradingEngine.GetActiveOrders();

                foreach (var order in ordersToShow.OrderByDescending(o => o.CreatedTime))
                {
                    Orders.Add(order);
                }
            });
        }

        #endregion

        #region Event Handlers

        private async void OnOrderFilled(object sender, OrderFilledEventArgs e)
        {
            RefreshPortfolio();
            RefreshPositions();
            RefreshOrders();

            // Update session in database
            await UpdateSessionInDatabaseAsync();
        }

        private void OnOrderStateChanged(object sender, OrderStateChangedEventArgs e)
        {
            RefreshOrders();
        }

        private async void OnPortfolioChanged(object sender, PortfolioChangedEventArgs e)
        {
            RefreshPortfolio();

            // Update session in database
            await UpdateSessionInDatabaseAsync();
        }

        /// <summary>
        /// Updates the current session statistics in the database
        /// </summary>
        private async System.Threading.Tasks.Task UpdateSessionInDatabaseAsync()
        {
            if (!_currentSessionId.HasValue || _portfolioManager == null || _tradingEngine == null)
                return;

            try
            {
                // Calculate trade statistics
                var fills = _tradingEngine.GetFills();
                int winningTrades = 0;
                int losingTrades = 0;

                foreach (var fill in fills)
                {
                    // Simple win/loss calculation based on whether we made or lost money
                    // In a real implementation, you'd track individual position P&L
                    if (_portfolioManager.RealizedPnL > 0)
                        winningTrades++;
                    else if (_portfolioManager.RealizedPnL < 0)
                        losingTrades++;
                }

                await _sessionService.UpdateSessionAsync(
                    _currentSessionId.Value,
                    _portfolioManager.CashBalance,
                    _portfolioManager.TotalValue,
                    _portfolioManager.RealizedPnL,
                    _portfolioManager.UnrealizedPnL,
                    fills.Count,
                    winningTrades,
                    losingTrades
                );
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating session in database: {ex.Message}");
            }
        }

        #endregion

        #region IDisposable

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_isDisposed)
            {
                if (disposing)
                {
                    // Unsubscribe from events
                    if (_tradingEngine != null)
                    {
                        _tradingEngine.OrderFilled -= OnOrderFilled;
                        _tradingEngine.OrderStateChanged -= OnOrderStateChanged;
                        _tradingEngine.Stop();
                    }

                    if (_portfolioManager != null)
                    {
                        _portfolioManager.PortfolioChanged -= OnPortfolioChanged;
                    }

                    _clock?.Dispose();
                }
                _isDisposed = true;
            }
        }

        ~PaperTradingViewModel()
        {
            Dispose(false);
        }

        #endregion
    }
}
