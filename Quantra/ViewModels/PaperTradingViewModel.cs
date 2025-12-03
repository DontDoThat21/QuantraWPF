using MaterialDesignThemes.Wpf;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
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
        private HistoricalDataSource _dataSource;
        private IPaperTradingPersistenceService _persistenceService;
        private PaperTradingSessionEntity _currentSession;
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

            // Initialize persistence service
            _persistenceService = new PaperTradingPersistenceService();
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public PaperTradingViewModel(IPaperTradingPersistenceService persistenceService)
        {
            _positions = new ObservableCollection<TradingPosition>();
            _orders = new ObservableCollection<Order>();

            // Initialize with default values
            _cashBalance = 100000m;
            _totalValue = 100000m;
            _buyingPower = 100000m;

            _persistenceService = persistenceService ?? throw new ArgumentNullException(nameof(persistenceService));
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
            _ = InitializeWithExceptionHandlingAsync();
        }

        /// <summary>
        /// Wrapper for async initialization with exception handling
        /// </summary>
        private async Task InitializeWithExceptionHandlingAsync()
        {
            try
            {
                await InitializeAsync();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error in background initialization: {ex.Message}");
            }
        }

        /// <summary>
        /// Initializes the trading engine and components asynchronously
        /// </summary>
        public async Task InitializeAsync()
        {
            try
            {
                // Create components
                _dataSource = new HistoricalDataSource();
                _clock = new RealTimeClock(1000); // 1 second tick interval
                _portfolioManager = new PortfolioManager(100000m);
                _tradingEngine = new TradingEngine();

                // Try to restore from an existing active session
                await RestoreSessionAsync();

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
        /// Restores session state from the database
        /// </summary>
        private async Task RestoreSessionAsync()
        {
            try
            {
                if (_persistenceService == null)
                {
                    return;
                }

                // Try to get active session
                _currentSession = await _persistenceService.GetActiveSessionAsync();

                if (_currentSession != null)
                {
                    // Restore portfolio state with saved cash balance
                    _portfolioManager.Reset(_currentSession.CashBalance);

                    // Restore realized P&L from the session
                    _portfolioManager.SetRealizedPnL(_currentSession.RealizedPnL);

                    // Restore positions from the database
                    var positions = await _persistenceService.RestorePositionsAsync(_currentSession.Id);
                    foreach (var position in positions.Values)
                    {
                        _portfolioManager.RestorePosition(position);
                    }

                    System.Diagnostics.Debug.WriteLine($"Restored session: {_currentSession.SessionId} with {positions.Count} positions");
                }
                else
                {
                    // Create a new session
                    _currentSession = await _persistenceService.CreateSessionAsync(null, 100000m);
                    System.Diagnostics.Debug.WriteLine($"Created new session: {_currentSession.SessionId}");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error restoring session: {ex.Message}");
                // Continue without persistence if it fails
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

                // Update session state in database
                await UpdateSessionStateAsync();
            }
            else
            {
                // Start the engine
                _tradingEngine.Start();
                IsEngineRunning = true;
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

            // Persist the order
            await SaveOrderAsync(order);

            RefreshOrders();
        }

        /// <summary>
        /// Saves an order to the database
        /// </summary>
        private async Task SaveOrderAsync(Order order)
        {
            try
            {
                if (_persistenceService != null && _currentSession != null)
                {
                    await _persistenceService.SaveOrderAsync(_currentSession.Id, order);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving order: {ex.Message}");
            }
        }

        /// <summary>
        /// Updates an order in the database
        /// </summary>
        private async Task UpdateOrderAsync(Order order)
        {
            try
            {
                if (_persistenceService != null && _currentSession != null)
                {
                    await _persistenceService.UpdateOrderAsync(order.Id, order);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating order: {ex.Message}");
            }
        }

        /// <summary>
        /// Saves a position to the database
        /// </summary>
        private async Task SavePositionAsync(TradingPosition position)
        {
            try
            {
                if (_persistenceService != null && _currentSession != null)
                {
                    await _persistenceService.SavePositionAsync(_currentSession.Id, position);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving position: {ex.Message}");
            }
        }

        /// <summary>
        /// Updates the session state in the database
        /// </summary>
        private async Task UpdateSessionStateAsync()
        {
            try
            {
                if (_persistenceService != null && _currentSession != null && _portfolioManager != null)
                {
                    await _persistenceService.UpdateSessionStateAsync(
                        _currentSession.Id, 
                        _portfolioManager.CashBalance, 
                        _portfolioManager.RealizedPnL);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating session state: {ex.Message}");
            }
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
                // Update order in database
                var order = _tradingEngine.GetOrder(orderId);
                if (order != null)
                {
                    _ = UpdateOrderWithExceptionHandlingAsync(order);
                }
                RefreshOrders();
            }
            return result;
        }

        /// <summary>
        /// Wrapper for async order update with exception handling
        /// </summary>
        private async Task UpdateOrderWithExceptionHandlingAsync(Order order)
        {
            try
            {
                await UpdateOrderAsync(order);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating order in background: {ex.Message}");
            }
        }

        /// <summary>
        /// Resets the paper trading account
        /// </summary>
        public async void ResetAccount()
        {
            _ = ResetAccountWithExceptionHandlingAsync();
        }

        /// <summary>
        /// Wrapper for async reset with exception handling
        /// </summary>
        private async Task ResetAccountWithExceptionHandlingAsync()
        {
            try
            {
                await ResetAccountAsync();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error in background reset: {ex.Message}");
            }
        }

        /// <summary>
        /// Resets the paper trading account asynchronously
        /// </summary>
        public async Task ResetAccountAsync()
        {
            if (_tradingEngine != null)
            {
                _tradingEngine.Stop();
            }

            _portfolioManager?.Reset(100000m);
            _tradingEngine?.Reset();

            // Clear persistence data and create new session
            try
            {
                if (_persistenceService != null && _currentSession != null)
                {
                    await _persistenceService.ClearPositionsAsync(_currentSession.Id);
                    await _persistenceService.ClearOrdersAsync(_currentSession.Id);
                    await _persistenceService.EndSessionAsync(_currentSession.Id);

                    // Create a new session
                    _currentSession = await _persistenceService.CreateSessionAsync(null, 100000m);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error resetting persistence: {ex.Message}");
            }

            IsEngineRunning = false;

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
            // Save the fill and update position/order in the background
            // Note: SaveFillAsync already has exception handling
            _ = SaveFillAsync(e);

            RefreshPortfolio();
            RefreshPositions();
            RefreshOrders();
        }

        private async Task SaveFillAsync(OrderFilledEventArgs e)
        {
            try
            {
                if (_persistenceService == null || _currentSession == null)
                {
                    return;
                }

                // Update the order
                await _persistenceService.UpdateOrderAsync(e.Order.Id, e.Order);

                // Save position if we have one
                var position = _portfolioManager?.GetPosition(e.Fill.Symbol);
                if (position != null)
                {
                    await _persistenceService.SavePositionAsync(_currentSession.Id, position);
                }

                // Update session state
                await UpdateSessionStateAsync();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving fill: {ex.Message}");
            }
        }

        private void OnOrderStateChanged(object sender, OrderStateChangedEventArgs e)
        {
            // Update order state in the background with exception handling
            _ = UpdateOrderWithExceptionHandlingAsync(e.Order);

            RefreshOrders();
        }

        private async void OnPortfolioChanged(object sender, PortfolioChangedEventArgs e)
        {
            // Update session state in the background with exception handling
            _ = UpdateSessionStateWithExceptionHandlingAsync();

            RefreshPortfolio();
        }

        /// <summary>
        /// Wrapper for async session state update with exception handling
        /// </summary>
        private async Task UpdateSessionStateWithExceptionHandlingAsync()
        {
            try
            {
                await UpdateSessionStateAsync();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating session state in background: {ex.Message}");
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
                    // Save final state before disposing using a synchronous-safe approach
                    SaveFinalStateSafe();

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

        /// <summary>
        /// Saves the final state synchronously without risking deadlock
        /// </summary>
        private void SaveFinalStateSafe()
        {
            try
            {
                // Use Task.Run to avoid deadlock in synchronization context
                Task.Run(async () => await SaveFinalStateAsync().ConfigureAwait(false)).Wait(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving final state: {ex.Message}");
            }
        }

        /// <summary>
        /// Saves the final state to the database before disposal
        /// </summary>
        private async Task SaveFinalStateAsync()
        {
            try
            {
                if (_persistenceService == null || _currentSession == null || _portfolioManager == null)
                {
                    return;
                }

                // Save all positions
                foreach (var position in _portfolioManager.Positions.Values)
                {
                    await _persistenceService.SavePositionAsync(_currentSession.Id, position);
                }

                // Save all orders
                if (_tradingEngine != null)
                {
                    foreach (var order in _tradingEngine.GetAllOrders())
                    {
                        await _persistenceService.SaveOrderAsync(_currentSession.Id, order);
                    }
                }

                // Update session state
                await UpdateSessionStateAsync();

                System.Diagnostics.Debug.WriteLine("Paper trading state saved successfully");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving final state: {ex.Message}");
            }
        }

        ~PaperTradingViewModel()
        {
            Dispose(false);
        }

        #endregion
    }
}
