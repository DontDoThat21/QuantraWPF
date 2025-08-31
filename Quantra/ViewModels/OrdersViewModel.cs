using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Media;
using Quantra.Commands;
using Quantra.Enums;
using Quantra.Models;
using Quantra.Services.Interfaces;
using Quantra.ViewModels.Base;
using MaterialDesignThemes.Wpf;
using System.Windows.Input;
using System.Windows;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.ViewModels
{
    public class OrdersViewModel : ViewModelBase
    {
        private readonly IOrderService _orderService;
        private bool _enableApiModalChecks;

        #region Properties

        private OrderModel _order;
        public OrderModel Order
        {
            get => _order;
            set => SetProperty(ref _order, value);
        }

        private ObservableCollection<OrderModel> _orderHistory;
        public ObservableCollection<OrderModel> OrderHistory
        {
            get => _orderHistory;
            set => SetProperty(ref _orderHistory, value);
        }

        private ObservableCollection<OrderModel> _filteredOrders;
        public ObservableCollection<OrderModel> FilteredOrders
        {
            get => _filteredOrders;
            set => SetProperty(ref _filteredOrders, value);
        }

        private string _notificationText;
        public string NotificationText
        {
            get => _notificationText;
            set => SetProperty(ref _notificationText, value);
        }

        private PackIconKind _notificationIcon;
        public PackIconKind NotificationIcon
        {
            get => _notificationIcon;
            set => SetProperty(ref _notificationIcon, value);
        }

        private Brush _notificationIconColor;
        public Brush NotificationIconColor
        {
            get => _notificationIconColor;
            set => SetProperty(ref _notificationIconColor, value);
        }

        private Brush _notificationBorderBrush;
        public Brush NotificationBorderBrush
        {
            get => _notificationBorderBrush;
            set => SetProperty(ref _notificationBorderBrush, value);
        }

        private string _tradeModeBadgeText;
        public string TradeModeBadgeText
        {
            get => _tradeModeBadgeText ?? (Order.IsPaperTrade ? "Paper Trade" : "Real Trade");
            set => SetProperty(ref _tradeModeBadgeText, value);
        }

        public Brush TradeModeBadgeBackground => Order.IsPaperTrade ?
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#2D6A4C")) :
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#6A2D2D"));

        public Brush TradeModeBadgeForeground => Order.IsPaperTrade ?
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#50E070")) :
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E05050"));

        private string _selectedFilter = "All";
        public string SelectedFilter
        {
            get => _selectedFilter;
            set
            {
                SetProperty(ref _selectedFilter, value);
                ApplyFilter();
            }
        }

        private bool _showApiConfirmationDialog;
        public bool ShowApiConfirmationDialog
        {
            get => _showApiConfirmationDialog;
            set => SetProperty(ref _showApiConfirmationDialog, value);
        }

        private bool _dontShowApiDialogAgain;
        public bool DontShowApiDialogAgain
        {
            get => _dontShowApiDialogAgain;
            set => SetProperty(ref _dontShowApiDialogAgain, value);
        }

        #endregion

        #region Commands

        public ICommand PlaceOrderCommand { get; }
        public ICommand CancelApiDialogCommand { get; }
        public ICommand ConfirmApiDialogCommand { get; }
        public ICommand RefreshPriceCommand { get; }
        public ICommand PaperTradeToggledCommand { get; }
        public ICommand CloseCommand { get; }
        public ICommand ClearHistoryCommand { get; }
        public ICommand FilterChangedCommand { get; }

        #endregion

        // Events to connect with the view
        public event Action RequestCloseWindow;
        public event Func<string, MessageBoxResult> ShowConfirmationDialog;

        // Constructor with dependencies
        public OrdersViewModel(IOrderService orderService)
        {
            _orderService = orderService ?? throw new ArgumentNullException(nameof(orderService));
            
            // Initialize order and collections
            Order = _orderService.CreateDefaultOrder();
            OrderHistory = _orderService.LoadOrderHistory();
            FilteredOrders = new ObservableCollection<OrderModel>(OrderHistory);
            
            // Get modal dialog settings
            _enableApiModalChecks = _orderService.GetApiModalCheckSetting();

            // Initialize commands
            PlaceOrderCommand = new RelayCommand(_ => ExecutePlaceOrder());
            CancelApiDialogCommand = new RelayCommand(_ => ShowApiConfirmationDialog = false);
            ConfirmApiDialogCommand = new RelayCommand(_ => ExecuteConfirmApiDialog(_));
            RefreshPriceCommand = new RelayCommand(_ => ExecuteRefreshPrice());
            PaperTradeToggledCommand = new RelayCommand(param => ExecutePaperTradeToggled(param));
            CloseCommand = new RelayCommand(_ => RequestCloseWindow?.Invoke());
            ClearHistoryCommand = new RelayCommand(_ => ExecuteClearHistory());
            FilterChangedCommand = new RelayCommand(param => ExecuteFilterChanged(param));
        }

        // Default constructor for XAML designer support
        public OrdersViewModel() : this(new OrderService()) { }

        #region Command Methods

        private void ExecutePlaceOrder()
        {
            // Validate input
            if (string.IsNullOrWhiteSpace(Order.Symbol))
            {
                SetNotification("Please enter a valid symbol", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }
            
            if (Order.Quantity <= 0)
            {
                SetNotification("Quantity must be greater than zero", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }
            
            if (Order.Price <= 0)
            {
                SetNotification("Price must be greater than zero", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }

            // If API modal checks are enabled, show confirmation dialog
            if (_enableApiModalChecks)
            {
                ShowApiConfirmationDialog = true;
                return;
            }
            
            // Otherwise place the order directly
            PlaceOrderSafely();
        }

        private async Task PlaceOrderAsync()
        {
            try
            {
                // Set trading mode in the service
                _orderService.SetTradingMode(Order.IsPaperTrade ? TradingMode.Paper : TradingMode.Market);
                
                // Place the order using the service
                bool success = await _orderService.PlaceLimitOrder(
                    Order.Symbol,
                    Order.Quantity,
                    Order.OrderType,
                    Order.Price
                );
                
                // Update the order status
                Order.Status = success ? "Executed" : "Failed";
                Order.Timestamp = DateTime.Now;
                
                // Add a copy of the current order to history
                var orderCopy = new OrderModel
                {
                    Symbol = Order.Symbol,
                    Quantity = Order.Quantity,
                    Price = Order.Price,
                    OrderType = Order.OrderType,
                    IsPaperTrade = Order.IsPaperTrade,
                    Status = Order.Status,
                    Timestamp = Order.Timestamp,
                    PredictionSource = Order.PredictionSource,
                    StopLoss = Order.StopLoss,
                    TakeProfit = Order.TakeProfit
                };
                
                OrderHistory.Insert(0, orderCopy);
                ApplyFilter();
                
                // Show success notification
                SetNotification($"Order successfully placed for {Order.Symbol}", 
                                PackIconKind.CheckCircle, Colors.LimeGreen);
                
                // Reset for next order
                ResetOrderForm();
            }
            catch (Exception ex)
            {
                // Update the status
                Order.Status = "Failed";
                Order.Timestamp = DateTime.Now;
                
                // Add error entry to history
                var orderCopy = new OrderModel
                {
                    Symbol = Order.Symbol,
                    Quantity = Order.Quantity,
                    Price = Order.Price,
                    OrderType = Order.OrderType,
                    IsPaperTrade = Order.IsPaperTrade,
                    Status = "Failed",
                    Timestamp = Order.Timestamp,
                    PredictionSource = Order.PredictionSource,
                    StopLoss = Order.StopLoss,
                    TakeProfit = Order.TakeProfit
                };
                
                OrderHistory.Insert(0, orderCopy);
                ApplyFilter();
                
                // Log the error
                DatabaseMonolith.Log("Error", "Failed to place order", ex.ToString());
                
                // Show error notification
                SetNotification($"Error placing order: {ex.Message}", 
                                PackIconKind.AlertCircle, Colors.Red);
            }
        }

        private async void ExecuteConfirmApiDialog(object _)
        {
            try
            {
                // Check if the user doesn't want to see this dialog again
                if (DontShowApiDialogAgain)
                {
                    _orderService.SaveApiModalCheckSetting(false);
                    _enableApiModalChecks = false;
                }
                
                // Hide the confirmation dialog
                ShowApiConfirmationDialog = false;
                
                // Execute the order
                await PlaceOrderAsync();
            }
            catch (Exception ex)
            {
                // Log error and show notification to user
                DatabaseMonolith.Log("Error", "Unexpected error in ExecuteConfirmApiDialog", ex.ToString());
                SetNotification($"Unexpected error: {ex.Message}", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        private async void ExecuteRefreshPrice()
        {
            if (string.IsNullOrWhiteSpace(Order.Symbol))
            {
                SetNotification("Please enter a valid symbol", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }
            
            try
            {
                // Fetch current market price
                double marketPrice = await _orderService.GetMarketPrice(Order.Symbol);
                
                if (marketPrice > 0)
                {
                    Order.Price = marketPrice;
                    SetNotification($"Price updated for {Order.Symbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                }
                else
                {
                    SetNotification($"Could not retrieve price for {Order.Symbol}", 
                                   PackIconKind.AlertCircle, Colors.Orange);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to refresh price", ex.ToString());
                SetNotification($"Error refreshing price: {ex.Message}", 
                               PackIconKind.AlertCircle, Colors.Red);
            }
        }

        private void ExecutePaperTradeToggled(object param)
        {
            bool isPaperTrade = (bool)param;
            
            Order.IsPaperTrade = isPaperTrade;
            TradeModeBadgeText = isPaperTrade ? "Paper Trade" : "Real Trade";
            
            // Notify property change for computed properties
            OnPropertyChanged(nameof(TradeModeBadgeBackground));
            OnPropertyChanged(nameof(TradeModeBadgeForeground));
        }

        private void ExecuteClearHistory()
        {
            if (ShowConfirmationDialog == null)
            {
                // Fallback if dialog service not available
                OrderHistory.Clear();
                FilteredOrders.Clear();
                SetNotification("Order history cleared", PackIconKind.Delete, Colors.LightBlue);
                return;
            }

            MessageBoxResult result = ShowConfirmationDialog(
                "Are you sure you want to clear your order history?");
                
            if (result == MessageBoxResult.Yes)
            {
                OrderHistory.Clear();
                FilteredOrders.Clear();
                SetNotification("Order history cleared", PackIconKind.Delete, Colors.LightBlue);
            }
        }

        private void ExecuteFilterChanged(object param)
        {
            if (param is string filter)
            {
                SelectedFilter = filter;
            }
        }

        #endregion

        #region Helper Methods

        private void ApplyFilter()
        {
            // Ensure OrderHistory is not null
            if (OrderHistory == null)
            {
                OrderHistory = new ObservableCollection<OrderModel>();
            }
            
            IEnumerable<OrderModel> orders = OrderHistory;
            
            switch (SelectedFilter)
            {
                case "Paper Trades":
                    orders = OrderHistory.Where(o => o.IsPaperTrade);
                    break;
                case "Real Trades":
                    orders = OrderHistory.Where(o => !o.IsPaperTrade);
                    break;
                case "Buy Orders":
                    orders = OrderHistory.Where(o => o.OrderType.Equals("BUY", StringComparison.OrdinalIgnoreCase));
                    break;
                case "Sell Orders":
                    orders = OrderHistory.Where(o => o.OrderType.Equals("SELL", StringComparison.OrdinalIgnoreCase));
                    break;
                default:
                    orders = OrderHistory;
                    break;
            }
            
            // Check if orders is null before applying OrderByDescending
            if (orders != null)
            {
                FilteredOrders = new ObservableCollection<OrderModel>(orders.OrderByDescending(o => o.Timestamp));
            }
            else
            {
                // If orders is null, initialize FilteredOrders with an empty collection
                FilteredOrders = new ObservableCollection<OrderModel>();
            }
        }

        private void ResetOrderForm()
        {
            Order = _orderService.CreateDefaultOrder();
        }

        private void SetNotification(string message, PackIconKind iconKind, Color iconColor)
        {
            NotificationText = message;
            NotificationIcon = iconKind;
            NotificationIconColor = new SolidColorBrush(iconColor);
            NotificationBorderBrush = new SolidColorBrush(Color.FromArgb(100, iconColor.R, iconColor.G, iconColor.B));
        }

        #endregion
        
        #region Public Methods

        public void CreateRuleFromPrediction(PredictionModel prediction)
        {
            try
            {
                if (prediction == null)
                {
                    DatabaseMonolith.Log("Error", "Cannot create rule from null prediction");
                    SetNotification("Invalid prediction data", PackIconKind.AlertCircle, Colors.Red);
                    return;
                }

                // Initialize a new order based on the prediction
                Order = new OrderModel
                {
                    Symbol = prediction.Symbol,
                    OrderType = prediction.PredictedAction, // BUY or SELL
                    Price = prediction.CurrentPrice,
                    Quantity = 100, // Default quantity, can be adjusted based on confidence
                    IsPaperTrade = true, // Default to paper trading for safety
                    Status = "New",
                    Timestamp = DateTime.Now,
                    PredictionSource = $"Prediction {prediction.Symbol} ({prediction.Confidence:P0})",
                    StopLoss = prediction.PredictedAction == "BUY" ? 
                        prediction.CurrentPrice * 0.95 : // 5% stop loss for buy orders
                        prediction.CurrentPrice * 1.05,  // 5% stop loss for sell orders
                    TakeProfit = prediction.PredictedAction == "BUY" ? 
                        prediction.TargetPrice : // Use target price for take profit on buy orders
                        prediction.CurrentPrice * 0.9   // 10% take profit for sell orders
                };

                // Adjust quantity based on confidence if needed
                if (prediction.Confidence > 0.8)
                {
                    Order.Quantity = 150; // Higher quantity for high-confidence predictions
                }

                // Ensure OrderHistory is initialized
                if (OrderHistory == null)
                {
                    OrderHistory = new ObservableCollection<OrderModel>();
                }

                // Notify UI to update
                OnPropertyChanged(nameof(Order));

                // Show notification to user
                Color notificationColor = prediction.PredictedAction == "BUY" ? Colors.Green : Colors.OrangeRed;
                SetNotification($"Trading rule created for {prediction.Symbol} ({prediction.PredictedAction})", 
                    prediction.PredictedAction == "BUY" ? PackIconKind.TrendingUp : PackIconKind.TrendingDown, 
                    notificationColor);

                DatabaseMonolith.Log("Info", $"Created trading rule from prediction for {prediction.Symbol} with {prediction.PredictedAction} signal");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error creating rule from prediction", ex.ToString());
                SetNotification("Failed to create trading rule", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        /// <summary>
        /// Safely executes the place order operation with proper error handling.
        /// </summary>
        private async void PlaceOrderSafely()
        {
            try
            {
                await PlaceOrderAsync();
            }
            catch (Exception ex)
            {
                // Log error and show notification to user
                DatabaseMonolith.Log("Error", "Unexpected error in PlaceOrderSafely", ex.ToString());
                SetNotification($"Unexpected error: {ex.Message}", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        #endregion
    }
}
