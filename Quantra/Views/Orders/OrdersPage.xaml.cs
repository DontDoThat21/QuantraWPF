using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using MaterialDesignThemes.Wpf;
using System.Windows.Media.Animation;
using System.Windows.Documents;
using Quantra.Controls; // Add explicit reference to Quantra.Controls namespace
// Import the specific PredictionModel we want to use
using PredictionModel = Quantra.Models.PredictionModel;
using Quantra.Adapters;
using Microsoft.Extensions.DependencyInjection;
using Quantra.DAL.Services;

namespace Quantra.Views.Orders
{
    public partial class OrdersPage : Window, INotifyPropertyChanged
    {
        private WebullTradingBot tradingBot;
        private OrderModel order;
        private ObservableCollection<OrderModel> orderHistory;
        private ObservableCollection<OrderModel> filteredOrders;
        private string notificationText;
        private PackIconKind notificationIcon;
        private Brush notificationIconColor;
        private Brush notificationBorderBrush;
        private bool enableApiModalChecks;
        private string tradeModeBadgeText;
        private readonly ITransactionService _transactionService;
        private readonly INotificationService _notificationService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly UserSettingsService _userSettingsService;
        private readonly HistoricalDataService _historicalDataService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private readonly ITradingService _tradingService;

        // Public properties
        public OrderModel Order
        {
            get => order;
            set
            {
                order = value;
                OnPropertyChanged(nameof(Order));
            }
        }

        public ObservableCollection<OrderModel> OrderHistory
        {
            get => orderHistory;
            set
            {
                orderHistory = value;
                OnPropertyChanged(nameof(OrderHistory));
            }
        }

        public ObservableCollection<OrderModel> FilteredOrders
        {
            get => filteredOrders;
            set
            {
                filteredOrders = value;
                OnPropertyChanged(nameof(FilteredOrders));
            }
        }

        public string NotificationText
        {
            get => notificationText;
            set
            {
                notificationText = value;
                OnPropertyChanged(nameof(NotificationText));
            }
        }

        public PackIconKind NotificationIcon
        {
            get => notificationIcon;
            set
            {
                notificationIcon = value;
                OnPropertyChanged(nameof(NotificationIcon));
            }
        }

        public Brush NotificationIconColor
        {
            get => notificationIconColor;
            set
            {
                notificationIconColor = value;
                OnPropertyChanged(nameof(NotificationIconColor));
            }
        }

        public Brush NotificationBorderBrush
        {
            get => notificationBorderBrush;
            set
            {
                notificationBorderBrush = value;
                OnPropertyChanged(nameof(NotificationBorderBrush));
            }
        }

        // Computed properties for UI
        public string TradeModeBadgeText
        {
            get => tradeModeBadgeText ?? (Order.IsPaperTrade ? "Paper Trade" : "Real Trade");
            set
            {
                tradeModeBadgeText = value;
                OnPropertyChanged(nameof(TradeModeBadgeText));
            }
        }

        public Brush TradeModeBadgeBackground => Order.IsPaperTrade ? 
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#2D6A4C")) : 
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#6A2D2D"));
        public Brush TradeModeBadgeForeground => Order.IsPaperTrade ? 
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#50E070")) : 
            new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E05050"));

        public OrdersPage(UserSettingsService userSettingsService, HistoricalDataService historicalDataService, AlphaVantageService alphaVantageService, TechnicalIndicatorService technicalIndicatorService)
        {
            InitializeComponent();

            // Initialize services
            _userSettingsService = userSettingsService;
            _historicalDataService = historicalDataService;
            _alphaVantageService = alphaVantageService;
            _technicalIndicatorService = technicalIndicatorService;

            // Initialize data
            tradingBot = new WebullTradingBot(_userSettingsService,
                _historicalDataService,
                _alphaVantageService,
                _technicalIndicatorService);
            Order = new OrderModel
            {
                Symbol = "",
                Quantity = 100,
                Price = 0,
                OrderType = "BUY",
                IsPaperTrade = true,
                Status = "New",
                Timestamp = DateTime.Now,
                StopLoss = 0,
                TakeProfit = 0, PredictionSource = ""
            };
            
            OrderHistory = new ObservableCollection<OrderModel>();
            FilteredOrders = new ObservableCollection<OrderModel>(OrderHistory);
            
            // Set data context
            this.DataContext = this;
            
            // Load settings
            var settings = _userSettingsService.GetUserSettings();
            enableApiModalChecks = settings.EnableApiModalChecks;
            
            // Get services from DI container
            _transactionService = App.ServiceProvider.GetService<ITransactionService>();
            _notificationService = App.ServiceProvider.GetService<INotificationService>();
            _tradingService = App.ServiceProvider.GetService<ITradingService>();
        }

        public OrdersPage(string symbol, string orderType, double price, int quantity = 100, 
                          string predictionSource = null, double stopLoss = 0, double takeProfit = 0)
        {
            Order.Symbol = symbol;
            Order.OrderType = orderType;
            Order.Price = price;
            Order.Quantity = quantity;
            Order.PredictionSource = predictionSource;
            Order.StopLoss = stopLoss;
            Order.TakeProfit = takeProfit;
            LoadOrderHistory();
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void LoadOrderHistory()
        {
            // In a real app, this would load from database
            // For now, we'll just use some sample data
            OrderHistory.Clear();
            
            // Add some sample orders for demonstration
            //OrderHistory.Add(new OrderModel 
            //{ 
            //    Symbol = "AAPL", 
            //    OrderType = "BUY", 
            //    Quantity = 100, 
            //    Price = 182.50, 
            //    IsPaperTrade = true, 
            //    Status = "Executed", 
            //    Timestamp = DateTime.Now.AddDays(-3)
            //});
            
            //OrderHistory.Add(new OrderModel 
            //{ 
            //    Symbol = "MSFT", 
            //    OrderType = "SELL", 
            //    Quantity = 50, 
            //    Price = 326.75, 
            //    IsPaperTrade = false, 
            //    Status = "Failed", 
            //    Timestamp = DateTime.Now.AddDays(-2)
            //});
            
            //OrderHistory.Add(new OrderModel 
            //{ 
            //    Symbol = "TSLA", 
            //    OrderType = "BUY", 
            //    Quantity = 25, 
            //    Price = 215.30, 
            //    IsPaperTrade = true, 
            //    Status = "Executed", 
            //    Timestamp = DateTime.Now.AddDays(-1)
            //});
            
            ApplyFilter();
        }

        private void ApplyFilter()
        {
            var filter = FilterComboBox.SelectedItem as ComboBoxItem;
            if (filter == null) return;
            
            string filterText = filter.Content.ToString();
            
            // Ensure OrderHistory is not null
            if (OrderHistory == null)
            {
                OrderHistory = new ObservableCollection<OrderModel>();
            }
            
            IEnumerable<OrderModel> orders = OrderHistory;
            
            switch (filterText)
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
                // Add a default case for "All" or any other filter text
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

        private async void PlaceOrderButton_Click(object sender, RoutedEventArgs e)
        {
            // Validate input
            if (string.IsNullOrWhiteSpace(Order.Symbol))
            {
                ShowNotification("Please enter a valid symbol", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }
            
            if (Order.Quantity <= 0)
            {
                ShowNotification("Quantity must be greater than zero", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }
            
            if (Order.Price <= 0)
            {
                ShowNotification("Price must be greater than zero", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }

            // If API modal checks are enabled, show confirmation dialog
            if (enableApiModalChecks)
            {
                ApiConfirmationOverlay.Visibility = Visibility.Visible;
                return;
            }
            
            // Otherwise place the order directly
            await ExecutePlaceOrder();
        }

        private async Task ExecutePlaceOrder()
        {
            try
            {
                tradingBot.SetTradingMode(Order.IsPaperTrade ? Enums.TradingMode.Paper : Enums.TradingMode.Market);
                
                // Calculate a reasonable target price (this would typically be based on analysis)
                double targetPrice = Order.OrderType == "BUY" ? 
                    Order.Price * 1.05 : // 5% profit target for buy orders
                    Order.Price * 0.95;  // 5% profit target for sell orders
                
                // Use trading service for order execution
                bool success = false;
                
                if (_tradingService != null)
                {
                    // Use trading service which will handle notifications
                    success = await _tradingService.ExecuteTradeAsync(
                        Order.Symbol, 
                        Order.OrderType, 
                        Order.Price,
                        targetPrice
                    );
                }
                else
                {
                    // Fallback to direct execution
                    if (Order.StopLoss > 0 && Order.TakeProfit > 0)
                    {
                        // Use bracket order when stop loss and take profit are specified
                        success = await tradingBot.PlaceBracketOrder(
                            Order.Symbol,
                            Order.Quantity,
                            Order.OrderType,
                            Order.Price,
                            Order.StopLoss,
                            Order.TakeProfit
                        );
                    }
                    else
                    {
                        // Use regular limit order if no stop loss or take profit
                        await tradingBot.PlaceLimitOrder(
                            Order.Symbol,
                            Order.Quantity,
                            Order.OrderType,
                            Order.Price
                        );
                        success = true;
                    }
                }
                
                if (success)
                {
                    Order.Status = "Executed";
                    Order.Timestamp = DateTime.Now;
                    
                    var orderCopy = new OrderModel
                    {
                        Symbol = Order.Symbol,
                        Quantity = Order.Quantity,
                        Price = Order.Price,
                        OrderType = Order.OrderType,
                        IsPaperTrade = Order.IsPaperTrade,
                        Status = Order.Status,
                        Timestamp = Order.Timestamp,
                        PredictionSource = Order.PredictionSource
                    };
                    
                    OrderHistory.Insert(0, orderCopy);
                    ApplyFilter();
                    
                    // Save to transaction history
                    if (Order.IsPaperTrade)
                    {
                        var transaction = new TransactionModel
                        {
                            Symbol = Order.Symbol,
                            TransactionType = Order.OrderType,
                            Quantity = Order.Quantity,
                            ExecutionPrice = Order.Price,
                            ExecutionTime = Order.Timestamp,
                            IsPaperTrade = true,
                            Fees = 0.0,
                            RealizedPnL = 0.0,
                            RealizedPnLPercentage = 0.0,
                            TotalValue = Order.Price * Order.Quantity,
                            Notes = Order.PredictionSource,
                            Status = Order.Status,
                            OrderSource = string.IsNullOrEmpty(Order.PredictionSource) ? "Manual" : "Automated"
                        };
                        
                        _transactionService?.SaveTransaction(transaction);
                    }
                    
                    // Show in-app notification if not already shown by TradingService
                    if (_tradingService == null)
                    {
                        ShowNotification($"Order successfully placed for {Order.Symbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                    }
                    
                    ResetOrderForm();
                }
            }
            catch (Exception ex)
            {
                Order.Status = "Failed";
                Order.Timestamp = DateTime.Now;
                
                var orderCopy = new OrderModel
                {
                    Symbol = Order.Symbol,
                    Quantity = Order.Quantity,
                    Price = Order.Price,
                    OrderType = Order.OrderType,
                    IsPaperTrade = Order.IsPaperTrade,
                    Status = "Failed",
                    Timestamp = Order.Timestamp,
                    PredictionSource = Order.PredictionSource
                };
                
                OrderHistory.Insert(0, orderCopy);
                ApplyFilter();
                //DatabaseMonolith.Log("Error", "Failed to place order", ex.ToString());
                ShowNotification($"Error placing order: {ex.Message}", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        private void ResetOrderForm()
        {
            Order = new OrderModel
            {
                Symbol = "",
                Quantity = 100,
                Price = 0,
                OrderType = "BUY",
                IsPaperTrade = true,
                Status = "New",
                Timestamp = DateTime.Now,
                StopLoss = 0,
                TakeProfit = 0,
                PredictionSource = ""
            };
        }

        private async void RefreshPriceButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(Order.Symbol))
            {
                ShowNotification("Please enter a valid symbol", PackIconKind.AlertCircle, Colors.Orange);
                return;
            }
            
            try
            {
                // Fetch current market price
                double marketPrice = await tradingBot.GetMarketPrice(Order.Symbol);
                
                if (marketPrice > 0)
                {
                    Order.Price = marketPrice;
                    ShowNotification($"Price updated for {Order.Symbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                }
                else
                {
                    ShowNotification($"Could not retrieve price for {Order.Symbol}", PackIconKind.AlertCircle, Colors.Orange);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to refresh price", ex.ToString());
                ShowNotification($"Error refreshing price: {ex.Message}", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        private void FilterComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ApplyFilter();
        }

        private void PaperTradeToggle_Checked(object sender, RoutedEventArgs e)
        {
            Order.IsPaperTrade = true;
            TradeModeBadgeText = "Paper Trade";
            OnPropertyChanged(nameof(TradeModeBadgeBackground));
            OnPropertyChanged(nameof(TradeModeBadgeForeground));
        }

        private void PaperTradeToggle_Unchecked(object sender, RoutedEventArgs e)
        {
            Order.IsPaperTrade = false;
            TradeModeBadgeText = "Real Trade";
            OnPropertyChanged(nameof(TradeModeBadgeBackground));
            OnPropertyChanged(nameof(TradeModeBadgeForeground));
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        private void ClearHistoryButton_Click(object sender, RoutedEventArgs e)
        {
            MessageBoxResult result = MessageBox.Show(
                "Are you sure you want to clear your order history?", 
                "Confirm Clear History", 
                MessageBoxButton.YesNo, 
                MessageBoxImage.Question);
                
            if (result == MessageBoxResult.Yes)
            {
                OrderHistory.Clear();
                FilteredOrders.Clear();
                ShowNotification("Order history cleared", PackIconKind.Delete, Colors.LightBlue);
            }
        }

        private async void ConfirmApiButton_Click(object sender, RoutedEventArgs e)
        {
            // Check if the user doesn't want to see this dialog again
            if (DontShowAgainCheckBox.IsChecked == true)
            {
                var settings = _userSettingsService.GetUserSettings();
                settings.EnableApiModalChecks = false;
                _userSettingsService.SaveUserSettings(settings);
                enableApiModalChecks = false;
            }
            
            // Hide the confirmation dialog
            ApiConfirmationOverlay.Visibility = Visibility.Collapsed;
            
            // Execute the order
            await ExecutePlaceOrder();
        }

        private void CancelApiButton_Click(object sender, RoutedEventArgs e)
        {
            // Just hide the confirmation dialog
            ApiConfirmationOverlay.Visibility = Visibility.Collapsed;
        }

        private void ShowNotification(string message, PackIconKind iconKind, Color iconColor)
        {
            NotificationText = message;
            NotificationIcon = iconKind;
            NotificationIconColor = new SolidColorBrush(iconColor);
            NotificationBorderBrush = new SolidColorBrush(Color.FromArgb(100, iconColor.R, iconColor.G, iconColor.B));
            
            // Show the notification
            NotificationPanel.Visibility = Visibility.Visible;
            
            // Create fade-in animation
            DoubleAnimation fadeIn = new DoubleAnimation
            {
                From = 0,
                To = 1,
                Duration = TimeSpan.FromMilliseconds(300)
            };
            
            NotificationPanel.BeginAnimation(UIElement.OpacityProperty, fadeIn);
            
            // Schedule fade-out
            var timer = new System.Windows.Threading.DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(4)
            };
            
            timer.Tick += (s, e) =>
            {
                timer.Stop();
                
                // Create fade-out animation
                DoubleAnimation fadeOut = new DoubleAnimation
                {
                    From = 1,
                    To = 0,
                    Duration = TimeSpan.FromMilliseconds(300)
                };
                
                fadeOut.Completed += (s2, e2) => NotificationPanel.Visibility = Visibility.Collapsed;
                NotificationPanel.BeginAnimation(UIElement.OpacityProperty, fadeOut);
            };
            
            timer.Start();
        }

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Creates a new trading rule based on a prediction model.
        /// </summary>
        /// <param name="prediction">The prediction model containing trading signal information</param>
        public void CreateRuleFromPrediction(Quantra.Models.PredictionModel prediction)
        {
            try
            {
                if (prediction == null)
                {
                    //DatabaseMonolith.Log("Error", "Cannot create rule from null prediction");
                    ShowNotification("Invalid prediction data", PackIconKind.AlertCircle, Colors.Red);
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

                // Add to the OrderHistory (optional, since it's not executed yet)
                // OrderHistory.Add(new OrderModel(Order)); // If you want to track the suggestion

                // Notify UI to update
                OnPropertyChanged(nameof(Order));

                // Show notification to user
                Color notificationColor = prediction.PredictedAction == "BUY" ? Colors.Green : Colors.OrangeRed;
                ShowNotification($"Trading rule created for {prediction.Symbol} ({prediction.PredictedAction})", 
                    prediction.PredictedAction == "BUY" ? PackIconKind.TrendingUp : PackIconKind.TrendingDown, 
                    notificationColor);

                //DatabaseMonolith.Log("Info", $"Created trading rule from prediction for {prediction.Symbol} with {prediction.PredictedAction} signal");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error creating rule from prediction", ex.ToString());
                ShowNotification("Failed to create trading rule", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        // Add this new method to create a rule dialog
        public void ShowRuleFromPredictionDialog(Quantra.Models.PredictionModel prediction, Action<string> onRuleUpdated)
        {
            try
            {
                if (prediction == null)
                {
                    //DatabaseMonolith.Log("Error", "Cannot create rule from null prediction");
                    ShowNotification("Invalid prediction data", PackIconKind.AlertCircle, Colors.Red);
                    return;
                }

                // Use the extension method to get the existing trading rule
                string existingRule = PredictionModelAdapter.GetExistingTradingRule(prediction);

                // Show the trading rule editor with the existing rule
                ShowTradingRuleEditor(prediction, false, onRuleUpdated);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error showing rule from prediction dialog", ex.ToString());
                ShowNotification("Failed to open trading rule dialog", PackIconKind.AlertCircle, Colors.Red);
            }
        }

        // Add this method to OrdersPage class
        public void ShowTradingRuleEditor(Quantra.Models.PredictionModel prediction, Action<string> onRuleUpdated)
        {
            ShowTradingRuleEditor(prediction, false, onRuleUpdated);
        }

        // Add an overload with embeddedMode parameter
        public void ShowTradingRuleEditor(Quantra.Models.PredictionModel prediction, bool embeddedMode, Action<string> onRuleUpdated)
        {
            try
            {
                if (prediction == null)
                {
                    //DatabaseMonolith.Log("Error", "Cannot edit trading rule for null prediction");
                    ShowNotification("Invalid prediction data", PackIconKind.AlertCircle, Colors.Red);
                    return;
                }

                // Create a Grid to hold both the Orders form and the Trading Rule editor
                var originalContent = this.Content as UIElement;
                var containerGrid = new Grid();
                containerGrid.Children.Add(originalContent);

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

                // Adjust quantity based on confidence
                if (prediction.Confidence > 0.8)
                {
                    Order.Quantity = 150; // Higher quantity for high-confidence predictions
                }

                // Create Trading Rule editor panel
                var rulePanel = new Border
                {
                    Background = (Brush)FindResource("DarkBlueAccentBrush"),
                    BorderBrush = new SolidColorBrush(Colors.Cyan),
                    BorderThickness = new Thickness(1),
                    CornerRadius = new CornerRadius(6),
                    Padding = new Thickness(15),
                    Width = 500,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Top,
                    Margin = new Thickness(0, 50, 0, 0)
                };

                var ruleGrid = new Grid();
                ruleGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
                ruleGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
                ruleGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

                // Title
                var titleText = new TextBlock
                {
                    Text = $"Trading Rule for {prediction.Symbol}",
                    FontSize = 18,
                    FontWeight = FontWeights.Bold,
                    Foreground = new SolidColorBrush(Colors.White),
                    Margin = new Thickness(0, 0, 0, 15)
                };
                Grid.SetRow(titleText, 0);
                ruleGrid.Children.Add(titleText);

                // Rule input
                var ruleInput = new TextBox
                {
                    Text = prediction.TradingRule ?? string.Empty,
                    FontSize = 14,
                    Foreground = new SolidColorBrush(Colors.Black),
                    Background = new SolidColorBrush(Colors.White),
                    Padding = new Thickness(8),
                    Height = 100,
                    TextWrapping = TextWrapping.Wrap,
                    AcceptsReturn = true,
                    Margin = new Thickness(0, 0, 0, 15),
                    VerticalContentAlignment = VerticalAlignment.Top
                };
                Grid.SetRow(ruleInput, 1);
                ruleGrid.Children.Add(ruleInput);

                // Buttons
                var buttonPanel = new StackPanel
                {
                    Orientation = Orientation.Horizontal,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Margin = new Thickness(0, 5, 0, 0)
                };

                var cancelButton = new Button
                {
                    Content = "Cancel",
                    Width = 100,
                    Height = 30,
                    Margin = new Thickness(0, 0, 10, 0),
                    Style = (Style)FindResource("ButtonStyle1"),
                    Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#3E3E56")),
                    Foreground = new SolidColorBrush(Colors.White),
                    BorderBrush = (Brush)FindResource("AccentBlueBrush")
                };

                var saveButton = new Button
                {
                    Content = "Save Rule",
                    Width = 100,
                    Height = 30,
                    Style = (Style)FindResource("ButtonStyle1"),
                    Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#3E90FF")),
                    Foreground = new SolidColorBrush(Colors.White),
                    BorderThickness = new Thickness(0)
                };

                buttonPanel.Children.Add(cancelButton);
                buttonPanel.Children.Add(saveButton);
                Grid.SetRow(buttonPanel, 2);
                ruleGrid.Children.Add(buttonPanel);

                rulePanel.Child = ruleGrid;
                containerGrid.Children.Add(rulePanel);

                // Set up button click handlers
                cancelButton.Click += (s, e) => {
                    // Restore the original content
                    this.Content = originalContent;
                    
                    // Call the callback with empty string to indicate cancellation
                    onRuleUpdated.Invoke(prediction.TradingRule ?? string.Empty);
                };

                saveButton.Click += (s, e) => {
                    // Save the rule and invoke the callback
                    string updatedRule = ruleInput.Text?.Trim() ?? "";
                    onRuleUpdated.Invoke(updatedRule);
                    
                    // Restore the original content
                    this.Content = originalContent;
                    
                    // Show notification
                    ShowNotification($"Trading rule updated for {prediction.Symbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                };

                // Replace the content
                this.Content = containerGrid;

                // If in embedded mode, adjust window placement and size
                if (embeddedMode)
                {
                    // Position the window to appear next to the prediction grid
                    if (this.Owner != null)
                    {
                        // Calculate position based on owner's position
                        var ownerLeft = this.Owner.Left;
                        var ownerWidth = this.Owner.Width;
                        
                        // Position this window to the right of the owner
                        this.Left = ownerLeft + (ownerWidth * 0.5) + 10; // Position next to the shrunk grid
                        this.Top = this.Owner.Top + 50; // Add some vertical offset
                        
                        // Make the window size smaller to fit nicely
                        this.Width = System.Math.Min(600, this.Width); // Max width of 600
                        this.Height = System.Math.Min(550, this.Height); // Max height of 550
                    }
                    
                    // Make window appear with animation
                    this.Opacity = 0;
                    this.Visibility = Visibility.Visible;
                    
                    var fadeIn = new DoubleAnimation
                    {
                        From = 0,
                        To = 1,
                        Duration = TimeSpan.FromMilliseconds(300)
                    };
                    this.BeginAnimation(UIElement.OpacityProperty, fadeIn);
                }

                // Show notification
                ShowNotification($"Editing trading rule for {prediction.Symbol}", PackIconKind.Pencil, Colors.LightBlue);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error editing trading rule", ex.ToString());
                ShowNotification("Failed to open trading rule editor", PackIconKind.AlertCircle, Colors.Red);
            }
        }
    }
}
