using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using Quantra.Enums;
using MaterialDesignThemes.Wpf;
using Quantra.DAL.Services;
using Quantra.DAL.Notifications;

namespace Quantra
{
    public partial class MainWindow
    {
        #region UI Event Handlers

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            // Ensure lastNonPlusTab is set after the window is fully loaded
            if (lastNonPlusTab == null)
            {
                SetInitialLastNonPlusTab();
            }

            // Load controls for each tab
            foreach (var item in MainTabControl.Items.Cast<TabItem>())
            {
                if (item.Header.ToString() != "+")
                {
                    LoadControlsForTab(item);
                }
            }

            // Select the first tab by default
            if (MainTabControl.Items.Count > 0)
            {
                MainTabControl.SelectedIndex = 0;
            }

            // Log initialization
            //DatabaseMonolith.Log("Info", "Main window loaded successfully");
        }

        private void Window_Initialized(object sender, EventArgs e)
        {
            // Load controls for each existing tab (skip the '+' tab)
            foreach (var item in MainTabControl.Items.OfType<TabItem>())
            {
                if (item.Header.ToString() != "+")
                {
                    LoadTabControls(item.Header.ToString());
                }
            }
        }

        private void SetInitialLastNonPlusTab()
        {
            // Find the first non-'+' tab and set it as the lastNonPlusTab
            foreach (var item in MainTabControl.Items)
            {
                if (item is TabItem tabItem && tabItem.Header.ToString() != "+")
                {
                    lastNonPlusTab = tabItem;
                    break;
                }
            }
        }

        // Handler for notifications from the NotificationService
        private async void OnNotificationReceived(string message, NotificationIcon icon, string iconColorHex)
        {
            // Update dispatcher monitoring before making the call
            SharedTitleBar.UpdateDispatcherMonitoring("OnNotificationReceived_DispatcherCall");

            // Use the AppendAlert method to show notifications consistently
            string intent = "neutral";

            // Determine intent based on icon type and color
            switch (icon)
            {
                case NotificationIcon.Success:
                    intent = "positive";
                    break;
                case NotificationIcon.Error:
                    intent = "negative";
                    break;
                case NotificationIcon.Warning:
                    intent = "warning";
                    break;
                default:
                    // For other cases, check the color hex
                    if (iconColorHex?.Contains("00C853") == true || iconColorHex?.ToLower().Contains("green") == true)
                        intent = "positive";
                    else if (iconColorHex?.Contains("FF1744") == true || iconColorHex?.ToLower().Contains("red") == true)
                        intent = "negative";
                    else if (iconColorHex?.Contains("FFA000") == true || iconColorHex?.ToLower().Contains("orange") == true)
                        intent = "warning";
                    break;
            }

            // We need to ensure we're on the UI thread
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                AppendAlert(message, intent);
            });
        }

        #endregion

        #region Symbol-Related UI Events

        private void SymbolSearchBox_KeyUp(object sender, KeyEventArgs e)
        {
            var symbolSearchBox = sender as TextBox;

            // Dynamically find or initialize filteredResultsListBox
            var parent = VisualTreeHelper.GetParent(symbolSearchBox) as Panel;
            var filteredResultsListBox = parent?.Children.OfType<ListBox>().FirstOrDefault(lb => lb.Name == "FilteredResultsListBox");

            if (filteredResultsListBox == null)
            {
                MessageBox.Show("Error: FilteredResultsListBox is not initialized.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            IsSymbolSelected = !string.IsNullOrEmpty(symbolSearchBox.Text);
            string query = symbolSearchBox.Text.ToUpper().Trim();
            if (query.Length > 1)
            {
                IEnumerable<string> filtered;
                if (string.IsNullOrEmpty(query))
                {
                    filtered = availableStocks.OrderBy(s => s);
                }
                else
                {
                    filtered = availableStocks.Where(s => s.StartsWith(query, StringComparison.OrdinalIgnoreCase)).OrderBy(s => s);
                }
                filteredResultsListBox.ItemsSource = filtered.ToList();
                filteredResultsListBox.Visibility = Visibility.Visible;
            }
            else
            {
                filteredResultsListBox.Visibility = Visibility.Collapsed;
            }
        }

        private void FilteredResultsListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var filteredResultsListBox = sender as ListBox;
            var symbolSearchBox = FindName("SymbolSearchBox") as TextBox;
            IsSymbolSelected = filteredResultsListBox.SelectedItem != null;

            if (filteredResultsListBox.SelectedItem != null)
            {
                symbolSearchBox.Text = filteredResultsListBox.SelectedItem.ToString();
                filteredResultsListBox.Visibility = Visibility.Collapsed;
                SymbolSearchBox_SelectionChanged(sender, e);
            }
        }

        private async void SymbolSearchBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var symbolSearchBox = sender as TextBox;
            var selectedSymbolLabel = FindName("SelectedSymbolLabel") as TextBlock;
            var companySymbolLabel = FindName("CompanySymbolLabel") as TextBlock;
            debounceTimer.Stop();

            if (symbolSearchBox.Text == null)
                return;

            currentTicker = symbolSearchBox.Text;
            selectedSymbolLabel.Text = $"Selected: {currentTicker}";
            companySymbolLabel.Text = $"Company: {currentTicker}";
            AppendAlert($"Selected ticker: {currentTicker}");

            // Invoke the SymbolChanged event
            OnSymbolChanged(currentTicker);

            ShowModalDialog(async () => await RefreshData());
            debounceTimer.Start();
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            ShowModalDialog(async () => await RefreshData());
        }

        private async Task RefreshData()
        {
            var currentPriceLabel = FindName("CurrentPriceLabel") as TextBlock;
            var rsiLabel = FindName("RSILabel") as TextBlock;
            try
            {
                double price = await tradingBot.GetMarketPrice(currentTicker);
                currentPriceLabel.Text = $"Price: ${price:F2}";

                // Get RSI for a default timeframe (e.g., "5min")
                double rsi = await tradingBot.GetRSI(currentTicker, "5min");
                rsiLabel.Text = $"RSI: {rsi:F2}";
                AppendAlert($"RSI for {currentTicker}: {rsi:F2}");

                // Update chart with current price
                StockPriceValues.Clear();
                StockPriceValues.Add(price);
                StockPriceLineValues.Clear();
                StockPriceLineValues.Add(price);

                // Get Bollinger Bands
                var (upperBand, middleBand, lowerBand) = await tradingBot.GetBollingerBands(currentTicker, 20, 2);
                UpperBandValues.Clear();
                MiddleBandValues.Clear();
                LowerBandValues.Clear();
                UpperBandValues.AddRange(upperBand);
                MiddleBandValues.AddRange(middleBand);
                LowerBandValues.AddRange(lowerBand);

                // Update RSI values
                RSIValues.Clear();
                RSIValues.Add(rsi);
            }
            catch (Exception ex)
            {
                AppendAlert($"Error fetching data for {currentTicker}: {ex.Message}");
                //DatabaseMonolith.Log("Error", $"Error fetching data for {currentTicker}", ex.ToString());
            }
        }

        private async void DebounceTimer_Tick(object sender, EventArgs e)
        {
            var symbolSearchBox = FindName("SymbolSearchBox") as TextBox;
            var selectedSymbolLabel = FindName("SelectedSymbolLabel") as TextBlock;
            var companySymbolLabel = FindName("CompanySymbolLabel") as TextBlock;
            debounceTimer.Stop();

            if (symbolSearchBox.Text == null)
                return;

            currentTicker = symbolSearchBox.Text;
            selectedSymbolLabel.Text = $"Selected: {currentTicker}";
            companySymbolLabel.Text = $"Company: {currentTicker}";
            AppendAlert($"Selected ticker: {currentTicker}");

            try
            {
                double price = await tradingBot.GetMarketPrice(currentTicker);
                var currentPriceLabel = FindName("CurrentPriceLabel") as TextBlock;
                currentPriceLabel.Text = $"Price: ${price:F2}";

                // Get RSI for a default timeframe (e.g., "5min")
                double rsi = await tradingBot.GetRSI(currentTicker, "5min");
                var rsiLabel = FindName("RSILabel") as TextBlock;
                rsiLabel.Text = $"RSI: {rsi:F2}";
                AppendAlert($"RSI for {currentTicker}: {rsi:F2}");

                // Update chart with current price
                StockPriceValues.Clear();
                StockPriceValues.Add(price);
                StockPriceLineValues.Clear();
                StockPriceLineValues.Add(price);

                // Get Bollinger Bands
                var (upperBand, middleBand, lowerBand) = await tradingBot.GetBollingerBands(currentTicker, 20, 2);
                UpperBandValues.Clear();
                MiddleBandValues.Clear();
                LowerBandValues.Clear();
                UpperBandValues.AddRange(upperBand);
                MiddleBandValues.AddRange(middleBand);
                LowerBandValues.AddRange(lowerBand);

                // Update RSI values
                RSIValues.Clear();
                RSIValues.Add(rsi);
            }
            catch (Exception ex)
            {
                AppendAlert($"Error fetching data for {currentTicker}: {ex.Message}");
            }
            finally
            {
                debounceTimer.Start();
            }
        }

        private void AddSymbolButton_Click(object sender, RoutedEventArgs e)
        {
            string newSymbol = "SPOT"; // NewSymbolTextBox.Text.Trim();
            if (!string.IsNullOrEmpty(newSymbol) && IsSymbolUnique(newSymbol))
            {
                var tradingSymbol = new TradingSymbol
                {
                    Symbol = newSymbol,
                    CurrentPrice = "{133.70}", // Placeholder, replace with actual price fetching logic
                    DiffFromPositionAvg = "3.33%" // Placeholder, replace with actual calculation logic
                };
                ActiveSymbols.Add(tradingSymbol);
                tradingBot.AddSymbol(newSymbol);
            }
            else
            {
                MessageBox.Show("Symbol must be unique and not empty.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void RemoveSymbolButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.DataContext is TradingSymbol tradingSymbol)
            {
                ActiveSymbols.Remove(tradingSymbol);
                tradingBot.RemoveSymbol(tradingSymbol.Symbol);
            }
        }

        private bool IsSymbolUnique(string symbol)
        {
            foreach (var item in ActiveSymbols)
            {
                if (item.Symbol == symbol)
                {
                    return false;
                }
            }
            return true;
        }

        #endregion

        #region Trading Rules and Alerts

        private void AddRuleButton_Click(object sender, RoutedEventArgs e)
        {
            // Assuming you have a method to get the selected stock details
            var selectedStock = GetSelectedStock();
            if (selectedStock != null)
            {
                TradingRules.Add(new StockItem
                {
                    Symbol = selectedStock.Symbol,
                    CurrentPrice = selectedStock.CurrentPrice,
                    PercentageDiff = selectedStock.DiffFromPositionAvg
                });
            }
        }

        private void TradingRulesDataGrid_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            var tradingRulesDataGrid = sender as DataGrid;
            if (tradingRulesDataGrid.SelectedItem is StockItem selectedStock)
            {
                TradingRules.Remove(selectedStock);
                AppendAlert($"Removed stock: {selectedStock.Symbol}", "negative");
            }
        }

        private async void TradeUpdateTimer_Tick(object sender, EventArgs e)
        {
            var currentPriceLabel = FindName("CurrentPriceLabel") as TextBlock;
            var rsiLabel = FindName("RSILabel") as TextBlock;
            if (isTradingActive && !string.IsNullOrEmpty(currentTicker))
            {
                try
                {
                    double price = await tradingBot.GetMarketPrice(currentTicker);
                    double rsi = await tradingBot.GetRSI(currentTicker, "5min");
                    currentPriceLabel.Text = $"Price: ${price:F2}";
                    rsiLabel.Text = $"RSI: {rsi:F2}";

                    StockPriceValues.Add(price);
                    StockPriceLineValues.Add(price);
                    if (StockPriceValues.Count > 50)
                        StockPriceValues.RemoveAt(0);
                    if (StockPriceLineValues.Count > 50)
                        StockPriceLineValues.RemoveAt(0);

                    // Get Bollinger Bands
                    var (upperBand, middleBand, lowerBand) = await tradingBot.GetBollingerBands(currentTicker, 20, 2);
                    UpperBandValues.Clear();
                    MiddleBandValues.Clear();
                    LowerBandValues.Clear();
                    UpperBandValues.AddRange(upperBand);
                    MiddleBandValues.AddRange(middleBand);
                    LowerBandValues.AddRange(lowerBand);

                    // Update RSI values
                    RSIValues.Add(rsi);
                    if (RSIValues.Count > 50)
                        RSIValues.RemoveAt(0);
                }
                catch (Exception ex)
                {
                    AppendAlert($"Error updating data for {currentTicker}: {ex.Message}");
                    //DatabaseMonolith.Log("Error", $"Error updating data for {currentTicker}", ex.ToString());
                }
            }
        }

        public void AppendAlert(string message, string intent = "neutral")
        {
            var listBoxItem = new ListBoxItem
            {
                Content = $"{DateTime.Now}: {message}",
                Foreground = intent switch
                {
                    "positive" => Brushes.Green,
                    "negative" => Brushes.Red,
                    "warning" => Brushes.Orange,
                    _ => Brushes.White
                }
            };

            // Find and update the alerts listbox
            var alertsListBox = FindAlertListBox();
            if (alertsListBox != null)
            {
                alertsListBox.Items.Add(listBoxItem);
                alertsListBox.ScrollIntoView(listBoxItem);
            }
        }

        private ListBox FindAlertListBox()
        {
            // First look for it by name
            var alertsListBox = FindName("AlertsListBox") as ListBox;

            // If not found by name, try to find it in the visual tree
            if (alertsListBox == null)
            {
                alertsListBox = FindVisualChild<ListBox>(this, lb => lb.Name == "AlertsListBox");
            }

            return alertsListBox;
        }

        /// <summary>
        /// Appends an error to AlertsControl and logs it to the database.
        /// </summary>
        public void AppendErrorAlert(string message, Exception ex = null)
        {
            string fullMessage = ex == null ? message : $"{message}: {ex.Message}";
            AppendAlert(fullMessage, "negative");
            //DatabaseMonolith.Log("Error", message, ex?.ToString());
        }

        #endregion

        #region Trading Operations

        private async void LoginButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                bool isAuthenticated = await tradingBot.Authenticate("your_username", "your_password");
                AppendAlert(isAuthenticated ? "Authentication successful." : "Authentication unsuccessful.");
            }
            catch (Exception ex)
            {
                AppendAlert($"Error during authentication: {ex.Message}");
                //DatabaseMonolith.Log("Error", "Authentication error", ex.ToString());
            }
        }

        private async void TradeButton_Click(object sender, RoutedEventArgs e)
        {
            if (!isTradingActive)
            {
                AppendAlert("Starting trades...");
                await StartTrading();
            }
            else
            {
                AppendAlert("Trading already in progress.");
            }
        }

        private async Task StartTrading()
        {
            try
            {
                await tradingBot.ExecuteOptimalRiskTrading();
                isTradingActive = true;
                AppendAlert("Trading started successfully.");
                tradeUpdateTimer.Start();
            }
            catch (Exception ex)
            {
                AppendAlert($"Error starting trading: {ex.Message}");
            }
        }

        #endregion

        #region Trading Configuration UI

        private void InitializeTradingModeUI()
        {
            var tradingModeComboBox = FindName("TradingModeComboBox") as ComboBox;
            tradingModeComboBox.Items.Clear();
            tradingModeComboBox.ItemsSource = Enum.GetValues(typeof(TradingMode));
            tradingModeComboBox.SelectionChanged += TradingModeComboBox_SelectionChanged;
        }

        private void InitializeRiskModeUI()
        {
            var riskModeComboBox = FindName("RiskModeComboBox") as ComboBox;
            riskModeComboBox.Items.Clear();
            riskModeComboBox.ItemsSource = Enum.GetValues(typeof(RiskMode));
            riskModeComboBox.SelectionChanged += RiskModeComboBox_SelectionChanged;
        }

        private void TradingModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var tradingModeComboBox = sender as ComboBox;
            if (tradingModeComboBox.SelectedItem != null)
            {
                tradingBot.SetTradingMode((TradingMode)tradingModeComboBox.SelectedItem);
            }
        }

        private void RiskModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var riskModeComboBox = sender as ComboBox;
            if (riskModeComboBox.SelectedItem != null)
            {
                tradingBot.SetRiskMode((RiskMode)riskModeComboBox.SelectedItem);
            }
        }

        #endregion

        #region Modal Dialog Management

        private void ShowModalDialog(Action confirmAction)
        {
            if (EnableApiModalChecks)
            {
                confirmApiCallAction = confirmAction;
                ModalDialog.Visibility = Visibility.Visible;
            }
            else
            {
                confirmAction?.Invoke();
            }
        }

        private void ConfirmApiCall_Click(object sender, RoutedEventArgs e)
        {
            ModalDialog.Visibility = Visibility.Collapsed;
            confirmApiCallAction?.Invoke();
        }

        private void CancelApiCall_Click(object sender, RoutedEventArgs e)
        {
            ModalDialog.Visibility = Visibility.Collapsed;
            confirmApiCallAction = null;
        }

        public void AddControlButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedTab = MainTabControl.SelectedItem as TabItem;
            if (selectedTab != null)
            {
                // Use the singleton pattern to ensure only one instance
                var addControlWindow = AddControlWindow.GetInstance();

                // Make sure the window has the latest tab list
                addControlWindow.RefreshTabs();

                // Ensure we have the TabAdded event connected
                if (this.TabAdded != null)
                {
                    this.TabAdded -= addControlWindow.RefreshTabs;
                    this.TabAdded += addControlWindow.RefreshTabs;
                    //DatabaseMonolith.Log("Info", "TabAdded event connected to AddControlWindow in AddControlButton_Click");
                }

                addControlWindow.Show();
                addControlWindow.Activate();

                // Ensure we have the "Add Tool" button in the "+" tab
                if (selectedTab.Header.ToString() == "+")
                {
                    EnsureAddToolButtonForPlusTab();
                }
            }
        }

        private void EnsureAddToolButtonForPlusTab()
        {
            var plusTab = MainTabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == "+");
            if (plusTab != null)
            {
                // Create the grid and border for the "Add Tool" button
                Grid grid = new Grid();
                Border border = new Border();
                Button addToolButton = new Button
                {
                    Content = "Add Tool",
                    Width = 300, // Increased from 200 to 300
                    Height = 150, // Increased from 100 to 150
                    FontSize = 18, // Added larger font size
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    Style = FindResource("ButtonStyle1") as Style
                };
                addToolButton.Click += AddControlButton_Click;

                border.Child = addToolButton;
                grid.Children.Add(border);

                // Set as the content of the '+' tab
                plusTab.Content = grid;
            }
        }

        private Button CreateSettingsButton()
        {
            // Create settings button with gear icon
            Button settingsButton = new Button
            {
                Width = 24,
                Height = 24,
                Margin = new Thickness(5, 0, 5, 0),
                Background = Brushes.Transparent,
                BorderBrush = Brushes.Transparent
            };

            // Create settings gear icon
            Path gearIcon = new Path
            {
                Data = Geometry.Parse("M9.405 1.05c-.413-1.4-2.397-1.4-2.81 0l-.1.34a1.464 1.464 0 0 1-2.105.872l-.31-.17c-1.283-.698-2.686.705-1.987 1.987l.169.311c.446.82.023 1.841-.872 2.105l-.34.1c-1.4.413-1.4 2.397 0 2.81l.34.1a1.464 1.464 0 0 1 .872 2.105l-.17.31c-.698 1.283.705 2.686 1.987 1.987l.311-.169a1.464 1.464 0 0 1 2.105.872l.1.34c.413 1.4 2.397 1.4 2.81 0l.1-.34a1.464 1.464 0 0 1 2.105-.872l.31.17c1.283.698 2.686-.705 1.987-1.987l-.169-.311a1.464 1.464 0 0 1 .872-2.105l.34-.1c1.4-.413 1.4-2.397 0-2.81l-.34-.1a1.464 1.464 0 0 1-.872-2.105l.17-.31c.698-1.283-.705-2.686-1.987-1.987l-.311.169a1.464 1.464 0 0 1-2.105-.872l-.1-.34zM8 10.93a2.929 2.929 0 1 1 0-5.86 2.929 2.929 0 0 1 0 5.858z"),
                Fill = Brushes.White,
                Stretch = Stretch.Uniform,
                Width = 16,
                Height = 16
            };

            settingsButton.Content = gearIcon;
            settingsButton.Click += SettingsButton_Click;

            return settingsButton;
        }

        private void SettingsButton_Click(object sender, RoutedEventArgs e)
        {
            // Open settings window as a dialog
            var settingsWindow = new SettingsWindow();
            settingsWindow.Owner = this;

            // Use ShowDialog() instead of Show() to open as modal dialog
            bool? result = settingsWindow.ShowDialog();

            // Handle the result if needed
            if (result == true)
            {
                // Settings were saved, possibly refresh UI that depends on settings
                // This will execute after the SettingsWindow's SaveButton_Click method completes
            }
        }

        private TradingSymbol GetSelectedStock()
        {
            // Placeholder method to get the selected stock details
            // Replace with actual implementation
            return new TradingSymbol
            {
                Symbol = "AAPL",
                CurrentPrice = "150.00",
                DiffFromPositionAvg = "5%"
            };
        }

        #endregion

        #region Window Resize Functionality

        // ResizeGrip event handler for MainWindow.xaml.cs
        private void ResizeGrip_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left && e.ButtonState == MouseButtonState.Pressed)
            {
                // Get the parent window
                Window window = Window.GetWindow(this);

                // Determine which corner was clicked and set resize mode
                if (sender == BottomRightResizeGrip)
                {
                    // Bottom-right corner resize
                    window.ResizeMode = ResizeMode.CanResizeWithGrip;
                    NativeMethods.SendMessage(
                        new System.Windows.Interop.WindowInteropHelper(window).Handle,
                        NativeMethods.WM_SYSCOMMAND,
                        NativeMethods.SC_SIZE + NativeMethods.HTBOTTOMRIGHT,
                        0);
                }
                else if (sender == BottomLeftResizeGrip)
                {
                    // Bottom-left corner resize
                    window.ResizeMode = ResizeMode.CanResizeWithGrip;
                    NativeMethods.SendMessage(
                        new System.Windows.Interop.WindowInteropHelper(window).Handle,
                        NativeMethods.WM_SYSCOMMAND,
                        NativeMethods.SC_SIZE + NativeMethods.HTBOTTOMLEFT,
                        0);
                }
                else if (sender == TopRightResizeGrip)
                {
                    // Top-right corner resize
                    window.ResizeMode = ResizeMode.CanResizeWithGrip;
                    NativeMethods.SendMessage(
                        new System.Windows.Interop.WindowInteropHelper(window).Handle,
                        NativeMethods.WM_SYSCOMMAND,
                        NativeMethods.SC_SIZE + NativeMethods.HTTOPRIGHT,
                        0);
                }
                else if (sender == TopLeftResizeGrip)
                {
                    // Top-left corner resize
                    window.ResizeMode = ResizeMode.CanResizeWithGrip;
                    NativeMethods.SendMessage(
                        new System.Windows.Interop.WindowInteropHelper(window).Handle,
                        NativeMethods.WM_SYSCOMMAND,
                        NativeMethods.SC_SIZE + NativeMethods.HTTOPLEFT,
                        0);
                }

                e.Handled = true;
            }
        }

        // NativeMethods class needed for window resizing
        internal static class NativeMethods
        {
            public const int WM_SYSCOMMAND = 0x0112;
            public const int SC_SIZE = 0xF000;
            public const int HTBOTTOMRIGHT = 17;
            public const int HTBOTTOMLEFT = 16;
            public const int HTTOPRIGHT = 14;
            public const int HTTOPLEFT = 13;

            [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto)]
            public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, int wParam, int lParam);
        }

        // Reset cursor when resizing is complete
        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            base.OnMouseUp(e);
            this.Cursor = Cursors.Arrow;
        }

        #endregion


        // Modify the EnsureGridInitialized method to ensure grid lines are visible
        private void EnsureGridInitialized(TabItem tabItem, string tabName)
        {
            if (tabItem.Content is not Grid grid)
            {
                grid = new Grid();
                grid.Background = new SolidColorBrush(Color.FromArgb(20, 100, 100, 100)); // Slightly visible background

                // Default grid dimensions (4x4)
                int rows = 4;
                int columns = 4;

                // Add rows and columns
                for (int i = 0; i < rows; i++)
                {
                    grid.RowDefinitions.Add(new RowDefinition());
                }
                for (int j = 0; j < columns; j++)
                {
                    grid.ColumnDefinitions.Add(new ColumnDefinition());
                }

                tabItem.Content = grid;

                // Draw the grid lines to make them visible
                DrawGridLines(grid);

                // Log grid initialization
                //DatabaseMonolith.Log("Info", $"Initialized grid for tab '{tabName}' with default dimensions {rows}x{columns}");
            }
        }
    }
}
