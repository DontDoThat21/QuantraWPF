using Dapper;
using Quantra.Controls;
using Quantra.Repositories;
using Quantra.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Documents;
using System.Windows.Shapes; // For AdornerLayer

namespace Quantra
{
    public partial class MainWindow
    {
        #region Tab Management Fields

        // Tab management private fields
        private TabItem lastNonPlusTab;
        // Flag to track recursive tab selection operations
        private bool isTabSelectionInProgress = false;
        private TabRepository _tabRepository;

        #endregion

        #region Constructor

        private void InitializeTabManagement()
        {
            var connection = DatabaseMonolith.GetConnection();
            _tabRepository = new TabRepository(connection);
            
            // Initialize TabManager but avoid the complex PreviewMouseMove hookup
            // Keep the existing TabManager approach for compatibility
            try 
            {
                TabManager = new Utilities.TabManager(this, MainTabControl);
                
                // Hook up TabManager events
                TabManager.TabAdded += (tabName) => {
                    // Raise the MainWindow's TabAdded event so AddControlWindow can be notified
                    TabAdded?.Invoke(tabName);
                    DatabaseMonolith.Log("Info", $"MainWindow raised TabAdded event for tab: {tabName}");
                };
            }
            catch
            {
                // Fallback if TabManager constructor fails
                TabManager = null;
            }
            
            // Wire up events directly
            MainTabControl.SelectionChanged += MainTabControl_SelectionChanged;
            MainTabControl.Drop += MainTabControl_Drop;
            MainTabControl.PreviewMouseMove += MainTabControl_PreviewMouseMove;
            
            // Ensure TabControl supports drag-over and drop
            MainTabControl.AllowDrop = true;
            MainTabControl.DragOver += MainTabControl_DragOver;
        }

        #endregion

        #region Tab Management Functions

        public delegate void TabAddedEventHandler(string tabName);
        public event TabAddedEventHandler TabAdded;

        private void LoadCustomTabs()
        {
            // Delegate to TabManager
            TabManager.LoadCustomTabs();
        }

        private void AddCustomTab(string tabName)
        {
            // Delegate to TabManager
            TabManager.AddCustomTab(tabName);
        }

        private void RemoveCustomTab(TabItem tabItem, string tabName)
        {
            // Delegate to TabManager
            TabManager.RemoveCustomTab(tabItem, tabName);
            AppendAlert($"Removed tab: {tabName}");
        }

        private void EditCustomTab(TabItem tabItem, string oldTabName)
        {
            // Delegate to TabManager
            TabManager.EditCustomTab(tabItem, oldTabName);
            AppendAlert($"Renamed tab from {oldTabName} to {tabItem.Header.ToString()}");
        }

        private void AddNewTabButton_Click(object sender, MouseButtonEventArgs e)
        {
            // Create and show the new tab creation dialog
            var createTabWindow = new CreateTabWindow();
            bool? result = createTabWindow.ShowDialog();

            // If the user clicked Create and provided a valid tab name
            if (result == true)
            {
                string newTabName = createTabWindow.NewTabName;
                int gridRows = createTabWindow.GridRows;
                int gridColumns = createTabWindow.GridColumns;

                // Add the tab to the UI
                TabManager.AddCustomTab(newTabName);

                // Save the tab to the database with specified grid dimensions
                TabManager.SaveCustomTabWithGrid(newTabName, gridRows, gridColumns);

                // Select the newly created tab in the MainTabControl
                try
                {
                    var newTabItem = MainTabControl.Items.OfType<TabItem>()
                        .FirstOrDefault(t => string.Equals(t.Header?.ToString(), newTabName, StringComparison.Ordinal));
                    if (newTabItem != null)
                    {
                        MainTabControl.SelectedItem = newTabItem;
                    }
                }
                catch (Exception selEx)
                {
                    DatabaseMonolith.Log("Warning", $"Failed to select newly created tab '{newTabName}'", selEx.ToString());
                }

                // If AddControlWindow is open, refresh and select the new tab there as well
                try
                {
                    var addControlWnd = Application.Current.Windows.OfType<AddControlWindow>().FirstOrDefault();
                    if (addControlWnd != null && addControlWnd.IsLoaded)
                    {
                        addControlWnd.RefreshTabs(newTabName);
                    }
                }
                catch (Exception acwEx)
                {
                    DatabaseMonolith.Log("Warning", "Failed to refresh AddControlWindow after creating tab", acwEx.ToString());
                }

                // Log the creation
                DatabaseMonolith.Log("Info", $"Created new tab: {newTabName} with grid dimensions {gridRows}x{gridColumns}");

                // Show success message
                AppendAlert($"Created new tab: {newTabName}", "positive");
            }
        }

        private void SaveCustomTabWithGrid(string tabName, int rows, int columns)
        {
            // Delegate to TabManager
            TabManager.SaveCustomTabWithGrid(tabName, rows, columns);
       }

// Removed commented-out MainTabControl_PreviewMouseMove method to clean up dead code.

        private void MainTabControl_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetData(typeof(TabItem)) is TabItem tabItem)
            {
                var tabControl = sender as TabControl;

                // Disallow drops from other TabControls to avoid reparenting exceptions
                var owner = ItemsControl.ItemsControlFromItemContainer(tabItem) as TabControl;
                if (!ReferenceEquals(owner, tabControl))
                {
                    e.Effects = DragDropEffects.None;
                    e.Handled = true;
                    return;
                }

                var point = e.GetPosition(tabControl);

                // Hit test to find visual under pointer, then walk up to TabItem
                var hit = tabControl.InputHitTest(point) as DependencyObject;
                var targetItem = VisualTreeUtility.FindAncestor<TabItem>(hit);

                // Prevent moving the + tab or moving any tab to the + tab position
                if (tabItem.Header.ToString() == "+" || (targetItem?.Header?.ToString() == "+"))
                {
                    return;
                }

                // Determine indices
                int oldIndex = tabControl.Items.IndexOf(tabItem);
                int newIndex = targetItem != null ? tabControl.Items.IndexOf(targetItem) : -1;

                // Find the + tab index to ensure we don't move past it
                var plusTab = tabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == "+");
                int plusTabIndex = plusTab != null ? tabControl.Items.IndexOf(plusTab) : tabControl.Items.Count;

                // If dropping on empty area, default to inserting before '+'
                if (newIndex == -1)
                {
                    newIndex = Math.Max(0, plusTabIndex - 1);
                }

                // Ensure new index doesn't place tab after the + tab
                if (newIndex >= plusTabIndex)
                {
                    newIndex = plusTabIndex - 1;
                }

                // No-op if indices invalid or same
                if (newIndex < 0 || newIndex == oldIndex)
                {
                    return;
                }

                // Reorder items
                tabControl.Items.Remove(tabItem);
                tabControl.Items.Insert(newIndex, tabItem);
                tabControl.SelectedItem = tabItem;

                // Persist new tab order (skip '+')
                int order = 0;
                foreach (var item in MainTabControl.Items.OfType<TabItem>())
                {
                    if (item.Header is string name && name != "+")
                    {
                        _tabRepository.UpdateTabOrder(name, order);
                        order++;
                    }
                }
            }
        }

        private void MainTabControl_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                var tabItem = VisualTreeUtility.FindAncestor<TabItem>((DependencyObject)e.OriginalSource);
                if (tabItem != null && tabItem.Header.ToString() != "+")
                {
                    // Only allow dragging non-plus tabs
                    DragDrop.DoDragDrop(tabItem, tabItem, DragDropEffects.Move);
                }
            }
        }

        private void MainTabControl_DragOver(object sender, DragEventArgs e)
        {
            if (sender is not TabControl tabControl)
            {
                return;
            }

            // Only handle TabItem drags
            if (!e.Data.GetDataPresent(typeof(TabItem)))
            {
                e.Effects = DragDropEffects.None;
                e.Handled = true;
                return;
            }

            var dragging = e.Data.GetData(typeof(TabItem)) as TabItem;

            // Disallow drags from other TabControls to avoid reparenting exceptions
            var owner = ItemsControl.ItemsControlFromItemContainer(dragging) as TabControl;
            if (!ReferenceEquals(owner, tabControl))
            {
                e.Effects = DragDropEffects.None;
                e.Handled = true;
                return;
            }

            if (dragging?.Header?.ToString() == "+")
            {
                e.Effects = DragDropEffects.None;
                e.Handled = true;
                return;
            }

            var point = e.GetPosition(tabControl);
            var hit = tabControl.InputHitTest(point) as DependencyObject;
            var targetItem = VisualTreeUtility.FindAncestor<TabItem>(hit);

            // If over '+', disallow, but still allow drop before it by dropping on empty area
            if (targetItem?.Header?.ToString() == "+")
            {
                e.Effects = DragDropEffects.None;
            }
            else
            {
                e.Effects = DragDropEffects.Move;
            }

            e.Handled = true;
        }



        private void MainTabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Guard against recursive tab selection operations
            if (isTabSelectionInProgress)
            {
                DatabaseMonolith.Log("Info", "MainWindow: Prevented recursive tab selection");
                return;
            }

            // Ignore SelectionChanged events that are not actual tab changes
            // (e.g., event was raised due to a child control interaction)
            // Only proceed if the added/removed items are TabItems
            bool isTabChange = false;
            foreach (var item in e.AddedItems)
            {
                if (item is TabItem)
                {
                    isTabChange = true;
                    break;
                }
            }
            foreach (var item in e.RemovedItems)
            {
                if (item is TabItem)
                {
                    isTabChange = true;
                    break;
                }
            }
            if (!isTabChange)
            {
                // Not a real tab change, ignore this event
                return;
            }

            try
            {
                // Set the flag to indicate we're processing a tab selection
                isTabSelectionInProgress = true;
                
                var selectedTab = MainTabControl.SelectedItem as TabItem;
                if (selectedTab != null)
                {
                    if (selectedTab.Header.ToString() == "+")
                    {
                        // When "+" tab is selected, ensure it has the "Add Tool" button
                        // This fixes the issue where the button disappears after adding a tool
                        var grid = new Grid();
                        var border = new Border();
                        var addToolButton = new Button
                        {
                            Content = "Add Tool",
                            Width = 200,
                            Height = 100,
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            Style = FindResource("ButtonStyle1") as Style
                        };
                        addToolButton.Click += AddControlButton_Click;
                        border.Child = addToolButton;
                        grid.Children.Add(border);
                        selectedTab.Content = grid;
                    }
                    else
                    {
                        // For other tabs, update the last non-'+' tab and load controls
                        lastNonPlusTab = selectedTab;
                        
                        // Load tab controls directly without complex dispatcher logic
                        try
                        {
                            if (TabManager != null)
                            {
                                // Use TabManager to load tab controls
                                TabManager.LoadTabControls(selectedTab.Header.ToString());
                            }
                            else
                            {
                                LoadTabControls(selectedTab.Header.ToString());
                            }
                        }
                        catch (Exception ex)
                        {
                            DatabaseMonolith.Log("Error", "Error in LoadTabControls during tab selection", ex.ToString());
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error in MainTabControl_SelectionChanged", ex.ToString());
            }
            finally
            {
                // Always reset the flag when done
                isTabSelectionInProgress = false;
            }
        }

        // Add this method to load tab controls using our local implementation
        // until all references are updated to use the TabManager
        private void LoadTabControls(string tabName)
        {
            var tabItem = MainTabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == tabName);
            if (tabItem == null)
            {
                DatabaseMonolith.Log("Warning", $"Tab '{tabName}' not found when loading controls");
                return;
            }

            EnsureGridInitialized(tabItem, tabName);

            if (tabItem != null)
            {
                LoadControlsForTab(tabItem);
            }
        }

        #endregion

        #region Control Management Functions

        private void LoadCardPositions()
        {
            var cardPositions = DatabaseMonolith.LoadCardPositions();
            if (!string.IsNullOrEmpty(cardPositions))
            {
                // Uncomment when implementing card position persistence
                // var layoutControl = (Dragablz.LayoutControl)FindName("LayoutControl");
                // layoutControl.RestoreLayout(cardPositions);
            }
        }

        private void SaveCardPositions()
        {
            var cardPositions = ""; // Serialize card positions here
            DatabaseMonolith.SaveCardPositions(cardPositions);
        }

        // Update the LoadControlsForTab method to ensure grid lines remain visible
        public void LoadControlsForTab(TabItem tabItem)
        {
            try
            {
                if (tabItem == null || !(tabItem.Header is string tabName))
                    return;

                var controlsConfig = DatabaseMonolith.LoadControlsConfig(tabName);
                var gridConfig = DatabaseMonolith.LoadGridConfig(tabName);

                // Always ensure we have at least 4x4 grid dimensions
                int rows = Math.Max(4, gridConfig.Rows);
                int columns = Math.Max(4, gridConfig.Columns);

                // Create a grid for the tab content
                var grid = new Grid();
                grid.Background = new SolidColorBrush(Color.FromArgb(20, 100, 100, 100)); // Slightly visible background

                // Set up the grid rows and columns
                for (int i = 0; i < rows; i++)
                {
                    grid.RowDefinitions.Add(new RowDefinition());
                }
                for (int j = 0; j < columns; j++)
                {
                    grid.ColumnDefinitions.Add(new ColumnDefinition());
                }

                // Add the grid to the tab
                tabItem.Content = grid;

                // Draw grid lines before adding controls
                DrawGridLines(grid);

                if (!string.IsNullOrEmpty(controlsConfig))
                {
                    var controls = DeserializeControls(controlsConfig);

                    // Only create UI if we have controls to display
                    if (controls.Count > 0)
                    {
                        // Add each control to the grid with appropriate spans
                        foreach (var control in controls)
                        {
                            // Make sure the position is valid for our grid dimensions
                            int row = Math.Min(control.Item2, rows - 1);
                            int col = Math.Min(control.Item3, columns - 1);
                            int rowSpan = Math.Min(control.Item4, rows - row);  // Ensure rowSpan doesn't go beyond grid
                            int colSpan = Math.Min(control.Item5, columns - col);  // Ensure colSpan doesn't go beyond grid

                            // Create a border for the control with context menu for movement
                            var borderedControl = CreateDraggableBorder(control.Item1, row, col, rowSpan, colSpan, tabName);

                            // Set grid placement
                            Grid.SetRow(borderedControl, row);
                            Grid.SetColumn(borderedControl, col);
                            Grid.SetRowSpan(borderedControl, rowSpan);
                            Grid.SetColumnSpan(borderedControl, colSpan);

                            grid.Children.Add(borderedControl);
                        }

                        // Log success for debugging
                        DatabaseMonolith.Log("Info", $"Successfully loaded {controls.Count} controls for tab {tabName}");
                    }
                }

                // Make sure the grid is draggable, even if there are no controls
                MakeGridDraggable(grid, tabName);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error loading controls for tab: {tabItem?.Header}", ex.ToString());
            }
        }

        // Modify the DeserializeControls method
        public List<(UIElement Control, int Row, int Column, int RowSpan, int ColumnSpan)> DeserializeControls(string controlsConfig)
        {
            var controls = new List<(UIElement, int, int, int, int)>();

            if (string.IsNullOrWhiteSpace(controlsConfig))
                return controls;

            // Handle both semicolons and newlines as potential separators
            var controlConfigs = controlsConfig
                .Replace("\r\n", ";")
                .Replace("\n", ";")
                .Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);

            foreach (var controlConfig in controlConfigs)
            {
                try
                {
                    var parts = controlConfig.Split(',');
                    if (parts.Length < 3)
                    {
                        DatabaseMonolith.Log("Error", $"Invalid control configuration format: {controlConfig}");
                        continue; // Skip invalid control definitions
                    }

                    var controlType = parts[0].Trim();
                    DatabaseMonolith.Log("Info", $"Deserializing control of type: {controlType}");

                    // Improved parsing with error handling for row and column
                    if (!int.TryParse(parts[1], out int row) || !int.TryParse(parts[2], out int column))
                    {
                        DatabaseMonolith.Log("Error", $"Invalid row or column value in control config: {controlConfig}");
                        continue; // Skip controls with invalid positions
                    }

                    // Parse span values if they exist, otherwise use default of 1
                    int rowSpan = 1;
                    int columnSpan = 1;

                    if (parts.Length >= 4 && int.TryParse(parts[3], out int parsedRowSpan))
                    {
                        rowSpan = parsedRowSpan;
                    }

                    if (parts.Length >= 5 && int.TryParse(parts[4], out int parsedColumnSpan))
                    {
                        columnSpan = parsedColumnSpan;
                    }

                    UIElement control;

                    try
                    {
                        control = controlType switch
                        {
                            "Symbol Charts" => CreateSymbolChartsCard(),
                            "Trading Rules" => CreateTradingRulesCard(),
                            "Alerts" => CreateAlertsCard(),
                            "Configuration" => CreateConfigurationCard(),
                            "Prediction Analysis" => CreatePredictionAnalysisCard(),
                            "Sector Momentum Heatmap" => CreateSectorMomentumHeatmapCard(),
                            "Transactions" => CreateTransactionsCard(),
                            "Backtesting" => CreateBacktestingCard(),
                            "Backtest Chart" => CreateBacktestingCard(), // Accept alias
                            "Market Chat" => CreateMarketChatCard(),
                            "Spreads Explorer" => CreateSpreadsExplorerCard(),
                            _ => throw new NotSupportedException($"Control type '{controlType}' is not supported.")
                        };
                    }
                    catch (Exception ex)
                    {
                        DatabaseMonolith.Log("Error", $"Failed to create control of type '{controlType}'", ex.ToString());

                        // Create a fallback control to show the error
                        var errorPanel = new StackPanel();
                        errorPanel.Children.Add(new TextBlock
                        {
                            Text = $"Error loading '{controlType}'",
                            Foreground = Brushes.Red,
                            FontWeight = FontWeights.Bold,
                            TextWrapping = TextWrapping.Wrap,
                            Margin = new Thickness(10)
                        });

                        control = errorPanel;
                    }

                    controls.Add((control, row, column, rowSpan, columnSpan));
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error deserializing control: {controlConfig}", ex.ToString());
                }
            }

            return controls;
        }

        private UIElement CreateSymbolChartsCard()
        {
            try
            {
                // Create a new instance of our custom StockExplorer
                var symbolChartControl = new StockExplorer();

                // Ensure the control has proper sizing and stretching behavior
                symbolChartControl.Width = double.NaN; // Auto width
                symbolChartControl.Height = double.NaN; // Auto height
                symbolChartControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                symbolChartControl.VerticalAlignment = VerticalAlignment.Stretch;
                symbolChartControl.MinWidth = 900;
                symbolChartControl.MinHeight = 700;

                // Force initial layout calculation to ensure IndicatorCharts is properly initialized
                symbolChartControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                symbolChartControl.Arrange(new Rect(0, 0, symbolChartControl.DesiredSize.Width, symbolChartControl.DesiredSize.Height));

                // Log success
                DatabaseMonolith.Log("Info", "Successfully created StockExplorer control for custom tab");

                // Return the control
                return symbolChartControl;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error creating StockExplorer control", ex.ToString());
                
                // Return a fallback UI element
                var errorText = new TextBlock 
                { 
                    Text = "Error loading Stock Explorer", 
                    Foreground = Brushes.Red, 
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                };
                return errorText;
            }
        }

        private UIElement CreateTradingRulesCard()
        {
            // Create and return the UIElement for Trading Rules Card
            var stackPanel = new StackPanel();
            var tradingRulesDataGrid = new DataGrid { Name = "TradingRulesDataGrid", Background = new SolidColorBrush(Color.FromRgb(45, 45, 68)), Foreground = Brushes.Black, BorderBrush = Brushes.Blue, BorderThickness = new Thickness(1), Margin = new Thickness(0, 10, 0, 0), AutoGenerateColumns = false, CanUserAddRows = false };
            tradingRulesDataGrid.MouseRightButtonUp += TradingRulesDataGrid_MouseRightButtonUp;
            var addRuleButton = new Button { Name = "AddRuleButton", Width = 60, Height = 60, IsEnabled = IsSymbolSelected, Background = Brushes.Red };
            addRuleButton.Click += AddRuleButton_Click;
            var addRuleButtonText = new TextBlock { Text = "+", FontSize = 28, FontWeight = FontWeights.Bold, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center };
            addRuleButton.Content = addRuleButtonText;

            stackPanel.Children.Add(tradingRulesDataGrid);
            stackPanel.Children.Add(addRuleButton);

            return stackPanel;
        }

        private UIElement CreateAlertsCard()
        {
            // Create and return the UIElement for Alerts Card
            var stackPanel = new StackPanel();
            var alertsListBox = new ListBox { Name = "AlertsListBox", VerticalAlignment = VerticalAlignment.Stretch, Background = new SolidColorBrush(Color.FromRgb(45, 45, 68)), Foreground = Brushes.GhostWhite, BorderBrush = Brushes.Blue, BorderThickness = new Thickness(1), Margin = new Thickness(5) };

            stackPanel.Children.Add(alertsListBox);

            return stackPanel;
        }

        private UIElement CreatePredictionAnalysisCard()
        {
            try
            {
                // Create a new instance of our custom PredictionAnalysisControl with explicit sizing
                var predictionAnalysisControl = new redictionAnalysisControl();

                // Ensure the control has proper sizing and stretching behavior
                predictionAnalysisControl.Width = double.NaN; // Auto width
                predictionAnalysisControl.Height = double.NaN; // Auto height
                predictionAnalysisControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                predictionAnalysisControl.VerticalAlignment = VerticalAlignment.Stretch;
                predictionAnalysisControl.MinWidth = 400;
                predictionAnalysisControl.MinHeight = 300;

                // Immediately force layout calculation to ensure control is properly sized
                predictionAnalysisControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                predictionAnalysisControl.Arrange(new Rect(0, 0, predictionAnalysisControl.DesiredSize.Width, predictionAnalysisControl.DesiredSize.Height));
                predictionAnalysisControl.UpdateLayout();

                // Force the control to initialize itself
                if (predictionAnalysisControl.ForceLayoutUpdate != null)
                {
                    predictionAnalysisControl.ForceLayoutUpdate();
                }

                // Log success with dimensions
                DatabaseMonolith.Log("Info", $"Successfully created PredictionAnalysisControl with size: {predictionAnalysisControl.DesiredSize.Width}x{predictionAnalysisControl.DesiredSize.Height}");

                // Return the control
                return predictionAnalysisControl;
            }
            catch (Exception ex)
            {
                // Log error and create fallback content
                DatabaseMonolith.Log("Error", "Failed to create PredictionAnalysisControl", ex.ToString());

                // Create a simple error display as fallback
                var errorPanel = new StackPanel();
                errorPanel.Children.Add(new TextBlock
                {
                    Text = "Error: Could not load Prediction Analysis",
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                });
                errorPanel.Children.Add(new TextBlock
                {
                    Text = ex.Message,
                    Foreground = Brushes.White,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center
                });
                return errorPanel;
            }
        }

        private UIElement CreateConfigurationCard()
        {
            try
            {
                var configurationControl = new ConfigurationControl();

                // Ensure the control has proper sizing and stretching behavior
                configurationControl.Width = double.NaN; // Auto width
                configurationControl.Height = double.NaN; // Auto height
                configurationControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                configurationControl.VerticalAlignment = VerticalAlignment.Stretch;
                configurationControl.MinWidth = 400;
                configurationControl.MinHeight = 300;

                // Force layout calculation to ensure control is properly sized
                configurationControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                configurationControl.Arrange(new Rect(0, 0, configurationControl.DesiredSize.Width, configurationControl.DesiredSize.Height));
                configurationControl.UpdateLayout();

                // Log success
                DatabaseMonolith.Log("Info", "Successfully created ConfigurationControl");

                return configurationControl;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to create ConfigurationControl: {ex.Message}", ex.ToString());
                
                // Return a fallback control to prevent null reference
                var fallbackControl = new TextBlock
                {
                    Text = "Configuration control failed to load. Please check the logs for details.",
                    Foreground = new SolidColorBrush(Colors.Red),
                    FontWeight = FontWeights.Bold,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10)
                };
                
                return fallbackControl;
            }
        }

        private UIElement CreateTransactionsCard()
        {
            try
            {
                var transactionsControl = new TransactionsControl();

                // Ensure the control has proper sizing and stretching behavior
                transactionsControl.Width = double.NaN; // Auto width
                transactionsControl.Height = double.NaN; // Auto height
                transactionsControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                transactionsControl.VerticalAlignment = VerticalAlignment.Stretch;
                transactionsControl.MinWidth = 400;
                transactionsControl.MinHeight = 300;

                // Force layout calculation to ensure control is properly sized
                transactionsControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                transactionsControl.Arrange(new Rect(0, 0, transactionsControl.DesiredSize.Width, transactionsControl.DesiredSize.Height));
                transactionsControl.UpdateLayout();

                // Log success
                DatabaseMonolith.Log("Info", "Successfully created TransactionsUserControl");

                // Return the control
                return transactionsControl;
            }
            catch (Exception ex)
            {
                // Log error and create fallback content
                DatabaseMonolith.Log("Error", "Failed to create TransactionsUserControl", ex.ToString());

                // Create a simple error display as fallback
                var errorPanel = new StackPanel();
                errorPanel.Children.Add(new TextBlock
                {
                    Text = "Error: Could not load Transactions",
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                });
                errorPanel.Children.Add(new TextBlock
                {
                    Text = ex.Message,
                    Foreground = Brushes.White,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center
                });
                return errorPanel;
            }
        }

        private UIElement CreateMarketChatCard()
        {
            try
            {
                var marketChatControl = new Controls.MarketChatControl();

                // Ensure the control has proper sizing and stretching behavior
                marketChatControl.Width = double.NaN; // Auto width
                marketChatControl.Height = double.NaN; // Auto height
                marketChatControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                marketChatControl.VerticalAlignment = VerticalAlignment.Stretch;
                marketChatControl.MinWidth = 400;
                marketChatControl.MinHeight = 300;

                // Force layout calculation to ensure control is properly sized
                marketChatControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                marketChatControl.Arrange(new Rect(0, 0, marketChatControl.DesiredSize.Width, marketChatControl.DesiredSize.Height));
                marketChatControl.UpdateLayout();

                // Log success
                DatabaseMonolith.Log("Info", "Successfully created MarketChatControl");

                // Return the control
                return marketChatControl;
            }
            catch (Exception ex)
            {
                // Log error and create fallback content
                DatabaseMonolith.Log("Error", "Failed to create MarketChatControl", ex.ToString());

                // Create a simple error display as fallback
                var errorPanel = new StackPanel();
                errorPanel.Children.Add(new TextBlock
                {
                    Text = "Error: Could not load Market Chat",
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                });
                errorPanel.Children.Add(new TextBlock
                {
                    Text = ex.Message,
                    Foreground = Brushes.White,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center
                });
                return errorPanel;
            }
        }

        private UIElement CreateSpreadsExplorerCard()
        {
            try
            {
                // Create a new instance of our custom SpreadsExplorer control
                var spreadsExplorerControl = new SpreadsExplorer();

                // Ensure the control has proper sizing and stretching behavior
                spreadsExplorerControl.Width = double.NaN; // Auto width
                spreadsExplorerControl.Height = double.NaN; // Auto height
                spreadsExplorerControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                spreadsExplorerControl.VerticalAlignment = VerticalAlignment.Stretch;
                spreadsExplorerControl.MinWidth = 800;
                spreadsExplorerControl.MinHeight = 600;

                // Force layout calculation to ensure control is properly sized
                spreadsExplorerControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                spreadsExplorerControl.Arrange(new Rect(0, 0, spreadsExplorerControl.DesiredSize.Width, spreadsExplorerControl.DesiredSize.Height));
                spreadsExplorerControl.UpdateLayout();

                // Log success
                DatabaseMonolith.Log("Info", "Successfully created SpreadsExplorerControl");

                // Return the control
                return spreadsExplorerControl;
            }
            catch (Exception ex)
            {
                // Log error and create fallback content
                DatabaseMonolith.Log("Error", "Failed to create SpreadsExplorerControl", ex.ToString());

                // Create a simple error display as fallback
                var errorPanel = new StackPanel();
                errorPanel.Children.Add(new TextBlock
                {
                    Text = "Error: Could not load Spreads Explorer",
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                });
                errorPanel.Children.Add(new TextBlock
                {
                    Text = ex.Message,
                    Foreground = Brushes.White,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center
                });
                return errorPanel;
            }
        }
        
        private UIElement CreateSectorMomentumHeatmapCard()
        {
            try
            {
                // Create a new instance of our custom SectorAnalysisHeatmapControl
                var sectorHeatmapControl = new SectorAnalysisHeatmapControl();

                // Ensure the control has proper sizing and stretching behavior
                sectorHeatmapControl.Width = double.NaN; // Auto width
                sectorHeatmapControl.Height = double.NaN; // Auto height
                sectorHeatmapControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                sectorHeatmapControl.VerticalAlignment = VerticalAlignment.Stretch;
                sectorHeatmapControl.MinWidth = 400;
                sectorHeatmapControl.MinHeight = 300;

                // Force initial layout calculation
                sectorHeatmapControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                sectorHeatmapControl.Arrange(new Rect(0, 0, sectorHeatmapControl.DesiredSize.Width, sectorHeatmapControl.DesiredSize.Height));

                // Log success
                DatabaseMonolith.Log("Info", "Successfully created SectorAnalysisHeatmapControl");

                // Return the control
                return sectorHeatmapControl;
            }
            catch (Exception ex)
            {
                // Log error and create fallback content
                DatabaseMonolith.Log("Error", "Failed to create SectorAnalysisHeatmapControl", ex.ToString());

                // Create a simple error display as fallback
                var errorPanel = new StackPanel();
                errorPanel.Children.Add(new TextBlock
                {
                    Text = "Error: Could not load Sector Momentum Heatmap",
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                });
                errorPanel.Children.Add(new TextBlock
                {
                    Text = ex.Message,
                    Foreground = Brushes.White,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center
                });
                return errorPanel;
            }
        }

        private UIElement CreateBacktestingCard()
        {
            try
            {
                var backtestResultsControl = new Views.Backtesting.BacktestResultsControl();
                backtestResultsControl.Width = double.NaN;
                backtestResultsControl.Height = double.NaN;
                backtestResultsControl.HorizontalAlignment = HorizontalAlignment.Stretch;
                backtestResultsControl.VerticalAlignment = VerticalAlignment.Stretch;
                backtestResultsControl.MinWidth = 400;
                backtestResultsControl.MinHeight = 300;
                backtestResultsControl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                backtestResultsControl.Arrange(new Rect(0, 0, backtestResultsControl.DesiredSize.Width, backtestResultsControl.DesiredSize.Height));
                backtestResultsControl.UpdateLayout();
                DatabaseMonolith.Log("Info", "Successfully created BacktestResultsControl");
                return backtestResultsControl;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to create BacktestResultsControl", ex.ToString());
                var errorPanel = new StackPanel();
                errorPanel.Children.Add(new TextBlock
                {
                    Text = "Error: Could not load Backtesting",
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                });
                errorPanel.Children.Add(new TextBlock
                {
                    Text = ex.Message,
                    Foreground = Brushes.White,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(10),
                    HorizontalAlignment = HorizontalAlignment.Center
                });
                return errorPanel;
            }
        }

        public Border CreateDraggableBorder(UIElement content, int row, int col, int rowSpan, int colSpan, string tabName)
        {
            // Create a border to hold the control
            var border = new Border
            {
                BorderBrush = Brushes.Cyan,
                BorderThickness = new Thickness(1),
                Child = content,
                Tag = new { Row = row, Column = col, RowSpan = rowSpan, ColumnSpan = colSpan } // Store initial position data
            };

            // Create context menu for this control
            var contextMenu = new ContextMenu();
            
            // Apply enhanced styling
            contextMenu.Style = (Style)Application.Current.FindResource("EnhancedContextMenuStyle");

            // Add menu items
            var removeMenuItem = new MenuItem { Header = "Remove" };
            removeMenuItem.Style = (Style)Application.Current.FindResource("EnhancedMenuItemStyle");
            removeMenuItem.Click += (s, e) => RemoveControl(border, tabName);
            contextMenu.Items.Add(removeMenuItem);

            // Add resize handles on mouse enter
            border.MouseEnter += (s, e) =>
            {
                // Add resize adorner when mouse enters the control
                ShowResizeAdorner(border, tabName);
            };

            border.MouseLeave += (s, e) =>
            {
                // Only remove adorner if we're not dragging or resizing
                if (!border.IsMouseCaptured)
                {
                    // Check if mouse is still over an adorner before removing it
                    Point mousePos = Mouse.GetPosition(border);
                    if (mousePos.X < 0 || mousePos.Y < 0 ||
                        mousePos.X > border.ActualWidth || mousePos.Y > border.ActualHeight)
                    {
                        // Remove resize adorner when mouse leaves
                        RemoveResizeAdorner(border);
                    }
                }
            };

            // Track whether we're dragging the control
            bool isDragging = false;
            Point dragStartPoint = new Point();

            // Make the border draggable
            border.MouseLeftButtonDown += (s, e) =>
            {
                // Get the parent grid
                if (VisualTreeHelper.GetParent(border) is Grid parentGrid)
                {
                    // Check if we're near a corner (for resizing) or in the middle (for dragging)
                    Point mousePos = e.GetPosition(border);
                    bool nearCorner = IsNearCorner(mousePos, border.ActualWidth, border.ActualHeight);

                    if (nearCorner)
                    {
                        // Near corner - let the adorner handle resize
                        // Don't start dragging
                        // The adorner will handle its own mouse capture
                        e.Handled = true;
                    }
                    else
                    {
                        // In middle of control - start dragging
                        isDragging = true;
                        dragStartPoint = e.GetPosition(parentGrid);

                        // Store original position
                        border.Tag = new
                        {
                            Row = Grid.GetRow(border),
                            Column = Grid.GetColumn(border),
                            RowSpan = Grid.GetRowSpan(border),
                            ColumnSpan = Grid.GetColumnSpan(border)
                        };

                        // Remove any resize adorners to avoid conflicts during drag
                        RemoveResizeAdorner(border);

                        // Capture mouse for dragging
                        border.CaptureMouse();
                        e.Handled = true;
                    }
                }
            };

            border.MouseMove += (s, e) =>
            {
                // Only handle mouse move if we're dragging (not resizing via adorner)
                if (isDragging && border.IsMouseCaptured && e.LeftButton == MouseButtonState.Pressed)
                {
                    // Provide visual feedback during drag
                    border.Opacity = 0.7;

                    // Get current position
                    Point currentPosition = e.GetPosition((Grid)VisualTreeHelper.GetParent(border));

                    // Get parent grid and cell dimensions
                    if (VisualTreeHelper.GetParent(border) is Grid parentGrid)
                    {
                        double cellWidth = parentGrid.ActualWidth / parentGrid.ColumnDefinitions.Count;
                        double cellHeight = parentGrid.ActualHeight / parentGrid.RowDefinitions.Count;

                        // Calculate delta movement in cells
                        double deltaX = currentPosition.X - dragStartPoint.X;
                        double deltaY = currentPosition.Y - dragStartPoint.Y;

                        // Only start drag once we've moved enough (prevents accidental drags)
                        if (Math.Abs(deltaX) > cellWidth / 3 || Math.Abs(deltaY) > cellHeight / 3)
                        {
                            // Start drag-drop operation
                            DragDrop.DoDragDrop(border,
                                new DataObject("ControlBorder", border),
                                DragDropEffects.Move);

                            // Reset state after drag completes
                            border.Opacity = 1.0;
                            isDragging = false;
                            border.ReleaseMouseCapture();
                        }
                    }
                }
                else if (!isDragging)
                {
                    // Update cursor based on position for better UX when not dragging
                    Point mousePos = e.GetPosition(border);
                    UpdateResizeCursor(border, mousePos);
                }
            };

            border.MouseLeftButtonUp += (s, e) =>
            {
                // Reset dragging state
                isDragging = false;
                if (border.IsMouseCaptured)
                {
                    border.ReleaseMouseCapture();
                }
                border.Opacity = 1.0;
            };

            // Add drag drop target support
            border.AllowDrop = true;
            border.DragEnter += (s, e) =>
            {
                if (e.Data.GetDataPresent("ControlBorder"))
                {
                    e.Effects = DragDropEffects.Move;
                    border.BorderBrush = Brushes.LimeGreen;
                    border.BorderThickness = new Thickness(3);
                }
                e.Handled = true;
            };

            border.DragLeave += (s, e) =>
            {
                border.BorderBrush = Brushes.Cyan;
                border.BorderThickness = new Thickness(1);
            };

            border.Drop += (s, e) =>
            {
                if (e.Data.GetDataPresent("ControlBorder"))
                {
                    var draggedBorder = e.Data.GetData("ControlBorder") as Border;
                    if (draggedBorder != null && draggedBorder != border)
                    {
                        // Swap positions between the dragged border and the target
                        SwapControlPositions(draggedBorder, border, tabName);
                        e.Handled = true;

                        // Log successful drag-drop operation
                        DatabaseMonolith.Log("Info", "Control moved via drag-and-drop in tab: " + tabName);
                    }
                }
            };

            border.ContextMenu = contextMenu;
            return border;
        }

        // Add these new methods to MainWindow.TabManagement.cs 

        /// <summary>
        /// Shows resize adorner on the control border
        /// </summary>
        private void ShowResizeAdorner(Border controlBorder, string tabName)
        {
            // Remove any existing adorners first
            RemoveResizeAdorner(controlBorder);
            
            // Get the parent grid
            if (VisualTreeHelper.GetParent(controlBorder) is Grid parentGrid)
            {
                // Get adorner layer
                var adornerLayer = AdornerLayer.GetAdornerLayer(controlBorder);
                if (adornerLayer != null)
                {
                    // Get grid config
                    var gridConfig = DatabaseMonolith.LoadGridConfig(tabName);
                    int rows = Math.Max(4, gridConfig.Rows);
                    int cols = Math.Max(4, gridConfig.Columns);
                    
                    // Create a new resize adorner
                    var resizeAdorner = new ControlResizingAdorner(
                        controlBorder, 
                        parentGrid, 
                        rows, 
                        cols,
                        (newRow, newCol, newRowSpan, newColSpan) => {
                            // This callback is called when resize is completed
                            
                            // Check if the new position would overlap with other controls
                            bool isValid = true;
                            for (int r = newRow; r < newRow + newRowSpan; r++)
                            {
                                for (int c = newCol; c < newCol + newColSpan; c++)
                                {
                                    // Skip cells occupied by the current control
                                    if (r >= Grid.GetRow(controlBorder) && 
                                        r < Grid.GetRow(controlBorder) + Grid.GetRowSpan(controlBorder) &&
                                        c >= Grid.GetColumn(controlBorder) && 
                                        c < Grid.GetColumn(controlBorder) + Grid.GetColumnSpan(controlBorder))
                                        continue;
                                    
                                    if (IsCellOccupied(tabName, r, c))
                                    {
                                        isValid = false;
                                        break;
                                    }
                                }
                                
                                if (!isValid) break;
                            }
                            
                            if (isValid)
                            {
                                // Update grid positioning
                                Grid.SetRow(controlBorder, newRow);
                                Grid.SetColumn(controlBorder, newCol);
                                Grid.SetRowSpan(controlBorder, newRowSpan);
                                Grid.SetColumnSpan(controlBorder, newColSpan);
                                
                                // Save to database
                                int controlIndex = parentGrid.Children.IndexOf(controlBorder);
                                DatabaseMonolith.UpdateControlPosition(tabName, controlIndex, 
                                    newRow, newCol, newRowSpan, newColSpan);
                                

                                // Log the resize operation
                                DatabaseMonolith.Log("Info", $"Resized control via direct manipulation in tab '{tabName}' to ({newRow},{newCol}) with spans ({newRowSpan},{newColSpan})");
                            }
                            else
                            {
                                // Show error message
                                AppendAlert("Cannot resize: position overlaps with another control", "warning");
                            }
                        });
                    
                    // Add the adorner
                    adornerLayer.Add(resizeAdorner);
                }
            }
        }

        /// <summary>
        /// Removes resize adorner from the control border
        /// </summary>
        private void RemoveResizeAdorner(Border controlBorder)
        {
            var adornerLayer = AdornerLayer.GetAdornerLayer(controlBorder);
            if (adornerLayer != null)
            {
                var adorners = adornerLayer.GetAdorners(controlBorder);
                if (adorners != null)
                {
                    foreach (var adorner in adorners)
                    {
                        if (adorner is ControlResizingAdorner)
                        {
                            adornerLayer.Remove(adorner);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Checks if the mouse is near a corner of the control
        /// </summary>
        private bool IsNearCorner(Point position, double width, double height, double threshold = 15) // Increased threshold
        {
            bool nearTop = position.Y <= threshold;
            bool nearBottom = position.Y >= height - threshold;
            bool nearLeft = position.X <= threshold;
            bool nearRight = position.X >= width - threshold;

            return (nearTop && (nearLeft || nearRight)) || (nearBottom && (nearLeft || nearRight));
        }

        /// <summary>
        /// Updates cursor based on position over the control - disabled to remove unwanted cursor changes
        /// </summary>
        private void UpdateResizeCursor(Border border, Point position)
        {
            // Disable automatic cursor changes - keep normal arrow cursor
            // Resize functionality is handled by the ControlResizingAdorner when explicitly activated
            border.Cursor = Cursors.Arrow;
        }

        // Update the ShowMoveDialog method to only use in-place resize through the adorner
        private void ShowMoveDialog(Border controlBorder, string tabName)
        {
            // Always use in-place resize now (removed resize window option)
            ShowResizeAdorner(controlBorder, tabName);
        }

        /// <summary>
        /// Makes a grid draggable by enabling drag-and-drop functionality for controls within it
        /// </summary>
        /// <param name="grid">The grid to make draggable</param>
        /// <param name="tabName">The name of the tab containing the grid</param>
        public void MakeGridDraggable(Grid grid, string tabName)
        {
            // Make sure grid lines are visible
            DrawGridLines(grid);
            
            // Apply drag-and-drop functionality to each grid cell
            foreach (UIElement child in grid.Children)
            {
                if (child is Border border)
                {
                    border.MouseLeftButtonDown += (s, e) =>
                    {
                        border.CaptureMouse();
                        border.Tag = e.GetPosition(grid); // Store initial drag position
                        e.Handled = true;
                    };

                    border.MouseMove += (s, e) =>
                    {
                        if (border.IsMouseCaptured && e.LeftButton == MouseButtonState.Pressed)
                        {
                            Point currentPosition = e.GetPosition(grid);
                            
                            // Fix: Replace dynamic pattern matching with explicit type checking
                            Point startPosition;
                            
                            if (border.Tag is Point point)
                            {
                                startPosition = point;
                            }
                            else if (border.Tag != null)
                            {
                                // Try to access StartPoint property using reflection instead of dynamic
                                var tagType = border.Tag.GetType();
                                var startPointProperty = tagType.GetProperty("StartPoint");
                                
                                if (startPointProperty != null && startPointProperty.PropertyType == typeof(Point))
                                {
                                    var startPointValue = startPointProperty.GetValue(border.Tag);
                                    if (startPointValue is Point startPoint)
                                    {
                                        startPosition = startPoint;
                                    }
                                    else
                                    {
                                        // Fallback to current position
                                        startPosition = currentPosition;
                                        DatabaseMonolith.Log("Warning", $"StartPoint property found but not a Point in MakeGridDraggable: {startPointValue?.GetType().ToString() ?? "null"}");
                                    }
                                }
                                else
                                {
                                    // Fallback to current position
                                    startPosition = currentPosition;
                                    DatabaseMonolith.Log("Warning", $"Tag does not have a StartPoint property in MakeGridDraggable. Tag type: {tagType.Name}");
                                }
                            }
                            else
                            {
                                // If we can't extract a valid Point, use current position as fallback
                                startPosition = currentPosition;
                                DatabaseMonolith.Log("Warning", "Tag is null in MakeGridDraggable");
                            }

                            double deltaX = currentPosition.X - startPosition.X;
                            double deltaY = currentPosition.Y - startPosition.Y;

                            double cellWidth = grid.ActualWidth / grid.ColumnDefinitions.Count;
                            double cellHeight = grid.ActualHeight / grid.RowDefinitions.Count;

                            int newColumn = Math.Max(0, Math.Min((int)((Grid.GetColumn(border) + deltaX / cellWidth)), grid.ColumnDefinitions.Count - 1));
                            int newRow = Math.Max(0, Math.Min((int)((Grid.GetRow(border) + deltaY / cellHeight)), grid.RowDefinitions.Count - 1));

                            if (newRow != Grid.GetRow(border) || newColumn != Grid.GetColumn(border))
                            {
                                Grid.SetRow(border, newRow);
                                Grid.SetColumn(border, newColumn);

                                // Update database with new position
                                int controlIndex = grid.Children.IndexOf(border);
                                DatabaseMonolith.UpdateControlPosition(tabName, controlIndex, newRow, newColumn, Grid.GetRowSpan(border), Grid.GetColumnSpan(border));
                            }

                            border.Tag = currentPosition; // Update drag position
                        }
                    };

                    border.MouseLeftButtonUp += (s, e) =>
                    {
                        if (border.IsMouseCaptured)
                        {
                            border.ReleaseMouseCapture();
                        }
                    };
                }
            }
            
            // Add click handler to empty grid cells to allow selecting a cell
            grid.MouseLeftButtonDown += (s, e) => {
                // Only handle clicks directly on the grid, not on its children
                if (e.OriginalSource == grid)
                {
                    Point clickPoint = e.GetPosition(grid);
                    double cellWidth = grid.ActualWidth / grid.ColumnDefinitions.Count;
                    double cellHeight = grid.ActualHeight / grid.RowDefinitions.Count;
                    
                    int col = (int)(clickPoint.X / cellWidth);
                    int row = (int)(clickPoint.Y / cellHeight);
                    
                    // Log that we detected a grid cell click
                    DatabaseMonolith.Log("Info", $"Grid cell clicked at ({row},{col}) in tab '{tabName}'");
                    
                    // Here you can add code to highlight the cell or prepare for control placement
                }
            };

            // Add the MouseRightButtonUp event handler for empty cell detection
            grid.MouseRightButtonUp += Grid_MouseRightButtonUp;
        }

        // New method to draw visible grid lines
        private void DrawGridLines(Grid grid)
        {
            // Clear any existing grid line visuals
            var gridLines = grid.Children.OfType<Line>().ToList();
            foreach (var line in gridLines)
            {
                grid.Children.Remove(line);
            }
            
            int rows = grid.RowDefinitions.Count;
            int cols = grid.ColumnDefinitions.Count;
            
            // Add horizontal lines
            for (int r = 1; r < rows; r++)
            {
                var line = new Line
                {
                    Stroke = new SolidColorBrush(Color.FromArgb(40, 255, 255, 255)),
                    StrokeThickness = 1,
                    X1 = 0,
                    X2 = grid.ActualWidth,
                    Y1 = r * (grid.ActualHeight / rows),
                    Y2 = r * (grid.ActualHeight / rows),
                    IsHitTestVisible = false, // Make sure it doesn't interfere with clicks
                    Tag = "IsGridLine" // Mark this element as a grid line
                };
                grid.Children.Add(line);
            }
            
            // Add vertical lines
            for (int c = 1; c < cols; c++)
            {
                var line = new Line
                {
                    Stroke = new SolidColorBrush(Color.FromArgb(40, 255, 255, 255)),
                    StrokeThickness = 1,
                    X1 = c * (grid.ActualWidth / cols),
                    X2 = c * (grid.ActualWidth / cols),
                    Y1 = 0,
                    Y2 = grid.ActualHeight,
                    IsHitTestVisible = false, // Make sure it doesn't interfere with clicks
                    Tag = "IsGridLine" // Mark this element as a grid line
                };
                grid.Children.Add(line);
            }
            
            // Ensure grid lines are updated when the grid resizes
            grid.SizeChanged += (s, e) => {
                foreach (var child in grid.Children)
                {
                    if (child is Line line && line.Tag as string == "IsGridLine")
                    {
                        if (line.X1 == line.X2) // Vertical line
                        {
                            if (e.PreviousSize.Width > 0 && cols > 0)
                            {
                                int col = (int)Math.Round(line.X1 / (e.PreviousSize.Width / cols));
                                line.X1 = line.X2 = col * (grid.ActualWidth / cols);
                            }
                            else
                            {
                                line.X1 = line.X2 = line.X1; // keep position
                            }
                            line.Y2 = grid.ActualHeight;
                        }
                        else // Horizontal line
                        {
                            if (e.PreviousSize.Height > 0 && rows > 0)
                            {
                                int row = (int)Math.Round(line.Y1 / (e.PreviousSize.Height / rows));
                                line.Y1 = line.Y2 = row * (grid.ActualHeight / rows);
                            }
                            else
                            {
                                line.Y1 = line.Y2 = line.Y1; // keep position
                            }
                            line.X2 = grid.ActualWidth;
                        }
                    }
                }
            };
        }

        #endregion

        #region Control Placement and Validation

        // Check if a cell is occupied (for use by other classes)
        public bool IsCellOccupied(string tabName, int row, int column, int rowSpan = 1, int columnSpan = 1)
        {
            var tab = MainTabControl.Items
                .OfType<TabItem>()
                .FirstOrDefault(t => t.Header.ToString() == tabName);

            if (tab == null || !(tab.Content is Grid grid))
                return false;

            return IsOverlapping(grid, row, column, rowSpan, columnSpan);
        }

        // Method to check for cell occupancy
        private bool IsOverlapping(Grid grid, int newRow, int newColumn, int newRowSpan, int newColumnSpan)
        {
            // Check each child in the grid to see if it would overlap with the new control
            foreach (UIElement child in grid.Children)
            {
                // Skip grid lines which are not actually controls (these should not count as occupying cells)
                if (child is Line)
                {
                    continue;
                }

                // Skip elements with a 'IsGridLine' tag 
                if (child.GetValue(FrameworkElement.TagProperty) is string tag && tag == "IsGridLine")
                {
                    continue;
                }

                // Get the current child's grid position and span
                int childRow = Grid.GetRow(child);
                int childColumn = Grid.GetColumn(child);
                int childRowSpan = Grid.GetRowSpan(child);
                int childColumnSpan = Grid.GetColumnSpan(child);

                // Default span is 1 if not specified
                if (childRowSpan == 0) childRowSpan = 1;
                if (childColumnSpan == 0) childColumnSpan = 1;

                // Check for overlap using rectangle intersection logic
                bool rowOverlap = (newRow < childRow + childRowSpan) && (childRow < newRow + newRowSpan);
                bool columnOverlap = (newColumn < childColumn + childColumnSpan) && (childColumn < newColumn + newColumnSpan);

                if (rowOverlap && columnOverlap)
                {
                    // Log what child type is causing the overlap for debugging
                    DatabaseMonolith.Log("Debug", $"Overlap detected with element of type {child.GetType().Name} at position ({childRow},{childColumn})");
                    return true; // Overlap detected
                }
            }

            return false; // No overlap
        }

        // Get all occupied cells in a grid
        private List<(int Row, int Col)> GetOccupiedCells(Grid grid)
        {
            var occupiedCells = new List<(int Row, int Col)>();

            foreach (UIElement child in grid.Children)
            {
                // Skip grid lines
                if (child is Line || 
                   (child.GetValue(FrameworkElement.TagProperty) is string tag && tag == "IsGridLine"))
                {
                    continue;
                }

                int row = Grid.GetRow(child);
                int col = Grid.GetColumn(child);
                int rowSpan = Grid.GetRowSpan(child);
                int colSpan = Grid.GetColumnSpan(child);

                // Default span is 1 if not specified
                if (rowSpan == 0) rowSpan = 1;
                if (colSpan == 0) colSpan = 1;

                // Add all cells occupied by this control
                for (int r = row; r < row + rowSpan; r++)
                {
                    for (int c = col; c < col + colSpan; c++)
                    {
                        occupiedCells.Add((r, c));
                    }
                }
            }

            return occupiedCells;
        }

        // Get all occupied cells for a specific tab
        public List<(int Row, int Col)> GetOccupiedCellsForTab(string tabName)
        {
            var tab = MainTabControl.Items
                .OfType<TabItem>()
                .FirstOrDefault(t => t.Header.ToString() == tabName);

            if (tab == null || !(tab.Content is Grid grid))
                return new List<(int Row, int Col)>();

            return GetOccupiedCells(grid);
        }

        #endregion

        #region Control Addition and Management

        public void AddControlToTab(string tabName, string controlType, int row, int column, int rowSpan = 1, int columnSpan = 1)
        {
            var tab = MainTabControl.Items
                .OfType<TabItem>()
                .FirstOrDefault(t => t.Header.ToString() == tabName);

            if (tab == null)
            {
                MessageBox.Show($"Tab '{tabName}' not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            var grid = tab.Content as Grid;
            if (grid == null)
            {
                // Initialize a new Grid with 4x4 dimensions
                grid = new Grid();
                var gridConfig = DatabaseMonolith.LoadGridConfig(tabName);

                // Always ensure we have at least 4x4 grid dimensions
                int rows = Math.Max(4, gridConfig.Rows);
                int columns = Math.Max(4, gridConfig.Columns);

                for (int i = 0; i < rows; i++)
                    grid.RowDefinitions.Add(new RowDefinition());
                for (int j = 0; j < columns; j++)
                    grid.ColumnDefinitions.Add(new ColumnDefinition());

                tab.Content = grid;
            }

            // Ensure grid has at least 4x4 dimensions
            while (grid.RowDefinitions.Count < 4)
                grid.RowDefinitions.Add(new RowDefinition());
            while (grid.ColumnDefinitions.Count < 4)
                grid.ColumnDefinitions.Add(new ColumnDefinition());

            // Ensure grid has enough rows and columns for the new control
            while (grid.RowDefinitions.Count <= row + rowSpan - 1)
                grid.RowDefinitions.Add(new RowDefinition());
            while (grid.ColumnDefinitions.Count <= column + columnSpan - 1)
                grid.ColumnDefinitions.Add(new ColumnDefinition());

            // Check if the new control would overlap with any existing control
            if (IsOverlapping(grid, row, column, rowSpan, columnSpan))
            {
                // Show error message with occupied cells info
                MessageBox.Show($"Cannot add control at position ({row},{column}) with spans ({rowSpan},{columnSpan}) " +
                                $"as it would overlap with existing controls.",
                                "Overlap Error", MessageBoxButton.OK, MessageBoxImage.Warning);

                DatabaseMonolith.Log("Warning", $"Attempted to add overlapping control to tab '{tabName}' at position ({row},{column})");
                return;
            }

            // Create the control
            UIElement control = null;
            try {
                // Log control creation attempt
                DatabaseMonolith.Log("Info", $"Creating control of type: {controlType}");
                
                control = controlType switch
                {
                    "Symbol Charts" => CreateSymbolChartsCard(),
                    "Trading Rules" => CreateTradingRulesCard(),
                    "Alerts" => CreateAlertsCard(),
                    "Configuration" => CreateConfigurationCard(),
                    "Prediction Analysis" => CreatePredictionAnalysisCard(),
                    "Sector Momentum Heatmap" => CreateSectorMomentumHeatmapCard(),
                    "Transactions" => CreateTransactionsCard(),
                    "Backtesting" => CreateBacktestingCard(),
                    "Backtest Chart" => CreateBacktestingCard(), // Accept alias
                    "Market Chat" => CreateMarketChatCard(),
                    "Spreads Explorer" => CreateSpreadsExplorerCard(),
                    _ => throw new NotSupportedException($"Control type '{controlType}' is not supported.")
                };
                
                // Additional validation to ensure control was created
                if (control == null)
                {
                    throw new InvalidOperationException($"Failed to create control of type '{controlType}' - null returned");
                }
            }
            catch (Exception ex) {
                DatabaseMonolith.Log("Error", $"Failed to create control: {controlType}", ex.ToString());
                throw; // Re-throw to handle in the calling code
            }

            // Create draggable border
            var borderedControl = CreateDraggableBorder(control, row, column, rowSpan, columnSpan, tabName);

            // Add the control to the grid with span
            Grid.SetRow(borderedControl, row);
            Grid.SetColumn(borderedControl, column);
            Grid.SetRowSpan(borderedControl, rowSpan);
            Grid.SetColumnSpan(borderedControl, columnSpan);
            grid.Children.Add(borderedControl);

            // Save the updated configuration with spans
            DatabaseMonolith.AddCustomControlWithSpans(tabName, controlType, row, column, rowSpan, columnSpan);

            // Special handling for Prediction Analysis control - update layout after adding to grid
            if (controlType == "Prediction Analysis" && control is PredictionAnalysisControl predictionControl)
            {
                // Update layout directly without complex dispatcher logic
                try
                {
                    borderedControl.UpdateLayout();
                    predictionControl.ForceLayoutUpdate();
                    DatabaseMonolith.Log("Info", $"Force updated Prediction Analysis layout in tab '{tabName}'");
                    
                    // Optional: Select the first tab in the control to ensure content is visible
                    predictionControl.SelectTopPredictionsTab();
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Warning", "Could not select initial tab in Prediction Analysis control", ex.ToString());
                }
            }

            DatabaseMonolith.Log("Info", $"Control '{controlType}' added to tab '{tabName}' at position ({row},{column}) with spans ({rowSpan},{columnSpan})");
        }

        public List<string> GetTabNames()
        {
            return MainTabControl.Items
                .OfType<TabItem>()
                .Where(tab => tab.Header.ToString() != "+")
                .Select(tab => tab.Header.ToString())
                .ToList();
        }

        // Add this public method to force a refresh of tab controls
        public void RefreshTabControls(string tabName)
        {
            try
            {
                // Find the tab
                var tabItem = MainTabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == tabName);
                if (tabItem != null)
                {
                    // Save the content temporarily
                    var grid = tabItem.Content as Grid;
                    if (grid != null)
                    {
                        // Force the grid to re-measure
                        grid.InvalidateMeasure();
                        grid.UpdateLayout();
                        
                        // Process each control in the grid
                        foreach (var child in grid.Children)
                        {
                            if (child is Border border && border.Child != null)
                            {
                                // Force the control inside the border to update layout
                                var control = border.Child as FrameworkElement;
                                if (control != null)
                                {
                                    control.InvalidateMeasure();
                                    control.InvalidateArrange();
                                    control.UpdateLayout();
                                    
                                    // Special handling for PredictionAnalysisControl
                                    if (control is Controls.PredictionAnalysisControl predictionControl)
                                    {
                                        predictionControl.ForceLayoutUpdate();
                                        
                                        // Also try to select the first tab in the control to ensure content is visible
                                        try {
                                            predictionControl.SelectTopPredictionsTab();
                                        } 
                                        catch (Exception ex) {
                                            DatabaseMonolith.Log("Warning", "Could not select tab in Prediction Analysis control during refresh", ex.ToString());
                                        }
                                        
                                        DatabaseMonolith.Log("Info", $"Force updated Prediction Analysis layout in tab '{tabName}'");
                                    }
                                }
                            }
                        }
                        
                        DatabaseMonolith.Log("Info", $"Refreshed controls in tab '{tabName}'");
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error refreshing tab controls: {ex.Message}", ex.ToString());
            }
        }

        /// <summary>
        /// Removes a control from the grid and the database
        /// </summary>
        /// <param name="controlBorder">The border containing the control to remove</param>
        /// <param name="tabName">The name of the tab containing the control</param>
        public void RemoveControl(Border controlBorder, string tabName)
        {
            try
            {
                // Get the parent grid
                if (VisualTreeHelper.GetParent(controlBorder) is Grid parentGrid)
                {
                    // Get the child control inside the border
                    var control = controlBorder.Child;
                    string controlType = control?.GetType().Name ?? "Unknown";
                    
                    // Get the position before removing
                    int row = Grid.GetRow(controlBorder);
                    int column = Grid.GetColumn(controlBorder);
                    
                    // Get the control index in the grid
                    int controlIndex = parentGrid.Children.IndexOf(controlBorder);
                    
                    // Remove the control from the grid
                    parentGrid.Children.Remove(controlBorder);
                    
                    // Remove the control from the database
                    DatabaseMonolith.RemoveControl(tabName, controlIndex);
                    
                    // Log the removal
                    DatabaseMonolith.Log("Info", $"Removed control of type '{controlType}' from position ({row},{column}) in tab '{tabName}'");
                    
                    // Show success message
                    AppendAlert($"Removed control from tab: {tabName}", "neutral");
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error removing control from tab '{tabName}'", ex.ToString());
                AppendAlert("Error removing control", "warning");
            }
        }

        /// <summary>
        /// Swaps the positions of two controls in the grid and updates the database
        /// </summary>
        /// <param name="sourceBorder">The first control border to swap</param>
        /// <param name="targetBorder">The second control border to swap</param>
        /// <param name="tabName">The name of the tab containing the controls</param>
        public void SwapControlPositions(Border sourceBorder, Border targetBorder, string tabName)
        {
            try
            {
                // Get the parent grid
                if (VisualTreeHelper.GetParent(sourceBorder) is Grid parentGrid &&
                    VisualTreeHelper.GetParent(targetBorder) == parentGrid)
                {
                    // Get source position and span
                    int sourceRow = Grid.GetRow(sourceBorder);
                    int sourceColumn = Grid.GetColumn(sourceBorder);
                    int sourceRowSpan = Grid.GetRowSpan(sourceBorder);
                    int sourceColumnSpan = Grid.GetColumnSpan(sourceBorder);
                    
                    // Get target position and span
                    int targetRow = Grid.GetRow(targetBorder);
                    int targetColumn = Grid.GetColumn(targetBorder);
                    int targetRowSpan = Grid.GetRowSpan(targetBorder);
                    int targetColumnSpan = Grid.GetColumnSpan(targetBorder);
                    
                    // Swap positions
                    Grid.SetRow(sourceBorder, targetRow);
                    Grid.SetColumn(sourceBorder, targetColumn);
                    Grid.SetRowSpan(sourceBorder, targetRowSpan);
                    Grid.SetColumnSpan(sourceBorder, targetColumnSpan);
                    
                    Grid.SetRow(targetBorder, sourceRow);
                    Grid.SetColumn(targetBorder, sourceColumn);
                    Grid.SetRowSpan(targetBorder, sourceRowSpan);
                    Grid.SetColumnSpan(targetBorder, sourceColumnSpan);
                    
                    // Get indices for database update
                    int sourceIndex = parentGrid.Children.IndexOf(sourceBorder);
                    int targetIndex = parentGrid.Children.IndexOf(targetBorder);
                    
                    // Update positions in the database
                    DatabaseMonolith.UpdateControlPosition(tabName, sourceIndex, targetRow, targetColumn, targetRowSpan, targetColumnSpan);
                    DatabaseMonolith.UpdateControlPosition(tabName, targetIndex, sourceRow, sourceColumn, sourceRowSpan, sourceColumnSpan);
                    
                    // Log the swap
                    DatabaseMonolith.Log("Info", $"Swapped control positions in tab '{tabName}': ({sourceRow},{sourceColumn}) <-> ({targetRow},{targetColumn})");
                    
                    // Reset borders
                    sourceBorder.BorderBrush = Brushes.Cyan;
                    sourceBorder.BorderThickness = new Thickness(1);
                    targetBorder.BorderBrush = Brushes.Cyan;
                    targetBorder.BorderThickness = new Thickness(1);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error swapping control positions in tab '{tabName}'", ex.ToString());
                AppendAlert("Error swapping controls", "warning");
            }
        }

        #endregion

        #region Grid Initialization

        // Ensure grids are initialized with default dimensions when a tab is created or loaded
        //private void EnsureGridInitialized(TabItem tabItem, string tabName)
        //{
        //    if (tabItem.Content is not Grid grid)
        //    {
        //        grid = new Grid();

        //        // Default grid dimensions (4x4)
        //        int rows = 4;
        //        int columns = 4;

        //        // Add rows and columns
        //        for (int i = 0; i < rows; i++)
        //        {
        //            grid.RowDefinitions.Add(new RowDefinition());
        //        }
        //        for (int j = 0; j < columns; j++)
        //        {
        //            grid.ColumnDefinitions.Add(new ColumnDefinition());
        //        }

        //        tabItem.Content = grid;

        //        // Log grid initialization
        //        DatabaseMonolith.Log("Info", $"Initialized grid for tab '{tabName}' with default dimensions {rows}x{columns}");
        //    }
        //}

        #endregion

        // Add this new method to detect right-clicks on empty grid cells
        private void Grid_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (sender is Grid grid)
            {
                // Get the point where the user clicked
                Point clickPoint = e.GetPosition(grid);

                // Calculate which cell was clicked
                double cellWidth = grid.ActualWidth / grid.ColumnDefinitions.Count;
                double cellHeight = grid.ActualHeight / grid.RowDefinitions.Count;
                
                int col = (int)(clickPoint.X / cellWidth);
                int row = (int)(clickPoint.Y / cellHeight);
                
                // Find the tab that contains this grid
                var tabItem = VisualTreeUtility.FindAncestor<TabItem>(grid);
                if (tabItem != null && tabItem.Header is string tabName)
                {
                    // Check if the cell is empty
                    if (!IsCellOccupied(tabName, row, col))
                    {
                        // Create context menu if needed
                        if (grid.ContextMenu == null)
                        {
                            grid.ContextMenu = new ContextMenu();
                            
                            // Apply enhanced styling
                            grid.ContextMenu.Style = (Style)Application.Current.FindResource("EnhancedContextMenuStyle");

                            var addToolMenuItem = new MenuItem 
                            { 
                                Header = "Add Tool",
                                Icon = new System.Windows.Controls.Image
                                {
                                    Source = new System.Windows.Media.Imaging.BitmapImage(
                                        new Uri("pack://application:,,,/Quantra;component/Resources/Icons/add.png", UriKind.Absolute)),
                                    Width = 16,
                                    Height = 16
                                }
                            };
                            

                            // Apply enhanced styling to menu item
                            addToolMenuItem.Style = (Style)Application.Current.FindResource("EnhancedMenuItemStyle");
                            addToolMenuItem.Click += (s, args) => OpenAddControlWindowForCell(tabName, row, col);
                            grid.ContextMenu.Items.Add(addToolMenuItem);
                        }

                        // Store the cell coordinates in the context menu tag
                        grid.ContextMenu.Tag = new { Row = row, Column = col, TabName = tabName };
                    }
                    else
                    {
                        // If cell is occupied, don't show the context menu
                        if (grid.ContextMenu != null)
                        {
                            grid.ContextMenu.IsOpen = false;
                        }
                        e.Handled = true;
                        return;
                    }
                }
            }
        }

        // Add this new method to open the AddControlWindow for a specific cell
        private void OpenAddControlWindowForCell(string tabName, int row, int col)
        {
            var addControlWindow = AddControlWindow.GetInstance();
            addControlWindow.InitializeWithPosition(tabName, row, col);
            addControlWindow.Show();
            addControlWindow.Activate();
            
            // Log this action
            DatabaseMonolith.Log("Info", $"Opened AddControlWindow for cell ({row+1},{col+1}) in tab '{tabName}'");
        }


    }
}