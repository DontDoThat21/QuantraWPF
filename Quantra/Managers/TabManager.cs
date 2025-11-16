using Dapper;
using Quantra.Controls;
using Quantra.DAL.Data;
using Quantra.DAL.Services;
using Quantra.Repositories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Quantra.Managers
{
    public class TabManager
    {
        #region Fields and Properties

        private readonly TabControl _mainTabControl;
        private readonly TabRepository _tabRepository;
        private readonly UserSettingsService _userSettingsService;
        private TabItem _lastNonPlusTab;
        // Flag to track recursive tab selection operations
        private bool _isTabSelectionInProgress = false;

        #endregion

        #region Events

        public delegate void TabAddedEventHandler(string tabName);
        public event TabAddedEventHandler TabAdded;

        #endregion

        #region Constructor

        public TabManager(TabControl tabControl, UserSettingsService userSettingsService)
        {
            _mainTabControl = tabControl ?? throw new ArgumentNullException(nameof(tabControl));
            var connection = ConnectionHelper.GetConnection();
            _tabRepository = new TabRepository(connection);
            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
        }

        #endregion

        #region Tab Management Methods

        public void LoadCustomTabs()
        {
            try
            {
                var tabs = _tabRepository.GetTabs();
                //DatabaseMonolith.Log("Info", $"Found {tabs.Count} tabs in database");

                // Filter out any auto-generated StockExplorer tabs and clean them from database
                var validTabs = new List<(string TabName, int TabOrder)>();
                foreach (var tab in tabs)
                {
                    if (tab.TabName.StartsWith("StockExplorer_"))
                    {
                        // Remove auto-generated StockExplorer tabs from database
                        _tabRepository.DeleteTab(tab.TabName);
                        //DatabaseMonolith.Log("Info", $"Cleaned up auto-generated tab: {tab.TabName}");
                    }
                    else
                    {
                        validTabs.Add(tab);
                    }
                }

                // Clear existing tabs except for the '+' tab
                var plusTab = _mainTabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == "+");
                _mainTabControl.Items.Clear();
                if (plusTab != null)
                {
                    _mainTabControl.Items.Add(plusTab);
                }

                // Add valid tabs from database
                foreach (var tab in validTabs)
                {
                    AddCustomTab(tab.TabName);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading custom tabs", ex.ToString());
                MessageBox.Show($"Error loading tabs: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public void AddCustomTab(string tabName)
        {
            // Create a new tab with proper grid structure
            var tabItem = new TabItem
            {
                Header = tabName
            };

            // Get grid settings from database
            var settings = _userSettingsService.GetUserSettings();

            // Parse the color
            SolidColorBrush gridBorderBrush = Brushes.Cyan; // Default
            try
            {
                if (!string.IsNullOrEmpty(settings.GridBorderColor))
                {
                    Color borderColor = (Color)ColorConverter.ConvertFromString(settings.GridBorderColor);
                    gridBorderBrush = new SolidColorBrush(borderColor);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Failed to parse grid border color", ex.ToString());
            }

            // Use default grid dimensions from settings
            int rows = settings.DefaultGridRows;
            int columns = settings.DefaultGridColumns;

            // Ensure we have at least 1�1
            rows = Math.Max(1, rows);
            columns = Math.Max(1, columns);

            // Create a grid with dimensions from settings
            var grid = new Grid();

            // Set up the grid rows and columns
            for (int i = 0; i < rows; i++)
            {
                grid.RowDefinitions.Add(new RowDefinition());
            }
            for (int j = 0; j < columns; j++)
            {
                grid.ColumnDefinitions.Add(new ColumnDefinition());
            }

            // Add borders to each cell in the grid
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < columns; col++)
                {
                    var cellBorder = new Border
                    {
                        BorderBrush = gridBorderBrush,
                        BorderThickness = new Thickness(1),
                        Margin = new Thickness(0),
                        Background = new SolidColorBrush(Color.FromArgb(20, gridBorderBrush.Color.R, gridBorderBrush.Color.G, gridBorderBrush.Color.B)),
                        Tag = new { IsGridCell = true, Row = row, Col = col }
                    };

                    // Set cell position
                    Grid.SetRow(cellBorder, row);
                    Grid.SetColumn(cellBorder, col);

                    // Make the cell accept drops
                    cellBorder.AllowDrop = true;

                    // Handle drag-and-drop on empty cells
                    cellBorder.Drop += (s, e) => {
                        if (e.Data.GetDataPresent("ControlBorder"))
                        {
                            var draggedBorder = e.Data.GetData("ControlBorder") as Border;
                            if (draggedBorder != null)
                            {
                                var targetBorder = s as Border;
                                var cellInfo = targetBorder.Tag as dynamic;

                                if (cellInfo?.IsGridCell == true)
                                {
                                    // Move the control to this empty cell
                                    int newRow = cellInfo.Row;
                                    int newCol = cellInfo.Col;

                                    // Update control position in grid
                                    Grid.SetRow(draggedBorder, newRow);
                                    Grid.SetColumn(draggedBorder, newCol);

                                    // Update in database
                                    int controlIndex = grid.Children.IndexOf(draggedBorder);
                                    if (controlIndex >= 0)
                                    {
                                        DatabaseMonolith.UpdateControlPosition(
                                            tabName,
                                            controlIndex,
                                            newRow,
                                            newCol,
                                            Grid.GetRowSpan(draggedBorder),
                                            Grid.GetColumnSpan(draggedBorder));

                                        // Log the move
                                        //DatabaseMonolith.Log("Info", $"Moved control to cell ({newRow},{newCol}) in tab '{tabName}'");
                                    }

                                    e.Handled = true;
                                }
                            }
                        }
                    };

                    grid.Children.Add(cellBorder);
                }
            }

            // Set the grid as the tab content
            tabItem.Content = grid;

            // Add context menu for removing and editing the tab
            var contextMenu = new ContextMenu();
            
            // Apply enhanced styling
            contextMenu.Style = (Style)Application.Current.FindResource("EnhancedContextMenuStyle");
            
            var removeMenuItem = new MenuItem { Header = "Remove Tab" };
            removeMenuItem.Style = (Style)Application.Current.FindResource("EnhancedMenuItemStyle");
            removeMenuItem.Click += (s, e) => RemoveCustomTab(tabItem, tabName);
            contextMenu.Items.Add(removeMenuItem);

            var editMenuItem = new MenuItem { Header = "Edit Tab" };
            editMenuItem.Style = (Style)Application.Current.FindResource("EnhancedMenuItemStyle");
            editMenuItem.Click += (s, e) => EditCustomTab(tabItem, tabName);
            contextMenu.Items.Add(editMenuItem);

            tabItem.ContextMenu = contextMenu;

            _mainTabControl.Items.Insert(_mainTabControl.Items.Count - 1, tabItem);

            // Make the grid support direct drag-and-drop
            MakeGridDraggable(grid, tabName);
            
            // Raise the TabAdded event to notify listeners
            TabAdded?.Invoke(tabName);

            // NEW: Select the newly added tab
            try
            {
                _mainTabControl.SelectedItem = tabItem;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to select new tab '{tabName}' after adding", ex.ToString());
            }

            // NEW: If AddControlWindow is open, refresh and select this new tab
            try
            {
                var addControlWnd = Application.Current.Windows.OfType<AddControlWindow>().FirstOrDefault();
                if (addControlWnd != null && addControlWnd.IsLoaded)
                {
                    addControlWnd.RefreshTabs(tabName);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Failed to refresh AddControlWindow after adding new tab", ex.ToString());
            }
        }

        public void RemoveCustomTab(TabItem tabItem, string tabName)
        {
            // Remove the tab from the UI
            _mainTabControl.Items.Remove(tabItem);

            // Remove the tab from the database
            _tabRepository.DeleteTab(tabName);

            // Notify the main window that the tab was removed
            OnTabAction($"Removed tab: {tabName}");
        }

        public void EditCustomTab(TabItem tabItem, string oldTabName)
        {
            string newTabName = Microsoft.VisualBasic.Interaction.InputBox("Enter the new name for the tab:", "Edit Tab", oldTabName);
            if (!string.IsNullOrEmpty(newTabName) && newTabName != oldTabName)
            {
                // Update the tab name in the UI
                tabItem.Header = newTabName;

                // Update the tab name in the database
                _tabRepository.UpdateTabName(oldTabName, newTabName);

                // Notify the main window that the tab was edited
                OnTabAction($"Renamed tab from {oldTabName} to {newTabName}");
            }
        }

        public void HandleAddNewTabButtonClick(object sender, MouseButtonEventArgs e)
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
                AddCustomTab(newTabName);

                // Save the tab to the database with specified grid dimensions
                SaveCustomTabWithGrid(newTabName, gridRows, gridColumns);

                // Log the creation
                //DatabaseMonolith.Log("Info", $"Created new tab: {newTabName} with grid dimensions {gridRows}x{gridColumns}");

                // Notify the main window that the tab was created
                OnTabAction($"Created new tab: {newTabName}", "positive");
            }
        }

        public void SaveCustomTabWithGrid(string tabName, int rows, int columns)
        {
            try
            {
                // Calculate new tab order (one before the '+' tab)
                int tabOrder = _mainTabControl.Items.Count - 2;
                _tabRepository.InsertTab(tabName, tabOrder, rows, columns);
                //DatabaseMonolith.Log("Info", $"Created new tab: {tabName} with grid dimensions {rows}x{columns}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error saving tab '{tabName}'", ex.ToString());
            }
        }

        public void HandleTabPreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                var tabItem = FindAncestor<TabItem>((DependencyObject)e.OriginalSource);
                if (tabItem != null)
                {
                    DragDrop.DoDragDrop(tabItem, tabItem, DragDropEffects.Move);
                }
            }
        }

        public void HandleTabDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetData(typeof(TabItem)) is TabItem tabItem)
            {
                var tabControl = sender as TabControl;
                var point = e.GetPosition(tabControl);
                var targetItem = tabControl.InputHitTest(point) as TabItem;

                if (targetItem != null && targetItem != tabItem)
                {
                    int oldIndex = tabControl.Items.IndexOf(tabItem);
                    int newIndex = tabControl.Items.IndexOf(targetItem);

                    tabControl.Items.Remove(tabItem);
                    tabControl.Items.Insert(newIndex, tabItem);

                    // Update the tab order in the database
                    for (int i = 0; i < _mainTabControl.Items.Count - 1; i++)
                    {
                        if (_mainTabControl.Items[i] is TabItem item && item.Header is string tabName)
                        {
                            _tabRepository.UpdateTabOrder(tabName, i);
                        }
                    }
                }
            }
        }

        public void HandleTabSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Guard against recursive tab selection operations
            if (_isTabSelectionInProgress)
            {
                //DatabaseMonolith.Log("Info", "TabManager: Prevented recursive tab selection");
                return;
            }

            try
            {
                // Set the flag to indicate we're processing a tab selection
                _isTabSelectionInProgress = true;
                
                var selectedTab = _mainTabControl.SelectedItem as TabItem;
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
                            Style = Application.Current.FindResource("ButtonStyle1") as Style
                        };
                        addToolButton.Click += (s, args) => AddToolButtonClick?.Invoke(s, args);
                        border.Child = addToolButton;
                        grid.Children.Add(border);
                        selectedTab.Content = grid;
                    }
                    else
                    {
                        // For other tabs, update the last non-'+' tab
                        _lastNonPlusTab = selectedTab;
                        
                        // Notify about tab selection - the host will handle loading the controls
                        TabSelectionChanged?.Invoke(selectedTab.Header.ToString());
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error in HandleTabSelectionChanged", ex.ToString());
            }
            finally
            {
                // Always reset the flag when done
                _isTabSelectionInProgress = false;
            }
        }

        #endregion

        #region Grid Management Methods

        public void MakeGridDraggable(Grid grid, string tabName)
        {
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
                            Point startPosition = (Point)border.Tag;

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
        }

        #endregion

        #region Helper Methods

        // Helper method to find a control's ancestor in the visual tree
        private static T FindAncestor<T>(DependencyObject current) where T : DependencyObject
        {
            while (current != null)
            {
                if (current is T)
                {
                    return (T)current;
                }
                current = VisualTreeHelper.GetParent(current);
            }
            return null;
        }

        public TabItem GetLastNonPlusTab()
        {
            return _lastNonPlusTab;
        }

        public void SetLastNonPlusTab(TabItem tabItem)
        {
            _lastNonPlusTab = tabItem;
        }

        #endregion

        #region Event Delegates

        // Event to notify MainWindow about tab actions (to show alerts)
        public delegate void TabActionEventHandler(string message, string type = "");
        public event TabActionEventHandler TabAction;

        private void OnTabAction(string message, string type = "")
        {
            TabAction?.Invoke(message, type);
        }

        // Event to notify when tab selection changes
        public delegate void TabSelectionChangedEventHandler(string tabName);
        public event TabSelectionChangedEventHandler TabSelectionChanged;

        // Event to forward AddTool button clicks
        public event EventHandler AddToolButtonClick;

        #endregion
    }
}
