using Dapper;
using Quantra.Controls;
using Quantra.Repositories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Quantra.Utilities
{
    /// <summary>
    /// Manages tab-related operations within the application including loading, adding,
    /// saving, editing, and removing tabs, as well as handling tab selection and drag-drop functionality.
    /// </summary>
    public class TabManager
    {
        #region Fields and Properties

        private readonly MainWindow _mainWindow;
        private readonly TabControl _tabControl;
        private TabItem _lastNonPlusTab;
        private bool _isTabSelectionInProgress = false;

        #endregion

        #region Events

        public delegate void TabAddedEventHandler(string tabName);
        public event TabAddedEventHandler TabAdded;

        #endregion

        #region Constructor

        public TabManager(MainWindow mainWindow, TabControl tabControl)
        {
            _mainWindow = mainWindow ?? throw new ArgumentNullException(nameof(mainWindow));
            _tabControl = tabControl ?? throw new ArgumentNullException(nameof(tabControl));
            
            // Attach event handlers
            _tabControl.PreviewMouseMove += TabControl_PreviewMouseMove;
            _tabControl.Drop += TabControl_Drop;
        }

        #endregion

        #region Public Tab Management Methods

        /// <summary>
        /// Loads all custom tabs from the database
        /// </summary>
        public void LoadCustomTabs()
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    var tabs = connection.Query<(string TabName, int TabOrder)>(
                        "SELECT TabName, TabOrder FROM UserAppSettings ORDER BY TabOrder").ToList();

                    DatabaseMonolith.Log("Info", $"Found {tabs.Count} tabs in database");

                    // Clear existing tabs except for the '+' tab
                    var plusTab = _tabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == "+");
                    _tabControl.Items.Clear();
                    if (plusTab != null)
                    {
                        _tabControl.Items.Add(plusTab);
                    }

                    // Add tabs from database
                    foreach (var tab in tabs)
                    {
                        AddCustomTab(tab.TabName);
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error loading custom tabs", ex.ToString());
                MessageBox.Show($"Error loading tabs: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Adds a new custom tab with the specified name
        /// </summary>
        /// <param name="tabName">Name of the tab to add</param>
        public void AddCustomTab(string tabName)
        {
            // Create a new tab with proper grid structure
            var tabItem = new TabItem
            {
                Header = tabName
            };

            // Get grid settings from database
            var settings = DatabaseMonolith.GetUserSettings();

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
                DatabaseMonolith.Log("Warning", "Failed to parse grid border color", ex.ToString());
            }

            // Use default grid dimensions from settings
            int rows = settings.DefaultGridRows;
            int columns = settings.DefaultGridColumns;

            // Ensure we have at least 1x1
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
                                        DatabaseMonolith.Log("Info", $"Moved control to cell ({newRow},{newCol}) in tab '{tabName}'");
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

            _tabControl.Items.Insert(_tabControl.Items.Count - 1, tabItem);

            // Make the grid support direct drag-and-drop
            _mainWindow.MakeGridDraggable(grid, tabName);
            
            // Raise the TabAdded event to notify listeners
            TabAdded?.Invoke(tabName);

            // NEW: Select the newly added tab in the main TabControl
            try
            {
                _tabControl.SelectedItem = tabItem;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Failed to select new tab '{tabName}' after adding", ex.ToString());
            }

            // NEW: If AddControlWindow is open, refresh and select this new tab there as well
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
                DatabaseMonolith.Log("Warning", "Failed to refresh AddControlWindow after adding new tab", ex.ToString());
            }
        }

        /// <summary>
        /// Saves a custom tab to the database
        /// </summary>
        /// <param name="tabName">Name of the tab to save</param>
        public void SaveCustomTab(string tabName)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    // Check if tab already exists
                    var existingTab = connection.QueryFirstOrDefault<string>(
                        "SELECT TabName FROM UserAppSettings WHERE TabName = @TabName", new { TabName = tabName });

                    if (existingTab == null)
                    {
                        // Default grid dimensions - always 4x4
                        int rows = 4;
                        int columns = 4;

                        // Calculate new tab order (one before the '+' tab)
                        int tabOrder = _tabControl.Items.Count - 2;

                        var insertQuery = @"
                            INSERT INTO UserAppSettings 
                                (TabName, TabOrder, GridRows, GridColumns) 
                            VALUES 
                                (@TabName, @TabOrder, @GridRows, @GridColumns)";

                        connection.Execute(insertQuery, new
                        {
                            TabName = tabName,
                            TabOrder = tabOrder,
                            GridRows = rows,
                            GridColumns = columns
                        });

                        DatabaseMonolith.Log("Info", $"Created new tab: {tabName} with grid dimensions {rows}x{columns}");
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error saving tab '{tabName}'", ex.ToString());
            }
        }

        /// <summary>
        /// Saves a custom tab with specific grid dimensions
        /// </summary>
        /// <param name="tabName">Name of the tab</param>
        /// <param name="rows">Number of grid rows</param>
        /// <param name="columns">Number of grid columns</param>
        public void SaveCustomTabWithGrid(string tabName, int rows, int columns)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    // Check if tab already exists
                    var existingTab = connection.QueryFirstOrDefault<string>(
                        "SELECT TabName FROM UserAppSettings WHERE TabName = @TabName", new { TabName = tabName });

                    if (existingTab == null)
                    {
                        // Calculate new tab order (one before the '+' tab)
                        int tabOrder = _tabControl.Items.Count - 2;

                        var insertQuery = @"
                            INSERT INTO UserAppSettings 
                                (TabName, TabOrder, GridRows, GridColumns) 
                            VALUES 
                                (@TabName, @TabOrder, @GridRows, @GridColumns)";

                        connection.Execute(insertQuery, new
                        {
                            TabName = tabName,
                            TabOrder = tabOrder,
                            GridRows = rows,
                            GridColumns = columns
                        });

                        DatabaseMonolith.Log("Info", $"Created new tab: {tabName} with grid dimensions {rows}x{columns}");
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error saving tab '{tabName}'", ex.ToString());
            }
        }

        /// <summary>
        /// Removes a custom tab from the UI and database
        /// </summary>
        /// <param name="tabItem">The tab item to remove</param>
        /// <param name="tabName">The name of the tab to remove</param>
        public void RemoveCustomTab(TabItem tabItem, string tabName)
        {
            // Remove the tab from the UI
            _tabControl.Items.Remove(tabItem);

            // Remove the tab from the database
            using (var connection = DatabaseMonolith.GetConnection())
            {
                connection.Open();
                var deleteQuery = "DELETE FROM UserAppSettings WHERE TabName = @TabName";
                connection.Execute(deleteQuery, new { TabName = tabName });
            }

            _mainWindow.AppendAlert($"Removed tab: {tabName}");
        }

        /// <summary>
        /// Edit a custom tab's name
        /// </summary>
        /// <param name="tabItem">The tab item to edit</param>
        /// <param name="oldTabName">The current name of the tab</param>
        public void EditCustomTab(TabItem tabItem, string oldTabName)
        {
            string newTabName = Microsoft.VisualBasic.Interaction.InputBox("Enter the new name for the tab:", "Edit Tab", oldTabName);
            if (!string.IsNullOrEmpty(newTabName) && newTabName != oldTabName)
            {
                // Update the tab name in the UI
                tabItem.Header = newTabName;

                // Update the tab name in the database
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    var updateQuery = "UPDATE UserAppSettings SET TabName = @NewTabName WHERE TabName = @OldTabName";
                    connection.Execute(updateQuery, new { NewTabName = newTabName, OldTabName = oldTabName });

                    // Ensure grid dimensions are at least 4x4
                    var gridQuery = "UPDATE UserAppSettings SET GridRows = MAX(GridRows, 4), GridColumns = MAX(GridColumns, 4) WHERE TabName = @NewTabName";
                    connection.Execute(gridQuery, new { NewTabName = newTabName });
                }

                _mainWindow.AppendAlert($"Renamed tab from {oldTabName} to {newTabName}");
            }
        }

        /// <summary>
        /// Creates a new tab based on user input from the CreateTabWindow
        /// </summary>
        public void AddNewTab()
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
                DatabaseMonolith.Log("Info", $"Created new tab: {newTabName} with grid dimensions {gridRows}x{gridColumns}");

                // Show success message
                _mainWindow.AppendAlert($"Created new tab: {newTabName}", "positive");
            }
        }

        /// <summary>
        /// Gets a list of all tab names except the '+' tab
        /// </summary>
        /// <returns>List of tab names</returns>
        public List<string> GetTabNames()
        {
            return _tabControl.Items
                .OfType<TabItem>()
                .Where(tab => tab.Header.ToString() != "+")
                .Select(tab => tab.Header.ToString())
                .ToList();
        }

        /// <summary>
        /// Loads controls for a specific tab
        /// </summary>
        /// <param name="tabName">Name of the tab to load controls for</param>
        public void LoadTabControls(string tabName)
        {
            var controlsConfig = DatabaseMonolith.LoadControlsConfig(tabName);
            var gridConfig = DatabaseMonolith.LoadGridConfig(tabName);

            // Always ensure we have at least 4x4 grid dimensions
            int rows = Math.Max(4, gridConfig.Rows);
            int columns = Math.Max(4, gridConfig.Columns);

            var grid = new Grid();

            // Setup grid rows and columns
            for (int i = 0; i < rows; i++)
            {
                grid.RowDefinitions.Add(new RowDefinition());
            }
            for (int j = 0; j < columns; j++)
            {
                grid.ColumnDefinitions.Add(new ColumnDefinition());
            }

            if (!string.IsNullOrEmpty(controlsConfig))
            {
                var controls = _mainWindow.DeserializeControls(controlsConfig);

                // Add controls to grid
                foreach (var control in controls)
                {
                    // Get row/column/span values ensuring they're within grid bounds
                    int row = Math.Min(control.Item2, rows - 1);
                    int column = Math.Min(control.Item3, columns - 1);
                    int rowSpan = Math.Min(control.Item4, rows - row);
                    int columnSpan = Math.Min(control.Item5, columns - column);

                    // Create draggable border for the control
                    var borderedControl = _mainWindow.CreateDraggableBorder(control.Item1, row, column, rowSpan, columnSpan, tabName);

                    // Set grid placement
                    Grid.SetRow(borderedControl, row);
                    Grid.SetColumn(borderedControl, column);
                    Grid.SetRowSpan(borderedControl, rowSpan);
                    Grid.SetColumnSpan(borderedControl, columnSpan);

                    grid.Children.Add(borderedControl);
                }
            }

            var tabItem = _tabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == tabName);
            if (tabItem != null) tabItem.Content = grid;

            // Make the grid support direct drag-and-drop
            if (grid != null)
            {
                _mainWindow.MakeGridDraggable(grid, tabName);
            }
        }

        /// <summary>
        /// Loads controls for the current tab without triggering layout updates
        /// </summary>
        /// <param name="tabName">Name of the tab to load controls for</param>
        public void LoadTabControlsWithoutLayout(string tabName)
        {
            try
            {
                _isTabSelectionInProgress = true;
                LoadTabControls(tabName);
            }
            finally
            {
                _isTabSelectionInProgress = false;
            }
        }

        /// <summary>
        /// Forces a refresh of the controls on a specific tab
        /// </summary>
        /// <param name="tabName">Name of the tab to refresh</param>
        public void RefreshTabControls(string tabName)
        {
            try
            {
                // Find the tab
                var tabItem = _tabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == tabName);
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
                                        DatabaseMonolith.Log("Info", $"Force updated PredictionAnalysisControl layout in tab '{tabName}'");
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
        /// Checks if a cell is occupied in a specific tab
        /// </summary>
        /// <returns>True if the cell is occupied, false otherwise</returns>
        public bool IsCellOccupied(string tabName, int row, int column, int rowSpan = 1, int columnSpan = 1)
        {
            var tab = _tabControl.Items
                .OfType<TabItem>()
                .FirstOrDefault(t => t.Header.ToString() == tabName);

            if (tab == null || !(tab.Content is Grid grid))
                return false;

            return IsOverlapping(grid, row, column, rowSpan, columnSpan);
        }

        /// <summary>
        /// Gets a list of all occupied cells in a specific tab
        /// </summary>
        /// <returns>A list of occupied cell coordinates</returns>
        public List<(int Row, int Col)> GetOccupiedCellsForTab(string tabName)
        {
            var tab = _tabControl.Items
                .OfType<TabItem>()
                .FirstOrDefault(t => t.Header.ToString() == tabName);

            if (tab == null || !(tab.Content is Grid grid))
                return new List<(int Row, int Col)>();

            return GetOccupiedCells(grid);
        }

        #endregion

        #region Private Helper Methods

        private bool IsOverlapping(Grid grid, int newRow, int newColumn, int newRowSpan, int newColumnSpan)
        {
            // Check each child in the grid to see if it would overlap with the new control
            foreach (UIElement child in grid.Children)
            {
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
                    return true; // Overlap detected
                }
            }

            return false; // No overlap
        }

        private List<(int Row, int Col)> GetOccupiedCells(Grid grid)
        {
            var occupiedCells = new List<(int Row, int Col)>();

            foreach (UIElement child in grid.Children)
            {
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

        #endregion

        #region Event Handlers

        private void TabControl_PreviewMouseMove(object sender, MouseEventArgs e)
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

        private void TabControl_Drop(object sender, DragEventArgs e)
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
                    using (var connection = DatabaseMonolith.GetConnection())
                    {
                        connection.Open();
                        for (int i = 0; i < _tabControl.Items.Count - 1; i++)
                        {
                            if (_tabControl.Items[i] is TabItem item && item.Header is string tabName)
                            {
                                var updateQuery = "UPDATE UserAppSettings SET TabOrder = @TabOrder WHERE TabName = @TabName";
                                connection.Execute(updateQuery, new { TabOrder = i, TabName = tabName });
                            }
                        }
                    }
                }
            }
        }

        #endregion
    }
}
