// Handles DataGrid and stock grid logic for StockExplorer
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media; // Added for VisualTreeHelper
using Quantra.Models;
using Quantra.DAL.Services.Interfaces; // Add this for StockDataCacheService
using Quantra.Utilities; // Added for VisualTreeHelperExtensions
using Quantra.Views.Shared; // Added for ConfirmationModal
using Quantra.ViewModels; // Added for StockExplorerViewModel
using Quantra; // Added for QuoteData class
using System.Windows.Threading;
using Quantra.DAL.Services; // Added for DispatcherTimer and StockSymbolCacheService
using Microsoft.Extensions.DependencyInjection; // Added for dependency injection

namespace Quantra.Controls
{
    public partial class StockExplorer
    {
        // DataGrid and stock grid-related fields, properties, and methods
        private DispatcherTimer _saveTimer;
        private bool _isLoadingSettings = false;
        private IUserSettingsService _userSettingsService;

        // Initialize DataGrid settings
        private void InitializeDataGridSettings()
        {
            // _userSettingsService is already initialized in the constructor
            // If for some reason it wasn't, get it from the service provider
            if (_userSettingsService == null)
            {
                _userSettingsService = App.ServiceProvider.GetRequiredService<IUserSettingsService>();
            }

            // Initialize save timer for delayed saving
            _saveTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(500) // Save 500ms after last change
            };
            _saveTimer.Tick += SaveTimer_Tick;

            // Wire up column width change events using DependencyPropertyDescriptor
            if (StockDataGrid != null)
            {
                foreach (var column in StockDataGrid.Columns)
                {
                    var dpd = System.ComponentModel.DependencyPropertyDescriptor.FromProperty(DataGridColumn.WidthProperty, typeof(DataGridColumn));
                    if (dpd != null)
                    {
                        dpd.AddValueChanged(column, Column_WidthChanged);
                    }
                }
            }
        }

        // Load saved DataGrid settings
        private void LoadDataGridSettings()
        {
            try
            {
                _isLoadingSettings = true;
                
                var tabName = GetTabName();
                if (string.IsNullOrEmpty(tabName))
                    return;

                var settings = _userSettingsService.LoadDataGridConfig(tabName, "StockDataGrid");
                
                // Apply DataGrid size if saved
                if (!double.IsNaN(settings.DataGridWidth) && settings.DataGridWidth > 0)
                {
                    StockDataGrid.Width = settings.DataGridWidth;
                }
                if (!double.IsNaN(settings.DataGridHeight) && settings.DataGridHeight > 0)
                {
                    StockDataGrid.Height = settings.DataGridHeight;
                }

                // Apply column widths if saved
                if (settings.ColumnWidths != null && settings.ColumnWidths.Count > 0)
                {
                    foreach (var column in StockDataGrid.Columns)
                    {
                        if (column.Header != null && settings.ColumnWidths.ContainsKey(column.Header.ToString()))
                        {
                            var savedWidth = settings.ColumnWidths[column.Header.ToString()];
                            if (savedWidth > 0)
                            {
                                column.Width = new DataGridLength(savedWidth);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //_loggingService.Log("Warning", "Failed to load DataGrid settings", ex.ToString());
            }
            finally
            {
                _isLoadingSettings = false;
            }
        }

        // Save DataGrid settings
        private void SaveDataGridSettings()
        {
            if (_isLoadingSettings)
                return;

            try
            {
                var tabName = GetTabName();
                if (string.IsNullOrEmpty(tabName))
                    return;

                var settings = new DataGridSettings
                {
                    DataGridWidth = StockDataGrid.ActualWidth,
                    DataGridHeight = StockDataGrid.ActualHeight
                };

                // Save column widths
                foreach (var column in StockDataGrid.Columns)
                {
                    if (column.Header != null && column.ActualWidth > 0)
                    {
                        settings.ColumnWidths[column.Header.ToString()] = column.ActualWidth;
                    }
                }

                _userSettingsService.SaveDataGridConfig(tabName, "StockDataGrid", settings);
            }
            catch (Exception ex)
            {
                //_loggingService.Log("Warning", "Failed to save DataGrid settings", ex.ToString());
            }
        }

        // Get the tab name that contains this StockExplorer
        private string GetTabName()
        {
            try
            {
                // Walk up the visual tree to find the tab name
                DependencyObject parent = this;
                while (parent != null)
                {
                    parent = VisualTreeHelper.GetParent(parent);
                    if (parent is TabItem tabItem && tabItem.Header != null)
                    {
                        return tabItem.Header.ToString();
                    }
                }

                // Fallback: try to get from main window if available
                if (Application.Current?.MainWindow != null)
                {
                    // Generate a unique identifier based on stable instance ID
                    return $"StockExplorer_{_instanceId}";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Failed to get tab name for DataGrid settings", ex.ToString());
            }

            return $"StockExplorer_{_instanceId}"; // Stable fallback per instance
        }

        // Event handlers
        private void Column_WidthChanged(object sender, EventArgs e)
        {
            if (!_isLoadingSettings)
            {
                _saveTimer.Stop();
                _saveTimer.Start();
            }
        }

        private void StockDataGrid_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (!_isLoadingSettings)
            {
                // Reset timer to delay saving
                _saveTimer.Stop();
                _saveTimer.Start();
            }
        }

        private void SaveTimer_Tick(object sender, EventArgs e)
        {
            _saveTimer.Stop();
            SaveDataGridSettings();
        }

        private void StockDataGrid_PreviewMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Prevent parent controls from handling the right-click event
            e.Handled = true;
        }

        private void StockDataGrid_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            try
            {
                // Prevent event from bubbling to parent controls immediately
                e.Handled = true;
                
                var hitTestResult = VisualTreeHelper.HitTest(StockDataGrid, e.GetPosition(StockDataGrid));
                if (hitTestResult?.VisualHit == null) 
                {
                    // Clear context menu if right-clicking empty space
                    StockDataGrid.ContextMenu = null;
                    return;
                }
                
                var row = VisualTreeHelperExtensions.GetParentOfType<DataGridRow>(hitTestResult.VisualHit);

                if (row != null && row.Item is QuoteData quoteData)
                {
                    // Select the row without triggering SelectionChanged event
                    _isHandlingSelectionChanged = true;
                    try
                    {
                        StockDataGrid.SelectedItem = row.Item;
                    }
                    finally
                    {
                        _isHandlingSelectionChanged = false;
                    }
                    
                    // Create context menu
                    var contextMenu = new ContextMenu
                    {
                        PlacementTarget = StockDataGrid,
                        Placement = System.Windows.Controls.Primitives.PlacementMode.MousePoint
                    };
                    
                    // Apply enhanced styling
                    contextMenu.Style = (Style)Application.Current.FindResource("EnhancedContextMenuStyle");
                    
                    // Delete Cache option (existing)
                    var deleteCacheMenuItem = new MenuItem { Header = $"Delete Cache for {quoteData.Symbol}" };
                    deleteCacheMenuItem.Style = (Style)Application.Current.FindResource("EnhancedMenuItemStyle");
                    deleteCacheMenuItem.Click += (s, args) => DeleteSymbolCache_Click(quoteData.Symbol);
                    contextMenu.Items.Add(deleteCacheMenuItem);
                    
                    // Add separator
                    var separator = new Separator();
                    separator.Style = (Style)Application.Current.FindResource("EnhancedSeparatorStyle");
                    contextMenu.Items.Add(separator);
                    
                    // Delete Entry option (new)
                    var deleteEntryMenuItem = new MenuItem { Header = $"Delete Entry for {quoteData.Symbol}" };
                    deleteEntryMenuItem.Style = (Style)Application.Current.FindResource("EnhancedMenuItemStyle");
                    deleteEntryMenuItem.Click += (s, args) => DeleteStockEntry_Click(quoteData.Symbol);
                    contextMenu.Items.Add(deleteEntryMenuItem);
                    
                    // Set and open context menu
                    StockDataGrid.ContextMenu = contextMenu;
                    contextMenu.IsOpen = true;
                }
                else
                {
                    // Clear context menu if right-clicking empty space
                    StockDataGrid.ContextMenu = null;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error in StockDataGrid_MouseRightButtonUp", ex.ToString());
                // Clear context menu on error
                StockDataGrid.ContextMenu = null;
            }
        }

        private void DeleteSymbolCache_Click(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                (Application.Current.MainWindow as MainWindow)?.AppendAlert("Symbol is invalid.", "warning");
                return;
            }

            var confirmed = ConfirmationModal.Show(
                $"Are you sure you want to delete all cached data for {symbol}?",
                "Confirm Cache Deletion",
                Application.Current.MainWindow);

            if (confirmed)
            {
                // Use the existing _cacheService instance
                int deletedEntries = _cacheService.DeleteCachedDataForSymbol(symbol);
                (Application.Current.MainWindow as MainWindow)?.AppendAlert($"Deleted {deletedEntries} cache entries for {symbol}.", "positive");
                //DatabaseMonolith.Log("Info", $"User deleted cache for symbol: {symbol}. Entries removed: {deletedEntries}.");
                
                // Optionally, refresh the grid or the specific row if needed
                // RefreshDataGrid(); 
            }
        }

        private void DeleteStockEntry_Click(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                (Application.Current.MainWindow as MainWindow)?.AppendAlert("Symbol is invalid.", "warning");
                return;
            }

            // Use ConfirmationModal instead of MessageBox
            var confirmed = ConfirmationModal.Show(
                $"Are you sure you want to permanently delete the stock entry for {symbol}?\n\nThis will remove the symbol from the database entirely, including all cached data.",
                "Confirm Stock Entry Deletion",
                Application.Current.MainWindow);

            if (confirmed)
            {
                // Delete from database using Entity Framework Core service
                bool success = _stockSymbolCacheService.DeleteStockSymbol(symbol);
                
                if (success)
                {
                    // Remove from the UI collection
                    if (DataContext is StockExplorerViewModel viewModel)
                    {
                        var itemToRemove = viewModel.CachedStocks.FirstOrDefault(s => s.Symbol.Equals(symbol, StringComparison.InvariantCultureIgnoreCase));
                        if (itemToRemove != null)
                        {
                            // Dispose the QuoteData object to free resources
                            itemToRemove.Dispose();
                            
                            // Remove from the ObservableCollection (this should trigger UI update)
                            viewModel.CachedStocks.Remove(itemToRemove);
                            
                            // Force DataGrid refresh to ensure the row is visually removed
                            StockDataGrid?.Items.Refresh();
                        }
                    }
                    
                    (Application.Current.MainWindow as MainWindow)?.AppendAlert($"Successfully deleted stock entry for {symbol}.", "positive");
                    //DatabaseMonolith.Log("Info", $"User deleted stock entry for symbol: {symbol}.");
                }
                else
                {
                    (Application.Current.MainWindow as MainWindow)?.AppendAlert($"Failed to delete stock entry for {symbol}.", "warning");
                }
            }
        }
    }
}
