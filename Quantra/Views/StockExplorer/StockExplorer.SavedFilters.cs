using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Quantra.DAL.Data.Entities;

namespace Quantra.Controls
{
    /// <summary>
    /// Partial class for StockExplorer - Saved Filter Management
    /// </summary>
    public partial class StockExplorer
    {
        /// <summary>
        /// Load all saved filters from the database
        /// </summary>
        private async Task LoadSavedFiltersAsync()
        {
            if (_savedFilterService == null)
            {
                _loggingService?.Log("Warning", "SavedFilterService not initialized, cannot load filters");
                return;
            }

            try
            {
                // Get filters for current user (pass null for now, or get from user session if implemented)
                var filters = await _savedFilterService.GetAllFiltersAsync();

                // Update collection on UI thread
                await Dispatcher.InvokeAsync(() =>
                {
                    SavedFilters.Clear();

                    // Add a "None" option at the beginning
                    SavedFilters.Add(new SavedFilter
                    {
                        Id = -1,
                        Name = "-- No Filter --",
                        IsSystemFilter = true
                    });

                    foreach (var filter in filters)
                    {
                        SavedFilters.Add(filter);
                    }

                    // Select "None" by default
                    SelectedSavedFilter = SavedFilters.FirstOrDefault();
                });

                _loggingService?.Log("Info", $"Loaded {filters.Count} saved filters");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to load saved filters", ex.ToString());
            }
        }

        /// <summary>
        /// Apply a selected filter to the filter textboxes
        /// </summary>
        private void ApplySavedFilter(SavedFilter filter)
        {
            if (filter == null || filter.Id == -1)
            {
                // Clear all filters
                ClearAllFilters();
                return;
            }

            try
            {
                // Apply filter values to textboxes
                SymbolFilterText = filter.SymbolFilter ?? "";
                PriceFilterText = filter.PriceFilter ?? "";
                PeRatioFilterText = filter.PeRatioFilter ?? "";
                VwapFilterText = filter.VwapFilter ?? "";
                RsiFilterText = filter.RsiFilter ?? "";
                ChangePercentFilterText = filter.ChangePercentFilter ?? "";
                MarketCapFilterText = filter.MarketCapFilter ?? "";

                _loggingService?.Log("Info", $"Applied filter '{filter.Name}'");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to apply filter '{filter?.Name}'", ex.ToString());
            }
        }

        /// <summary>
        /// Clear all filter values
        /// </summary>
        private void ClearAllFilters()
        {
            SymbolFilterText = "";
            PriceFilterText = "";
            PeRatioFilterText = "";
            VwapFilterText = "";
            RsiFilterText = "";
            ChangePercentFilterText = "";
            MarketCapFilterText = "";
        }

        /// <summary>
        /// Save the current filter configuration
        /// </summary>
        private async Task SaveCurrentFilterAsync()
        {
            if (_savedFilterService == null)
            {
                MessageBox.Show("Filter service not available. Cannot save filter.", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // Check if any filters are set
            if (string.IsNullOrWhiteSpace(SymbolFilterText) &&
                string.IsNullOrWhiteSpace(PriceFilterText) &&
                string.IsNullOrWhiteSpace(PeRatioFilterText) &&
                string.IsNullOrWhiteSpace(VwapFilterText) &&
                string.IsNullOrWhiteSpace(RsiFilterText) &&
                string.IsNullOrWhiteSpace(ChangePercentFilterText) &&
                string.IsNullOrWhiteSpace(MarketCapFilterText))
            {
                MessageBox.Show("No filters are currently set. Please set at least one filter before saving.",
                    "No Filters", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            // Prompt for filter name
            var dialog = new FilterNameDialog();
            if (dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(dialog.FilterName))
            {
                var filterName = dialog.FilterName.Trim();

                // Check if name already exists
                if (await _savedFilterService.FilterNameExistsAsync(filterName))
                {
                    var result = MessageBox.Show(
                        $"A filter named '{filterName}' already exists. Do you want to overwrite it?",
                        "Filter Exists",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Question);

                    if (result != MessageBoxResult.Yes)
                        return;
                }

                // Create new filter
                var newFilter = new SavedFilter
                {
                    Name = filterName,
                    Description = dialog.FilterDescription,
                    UserId = null, // TODO: Set from current user session
                    IsSystemFilter = false,
                    SymbolFilter = string.IsNullOrWhiteSpace(SymbolFilterText) ? null : SymbolFilterText,
                    PriceFilter = string.IsNullOrWhiteSpace(PriceFilterText) ? null : PriceFilterText,
                    PeRatioFilter = string.IsNullOrWhiteSpace(PeRatioFilterText) ? null : PeRatioFilterText,
                    VwapFilter = string.IsNullOrWhiteSpace(VwapFilterText) ? null : VwapFilterText,
                    RsiFilter = string.IsNullOrWhiteSpace(RsiFilterText) ? null : RsiFilterText,
                    ChangePercentFilter = string.IsNullOrWhiteSpace(ChangePercentFilterText) ? null : ChangePercentFilterText,
                    MarketCapFilter = string.IsNullOrWhiteSpace(MarketCapFilterText) ? null : MarketCapFilterText
                };

                // Save filter
                var saved = await _savedFilterService.SaveFilterAsync(newFilter);
                if (saved != null)
                {
                    MessageBox.Show($"Filter '{filterName}' saved successfully!", "Success",
                        MessageBoxButton.OK, MessageBoxImage.Information);

                    // Reload filters
                    await LoadSavedFiltersAsync();

                    // Select the newly saved filter
                    SelectedSavedFilter = SavedFilters.FirstOrDefault(f => f.Id == saved.Id);
                }
                else
                {
                    MessageBox.Show("Failed to save filter. Please try again.", "Error",
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        /// <summary>
        /// Delete the currently selected filter
        /// </summary>
        private async Task DeleteCurrentFilterAsync()
        {
            if (SelectedSavedFilter == null || SelectedSavedFilter.Id == -1)
            {
                MessageBox.Show("Please select a filter to delete.", "No Filter Selected",
                    MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            if (SelectedSavedFilter.IsSystemFilter)
            {
                MessageBox.Show("System filters cannot be deleted.", "Cannot Delete",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var result = MessageBox.Show(
                $"Are you sure you want to delete the filter '{SelectedSavedFilter.Name}'?",
                "Confirm Delete",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question);

            if (result == MessageBoxResult.Yes)
            {
                var deleted = await _savedFilterService.DeleteFilterAsync(SelectedSavedFilter.Id);
                if (deleted)
                {
                    MessageBox.Show($"Filter '{SelectedSavedFilter.Name}' deleted successfully!", "Success",
                        MessageBoxButton.OK, MessageBoxImage.Information);

                    // Reload filters
                    await LoadSavedFiltersAsync();
                }
                else
                {
                    MessageBox.Show("Failed to delete filter. Please try again.", "Error",
                        MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }
    }
}
