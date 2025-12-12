using System.Windows;
using System.Windows.Controls;

namespace Quantra.Controls
{
    /// <summary>
    /// Partial class for StockExplorer - Filter Event Handlers
    /// </summary>
    public partial class StockExplorer
    {
        /// <summary>
        /// Event handler when saved filter selection changes
        /// </summary>
        private void SavedFiltersComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (sender is ComboBox comboBox && comboBox.SelectedItem != null)
            {
                if (SelectedSavedFilter != null)
                {
                    ApplySavedFilter(SelectedSavedFilter);
                }
            }
        }

        /// <summary>
        /// Event handler for Save Filter button
        /// </summary>
        private async void SaveFilterButton_Click(object sender, RoutedEventArgs e)
        {
            await SaveCurrentFilterAsync();
        }

        /// <summary>
        /// Event handler for Delete Filter button
        /// </summary>
        private async void DeleteFilterButton_Click(object sender, RoutedEventArgs e)
        {
            await DeleteCurrentFilterAsync();
        }
    }
}
