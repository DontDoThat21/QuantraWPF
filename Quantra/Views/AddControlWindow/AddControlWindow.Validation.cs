using System;
using System.Windows;
using System.Windows.Controls;

namespace Quantra
{
    /// <summary>
    /// Partial class for validation functionality
    /// </summary>
    public partial class AddControlWindow : Window
    {
        private void ValidatePosition(object sender, TextChangedEventArgs e)
        {
            UpdatePositionValidation();
            UpdateGridVisualization();
        }

        private void UpdatePositionValidation()
        {
            // First check if controls are initialized
            if (RowTextBox == null || ColumnTextBox == null ||
                RowSpanTextBox == null || ColumnSpanTextBox == null ||
                AddButton == null || OverlapWarningTextBlock == null)
            {
                // Controls not yet initialized, exit gracefully
                return;
            }

            if (!int.TryParse(RowTextBox.Text, out int row) ||
                !int.TryParse(ColumnTextBox.Text, out int column) ||
                !int.TryParse(RowSpanTextBox.Text, out int rowSpan) ||
                !int.TryParse(ColumnSpanTextBox.Text, out int columnSpan))
            {
                // Invalid number format - keep the button enabled but we'll validate further on click
                AddButton.IsEnabled = true;
                OverlapWarningTextBlock.Visibility = Visibility.Collapsed;
                return;
            }

            // Convert from 1-based to 0-based if needed for validation
            if (useOneBased)
            {
                row--;
                column--;
            }

            // Check for overlap with any existing controls on the selected tab
            var selectedTab = TabComboBox?.SelectedItem as string;
            if (selectedTab != null)
            {
                var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                if (mainWindow != null && mainWindow.IsCellOccupied(selectedTab, row, column, rowSpan, columnSpan))
                {
                    // Position would overlap with existing control - disable add button and show warning
                    AddButton.IsEnabled = false;
                    OverlapWarningTextBlock.Text = "This position would overlap with an existing control!";
                    OverlapWarningTextBlock.Visibility = Visibility.Visible;
                }
                else
                {
                    // Position is valid - enable add button and hide warning
                    AddButton.IsEnabled = true;
                    OverlapWarningTextBlock.Visibility = Visibility.Collapsed;
                }
            }
        }
    }
}
