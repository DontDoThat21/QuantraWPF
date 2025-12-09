using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;

namespace Quantra.Views.StockExplorer
{
    public partial class RefreshIntervalDialog : Window
    {
        public int SelectedInterval { get; private set; }
        public bool SaveAsFavorite { get; private set; }

        public RefreshIntervalDialog(int currentInterval)
        {
            InitializeComponent();
            
            SelectedInterval = currentInterval;
            
            // Select the radio button matching the current interval
            foreach (RadioButton radio in IntervalOptionsPanel.Children.OfType<RadioButton>())
            {
                if (int.TryParse(radio.Tag?.ToString(), out int interval) && interval == currentInterval)
                {
                    radio.IsChecked = true;
                    break;
                }
            }
        }

        private void IntervalRadioButton_Checked(object sender, RoutedEventArgs e)
        {
            if (sender is RadioButton radio && int.TryParse(radio.Tag?.ToString(), out int interval))
            {
                SelectedInterval = interval;
            }
        }

        private void ApplyButton_Click(object sender, RoutedEventArgs e)
        {
            SaveAsFavorite = SaveAsFavoriteCheckBox?.IsChecked ?? false;
            DialogResult = true;
            Close();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
