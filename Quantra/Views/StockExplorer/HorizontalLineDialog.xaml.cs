using System;
using System.Windows;

namespace Quantra.Views.StockExplorer
{
    public partial class HorizontalLineDialog : Window
    {
        public double PriceLevel { get; private set; }
        public string Label { get; private set; }

        public HorizontalLineDialog()
        {
            InitializeComponent();
            PriceLevelTextBox.Focus();
        }

        private void AddButton_Click(object sender, RoutedEventArgs e)
        {
            if (double.TryParse(PriceLevelTextBox.Text, out double price))
            {
                PriceLevel = price;
                Label = string.IsNullOrWhiteSpace(LabelTextBox.Text) 
                    ? $"Level {price:F2}" 
                    : LabelTextBox.Text;
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Please enter a valid price level.", "Invalid Input", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
