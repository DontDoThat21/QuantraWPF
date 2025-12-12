using System.Windows;

namespace Quantra.Controls
{
    /// <summary>
    /// Dialog for entering filter name and description
    /// </summary>
    public partial class FilterNameDialog : Window
    {
        public string FilterName => FilterNameTextBox.Text.Trim();
        public string? FilterDescription => string.IsNullOrWhiteSpace(DescriptionTextBox.Text)
            ? null
            : DescriptionTextBox.Text.Trim();

        public FilterNameDialog()
        {
            InitializeComponent();
            FilterNameTextBox.Focus();
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(FilterNameTextBox.Text))
            {
                MessageBox.Show("Please enter a name for the filter.", "Name Required",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                FilterNameTextBox.Focus();
                return;
            }

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
