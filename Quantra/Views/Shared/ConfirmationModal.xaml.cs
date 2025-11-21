using System;
using System.Windows;
using System.Windows.Input;

namespace Quantra.Views.Shared
{
    /// <summary>
    /// Interaction logic for ConfirmationModal.xaml
    /// </summary>
    public partial class ConfirmationModal : Window
    {
        private bool result = false;

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public ConfirmationModal()
        {
            InitializeComponent();
            result = false;
        }

        /// <summary>
        /// Initializes a new instance of the ConfirmationModal class.
        /// </summary>
        /// <param name="message">The message to display</param>
        /// <param name="title">The title of the dialog</param>
        public ConfirmationModal(string message, string title = "Confirmation")
        {
            InitializeComponent();
            
            // Set window properties
            TitleTextBlock.Text = title;
            MessageTextBlock.Text = message;
            
            // Set focus to the No button by default (safer option)
            this.Loaded += (s, e) => NoButton.Focus();
        }
        
        /// <summary>
        /// Shows a confirmation dialog with the specified message and title.
        /// </summary>
        /// <param name="message">The message to display</param>
        /// <param name="title">The title of the dialog</param>
        /// <param name="owner">The owner window</param>
        /// <returns>True if the user clicked Yes, false otherwise</returns>
        public static bool Show(string message, string title = "Confirmation", Window owner = null)
        {
            var dialog = new ConfirmationModal(message, title);
            
            if (owner != null)
            {
                dialog.Owner = owner;
            }
            else if (Application.Current.MainWindow != null)
            {
                dialog.Owner = Application.Current.MainWindow;
            }
            
            dialog.ShowDialog();
            return dialog.result;
        }

        private void YesButton_Click(object sender, RoutedEventArgs e)
        {
            result = true;
            Close();
        }

        private void NoButton_Click(object sender, RoutedEventArgs e)
        {
            result = false;
            Close();
        }
        
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            result = false;
            Close();
        }
        
        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                result = false;
                Close();
                e.Handled = true;
            }
            else if (e.Key == Key.Enter)
            {
                // If focus is on one of the buttons, click that button
                if (FocusManager.GetFocusedElement(this) == YesButton)
                {
                    YesButton_Click(YesButton, new RoutedEventArgs());
                    e.Handled = true;
                }
                else if (FocusManager.GetFocusedElement(this) == NoButton)
                {
                    NoButton_Click(NoButton, new RoutedEventArgs());
                    e.Handled = true;
                }
                else
                {
                    // If focus isn't on a button, default to Yes
                    YesButton_Click(YesButton, new RoutedEventArgs());
                    e.Handled = true;
                }
            }
        }
    }
}
