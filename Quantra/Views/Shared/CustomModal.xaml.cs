using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace Quantra.Views.Shared
{
    /// <summary>
    /// Message types for the custom modal
    /// </summary>
    public enum CustomModalType
    {
        Information,
        Warning,
        Error
    }

    /// <summary>
    /// Interaction logic for CustomModal.xaml
    /// A reusable modal for displaying Information, Warning, and Error messages
    /// </summary>
    public partial class CustomModal : Window
    {
        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public CustomModal()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Initializes a new instance of the CustomModal class.
        /// </summary>
        /// <param name="message">The message to display</param>
        /// <param name="title">The title of the dialog</param>
        /// <param name="messageType">The type of message (Information, Warning, Error)</param>
        public CustomModal(string message, string title = null, CustomModalType messageType = CustomModalType.Information)
        {
            InitializeComponent();
            
            // Set window properties based on message type
            SetMessageTypeProperties(messageType, title);
            MessageTextBlock.Text = message;
            
            // Set focus to the OK button
            this.Loaded += (s, e) => OkButton.Focus();
        }
        
        /// <summary>
        /// Shows a custom modal with the specified message and type.
        /// </summary>
        /// <param name="message">The message to display</param>
        /// <param name="title">The title of the dialog (optional, will be set based on type if not provided)</param>
        /// <param name="messageType">The type of message</param>
        /// <param name="owner">The owner window</param>
        public static void Show(string message, string title = null, CustomModalType messageType = CustomModalType.Information, Window owner = null)
        {
            var dialog = new CustomModal(message, title, messageType);
            
            if (owner != null)
            {
                dialog.Owner = owner;
            }
            else if (Application.Current.MainWindow != null)
            {
                dialog.Owner = Application.Current.MainWindow;
            }
            
            dialog.ShowDialog();
        }

        /// <summary>
        /// Convenience method for showing error messages
        /// </summary>
        public static void ShowError(string message, string title = "Error", Window owner = null)
        {
            Show(message, title, CustomModalType.Error, owner);
        }

        /// <summary>
        /// Convenience method for showing warning messages
        /// </summary>
        public static void ShowWarning(string message, string title = "Warning", Window owner = null)
        {
            Show(message, title, CustomModalType.Warning, owner);
        }

        /// <summary>
        /// Convenience method for showing information messages
        /// </summary>
        public static void ShowInformation(string message, string title = "Information", Window owner = null)
        {
            Show(message, title, CustomModalType.Information, owner);
        }

        private void SetMessageTypeProperties(CustomModalType messageType, string customTitle)
        {
            switch (messageType)
            {
                case CustomModalType.Error:
                    TitleTextBlock.Text = customTitle ?? "Error";
                    TitleTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0x4D, 0x4D)); // Red
                    IconTextBlock.Text = "⚠";
                    IconTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0x4D, 0x4D)); // Red
                    OkButton.Background = new SolidColorBrush(Color.FromRgb(0xFF, 0x4D, 0x4D)); // Red background for error
                    break;
                    
                case CustomModalType.Warning:
                    TitleTextBlock.Text = customTitle ?? "Warning";
                    TitleTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0xA5, 0x00)); // Orange
                    IconTextBlock.Text = "⚠";
                    IconTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0xA5, 0x00)); // Orange
                    OkButton.Background = new SolidColorBrush(Color.FromRgb(0xFF, 0xA5, 0x00)); // Orange background for warning
                    break;
                    
                case CustomModalType.Information:
                default:
                    TitleTextBlock.Text = customTitle ?? "Information";
                    TitleTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(0x1E, 0x90, 0xFF)); // Blue
                    IconTextBlock.Text = "ℹ";
                    IconTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(0x1E, 0x90, 0xFF)); // Blue
                    OkButton.Background = new SolidColorBrush(Color.FromRgb(0x3A, 0x6E, 0xA5)); // Blue background for info
                    break;
            }
        }

        private void OkButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
        
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
        
        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape || e.Key == Key.Enter)
            {
                Close();
                e.Handled = true;
            }
        }
    }
}