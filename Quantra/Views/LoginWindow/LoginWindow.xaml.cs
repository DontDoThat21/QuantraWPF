using Quantra.DAL.Services.Interfaces;
using System.IO; // Add this for Path operations
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using static Quantra.ResizableBorder;
using System.Text.Json;
using Quantra.DAL.Services; // For JSON deserialization

namespace Quantra
{
    public partial class LoginWindow : Window
    {
        private Dictionary<string, (string Username, string Password, string Pin)> rememberedAccounts;

        public LoginWindow()
        {
            InitializeComponent();
            //DatabaseMonolith.EnsureDatabaseAndTables();
            LoadRememberedAccounts();
            //UsernameTextBox.Text = DictionaryEn.DefaultUsername; // Set default value programmatically
        }

        private void LoadRememberedAccounts()
        {
            try 
            {
                rememberedAccounts = DatabaseMonolith.GetRememberedAccounts();
                AccountComboBox.ItemsSource = rememberedAccounts.Keys.ToList();
            }
            catch (Exception ex)
            {
                // Handle exception gracefully - still allow login window to open
                MessageBox.Show($"Unable to load saved accounts: {ex.Message}", 
                    "Account Loading Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                
                // Initialize with empty dictionary to prevent null reference exceptions
                rememberedAccounts = new Dictionary<string, (string Username, string Password, string Pin)>();
            }
        }

        private async void LoginButton_Click(object sender, RoutedEventArgs e)
        {
            var username = UsernameTextBox.Text;
            var password = PasswordBox.Password;
            var pin = PinTextBox.Text;

            // Example: If you need to pass the API key to a service, use the new method
            string alphaVantageApiKey = AlphaVantageService.GetApiKey();

            var tradingBot = new WebullTradingBot(/* pass what is needed, remove configuration */);
            bool isAuthenticated = false;

            if (!string.IsNullOrEmpty(pin) && rememberedAccounts.Values.Any(a => a.Pin == pin))
            {
                var account = rememberedAccounts.Values.First(a => a.Pin == pin);
                isAuthenticated = await tradingBot.Authenticate(account.Username, account.Password);
            }
            else
            {
                isAuthenticated = await tradingBot.Authenticate(username, password);
            }

            if (isAuthenticated)
            {
                if (RememberMeCheckBox.IsChecked == true)
                {
                    DatabaseMonolith.RememberAccount(username, password, pin);
                }

                var mainWindow = new MainWindow();
                
                // Restore window state if enabled
                var savedWindowState = UserSettingsService.GetSavedWindowState();
                if (savedWindowState.HasValue)
                {
                    mainWindow.WindowState = savedWindowState.Value;
                }
                
                mainWindow.Show();
                this.Close();
            }
            else
            {
                MessageBox.Show("Authentication unsuccessful.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void SettingsButton_Click(object sender, RoutedEventArgs e)
        {
            var settingsWindow = new SettingsWindow();
            settingsWindow.Show();
        }

        private void AccountComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (AccountComboBox.SelectedItem != null)
            {
                var selectedAccount = AccountComboBox.SelectedItem.ToString();
                if (rememberedAccounts.ContainsKey(selectedAccount))
                {
                    var account = rememberedAccounts[selectedAccount];
                    UsernameTextBox.Text = account.Username;
                    PasswordBox.Password = account.Password;
                    PinTextBox.Text = account.Pin;
                    
                    // Use the container for better visibility handling
                    PinTextBoxContainer.Visibility = Visibility.Visible;
                }
            }
        }

        private void UsernameTextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            if (UsernameTextBox.Text == "Username")
            {
                UsernameTextBox.SelectAll();
            }
        }

        private void UsernameTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            //if (e.Key == Key.Back && UsernameTextBox.Text == DictionaryEn.DefaultUsername)
            //{
            //    UsernameTextBox.Clear();
            //    e.Handled = true;
            //}
        }
        
        // New resize grip functionality
        private void ResizeGrip_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left && e.ButtonState == MouseButtonState.Pressed)
            {
                if (sender == BottomRightGrip)
                {
                    this.ResizeMode = ResizeMode.CanResizeWithGrip;
                    this.Cursor = Cursors.SizeNWSE;
                    this.DragResize(ResizeDirection.BottomRight);
                }
                else if (sender == BottomLeftGrip)
                {
                    this.Cursor = Cursors.SizeNESW;
                    this.DragResize(ResizeDirection.BottomLeft);
                }
                else if (sender == BottomGrip)
                {
                    this.Cursor = Cursors.SizeNS;
                    this.DragResize(ResizeDirection.Bottom);
                }
            }
        }

        private void DragResize(ResizeDirection direction)
        {
            switch (direction)
            {
                case ResizeDirection.Bottom:
                    this.ResizeMode = ResizeMode.CanResize;
                    SendMessage(this, WM.SYSCOMMAND, (IntPtr)SC.SIZE + (IntPtr)ResizeDirection.Bottom, IntPtr.Zero);
                    break;
                case ResizeDirection.BottomLeft:
                    this.ResizeMode = ResizeMode.CanResize;
                    SendMessage(this, WM.SYSCOMMAND, (IntPtr)SC.SIZE + (IntPtr)ResizeDirection.BottomLeft, IntPtr.Zero);
                    break;
                case ResizeDirection.BottomRight:
                    this.ResizeMode = ResizeMode.CanResize;
                    SendMessage(this, WM.SYSCOMMAND, (IntPtr)SC.SIZE + (IntPtr)ResizeDirection.BottomRight, IntPtr.Zero);
                    break;
            }
        }        

        [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto)]
        private static extern IntPtr SendMessage(Window window, WM msg, IntPtr wParam, IntPtr lParam);
    }
}

