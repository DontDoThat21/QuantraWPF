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
using Quantra.ViewModels;

namespace Quantra
{
    public partial class LoginWindow : Window
    {
        private readonly LoginWindowViewModel _viewModel;

        // Parameterless constructor for XAML designer support
        public LoginWindow()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public LoginWindow(LoginWindowViewModel viewModel)
        {
            InitializeComponent();
            
            _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
            DataContext = _viewModel;
            
            // Subscribe to ViewModel events
            _viewModel.LoginSuccessful += OnLoginSuccessful;
            _viewModel.LoginFailed += OnLoginFailed;
            _viewModel.SettingsRequested += OnSettingsRequested;
        }

        /// <summary>
        /// Legacy constructor for compatibility - creates ViewModel internally
        /// </summary>
        public LoginWindow(UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService)
            : this(new LoginWindowViewModel(userSettingsService, historicalDataService, alphaVantageService, technicalIndicatorService))
        {
        }

        private void OnLoginSuccessful(object sender, LoginSuccessEventArgs e)
        {
            var mainWindow = new MainWindow(
                e.UserSettingsService,
                e.HistoricalDataService,
                e.AlphaVantageService,
                e.TechnicalIndicatorService);

            if (e.SavedWindowState.HasValue)
            {
                mainWindow.WindowState = e.SavedWindowState.Value;
            }

            mainWindow.Show();
            this.Close();
        }

        private void OnLoginFailed(object sender, string errorMessage)
        {
            MessageBox.Show(errorMessage, "Login Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        private void OnSettingsRequested(object sender, EventArgs e)
        {
            var settingsWindow = new SettingsWindow();
            settingsWindow.Show();
        }

        protected override void OnClosed(EventArgs e)
        {
            // Unsubscribe from events to prevent memory leaks
            if (_viewModel != null)
            {
                _viewModel.LoginSuccessful -= OnLoginSuccessful;
                _viewModel.LoginFailed -= OnLoginFailed;
                _viewModel.SettingsRequested -= OnSettingsRequested;
            }
            base.OnClosed(e);
        }

        #region UI Event Handlers (XAML-referenced)

        private void LoginButton_Click(object sender, RoutedEventArgs e)
        {
            // Update ViewModel properties from UI (PasswordBox can't bind directly)
            if (_viewModel != null)
            {
                _viewModel.Username = UsernameTextBox.Text;
                _viewModel.Password = PasswordBox.Password;
                _viewModel.Pin = PinTextBox.Text;
                _viewModel.RememberMe = RememberMeCheckBox.IsChecked == true;
                
                // Execute login command
                if (_viewModel.LoginCommand.CanExecute(null))
                {
                    _viewModel.LoginCommand.Execute(null);
                }
            }
        }

        private void SettingsButton_Click(object sender, RoutedEventArgs e)
        {
            _viewModel?.OpenSettingsCommand?.Execute(null);
        }

        private void AccountComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (_viewModel != null && AccountComboBox.SelectedItem != null)
            {
                _viewModel.SelectedAccount = AccountComboBox.SelectedItem.ToString();
                
                // Update UI from ViewModel
                UsernameTextBox.Text = _viewModel.Username;
                PasswordBox.Password = _viewModel.Password;
                PinTextBox.Text = _viewModel.Pin;
                PinTextBoxContainer.Visibility = _viewModel.IsPinVisible ? Visibility.Visible : Visibility.Collapsed;
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

