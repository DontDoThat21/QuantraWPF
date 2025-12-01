using Quantra.DAL.Services.Interfaces;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using static Quantra.ResizableBorder;
using System.Text.Json;
using Quantra.DAL.Services;
using Quantra.ViewModels;
using System.ComponentModel;
using Microsoft.EntityFrameworkCore;

namespace Quantra
{
    public partial class LoginWindow : Window
    {
        private readonly LoginWindowViewModel _viewModel;
        private bool _isRegistrationMode = false;

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
            _viewModel.RegistrationSuccessful += OnRegistrationSuccessful;
            _viewModel.RegistrationFailed += OnRegistrationFailed;
            _viewModel.SettingsRequested += OnSettingsRequested;
            
            // Subscribe to PropertyChanged to monitor state changes
            _viewModel.PropertyChanged += OnViewModelPropertyChanged;
        }

        /// <summary>
        /// Legacy constructor for compatibility - creates ViewModel internally
        /// NOTE: This constructor creates a temporary AuthenticationService for backward compatibility.
        /// Prefer using the constructor that accepts LoginWindowViewModel.
        /// </summary>
        public LoginWindow(UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService)
            : this(CreateViewModelWithInternalAuth(userSettingsService, historicalDataService, alphaVantageService, technicalIndicatorService))
        {
        }

        /// <summary>
        /// Creates a LoginWindowViewModel with an internally created AuthenticationService
        /// </summary>
        private static LoginWindowViewModel CreateViewModelWithInternalAuth(
            UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService)
        {
            var optionsBuilder = new Microsoft.EntityFrameworkCore.DbContextOptionsBuilder<Quantra.DAL.Data.QuantraDbContext>();
            optionsBuilder.UseSqlServer(Quantra.DAL.Data.ConnectionHelper.ConnectionString);
            var dbContext = new Quantra.DAL.Data.QuantraDbContext(optionsBuilder.Options);
            var loggingService = new LoggingService();
            var authService = new AuthenticationService(dbContext, loggingService);
            
            return new LoginWindowViewModel(userSettingsService, historicalDataService, alphaVantageService, technicalIndicatorService, authService);
        }

        private void OnViewModelPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(LoginWindowViewModel.IsLoggingIn))
            {
                // Set cursor based on IsLoggingIn state
                this.Cursor = _viewModel.IsLoggingIn ? Cursors.Wait : Cursors.Arrow;
            }
            else if (e.PropertyName == nameof(LoginWindowViewModel.IsRegistrationMode))
            {
                UpdateUIForMode(_viewModel.IsRegistrationMode);
            }
            else if (e.PropertyName == nameof(LoginWindowViewModel.StatusMessage))
            {
                // Update status message visibility
                if (!string.IsNullOrEmpty(_viewModel.StatusMessage))
                {
                    StatusMessageText.Visibility = Visibility.Visible;
                }
                else
                {
                    StatusMessageText.Visibility = Visibility.Collapsed;
                }
            }
        }

        private void UpdateUIForMode(bool isRegistrationMode)
        {
            _isRegistrationMode = isRegistrationMode;
            
            if (isRegistrationMode)
            {
                // Switch to registration mode
                FormTitleText.Text = "Create Account";
                TitleBar.Title = "Register";
                LoginButton.Visibility = Visibility.Collapsed;
                RegisterButton.Visibility = Visibility.Visible;
                ConfirmPasswordContainer.Visibility = Visibility.Visible;
                EmailContainer.Visibility = Visibility.Visible;
                AccountSelectionContainer.Visibility = Visibility.Collapsed;
                RememberMeCheckBox.Visibility = Visibility.Collapsed;
                PinTextBoxContainer.Visibility = Visibility.Collapsed;
                ToggleModeButton.Content = "Back to Login";
            }
            else
            {
                // Switch to login mode
                FormTitleText.Text = "Quantra Login";
                TitleBar.Title = "Login";
                LoginButton.Visibility = Visibility.Visible;
                RegisterButton.Visibility = Visibility.Collapsed;
                ConfirmPasswordContainer.Visibility = Visibility.Collapsed;
                EmailContainer.Visibility = Visibility.Collapsed;
                AccountSelectionContainer.Visibility = Visibility.Visible;
                RememberMeCheckBox.Visibility = Visibility.Visible;
                ToggleModeButton.Content = "Create Account";
            }
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

        private void OnRegistrationSuccessful(object sender, string message)
        {
            MessageBox.Show(message, "Registration Successful", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void OnRegistrationFailed(object sender, string errorMessage)
        {
            MessageBox.Show(errorMessage, "Registration Error", MessageBoxButton.OK, MessageBoxImage.Error);
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
                _viewModel.RegistrationSuccessful -= OnRegistrationSuccessful;
                _viewModel.RegistrationFailed -= OnRegistrationFailed;
                _viewModel.SettingsRequested -= OnSettingsRequested;
                _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
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

        private void RegisterButton_Click(object sender, RoutedEventArgs e)
        {
            // Update ViewModel properties from UI (PasswordBox can't bind directly)
            if (_viewModel != null)
            {
                _viewModel.Username = UsernameTextBox.Text;
                _viewModel.Password = PasswordBox.Password;
                _viewModel.ConfirmPassword = ConfirmPasswordBox.Password;
                _viewModel.Email = EmailTextBox.Text;
                
                // Execute register command
                if (_viewModel.RegisterCommand.CanExecute(null))
                {
                    _viewModel.RegisterCommand.Execute(null);
                }
            }
        }

        private void ToggleModeButton_Click(object sender, RoutedEventArgs e)
        {
            if (_viewModel != null)
            {
                _viewModel.ToggleRegistrationModeCommand?.Execute(null);
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

        #endregion
    }
}