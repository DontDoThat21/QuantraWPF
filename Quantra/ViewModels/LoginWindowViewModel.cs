using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Login Window with registration support
    /// </summary>
    public class LoginWindowViewModel : ViewModelBase
    {
        private readonly UserSettingsService _userSettingsService;
        private readonly HistoricalDataService _historicalDataService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private readonly AuthenticationService _authenticationService;
        private Dictionary<string, (string Username, string Password, string Pin)> _rememberedAccounts;

        private string _username;
        private string _password;
        private string _confirmPassword;
        private string _email;
        private string _pin;
        private bool _rememberMe;
        private string _selectedAccount;
        private bool _isPinVisible;
        private bool _isLoggingIn;
        private bool _isRegistrationMode;
        private string _statusMessage;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public LoginWindowViewModel(
            UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService,
            AuthenticationService authenticationService)
        {
            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            _historicalDataService = historicalDataService ?? throw new ArgumentNullException(nameof(historicalDataService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _technicalIndicatorService = technicalIndicatorService ?? throw new ArgumentNullException(nameof(technicalIndicatorService));
            _authenticationService = authenticationService ?? throw new ArgumentNullException(nameof(authenticationService));

            _rememberedAccounts = new Dictionary<string, (string Username, string Password, string Pin)>();
            RememberedAccountsList = new ObservableCollection<string>();
            PreviouslyLoggedInUsersList = new ObservableCollection<string>();

            InitializeCommands();
            LoadRememberedAccounts();
            _ = LoadPreviouslyLoggedInUsersAsync();
        }

        #region Properties

        /// <summary>
        /// Username for login
        /// </summary>
        public string Username
        {
            get => _username;
            set => SetProperty(ref _username, value);
        }

        /// <summary>
        /// Password for login
        /// </summary>
        public string Password
        {
            get => _password;
            set => SetProperty(ref _password, value);
        }

        /// <summary>
        /// Confirm password for registration
        /// </summary>
        public string ConfirmPassword
        {
            get => _confirmPassword;
            set => SetProperty(ref _confirmPassword, value);
        }

        /// <summary>
        /// Email for registration (optional)
        /// </summary>
        public string Email
        {
            get => _email;
            set => SetProperty(ref _email, value);
        }

        /// <summary>
        /// PIN for saved account
        /// </summary>
        public string Pin
        {
            get => _pin;
            set => SetProperty(ref _pin, value);
        }

        /// <summary>
        /// Remember credentials flag
        /// </summary>
        public bool RememberMe
        {
            get => _rememberMe;
            set => SetProperty(ref _rememberMe, value);
        }

        /// <summary>
        /// Selected saved account
        /// </summary>
        public string SelectedAccount
        {
            get => _selectedAccount;
            set
            {
                if (SetProperty(ref _selectedAccount, value))
                {
                    LoadSelectedAccount();
                }
            }
        }

        /// <summary>
        /// List of remembered account names
        /// </summary>
        public ObservableCollection<string> RememberedAccountsList { get; }

        /// <summary>
        /// List of previously logged-in users from the database
        /// </summary>
        public ObservableCollection<string> PreviouslyLoggedInUsersList { get; }

        /// <summary>
        /// Selected user from previously logged-in users dropdown
        /// </summary>
        private string _selectedPreviousUser;
        public string SelectedPreviousUser
        {
            get => _selectedPreviousUser;
            set
            {
                if (SetProperty(ref _selectedPreviousUser, value))
                {
                    LoadSelectedPreviousUser();
                }
            }
        }

        /// <summary>
        /// PIN visibility flag
        /// </summary>
        public bool IsPinVisible
        {
            get => _isPinVisible;
            set => SetProperty(ref _isPinVisible, value);
        }

        /// <summary>
        /// Login operation in progress flag
        /// </summary>
        public bool IsLoggingIn
        {
            get => _isLoggingIn;
            set => SetProperty(ref _isLoggingIn, value);
        }

        /// <summary>
        /// Whether the form is in registration mode
        /// </summary>
        public bool IsRegistrationMode
        {
            get => _isRegistrationMode;
            set
            {
                if (SetProperty(ref _isRegistrationMode, value))
                {
                    OnPropertyChanged(nameof(IsLoginMode));
                    StatusMessage = string.Empty;
                }
            }
        }

        /// <summary>
        /// Whether the form is in login mode (inverse of registration mode)
        /// </summary>
        public bool IsLoginMode => !IsRegistrationMode;

        /// <summary>
        /// Status message to display to user
        /// </summary>
        public string StatusMessage
        {
            get => _statusMessage;
            set => SetProperty(ref _statusMessage, value);
        }

        #endregion

        #region Commands

        public ICommand LoginCommand { get; private set; }
        public ICommand RegisterCommand { get; private set; }
        public ICommand ToggleRegistrationModeCommand { get; private set; }
        public ICommand OpenSettingsCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when login is successful
        /// </summary>
        public event EventHandler<LoginSuccessEventArgs> LoginSuccessful;

        /// <summary>
        /// Event fired when login fails
        /// </summary>
        public event EventHandler<string> LoginFailed;

        /// <summary>
        /// Event fired when registration succeeds
        /// </summary>
        public event EventHandler<string> RegistrationSuccessful;

        /// <summary>
        /// Event fired when registration fails
        /// </summary>
        public event EventHandler<string> RegistrationFailed;

        /// <summary>
        /// Event fired when settings should be opened
        /// </summary>
        public event EventHandler SettingsRequested;

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            LoginCommand = new RelayCommand(async param => await ExecuteLoginAsync(), CanExecuteLogin);
            RegisterCommand = new RelayCommand(async param => await ExecuteRegisterAsync(), CanExecuteRegister);
            ToggleRegistrationModeCommand = new RelayCommand(ExecuteToggleRegistrationMode);
            OpenSettingsCommand = new RelayCommand(ExecuteOpenSettings);
        }

        private void LoadRememberedAccounts()
        {
            try
            {
                _rememberedAccounts = DatabaseMonolith.GetRememberedAccounts();
                RememberedAccountsList.Clear();
                foreach (var accountName in _rememberedAccounts.Keys)
                {
                    RememberedAccountsList.Add(accountName);
                }
            }
            catch (Exception ex)
            {
                // Log error but don't prevent login window from opening
                System.Diagnostics.Debug.WriteLine($"Error loading remembered accounts: {ex.Message}");
            }
        }

        /// <summary>
        /// Loads previously logged-in users from the database
        /// </summary>
        private async Task LoadPreviouslyLoggedInUsersAsync()
        {
            try
            {
                var users = await _authenticationService.GetPreviouslyLoggedInUsersAsync();
                
                // Ensure we're on the UI thread when updating the collection
                if (Application.Current?.Dispatcher?.CheckAccess() == false)
                {
                    await Application.Current.Dispatcher.InvokeAsync(() =>
                    {
                        PreviouslyLoggedInUsersList.Clear();
                        foreach (var username in users)
                        {
                            PreviouslyLoggedInUsersList.Add(username);
                        }
                    });
                }
                else
                {
                    PreviouslyLoggedInUsersList.Clear();
                    foreach (var username in users)
                    {
                        PreviouslyLoggedInUsersList.Add(username);
                    }
                }
            }
            catch (Exception ex)
            {
                // Log error but don't prevent login window from opening
                System.Diagnostics.Debug.WriteLine($"Error loading previously logged-in users: {ex.Message}");
            }
        }

        /// <summary>
        /// Loads the selected user from the previously logged-in users dropdown
        /// This properly updates the Username property which is bound to the UI
        /// </summary>
        private void LoadSelectedPreviousUser()
        {
            if (!string.IsNullOrEmpty(SelectedPreviousUser))
            {
                Username = SelectedPreviousUser;
                // Clear password - user must enter it again for security
                Password = string.Empty;
                // Notify the UI that Username has changed
                OnPropertyChanged(nameof(Username));
            }
        }

        private void LoadSelectedAccount()
        {
            if (!string.IsNullOrEmpty(SelectedAccount) && _rememberedAccounts.ContainsKey(SelectedAccount))
            {
                var account = _rememberedAccounts[SelectedAccount];
                Username = account.Username;
                Password = account.Password;
                Pin = account.Pin;
                IsPinVisible = true;
            }
        }

        #endregion

        #region Command Implementations

        private bool CanExecuteLogin(object parameter)
        {
            return !IsLoggingIn && !IsRegistrationMode && (!string.IsNullOrWhiteSpace(Username) || !string.IsNullOrWhiteSpace(Pin));
        }

        private bool CanExecuteRegister(object parameter)
        {
            return !IsLoggingIn && IsRegistrationMode && !string.IsNullOrWhiteSpace(Username) && !string.IsNullOrWhiteSpace(Password);
        }

        private async Task ExecuteLoginAsync()
        {
            if (IsLoggingIn) return;

            // Validate username
            if (string.IsNullOrWhiteSpace(Username))
            {
                StatusMessage = "Please enter a username.";
                LoginFailed?.Invoke(this, "Username is required.");
                return;
            }

            // Validate password
            if (string.IsNullOrWhiteSpace(Password))
            {
                StatusMessage = "Please enter a password.";
                LoginFailed?.Invoke(this, "Password is required.");
                return;
            }

            IsLoggingIn = true;
            StatusMessage = string.Empty;
            try
            {
                // First try to authenticate with the AuthenticationService (registered users)
                var authResult = await _authenticationService.AuthenticateAsync(Username, Password);
                
                if (authResult.Success)
                {
                    // Save credentials if remember me is checked
                    if (RememberMe)
                    {
                        DatabaseMonolith.RememberAccount(Username, Password, Pin);
                    }

                    // Get saved window state
                    var savedWindowState = _userSettingsService.GetSavedWindowState();

                    // Fire success event with context needed to open MainWindow
                    LoginSuccessful?.Invoke(this, new LoginSuccessEventArgs
                    {
                        UserSettingsService = _userSettingsService,
                        HistoricalDataService = _historicalDataService,
                        AlphaVantageService = _alphaVantageService,
                        TechnicalIndicatorService = _technicalIndicatorService,
                        SavedWindowState = savedWindowState,
                        UserId = authResult.UserId,
                        Username = authResult.Username
                    });
                }
                else
                {
                    // Fall back to WebullTradingBot authentication for backward compatibility
                    var tradingBot = new WebullTradingBot(
                        _userSettingsService,
                        _historicalDataService,
                        _alphaVantageService,
                        _technicalIndicatorService);

                    bool isAuthenticated = false;

                    // Authenticate with PIN if available and remembered
                    if (!string.IsNullOrEmpty(Pin) && _rememberedAccounts.Values.Any(a => a.Pin == Pin))
                    {
                        var account = _rememberedAccounts.Values.First(a => a.Pin == Pin);
                        isAuthenticated = await tradingBot.Authenticate(account.Username, account.Password);
                    }
                    else
                    {
                        isAuthenticated = await tradingBot.Authenticate(Username, Password);
                    }

                    if (isAuthenticated)
                    {
                        // Clear user context for legacy authentication (no user-specific settings)
                        AuthenticationService.SetCurrentUserId(null);

                        // Save credentials if remember me is checked
                        if (RememberMe)
                        {
                            DatabaseMonolith.RememberAccount(Username, Password, Pin);
                        }

                        // Get saved window state
                        var savedWindowState = _userSettingsService.GetSavedWindowState();

                        // Fire success event with context needed to open MainWindow
                        LoginSuccessful?.Invoke(this, new LoginSuccessEventArgs
                        {
                            UserSettingsService = _userSettingsService,
                            HistoricalDataService = _historicalDataService,
                            AlphaVantageService = _alphaVantageService,
                            TechnicalIndicatorService = _technicalIndicatorService,
                            SavedWindowState = savedWindowState,
                            UserId = null, // Legacy authentication doesn't have user ID
                            Username = Username
                        });
                    }
                    else
                    {
                        StatusMessage = "Invalid username or password.";
                        LoginFailed?.Invoke(this, "Authentication unsuccessful.");
                    }
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Login error: {ex.Message}";
                LoginFailed?.Invoke(this, $"Login error: {ex.Message}");
            }
            finally
            {
                IsLoggingIn = false;
            }
        }

        private async Task ExecuteRegisterAsync()
        {
            if (IsLoggingIn) return;

            // Validate passwords match
            if (Password != ConfirmPassword)
            {
                StatusMessage = "Passwords do not match.";
                RegistrationFailed?.Invoke(this, "Passwords do not match.");
                return;
            }

            IsLoggingIn = true;
            StatusMessage = string.Empty;
            try
            {
                var result = await _authenticationService.RegisterUserAsync(Username, Password, Email);
                
                if (result.Success)
                {
                    StatusMessage = "Registration successful! Please login.";
                    RegistrationSuccessful?.Invoke(this, result.Message);
                    
                    // Switch to login mode
                    IsRegistrationMode = false;
                    ConfirmPassword = string.Empty;
                    Email = string.Empty;
                }
                else
                {
                    StatusMessage = result.ErrorMessage;
                    RegistrationFailed?.Invoke(this, result.ErrorMessage);
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Registration error: {ex.Message}";
                RegistrationFailed?.Invoke(this, $"Registration error: {ex.Message}");
            }
            finally
            {
                IsLoggingIn = false;
            }
        }

        private void ExecuteToggleRegistrationMode(object parameter)
        {
            IsRegistrationMode = !IsRegistrationMode;
            // Clear fields when switching modes
            Password = string.Empty;
            ConfirmPassword = string.Empty;
            StatusMessage = string.Empty;
        }

        private void ExecuteOpenSettings(object parameter)
        {
            SettingsRequested?.Invoke(this, EventArgs.Empty);
        }

        #endregion
    }

    /// <summary>
    /// Event arguments for successful login
    /// </summary>
    public class LoginSuccessEventArgs : EventArgs
    {
        public UserSettingsService UserSettingsService { get; set; }
        public HistoricalDataService HistoricalDataService { get; set; }
        public AlphaVantageService AlphaVantageService { get; set; }
        public TechnicalIndicatorService TechnicalIndicatorService { get; set; }
        public WindowState? SavedWindowState { get; set; }
        public int? UserId { get; set; }
        public string Username { get; set; }
    }
}
