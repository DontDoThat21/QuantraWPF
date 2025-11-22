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
    /// ViewModel for the Login Window
    /// </summary>
    public class LoginWindowViewModel : ViewModelBase
    {
        private readonly UserSettingsService _userSettingsService;
        private readonly HistoricalDataService _historicalDataService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private Dictionary<string, (string Username, string Password, string Pin)> _rememberedAccounts;
        
        private string _username;
        private string _password;
        private string _pin;
        private bool _rememberMe;
        private string _selectedAccount;
        private bool _isPinVisible;
        private bool _isLoggingIn;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public LoginWindowViewModel(
            UserSettingsService userSettingsService,
            HistoricalDataService historicalDataService,
            AlphaVantageService alphaVantageService,
            TechnicalIndicatorService technicalIndicatorService)
        {
            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            _historicalDataService = historicalDataService ?? throw new ArgumentNullException(nameof(historicalDataService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _technicalIndicatorService = technicalIndicatorService ?? throw new ArgumentNullException(nameof(technicalIndicatorService));
            
            _rememberedAccounts = new Dictionary<string, (string Username, string Password, string Pin)>();
            RememberedAccountsList = new ObservableCollection<string>();
            
            InitializeCommands();
            LoadRememberedAccounts();
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

        #endregion

        #region Commands

        public ICommand LoginCommand { get; private set; }
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
        /// Event fired when settings should be opened
        /// </summary>
        public event EventHandler SettingsRequested;

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            LoginCommand = new RelayCommand(async param => await ExecuteLoginAsync(), CanExecuteLogin);
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
            return !IsLoggingIn && (!string.IsNullOrWhiteSpace(Username) || !string.IsNullOrWhiteSpace(Pin));
        }

        private async Task ExecuteLoginAsync()
        {
            if (IsLoggingIn) return;

            IsLoggingIn = true;
            try
            {
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
                        SavedWindowState = savedWindowState
                    });
                }
                else
                {
                    LoginFailed?.Invoke(this, "Authentication unsuccessful.");
                }
            }
            catch (Exception ex)
            {
                LoginFailed?.Invoke(this, $"Login error: {ex.Message}");
            }
            finally
            {
                IsLoggingIn = false;
            }
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
    }
}
