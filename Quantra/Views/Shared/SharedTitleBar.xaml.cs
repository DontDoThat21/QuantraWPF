using Microsoft.Extensions.DependencyInjection;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;

namespace Quantra
{
    public partial class SharedTitleBar : UserControl, INotifyPropertyChanged, IDisposable
    {
        private static SharedTitleBar _currentInstance;
        
        private bool isDragging = false;
        private Point startPoint;
        private DispatcherTimer apiUsageTimer;
        private DispatcherTimer emergencyStopCheckTimer;
        private DispatcherTimer vixMonitoringTimer;
        private DispatcherTimer monitoringClearTimer;
        private IOrderService _orderService;
        private ISettingsService _settingsService;

        // Add AlphaVantageService field
        private AlphaVantageService _alphaVantageService;

        private bool _isFullscreen = false;
        private WindowState _previousWindowState;
        private WindowStyle _previousWindowStyle;
        private bool _previousTopmost;
        public static readonly DependencyProperty TitleProperty =
            DependencyProperty.Register("Title", typeof(string), typeof(SharedTitleBar), new PropertyMetadata(string.Empty));

        public string Title
        {
            get { return (string)GetValue(TitleProperty); }
            set { SetValue(TitleProperty, value); }
        }

        // New property for the settings button
        public Button SettingsButton
        {
            get
            {
                return _settingsButton;
            }
            set
            {
                _settingsButton = value;
                if (SettingsPlaceholder != null)
                {
                    SettingsPlaceholder.Content = _settingsButton;
                }
            }
        }
        private Button _settingsButton;

        // API usage display property
        private string _apiUsageDisplay = "API: --/--";
        public string ApiUsageDisplay
        {
            get => _apiUsageDisplay;
            set
            {
                if (_apiUsageDisplay != value)
                {
                    _apiUsageDisplay = value;
                    OnPropertyChanged(nameof(ApiUsageDisplay));
                }
            }
        }
        
        // VIX display property
        private string _vixDisplay = "VIX: --";
        public string VixDisplay
        {
            get => _vixDisplay;
            set
            {
                if (_vixDisplay != value)
                {
                    // Ensure property updates happen on UI thread
                    if (Dispatcher.CheckAccess())
                    {
                        _vixDisplay = value;
                        OnPropertyChanged(nameof(VixDisplay));
                    }
                    else
                    {
                        Dispatcher.Invoke(() =>
                        {
                            _vixDisplay = value;
                            OnPropertyChanged(nameof(VixDisplay));
                        });
                    }
                }
            }
        }
        
        // Emergency stop active property
        private bool _isEmergencyStopActive = false;
        public bool IsEmergencyStopActive
        {
            get => _isEmergencyStopActive;
            set
            {
                if (_isEmergencyStopActive != value)
                {
                    _isEmergencyStopActive = value;
                    OnPropertyChanged(nameof(IsEmergencyStopActive));
                    
                    // Update button text based on state
                    EmergencyStopText.Text = _isEmergencyStopActive ? "RESUME TRADING" : "EMERGENCY STOP";
                }
            }
        }

        // Global loading state property
        private bool _isGlobalLoading = false;
        public bool IsGlobalLoading
        {
            get => _isGlobalLoading;
            set
            {
                if (_isGlobalLoading != value)
                {
                    _isGlobalLoading = value;
                    OnPropertyChanged(nameof(IsGlobalLoading));
                }
            }
        }

        // Callee property for monitoring dispatcher calls
        private string _currentCallee = "None";
        public string CurrentCallee
        {
            get => _currentCallee;
            set
            {
                if (_currentCallee != value)
                {
                    _currentCallee = value;
                    OnPropertyChanged(nameof(CurrentCallee));
                }
            }
        }

        // Current call property for monitoring dispatcher calls
        private string _currentCall = "None";
        public string CurrentCall
        {
            get => _currentCall;
            set
            {
                if (_currentCall != value)
                {
                    _currentCall = value;
                    OnPropertyChanged(nameof(CurrentCall));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        /// <summary>
        /// Updates the dispatcher monitoring display with current method information
        /// </summary>
        /// <param name="skipFrames">Number of frames to skip to get the calling method</param>
        private void UpdateDispatcherMonitoring(int skipFrames = 1)
        {
            try
            {
                var stackTrace = new StackTrace();
                
                // Get current method (the method that called this)
                var currentFrame = stackTrace.GetFrame(skipFrames);
                var currentMethod = currentFrame?.GetMethod();
                CurrentCall = currentMethod?.Name ?? "Unknown";
                
                // Get calling method (one frame up)
                var callerFrame = stackTrace.GetFrame(skipFrames + 1);
                var callerMethod = callerFrame?.GetMethod();
                CurrentCallee = callerMethod?.Name ?? "Unknown";
                
                // Start or restart the monitoring clear timer
                StartMonitoringClearTimer();
            }
            catch (Exception)
            {
                // If reflection fails, fall back to default values
                CurrentCall = "Error";
                CurrentCallee = "Error";
            }
        }

        /// <summary>
        /// Updates monitoring with a custom method name instead of using reflection
        /// </summary>
        /// <param name="methodName">The name of the method to display</param>
        /// <param name="callerName">The name of the calling method</param>
        private void UpdateDispatcherMonitoringManual(string methodName, string callerName = null)
        {
            CurrentCall = methodName ?? "Unknown";
            
            if (callerName != null)
            {
                CurrentCallee = callerName;
            }
            else
            {
                // Try to get caller from stack trace
                try
                {
                    var stackTrace = new StackTrace();
                    var callerFrame = stackTrace.GetFrame(2); // Skip this method and the calling method
                    var callerMethod = callerFrame?.GetMethod();
                    CurrentCallee = callerMethod?.Name ?? "Unknown";
                }
                catch
                {
                    CurrentCallee = "Unknown";
                }
            }
            
            // Start or restart the monitoring clear timer (5 seconds)
            StartMonitoringClearTimer();
        }

        /// <summary>
        /// Starts a timer to clear the monitoring display after 5 seconds of inactivity
        /// </summary>
        private void StartMonitoringClearTimer()
        {
            // Stop any existing timer
            if (monitoringClearTimer != null)
            {
                monitoringClearTimer.Stop();
                monitoringClearTimer = null;
            }
            
            // Create a new timer that clears the display after 5 seconds
            monitoringClearTimer = new DispatcherTimer();
            monitoringClearTimer.Interval = TimeSpan.FromSeconds(5);
            
            // Capture the timer reference locally to prevent race conditions
            var timerRef = monitoringClearTimer;
            timerRef.Tick += (s, e) => 
            {
                CurrentCall = "None";
                CurrentCallee = "None";
                
                // Use the captured reference to safely stop the timer
                timerRef.Stop();
                
                // Only set the field to null if it still references this timer
                if (monitoringClearTimer == timerRef)
                {
                    monitoringClearTimer = null;
                }
            };
            monitoringClearTimer.Start();
        }

        public SharedTitleBar()
        {
            InitializeComponent();
            this.DataContext = this;
            
            // Set current instance for global access
            _currentInstance = this;
            
            // Initialize monitoring display
            CurrentCallee = "None";
            CurrentCall = "None";
            
            // Subscribe to global loading state changes
            GlobalLoadingStateService.LoadingStateChanged += OnGlobalLoadingStateChanged;

            // Get the order service
            _orderService = App.ServiceProvider.GetService<IOrderService>();

            // Instantiate AlphaVantageService
            _alphaVantageService = new AlphaVantageService();

            // Ensure database and settings are properly initialized before starting VIX monitoring
            try
            {
                // Ensure database is initialized first
                DatabaseMonolith.Initialize();

                // Then ensure settings profiles exist
                _settingsService.EnsureSettingsProfiles();
                //DatabaseMonolith.Log("Info", "SharedTitleBar: Database and settings profiles initialized successfully");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "SharedTitleBar: Failed to initialize database or settings profiles", ex.ToString());
            }

            // Start updating API usage count
            StartApiUsageTimer();
            
            // Start checking emergency stop status
            StartEmergencyStopCheckTimer();
            
            // Start VIX monitoring
            StartVixMonitoringTimer();
        }

        private void StartApiUsageTimer()
        {
            UpdateDispatcherMonitoringManual("StartApiUsageTimer");
            
            apiUsageTimer = new DispatcherTimer();
            apiUsageTimer.Interval = TimeSpan.FromSeconds(10);
            apiUsageTimer.Tick += async (s, e) => {
                UpdateDispatcherMonitoringManual("ApiUsageTimer_Tick");
                await RefreshApiUsageDisplay();
            };
            apiUsageTimer.Start();
            _ = RefreshApiUsageDisplay();
        }
        
        private void StartEmergencyStopCheckTimer()
        {
            UpdateDispatcherMonitoringManual("StartEmergencyStopCheckTimer");
            
            if (_orderService != null)
            {
                emergencyStopCheckTimer = new DispatcherTimer();
                emergencyStopCheckTimer.Interval = TimeSpan.FromSeconds(2);
                emergencyStopCheckTimer.Tick += (s, e) => 
                {
                    UpdateDispatcherMonitoringManual("EmergencyStopTimer_Tick");
                    IsEmergencyStopActive = _orderService.IsEmergencyStopActive();
                };
                emergencyStopCheckTimer.Start();
                IsEmergencyStopActive = _orderService.IsEmergencyStopActive();
            }
        }

        private void StartVixMonitoringTimer()
        {
            UpdateDispatcherMonitoringManual("StartVixMonitoringTimer");
            
            vixMonitoringTimer = new DispatcherTimer();
            vixMonitoringTimer.Interval = TimeSpan.FromMinutes(1); // Update every minute
            vixMonitoringTimer.Tick += async (s, e) => {
                UpdateDispatcherMonitoringManual("VixTimer_Tick");
                await RefreshVixDisplay();
            };
            vixMonitoringTimer.Start();
            
            // Delay the initial VIX refresh slightly to ensure database/settings are fully initialized
            // Use proper async/await pattern instead of ContinueWith
            _ = Task.Run(async () =>
            {
                await Task.Delay(TimeSpan.FromSeconds(2));
                await RefreshVixDisplay();
            });
        }

        private async Task RefreshVixDisplay()
        {
            UpdateDispatcherMonitoringManual("RefreshVixDisplay");
            
            try
            {
                //DatabaseMonolith.Log("Info", "RefreshVixDisplay: Starting VIX refresh");
                
                // Check if VIX monitoring is enabled in settings using resilient retry logic
                var activeProfile = ResilienceHelper.Retry(() => _settingsService.GetDefaultSettingsProfile(), 
                    RetryOptions.ForCriticalOperation());
                
                // If no profile found, try to initialize settings
                if (activeProfile == null)
                {
                    //DatabaseMonolith.Log("Warning", "RefreshVixDisplay: No settings profile found, attempting initialization");
                    
                    try
                    {
                        ResilienceHelper.Retry(() => _settingsService.EnsureSettingsProfiles(), 
                            RetryOptions.ForCriticalOperation());
                        
                        activeProfile = ResilienceHelper.Retry(() => _settingsService.GetDefaultSettingsProfile(), 
                            RetryOptions.ForCriticalOperation());
                        
                        if (activeProfile != null)
                        {
                            //DatabaseMonolith.Log("Info", "RefreshVixDisplay: Settings profile initialized successfully");
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", "RefreshVixDisplay: Failed to initialize settings", ex.ToString());
                    }
                    
                    if (activeProfile == null)
                    {
                        //DatabaseMonolith.Log("Warning", "RefreshVixDisplay: Setting VIX to disabled - no profile");
                        VixDisplay = "VIX: Disabled";
                        return;
                    }
                }
                
                if (activeProfile.EnableVixMonitoring != true)
                {
                    //DatabaseMonolith.Log("Info", "RefreshVixDisplay: VIX monitoring disabled in settings");
                    VixDisplay = "VIX: Disabled";
                    return;
                }

                // Check if market is open (use trading hours logic)
                bool marketOpen = IsMarketOpen();
                //DatabaseMonolith.Log("Info", $"RefreshVixDisplay: Market open status: {marketOpen}");
                
                if (!marketOpen)
                {
                    // Market is closed, try to get the latest cached VIX value
                    var (cachedVixValue, lastUpdate) = await GetLatestCachedVixValue();
                    
                    if (lastUpdate.HasValue)
                    {
                        // Format the date for display (show date if not today)
                        var updateDate = lastUpdate.Value.Date;
                        var today = DateTime.Today;
                        
                        string dateDisplay;
                        if (updateDate == today)
                        {
                            dateDisplay = "Today";
                        }
                        else if (updateDate == today.AddDays(-1))
                        {
                            dateDisplay = "Yesterday";
                        }
                        else
                        {
                            dateDisplay = updateDate.ToString("MM/dd");
                        }
                        
                        VixDisplay = $"VIX: {cachedVixValue:F1} ({dateDisplay})";
                        //DatabaseMonolith.Log("Info", $"RefreshVixDisplay: Set cached VIX value: {VixDisplay}");
                    }
                    else
                    {
                        // Fallback to Market Closed if no cached data available
                        VixDisplay = "VIX: Market Closed";
                        //DatabaseMonolith.Log("Info", "RefreshVixDisplay: No cached data, market closed");
                    }
                    return;
                }

                // Fetch VIX data (market is open)
                //DatabaseMonolith.Log("Info", "RefreshVixDisplay: Fetching live VIX data");
                double vixValue = await GetVixValue();
                string timestamp = DateTime.Now.ToString("HH:mm");
                VixDisplay = $"VIX: {vixValue:F1} ({timestamp})";
                //DatabaseMonolith.Log("Info", $"RefreshVixDisplay: Set live VIX value: {VixDisplay}");
            }
            catch (Exception ex)
            {
                VixDisplay = "VIX: Error";
                //DatabaseMonolith.Log("Error", "Failed to refresh VIX display", ex.ToString());
            }
        }

        private async Task<double> GetVixValue()
        {
            try
            {
                //DatabaseMonolith.Log("Info", "GetVixValue: Calling AlphaVantage API for VIX data");
                // Use AlphaVantage service directly to get VIX data
                double vixValue = await _alphaVantageService.GetQuoteData("^VIX");
                //DatabaseMonolith.Log("Info", $"GetVixValue: Received VIX value: {vixValue}");
                return vixValue;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "GetVixValue: Failed to get VIX data from API", ex.ToString());
                return 15.0; // Default moderate volatility fallback
            }
        }

        private async Task<(double value, DateTime? lastUpdate)> GetLatestCachedVixValue()
        {
            try
            {
                // Use the AlphaVantage service's cached historical data method to get the latest VIX value
                var cachedData = await _alphaVantageService.GetExtendedHistoricalData("^VIX", "daily", "compact");
                
                if (cachedData != null && cachedData.Count > 0)
                {
                    // Get the most recent data point
                    var latestData = cachedData.OrderByDescending(d => d.Date).First();
                    return (latestData.Close, latestData.Date);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to get cached VIX value", ex.ToString());
            }
            
            // Return default if no cached data available
            return (15.0, null);
        }

        private bool IsMarketOpen()
        {
            try
            {
                // Get current time
                TimeOnly now = TimeOnly.FromDateTime(DateTime.Now);
                
                // Use standard market hours (pre-market 4:00 AM through after-hours 8:00 PM)
                TimeOnly preMarketOpen = new TimeOnly(4, 0, 0);   // 4:00 AM
                TimeOnly afterHoursClose = new TimeOnly(20, 0, 0); // 8:00 PM
                
                // Check if we're within trading hours (including pre-market and after-hours)
                return now >= preMarketOpen && now <= afterHoursClose;
            }
            catch
            {
                return false; // Default to market closed if there's an error
            }
        }

        // Get the API usage count from AlphaVantageService (or a central tracker)
        private async Task RefreshApiUsageDisplay()
        {
            UpdateDispatcherMonitoringManual("RefreshApiUsageDisplay");
            
            try
            {
                // Set loading state when starting API call
                GlobalLoadingStateService.SetLoadingState(true);
                
                // Use DB-based count for accuracy
                int used = await Task.Run(() => _alphaVantageService.GetCurrentDbApiCallCount());
                int limit = AlphaVantageService.ApiCallLimit;
                ApiUsageDisplay = $"AlphaVantage API: {used}/{limit}/min";
            }
            finally
            {
                // Clear loading state when done
                GlobalLoadingStateService.SetLoadingState(false);
            }
        }
        
        private void EmergencyStopButton_Click(object sender, RoutedEventArgs e)
        {
            if (_orderService == null)
            {
                MessageBox.Show("Trading service is not available.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            
            if (IsEmergencyStopActive)
            {
                // Ask for confirmation before resuming trading
                var resumeResult = MessageBox.Show(
                    "Are you sure you want to resume trading?", 
                    "Resume Trading", 
                    MessageBoxButton.YesNo, 
                    MessageBoxImage.Question);
                    
                if (resumeResult == MessageBoxResult.Yes)
                {
                    bool success = _orderService.DeactivateEmergencyStop();
                    IsEmergencyStopActive = _orderService.IsEmergencyStopActive();
                    
                    if (success)
                    {
                        MessageBox.Show("Trading has been resumed.", "Trading Resumed", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
            }
            else
            {
                // Ask for confirmation before activating emergency stop
                var stopResult = MessageBox.Show(
                    "EMERGENCY STOP: This will cancel all pending orders and halt all trading activity. Continue?", 
                    "Emergency Stop", 
                    MessageBoxButton.YesNo, 
                    MessageBoxImage.Warning);
                    
                if (stopResult == MessageBoxResult.Yes)
                {
                    bool success = _orderService.ActivateEmergencyStop();
                    IsEmergencyStopActive = _orderService.IsEmergencyStopActive();
                    
                    if (success)
                    {
                        MessageBox.Show("Emergency stop activated. All trading has been halted.", "Emergency Stop", MessageBoxButton.OK, MessageBoxImage.Warning);
                    }
                }
            }
        }

        private void MinimizeButton_Click(object sender, RoutedEventArgs e)
        {
            Window.GetWindow(this).WindowState = WindowState.Minimized;
        }

        private void MaximizeButton_Click(object sender, RoutedEventArgs e)
        {
            var window = Window.GetWindow(this);
            window.WindowState = window.WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            var currentWindow = Window.GetWindow(this);
            if (currentWindow is MainWindow)
            {
                var loginWindow = new LoginWindow();
                loginWindow.Show();
            }
            currentWindow.Close();
        }

        private void TitleBar_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                isDragging = true;
                startPoint = e.GetPosition(null);
                
                // Handle double-click to maximize/restore the window
                if (e.ClickCount == 2)
                {
                    var window = Window.GetWindow(this);
                    if (window != null)
                    {
                        window.WindowState = window.WindowState == WindowState.Maximized ? 
                            WindowState.Normal : WindowState.Maximized;
                    }
                    return;
                }
                
                Window.GetWindow(this)?.DragMove();
            }
        }

        private void TitleBar_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Released)
            {
                isDragging = false;
            }
        }

        private void TitleBar_MouseMove(object sender, MouseEventArgs e)
        {
            if (isDragging && e.LeftButton == MouseButtonState.Pressed)
            {
                Window.GetWindow(this)?.DragMove();
            }
        }
        
        private async void OnGlobalLoadingStateChanged(bool isLoading)
        {
            // Update monitoring for this dispatcher call
            UpdateDispatcherMonitoring();
            
            // Update the UI on the dispatcher thread
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                IsGlobalLoading = isLoading;
            });
        }
        
        /// <summary>
        /// Sets the global loading state from any part of the application
        /// </summary>
        /// <param name="isLoading">True to show loading spinner, false to hide it</param>
        public static void SetGlobalLoadingState(bool isLoading)
        {
            GlobalLoadingStateService.SetLoadingState(isLoading);
        }
        
        /// <summary>
        /// Gets the current global loading state
        /// </summary>
        /// <returns>True if the app is currently showing as busy/loading</returns>
        public static bool GetGlobalLoadingState()
        {
            return GlobalLoadingStateService.IsLoading;
        }
        
        /// <summary>
        /// Updates the dispatcher monitoring display from any part of the application
        /// </summary>
        /// <param name="methodName">The name of the method making the dispatcher call</param>
        /// <param name="callerName">Optional: The name of the calling method</param>
        public static void UpdateDispatcherMonitoring(string methodName, string callerName = null)
        {
            if (_currentInstance != null)
            {
                if (callerName == null)
                {
                    // Try to get caller from stack trace
                    try
                    {
                        var stackTrace = new StackTrace();
                        var callerFrame = stackTrace.GetFrame(2); // Skip this method and the calling method
                        var callerMethod = callerFrame?.GetMethod();
                        callerName = callerMethod?.Name ?? "Unknown";
                    }
                    catch
                    {
                        callerName = "Unknown";
                    }
                }
                
                _currentInstance.UpdateDispatcherMonitoringManual(methodName, callerName);
            }
        }
        
        /// <summary>
        /// Clears the dispatcher monitoring display
        /// </summary>
        public static void ClearDispatcherMonitoring()
        {
            if (_currentInstance != null)
            {
                _currentInstance.CurrentCall = "None";
                _currentInstance.CurrentCallee = "None";
            }
        }
        
        // Removed TitleBar_MouseDoubleClick method as its functionality is now handled in TitleBar_MouseDown
        
        /// <summary>
        /// Safely disposes of all timers to prevent null reference exceptions
        /// </summary>
        public void Dispose()
        {
            // Safely stop and dispose monitoring clear timer
            if (monitoringClearTimer != null)
            {
                monitoringClearTimer.Stop();
                monitoringClearTimer = null;
            }
            
            // Safely stop and dispose API usage timer
            if (apiUsageTimer != null)
            {
                apiUsageTimer.Stop();
                apiUsageTimer = null;
            }
            
            // Safely stop and dispose emergency stop check timer
            if (emergencyStopCheckTimer != null)
            {
                emergencyStopCheckTimer.Stop();
                emergencyStopCheckTimer = null;
            }
            
            // Safely stop and dispose VIX monitoring timer
            if (vixMonitoringTimer != null)
            {
                vixMonitoringTimer.Stop();
                vixMonitoringTimer = null;
            }
            
            // Unsubscribe from global loading state changes
            GlobalLoadingStateService.LoadingStateChanged -= OnGlobalLoadingStateChanged;
        }

        private void FullscreenButton_Click(object sender, RoutedEventArgs e)
        {
            var window = Window.GetWindow(this);
            if (window == null) return;

            if (_isFullscreen)
            {
                ExitFullscreen(window);
            }
            else
            {
                EnterFullscreen(window);
            }
        }

        private void ExitFullscreen(Window window)
        {
            // Show taskbar
            IntPtr taskbarHandle = FindWindow("Shell_TrayWnd", null);
            if (taskbarHandle != IntPtr.Zero)
            {
                ShowWindow(taskbarHandle, SW_SHOW);
            }

            // Restore previous window state
            window.WindowStyle = _previousWindowStyle;
            window.Topmost = _previousTopmost;
            window.WindowState = _previousWindowState;

            _isFullscreen = false;
        }

        private void EnterFullscreen(Window window)
        {
            // Save current window state
            _previousWindowState = window.WindowState;
            _previousWindowStyle = window.WindowStyle;
            _previousTopmost = window.Topmost;

            // Hide taskbar
            IntPtr taskbarHandle = FindWindow("Shell_TrayWnd", null);
            if (taskbarHandle != IntPtr.Zero)
            {
                ShowWindow(taskbarHandle, SW_HIDE);
            }

            // Set window to fullscreen
            window.WindowStyle = WindowStyle.None;
            window.WindowState = WindowState.Maximized;
            window.Topmost = true;

            _isFullscreen = true;
        }

        [DllImport("user32.dll")]
        private static extern IntPtr FindWindow(string lpClassName, string lpWindowName);


        [DllImport("user32.dll")]
        private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        private const int SW_HIDE = 0;
        private const int SW_SHOW = 1;
    }
}
