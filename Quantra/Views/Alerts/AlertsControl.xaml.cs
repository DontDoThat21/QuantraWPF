using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Media;
using System.Windows.Threading;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System.Threading.Tasks;
using Quantra.DAL.Services;
using Quantra.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Quantra.Controls
{
    public partial class AlertsControl : UserControl
    {
        private readonly AlertsControlViewModel _viewModel;

        // Static event for global alert emission
        public static event Action<AlertModel> GlobalAlertEmitted;

        // Parameterless constructor for XAML designer support
        public AlertsControl()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public AlertsControl(AlertsControlViewModel viewModel)
        {
            InitializeComponent();
            
            _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
            DataContext = _viewModel;
            
            // Subscribe to ViewModel events
            _viewModel.AlertTriggered += OnAlertTriggered;
            
            // Subscribe to global alert event
            GlobalAlertEmitted += _viewModel.OnGlobalAlertEmitted;
            
            // Start monitoring
            _viewModel.StartMonitoring();
        }

        /// <summary>
        /// Legacy constructor using ServiceLocator for compatibility
        /// </summary>
        public AlertsControl(
            ITechnicalIndicatorService indicatorService,
            IHistoricalDataService historicalDataService,
            SettingsService settingsService,
            StockDataCacheService stockDataCacheService)
            : this(new AlertsControlViewModel(indicatorService, historicalDataService, settingsService, stockDataCacheService))
        {
        }

        private void OnAlertTriggered(object sender, AlertModel alert)
        {
            // Handle alert triggered in UI if needed
            System.Diagnostics.Debug.WriteLine($"Alert triggered: {alert.Message}");
        }

        protected override void OnUnloaded(RoutedEventArgs e)
        {
            // Clean up
            if (_viewModel != null)
            {
                _viewModel.StopMonitoring();
                _viewModel.AlertTriggered -= OnAlertTriggered;
                GlobalAlertEmitted -= _viewModel.OnGlobalAlertEmitted;
                _viewModel.Dispose();
            }
            base.OnUnloaded(e);
        }

        #region Remaining UI Event Handlers
        
        // Note: These methods remain in code-behind for XAML event bindings.
        // They delegate to ViewModel commands and properties where appropriate.
        // Further refactoring to pure XAML bindings can be done in a future PR.
        
        private void OnGlobalAlertEmitted_Legacy(AlertModel alert)
        {
            try
            {
                // Only check active, non-triggered technical indicator alerts
                var indicatorAlerts = alerts.Where(a => 
                    a.Category == AlertCategory.TechnicalIndicator && 
                    a.IsActive && 
                    !a.IsTriggered).ToList();
                
                if (indicatorAlerts.Count > 0)
                {
                    int triggeredCount = await technicalIndicatorAlertService.CheckAllAlertsAsync(indicatorAlerts);
                    
                    // Also check for volume spike alerts
                    int volumeTriggeredCount = await volumeAlertService.CheckAllVolumeAlertsAsync(indicatorAlerts);
                    triggeredCount += volumeTriggeredCount;
                    
                    // Check for pattern alerts
                    var symbols = alerts
                        .Where(a => a.IsActive && !string.IsNullOrWhiteSpace(a.Symbol))
                        .Select(a => a.Symbol)
                        .Distinct()
                        .Take(MaxSymbolsToCheck) // Limit to top symbols to avoid API overuse
                        .ToList();
                    
                    if (symbols.Count > 0)
                    {
                        int patternTriggeredCount = await patternAlertService.DetectPatternsForSymbolsAsync(symbols);
                        triggeredCount += patternTriggeredCount;
                    }
                    
                    if (triggeredCount > 0)
                    {
                        // Refresh UI if any alerts were triggered
                        ApplyFilter();
                    }
                }
            }
            catch (Exception ex)
            {
                EmitGlobalError("Error checking technical indicator alerts", ex);
            }
        }

        private async void OnGlobalAlertEmitted(AlertModel alert)
        {
            if (alert == null) return;
            // Insert at the top for newest first
            alerts.Insert(0, alert);
            // Optionally limit the number of alerts to avoid memory issues
            if (alerts.Count > 500)
                alerts.RemoveAt(alerts.Count - 1);
            // Refresh UI directly, ensuring UI-bound calls are marshaled to the UI thread
            await Dispatcher.InvokeAsync(() => ApplyFilter());
        }

        private void AddAlert(AlertModel alert)
        {
            if (alert == null) return;
            alerts.Insert(0, alert);
            if (alerts.Count > 500)
                alerts.RemoveAt(alerts.Count - 1);
            ApplyFilter();
        }

        private void CategoryFilterComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var comboBoxItem = CategoryFilterComboBox.SelectedItem as ComboBoxItem;
            selectedCategoryFilter = comboBoxItem?.Content.ToString() ?? "All Categories";
            ApplyFilter();
        }

        private void ApplyFilter()
        {
            string query = SearchBox.Text.ToUpper().Trim();
            var comboBoxItem = FilterComboBox.SelectedItem as ComboBoxItem;
            string filterOption = comboBoxItem?.Content.ToString() ?? "All";
            var categoryComboBoxItem = CategoryFilterComboBox.SelectedItem as ComboBoxItem;
            string categoryFilter = categoryComboBoxItem?.Content.ToString() ?? "All Categories";

            // Start with all alerts
            IEnumerable<AlertModel> filtered = alerts;

            // Apply filter based on selection
            switch (filterOption)
            {
                case "Active":
                    filtered = filtered.Where(a => a.IsActive);
                    break;
                case "Triggered":
                    filtered = filtered.Where(a => a.IsTriggered);
                    break;
                case "High Priority":
                    filtered = filtered.Where(a => a.Priority == 1);
                    break;
            }

            // Apply category filter
            if (!string.IsNullOrWhiteSpace(categoryFilter) && categoryFilter != "All Categories")
            {
                if (Enum.TryParse<AlertCategory>(categoryFilter, out var cat))
                {
                    filtered = filtered.Where(a => a.Category == cat);
                }
            }

            // Apply search query if any
            if (!string.IsNullOrWhiteSpace(query))
            {
                filtered = filtered.Where(a =>
                    (a.Name != null && a.Name.ToUpper().Contains(query)) ||
                    (a.Symbol != null && a.Symbol.ToUpper().Contains(query)) ||
                    (a.Condition != null && a.Condition.ToUpper().Contains(query)) ||
                    (a.AlertType != null && a.AlertType.ToUpper().Contains(query)) ||
                    (a.Notes != null && a.Notes.ToUpper().Contains(query)) ||
                    (a.IndicatorName != null && a.IndicatorName.ToUpper().Contains(query))
                );
            }

            // Sort: Newest first, then by priority, then by category
            filtered = filtered.OrderByDescending(a => a.CreatedDate)
                               .ThenBy(a => a.Priority)
                               .ThenBy(a => a.Category);

            AlertsDataGrid.ItemsSource = filtered.ToList();
        }

        // Add missing event handler stubs for XAML events
        private void DeleteAlert_Click(object sender, RoutedEventArgs e)
        {
            // TODO: Implement delete alert logic
        }

        private void EditAlert_Click(object sender, RoutedEventArgs e)
        {
            var button = sender as Button;
            var alert = button?.DataContext as AlertModel;
            
            if (alert != null)
            {
                currentEditAlert = alert;
                
                // Fill in the form with the alert details
                AlertNameTextBox.Text = alert.Name;
                SymbolTextBox.Text = alert.Symbol;
                AlertTypeComboBox.SelectedItem = AlertTypeComboBox.Items.Cast<ComboBoxItem>()
                    .FirstOrDefault(i => i.Content.ToString() == alert.AlertType);
                NotesTextBox.Text = alert.Notes;
                IsActiveCheckBox.IsChecked = alert.IsActive;
                
                // Set priority
                foreach (ComboBoxItem item in PriorityComboBox.Items)
                {
                    if ((int)item.Tag == alert.Priority)
                    {
                        PriorityComboBox.SelectedItem = item;
                        break;
                    }
                }
                
                // Set category
                foreach (ComboBoxItem item in AlertCategoryComboBox.Items)
                {
                    if (item.Tag.ToString() == alert.Category.ToString())
                    {
                        AlertCategoryComboBox.SelectedItem = item;
                        break;
                    }
                }
                
                // Handle category-specific fields
                if (alert.Category == AlertCategory.TechnicalIndicator)
                {
                    // Show indicator panel
                    PriceConditionPanel.Visibility = Visibility.Collapsed;
                    IndicatorConditionPanel.Visibility = Visibility.Visible;
                    
                    // Set indicator name
                    foreach (ComboBoxItem item in IndicatorNameComboBox.Items)
                    {
                        if (item.Content.ToString() == alert.IndicatorName)
                        {
                            IndicatorNameComboBox.SelectedItem = item;
                            break;
                        }
                    }
                    
                    // Set comparison operator
                    foreach (ComboBoxItem item in ComparisonOperatorComboBox.Items)
                    {
                        if (item.Tag.ToString() == alert.ComparisonOperator.ToString())
                        {
                            ComparisonOperatorComboBox.SelectedItem = item;
                            break;
                        }
                    }
                    
                    ThresholdValueTextBox.Text = alert.ThresholdValue.ToString();
                }
                else
                {
                    // Show price panel
                    PriceConditionPanel.Visibility = Visibility.Visible;
                    IndicatorConditionPanel.Visibility = Visibility.Collapsed;
                    
                    // Set condition
                    foreach (ComboBoxItem item in ConditionComboBox.Items)
                    {
                        if (item.Content.ToString() == alert.Condition)
                        {
                            ConditionComboBox.SelectedItem = item;
                            break;
                        }
                    }
                    
                    TriggerPriceTextBox.Text = alert.TriggerPrice.ToString();
                }
                
                // Set sound preferences and visual indicators for existing alert
                EnableSoundCheckBox.IsChecked = alert.EnableSound;
                
                // Set sound file
                foreach (ComboBoxItem item in SoundFileComboBox.Items)
                {
                    if (item.Tag.ToString() == alert.SoundFileName)
                    {
                        SoundFileComboBox.SelectedItem = item;
                        break;
                    }
                }
                
                // Set visual indicator type
                foreach (ComboBoxItem item in VisualIndicatorTypeComboBox.Items)
                {
                    if (item.Tag.ToString() == alert.VisualIndicatorType.ToString())
                    {
                        VisualIndicatorTypeComboBox.SelectedItem = item;
                        break;
                    }
                }
                
                // Set visual indicator color
                foreach (ComboBoxItem item in VisualIndicatorColorComboBox.Items)
                {
                    if (item.Tag.ToString() == alert.VisualIndicatorColor)
                    {
                        VisualIndicatorColorComboBox.SelectedItem = item;
                        break;
                    }
                }
                
                // Show the popup
                AlertEditorPopup.IsOpen = true;
            }
        }

        private void AddAlertButton_Click(object sender, RoutedEventArgs e)
        {
            // Reset the form for a new alert
            AlertNameTextBox.Text = string.Empty;
            SymbolTextBox.Text = string.Empty;
            ConditionComboBox.SelectedIndex = 0;
            TriggerPriceTextBox.Text = string.Empty;
            AlertTypeComboBox.SelectedIndex = 0;
            PriorityComboBox.SelectedIndex = 1; // Medium priority
            NotesTextBox.Text = string.Empty;
            IsActiveCheckBox.IsChecked = true;
            AlertCategoryComboBox.SelectedIndex = 0; // Standard
            
            // Reset sound and visual indicator settings
            EnableSoundCheckBox.IsChecked = true;
            SoundFileComboBox.SelectedIndex = 0; // Default based on category
            VisualIndicatorTypeComboBox.SelectedIndex = 0; // Toast
            VisualIndicatorColorComboBox.SelectedIndex = 0; // Yellow
            
            // Show price condition panel by default
            PriceConditionPanel.Visibility = Visibility.Visible;
            IndicatorConditionPanel.Visibility = Visibility.Collapsed;
            
            currentEditAlert = null; // Creating a new alert, not editing
            
            // Show the popup
            AlertEditorPopup.IsOpen = true;
        }

        private void SaveAlertButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Create or update an alert model
                var alert = currentEditAlert ?? new AlertModel();
                
                alert.Name = AlertNameTextBox.Text;
                alert.Symbol = SymbolTextBox.Text.ToUpper();
                alert.AlertType = (AlertTypeComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();
                alert.Notes = NotesTextBox.Text;
                alert.IsActive = IsActiveCheckBox.IsChecked ?? true;
                
                // Set sound preferences
                alert.EnableSound = EnableSoundCheckBox.IsChecked ?? true;
                var soundFileItem = SoundFileComboBox.SelectedItem as ComboBoxItem;
                alert.SoundFileName = soundFileItem?.Tag?.ToString() ?? "";
                
                // Set visual indicator preferences
                var visualTypeItem = VisualIndicatorTypeComboBox.SelectedItem as ComboBoxItem;
                if (visualTypeItem != null && Enum.TryParse(visualTypeItem.Tag.ToString(), out VisualIndicatorType visualType))
                {
                    alert.VisualIndicatorType = visualType;
                }
                
                var colorItem = VisualIndicatorColorComboBox.SelectedItem as ComboBoxItem;
                alert.VisualIndicatorColor = colorItem?.Tag?.ToString() ?? "#FFFF00"; // Default to yellow
                
                // Handle priority setting
                var selectedPriorityItem = PriorityComboBox.SelectedItem as ComboBoxItem;
                alert.Priority = int.Parse(selectedPriorityItem?.Tag?.ToString() ?? "2");
                
                // Get selected alert category
                var selectedCategoryItem = AlertCategoryComboBox.SelectedItem as ComboBoxItem;
                alert.Category = Enum.Parse<AlertCategory>(selectedCategoryItem?.Tag?.ToString() ?? "Standard");
                
                if (alert.Category == AlertCategory.TechnicalIndicator)
                {
                    // Technical indicator alert
                    alert.IndicatorName = (IndicatorNameComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();
                    
                    var comparisonOpItem = ComparisonOperatorComboBox.SelectedItem as ComboBoxItem;
                    alert.ComparisonOperator = Enum.Parse<ComparisonOperator>(comparisonOpItem?.Tag?.ToString() ?? "GreaterThan");
                    
                    if (double.TryParse(ThresholdValueTextBox.Text, out double thresholdValue))
                    {
                        alert.ThresholdValue = thresholdValue;
                    }
                    
                    // Format the condition string for display
                    alert.Condition = $"{alert.IndicatorName} {TechnicalIndicatorAlertService.GetOperatorDisplayString(alert.ComparisonOperator)} {alert.ThresholdValue:F2}";
                    
                    // Add specific notes for volume spike alerts
                    if (alert.IndicatorName == "VolumeSpike" && string.IsNullOrWhiteSpace(alert.Notes))
                    {
                        alert.Notes = "Detects unusual trading volume increases that may indicate breakout opportunities or significant market events.";
                    }
                    
                    // Set default sound if none selected
                    if (string.IsNullOrEmpty(alert.SoundFileName))
                    {
                        alert.SoundFileName = "indicator.wav";
                    }
                }
                else
                {
                    // Standard price alert
                    alert.Condition = (ConditionComboBox.SelectedItem as ComboBoxItem)?.Content.ToString();
                    
                    if (double.TryParse(TriggerPriceTextBox.Text, out double triggerPrice))
                    {
                        alert.TriggerPrice = triggerPrice;
                    }
                    
                    // Set default sound if none selected
                    if (string.IsNullOrEmpty(alert.SoundFileName))
                    {
                        switch (alert.Category)
                        {
                            case AlertCategory.Opportunity:
                                alert.SoundFileName = "opportunity.wav";
                                break;
                            case AlertCategory.Prediction:
                                alert.SoundFileName = "prediction.wav";
                                break;
                            default:
                                alert.SoundFileName = "alert.wav";
                                break;
                        }
                    }
                }
                
                // If it's a new alert, add it
                if (currentEditAlert == null)
                {
                    AddAlert(alert);
                }
                else
                {
                    // Just refresh the UI since we modified the alert in-place
                    ApplyFilter();
                }
                
                // Close the popup
                AlertEditorPopup.IsOpen = false;
                
                // If it's a high priority alert, trigger a notification to test the sound/visual settings
                if (alert.Priority == 1)
                {
                    var notificationService = ServiceLocator.Resolve<INotificationService>();
                    notificationService?.ShowAlertNotification(alert);
                }
            }
            catch (Exception ex)
            {
                EmitGlobalError($"Error saving alert: {ex.Message}", ex);
            }
        }

        private void CancelAlertButton_Click(object sender, RoutedEventArgs e)
        {
            // Just close the popup
            AlertEditorPopup.IsOpen = false;
        }

        private void AlertCategoryComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var comboBoxItem = AlertCategoryComboBox.SelectedItem as ComboBoxItem;
            if (comboBoxItem != null)
            {
                string selectedCategory = comboBoxItem.Tag.ToString();
                
                // Show/hide panels based on selected category
                if (selectedCategory == "TechnicalIndicator")
                {
                    PriceConditionPanel.Visibility = Visibility.Collapsed;
                    IndicatorConditionPanel.Visibility = Visibility.Visible;
                }
                else
                {
                    PriceConditionPanel.Visibility = Visibility.Visible;
                    IndicatorConditionPanel.Visibility = Visibility.Collapsed;
                }
            }
        }
        
        private void IndicatorNameComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var comboBoxItem = IndicatorNameComboBox.SelectedItem as ComboBoxItem;
            if (comboBoxItem != null)
            {
                string selectedIndicator = comboBoxItem.Content.ToString();
                
                // Set recommended default threshold values for certain indicators
                if (selectedIndicator == "VolumeSpike")
                {
                    // Set default threshold for volume spike (2.0 by default)
                    ThresholdValueTextBox.Text = VolumeAlertService.GetRecommendedVolumeRatioThreshold().ToString();
                    
                    // Set default operator to GreaterThan
                    foreach (ComboBoxItem item in ComparisonOperatorComboBox.Items)
                    {
                        if (item.Tag.ToString() == "GreaterThan")
                        {
                            ComparisonOperatorComboBox.SelectedItem = item;
                            break;
                        }
                    }
                    
                    // Set explanation text
                    NotesTextBox.Text = "Detects unusual trading volume increases that may indicate breakout opportunities or significant market events.";
                }
                else if (selectedIndicator == "Volume")
                {
                    // For regular volume, no special defaults needed
                }
            }
        }

        private void FilterComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ApplyFilter();
        }

        private void SearchBox_KeyUp(object sender, System.Windows.Input.KeyEventArgs e)
        {
            ApplyFilter();
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            // Check for new technical indicator alerts directly
            try
            {
                await CheckTechnicalIndicatorAlerts();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error refreshing alerts", ex.ToString());
            }
            
            ApplyFilter();
        }

        private void AlertsDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // TODO: Implement logic for when alert selection changes
        }

        /// <summary>
        /// Emits a global alert that will be displayed in all AlertsControl instances.
        /// Usage: AlertsControl.EmitGlobalAlert(new AlertModel { ... });
        /// </summary>
        public void EmitGlobalAlert(AlertModel alert)
        {
            GlobalAlertEmitted?.Invoke(alert);
            // Send email if enabled in settings
            var settings = settingsService.GetDefaultSettingsProfile();
            EmailAlertService.SendAlertEmail(alert, settings);
        }

        /// <summary>
        /// Helper method to create a volume spike alert with recommended defaults
        /// </summary>
        /// <param name="symbol">The stock symbol to monitor</param>
        /// <returns>A new AlertModel configured for volume spike detection</returns>
        public static AlertModel CreateVolumeAlertWithDefaults(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol is required", nameof(symbol));

            // Get recommended threshold from VolumeAlertService
            double recommendedThreshold = VolumeAlertService.GetRecommendedVolumeRatioThreshold();

            var alert = new AlertModel
            {
                Name = $"{symbol} Volume Spike Alert",
                Symbol = symbol.ToUpper(),
                Category = AlertCategory.TechnicalIndicator,
                IndicatorName = "VolumeSpike",
                ComparisonOperator = ComparisonOperator.GreaterThan,
                ThresholdValue = recommendedThreshold,
                Condition = $"VolumeSpike > {recommendedThreshold:F2}",
                AlertType = "Volume Spike",
                Priority = 1, // High priority
                IsActive = true,
                Notes = "Automatically detects unusual trading volume increases that may indicate potential breakout opportunities or market events."
            };

            return alert;
        }
        
        /// <summary>
        /// Helper method to create a pattern alert for a symbol
        /// </summary>
        /// <param name="symbol">The stock symbol to monitor</param>
        /// <returns>A new AlertModel configured for pattern detection</returns>
        public static AlertModel CreatePatternAlertWithDefaults(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol is required", nameof(symbol));
            
            var alert = new AlertModel
            {
                Name = $"{symbol} Pattern Alert",
                Symbol = symbol.ToUpper(),
                Category = AlertCategory.Pattern,
                Condition = "Pattern Detection",
                AlertType = "Price Patterns",
                Priority = 2, // Medium priority
                IsActive = true,
                Notes = "Automatically detects chart patterns like double tops/bottoms, head & shoulders, triangles, and engulfing patterns."
            };
            
            return alert;
        }

        /// <summary>
        /// Emits a global error alert from any exception or error message.
        /// </summary>
        public void EmitGlobalError(string message, Exception ex = null)
        {
            var alert = new AlertModel
            {
                Name = message,
                Condition = "Error",
                AlertType = "Error",
                IsActive = true,
                Priority = 1,
                CreatedDate = DateTime.Now,
                Category = AlertCategory.Global,
                Notes = ex?.ToString() ?? string.Empty
            };
            EmitGlobalAlert(alert);
        }
    }

    public class AlertCategoryToBrushConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value is AlertCategory category)
            {
                return category switch
                {
                    AlertCategory.Standard => Brushes.Transparent,
                    AlertCategory.Opportunity => Brushes.MediumPurple,
                    AlertCategory.Prediction => Brushes.DodgerBlue,
                    AlertCategory.Global => Brushes.Gold,
                    AlertCategory.Pattern => Brushes.MediumSeaGreen,
                    _ => Brushes.Transparent
                };
            }
            return Brushes.Transparent;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
