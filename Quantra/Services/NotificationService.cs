using Quantra.Services.Interfaces;
using MaterialDesignThemes.Wpf;
using System;
using System.Windows;
using System.Windows.Media;
using Quantra.Models;
using System.Threading.Tasks;

namespace Quantra.Services
{
    public class NotificationService : INotificationService
    {
        private readonly IAudioService _audioService;
        private readonly UserSettings _userSettings;
        
        public delegate void ShowNotificationHandler(string message, PackIconKind icon, Color iconColor);
        public event ShowNotificationHandler OnShowNotification;
        
        public delegate void ShowCustomNotificationHandler(string message, string visualType, Color backgroundColor, double duration);
        public event ShowCustomNotificationHandler OnShowCustomNotification;

        public NotificationService(UserSettings userSettings, IAudioService audioService)
        {
            _userSettings = userSettings ?? throw new ArgumentNullException(nameof(userSettings));
            _audioService = audioService ?? throw new ArgumentNullException(nameof(audioService));
        }

        public void ShowNotification(string message, PackIconKind icon, Color iconColor)
        {
            OnShowNotification?.Invoke(message, icon, iconColor);
        }

        public void ShowSuccess(string message)
        {
            OnShowNotification?.Invoke(message, PackIconKind.Check, Colors.Green);
        }

        public void ShowError(string message)
        {
            OnShowNotification?.Invoke(message, PackIconKind.Error, Colors.Red);
        }

        public void ShowWarning(string message)
        {
            OnShowNotification?.Invoke(message, PackIconKind.AlertCircle, Colors.Orange);
        }

        public void ShowInfo(string message)
        {
            OnShowNotification?.Invoke(message, PackIconKind.Information, Colors.Blue);
        }
        
        public void ShowTradeNotification(TradeRecord trade)
        {
            // Format a detailed trade notification message
            string message = $"Trade Executed: {trade.Action} {trade.Symbol}\n" +
                            $"Quantity: {trade.Quantity}\n" + 
                            $"Price: ${trade.Price:F2}\n" +
                            $"Target: ${trade.TargetPrice:F2}\n" + 
                            $"Time: {trade.ExecutionTime:g}";
            
            // Use success icon and color for trade notifications
            OnShowNotification?.Invoke(message, PackIconKind.TrendingUp, Colors.Green);
            
            // Send push notification if enabled
            try
            {
                var settings = SettingsService.GetDefaultSettingsProfile();
                if (settings != null && settings.EnablePushNotifications && settings.EnableTradeExecutionPushNotifications)
                {
                    var pushService = new PushNotificationService();
                    string title = $"Quantra Trade: {trade.Action} {trade.Symbol}";
                    string pushMessage = $"{trade.Action} {trade.Quantity} {trade.Symbol} @ ${trade.Price:F2}";
                    var payload = new
                    {
                        TradeId = trade.Id,
                        Symbol = trade.Symbol,
                        Action = trade.Action,
                        Price = trade.Price,
                        Quantity = trade.Quantity,
                        ExecutionTime = trade.ExecutionTime
                    };
                    _ = pushService.SendNotificationToUserAsync(settings.PushNotificationUserId, title, pushMessage, payload);
                }
            }
            catch (Exception ex)
            {
                // Log error but don't throw - push notifications shouldn't break the main flow
                DatabaseMonolith.Log("Error", $"Failed to send push notification for trade: {ex.Message}", ex.ToString());
            }
        }
        
        public void ShowAlertNotification(AlertModel alert)
        {
            if (alert == null)
                return;
                
            // Play the alert sound if enabled
            if (_audioService != null && _userSettings.EnableAlertSounds && alert.EnableSound)
            {
                _audioService.PlayAlertSound(alert);
            }
            
            // Prepare notification message
            string message = $"Alert Triggered: {alert.Name}\n" +
                            $"Symbol: {alert.Symbol}\n" +
                            $"Condition: {alert.Condition}\n" +
                            $"Time: {DateTime.Now:g}";
            
            // Show visual indicator if enabled
            if (_userSettings.EnableVisualIndicators)
            {
                // Determine which type of visual indicator to show
                string visualType = alert.VisualIndicatorType.ToString();
                
                // Get the color for the visual indicator, use the alert-specific color if set
                Color backgroundColor;
                if (!string.IsNullOrEmpty(alert.VisualIndicatorColor))
                {
                    try
                    {
                        var brush = new BrushConverter().ConvertFrom(alert.VisualIndicatorColor) as SolidColorBrush;
                        backgroundColor = brush?.Color ?? Colors.Yellow;
                    }
                    catch
                    {
                        backgroundColor = Colors.Yellow; // Default if conversion fails
                    }
                }
                else
                {
                    // Default colors based on alert category
                    backgroundColor = alert.Category switch
                    {
                        AlertCategory.Opportunity => Colors.Green,
                        AlertCategory.Prediction => Colors.Blue,
                        AlertCategory.TechnicalIndicator => Colors.Purple,
                        _ => Colors.Orange
                    };
                }
                
                // Show the custom visual notification
                OnShowCustomNotification?.Invoke(message, visualType, backgroundColor, _userSettings.VisualIndicatorDuration);
            }
            else
            {
                // Fall back to standard notification if custom visuals are disabled
                PackIconKind icon = alert.Category switch
                {
                    AlertCategory.Opportunity => PackIconKind.ChartLine,
                    AlertCategory.Prediction => PackIconKind.ChartBubble,
                    AlertCategory.TechnicalIndicator => PackIconKind.Calculator,
                    _ => PackIconKind.AlertCircle
                };
                
                Color iconColor = alert.Priority switch
                {
                    1 => Colors.Red,      // High priority
                    3 => Colors.Blue,     // Low priority
                    _ => Colors.Orange    // Medium priority or default
                };
                
                OnShowNotification?.Invoke(message, icon, iconColor);
            }
        }
    }
}
