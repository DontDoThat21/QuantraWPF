using System;
using MaterialDesignThemes.Wpf;
using Quantra.DAL.Notifications;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using System.Windows.Media;

namespace Quantra.DAL.Services
{
    public class NotificationService : INotificationService
    {
        private readonly IAudioService _audioService;
        private readonly UserSettings _userSettings;
        private readonly ISettingsService _settingsService;
        
        public delegate void ShowNotificationHandler(string message, NotificationIcon icon, string iconColorHex);
        public event ShowNotificationHandler OnShowNotification;
        
        public delegate void ShowCustomNotificationHandler(string message, string visualType, string backgroundColorHex, double duration);
        public event ShowCustomNotificationHandler OnShowCustomNotification;

        public NotificationService(UserSettings userSettings, IAudioService audioService, ISettingsService settingsService)
        {
            _userSettings = userSettings ?? throw new ArgumentNullException(nameof(userSettings));
            _audioService = audioService ?? throw new ArgumentNullException(nameof(audioService));
            _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
        }

        public void ShowNotification(string message, PackIconKind icon, Color color)
        {
            var notificationIcon = ConvertPackIconKindToNotificationIcon(icon);
            var colorHex = ConvertColorToHex(color);
            OnShowNotification?.Invoke(message, notificationIcon, colorHex);
        }

        public void ShowSuccess(string message)
        {
            OnShowNotification?.Invoke(message, NotificationIcon.Success, "#00C853"); // green
        }

        public void ShowError(string message)
        {
            OnShowNotification?.Invoke(message, NotificationIcon.Error, "#FF1744"); // red
        }

        public void ShowWarning(string message)
        {
            OnShowNotification?.Invoke(message, NotificationIcon.Warning, "#FFA000"); // orange
        }

        public void ShowInfo(string message)
        {
            OnShowNotification?.Invoke(message, NotificationIcon.Info, "#2196F3"); // blue
        }
        
        public void ShowTradeNotification(TradeRecord trade)
        {
            if (trade == null) return;

            string message = $"Trade Executed: {trade.Action} {trade.Symbol}\n" +
                            $"Quantity: {trade.Quantity}\n" + 
                            $"Price: ${trade.Price:F2}\n" +
                            $"Target: ${trade.TargetPrice:F2}\n" + 
                            $"Time: {trade.ExecutionTime:g}";
            
            OnShowNotification?.Invoke(message, NotificationIcon.TrendingUp, "#00C853");
            
            try
            {
                var settings = _settingsService.GetDefaultSettingsProfile();
                if (settings != null && settings.EnablePushNotifications && settings.EnableTradeExecutionPushNotifications)
                {
                    var pushService = new PushNotificationService();
                    string title = $"Quantra Trade: {trade.Action} {trade.Symbol}";
                    string pushMessage = $"{trade.Action} {trade.Quantity} {trade.Symbol} @ ${trade.Price:F2}";
                    var payload = new
                    {
                        TradeId = trade.Id,
                        trade.Symbol,
                        trade.Action,
                        trade.Price,
                        trade.Quantity,
                        trade.ExecutionTime
                    };
                    _ = pushService.SendNotificationToUserAsync(settings.PushNotificationUserId, title, pushMessage, payload);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to send push notification for trade: {ex.Message}", ex.ToString());
            }
        }
        
        public void ShowAlertNotification(AlertModel alert)
        {
            if (alert == null)
                return;
                
            if (_audioService != null && _userSettings.EnableAlertSounds && alert.EnableSound)
            {
                _audioService.PlayAlertSound(alert);
            }
            
            string message = $"Alert Triggered: {alert.Name}\n" +
                            $"Symbol: {alert.Symbol}\n" +
                            $"Condition: {alert.Condition}\n" +
                            $"Time: {DateTime.Now:g}";
            
            if (_userSettings.EnableVisualIndicators)
            {
                string visualType = alert.VisualIndicatorType.ToString();
                
                string backgroundColorHex;
                if (!string.IsNullOrEmpty(alert.VisualIndicatorColor))
                {
                    backgroundColorHex = alert.VisualIndicatorColor;
                }
                else
                {
                    backgroundColorHex = alert.Category switch
                    {
                        AlertCategory.Opportunity => "#00C853", // green
                        AlertCategory.Prediction => "#2196F3", // blue
                        AlertCategory.TechnicalIndicator => "#9C27B0", // purple
                        _ => "#FFA000" // orange
                    };
                }
                
                OnShowCustomNotification?.Invoke(message, visualType, backgroundColorHex, _userSettings.VisualIndicatorDuration);
            }
            else
            {
                NotificationIcon icon = alert.Category switch
                {
                    AlertCategory.Opportunity => NotificationIcon.ChartLine,
                    AlertCategory.Prediction => NotificationIcon.ChartBubble,
                    AlertCategory.TechnicalIndicator => NotificationIcon.Calculator,
                    _ => NotificationIcon.Warning
                };
                
                string iconColorHex = alert.Priority switch
                {
                    1 => "#FF1744", // red high priority
                    3 => "#2196F3", // blue low priority
                    _ => "#FFA000"  // orange medium/default
                };
                
                OnShowNotification?.Invoke(message, icon, iconColorHex);
            }
        }

        private NotificationIcon ConvertPackIconKindToNotificationIcon(PackIconKind packIcon)
        {
            return packIcon switch
            {
                PackIconKind.Info => NotificationIcon.Info,
                PackIconKind.CheckCircle => NotificationIcon.Success,
                PackIconKind.Warning => NotificationIcon.Warning,
                PackIconKind.Error => NotificationIcon.Error,
                PackIconKind.TrendingUp => NotificationIcon.TrendingUp,
                PackIconKind.ChartLine => NotificationIcon.ChartLine,
                PackIconKind.ChartBubble => NotificationIcon.ChartBubble,
                PackIconKind.Calculator => NotificationIcon.Calculator,
                _ => NotificationIcon.Info
            };
        }

        private string ConvertColorToHex(Color color)
        {
            return $"#{color.R:X2}{color.G:X2}{color.B:X2}";
        }
    }
}
