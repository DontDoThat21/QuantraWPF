using System;
using Quantra.Models;

namespace Quantra.Services
{
    /// <summary>
    /// Service for sending alerts as push notifications to mobile devices
    /// </summary>
    public static class PushNotificationAlertService
    {
        private static readonly Lazy<PushNotificationService> _pushService = 
            new Lazy<PushNotificationService>(() => new PushNotificationService());
            
        // Send an alert as a push notification if enabled for the alert type
        public static void SendAlertPushNotification(AlertModel alert, DatabaseSettingsProfile settings)
        {
            if (settings == null || !settings.EnablePushNotifications || string.IsNullOrWhiteSpace(settings.PushNotificationUserId))
                return;

            bool shouldSend = false;
            switch (alert.Category)
            {
                case AlertCategory.Standard:
                    shouldSend = settings.EnableStandardAlertPushNotifications;
                    break;
                case AlertCategory.Opportunity:
                    shouldSend = settings.EnableOpportunityAlertPushNotifications;
                    break;
                case AlertCategory.Prediction:
                    shouldSend = settings.EnablePredictionAlertPushNotifications;
                    break;
                case AlertCategory.Global:
                    shouldSend = settings.EnableGlobalAlertPushNotifications;
                    break;
                case AlertCategory.TechnicalIndicator:
                    shouldSend = settings.EnableTechnicalIndicatorAlertPushNotifications;
                    break;
                case AlertCategory.Sentiment:
                    shouldSend = settings.EnableSentimentShiftAlertPushNotifications;
                    break;
                case AlertCategory.SystemHealth:
                    shouldSend = settings.EnableSystemHealthAlertPushNotifications;
                    break;
            }
            if (!shouldSend) return;

            try
            {
                // Log the push notification attempt
                DatabaseMonolith.Log("Info", $"Sending push notification for alert {alert.Id} to user {settings.PushNotificationUserId}", 
                    $"Alert: {alert.Name}, Category: {alert.Category}");
                
                // Send the notification asynchronously but don't wait for it
                _ = _pushService.Value.SendAlertNotificationAsync(alert, settings.PushNotificationUserId);
            }
            catch (Exception ex)
            {
                // Log any errors but don't rethrow - push notifications shouldn't break the main flow
                DatabaseMonolith.Log("Error", $"Failed to send push notification for alert {alert.Id}", ex.ToString());
            }
        }
    }
}