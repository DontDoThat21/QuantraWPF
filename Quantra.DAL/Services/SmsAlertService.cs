using System;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    public static class SmsAlertService
    {
        // Send a generic SMS
        public static void SendSms(string to, string message)
        {
            var service = new SmsService();
            service.SendSmsAsync(to, message).Wait(); // Blocking call for simplicity
        }

        // Send an alert SMS if enabled for the alert type
        public static void SendAlertSms(AlertModel alert, DatabaseSettingsProfile settings)
        {
            if (settings == null || !settings.EnableSmsAlerts || string.IsNullOrWhiteSpace(settings.AlertPhoneNumber))
                return;

            bool shouldSend = false;
            switch (alert.Category)
            {
                case AlertCategory.Standard:
                    shouldSend = settings.EnableStandardAlertSms;
                    break;
                case AlertCategory.Opportunity:
                    shouldSend = settings.EnableOpportunityAlertSms;
                    break;
                case AlertCategory.Prediction:
                    shouldSend = settings.EnablePredictionAlertSms;
                    break;
                case AlertCategory.Global:
                    shouldSend = settings.EnableGlobalAlertSms;
                    break;
            }
            if (!shouldSend) return;

            string message = $"QUANTRA ALERT: {alert.Name} [{alert.Category}] - {alert.Symbol}: {alert.Condition}";
            
            // Add technical indicator specific details if applicable
            if (alert.Category == AlertCategory.TechnicalIndicator && alert.CurrentIndicatorValue > 0)
            {
                message += $"\nIndicator: {alert.IndicatorName} {alert.ComparisonOperator} {alert.ThresholdValue}";
            }
            
            SendSms(settings.AlertPhoneNumber, message);
        }
    }
}