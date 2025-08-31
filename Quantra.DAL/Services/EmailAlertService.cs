using System.Net;
using System.Net.Mail;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public static class EmailAlertService
    {
        // Send a generic email
        public static void SendEmail(string to, string subject, string body)
        {
            var from = "daytrader.alerts@gmail.com"; // Use a real sender or config
            var smtp = new SmtpClient("smtp.gmail.com", 587)
            {
                Credentials = new NetworkCredential(from, "your_app_password_here"),
                EnableSsl = true
            };
            var mail = new MailMessage(from, to, subject, body);
            smtp.Send(mail);
        }

        // Send an alert email if enabled for the alert type
        public static void SendAlertEmail(AlertModel alert, DatabaseSettingsProfile settings)
        {
            if (settings == null || !settings.EnableEmailAlerts || string.IsNullOrWhiteSpace(settings.AlertEmail))
                return;

            bool shouldSend = false;
            switch (alert.Category)
            {
                case AlertCategory.Standard:
                    shouldSend = settings.EnableStandardAlertEmails;
                    break;
                case AlertCategory.Opportunity:
                    shouldSend = settings.EnableOpportunityAlertEmails;
                    break;
                case AlertCategory.Prediction:
                    shouldSend = settings.EnablePredictionAlertEmails;
                    break;
                case AlertCategory.Global:
                    shouldSend = settings.EnableGlobalAlertEmails;
                    break;
                case AlertCategory.SystemHealth:
                    shouldSend = settings.EnableSystemHealthAlertEmails;
                    break;
            }
            if (!shouldSend) return;

            string subject = $"Quantra Alert: {alert.Name} [{alert.Category}]";
            string body = $"Alert Name: {alert.Name}\nSymbol: {alert.Symbol}\nCondition: {alert.Condition}\nType: {alert.AlertType}\nPriority: {alert.Priority}\nNotes: {alert.Notes}\nCreated: {alert.CreatedDate}\n";
            
            // Add technical indicator specific details if applicable
            if (alert.Category == AlertCategory.TechnicalIndicator && alert.CurrentIndicatorValue > 0)
            {
                body += $"\nTechnical Indicator: {alert.IndicatorName}\n";
                body += $"Threshold: {alert.ThresholdValue}\n";
                body += $"Current Value: {alert.CurrentIndicatorValue}\n";
                body += $"Operator: {alert.ComparisonOperator}\n";
            }
            
            SendEmail(settings.AlertEmail, subject, body);
        }
    }
}
