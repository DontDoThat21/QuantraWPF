using System;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Data;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class TradingService : ITradingService
    {
        private readonly IEmailService _emailService;
        private readonly INotificationService _notificationService;
        private readonly ISmsService _smsService;
        private readonly UserSettings _userSettings;

        public TradingService(IEmailService emailService, INotificationService notificationService, ISmsService smsService)
        {
            _emailService = emailService ?? throw new ArgumentNullException(nameof(emailService));
            _notificationService = notificationService ?? throw new ArgumentNullException(nameof(notificationService));
            _smsService = smsService ?? throw new ArgumentNullException(nameof(smsService));
            
            // Load user settings for notification preferences
            _userSettings = DatabaseMonolith.GetUserSettings();
        }

        public async Task<bool> ExecuteTradeAsync(string symbol, string action, double price, double targetPrice)
        {
            try
            {
                // Create a trade record
                var trade = new TradeRecord
                {
                    Symbol = symbol,
                    Action = action,
                    Price = price,
                    TargetPrice = targetPrice,
                    ExecutionTime = DateTime.UtcNow,
                    Status = "Executed",
                    Quantity = 100 // Default quantity, should be parameterized in a real implementation
                };

                // Save to database
                DatabaseMonolith.SaveTradeRecord(trade);
                
                // Display in-app notification if enabled
                if (_userSettings.EnableTradeNotifications)
                {
                    _notificationService.ShowTradeNotification(trade);
                }

                // Send email notification if enabled
                if (_userSettings.EnableEmailAlerts && _userSettings.EnableTradeNotifications)
                {
                    await _emailService.SendEmailAsync(
                        _userSettings.AlertEmail,
                        $"Trade Executed: {action} {symbol}",
                        $"Trade Details:\nSymbol: {symbol}\nAction: {action}\nQuantity: {trade.Quantity}\nPrice: {price:C}\nTarget: {targetPrice:C}\nTime: {trade.ExecutionTime}"
                    );
                }
                
                // Send SMS notification if enabled
                if (_userSettings.EnableSmsAlerts && _userSettings.EnableTradeNotifications && !string.IsNullOrEmpty(_userSettings.AlertPhoneNumber))
                {
                    await _smsService.SendSmsAsync(
                        _userSettings.AlertPhoneNumber,
                        $"Trade Executed: {action} {symbol} - {trade.Quantity} @ ${price:F2}"
                    );
                }

                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error executing trade: {ex.Message}", ex.ToString());
                _notificationService.ShowError($"Error executing trade: {ex.Message}");
                return false;
            }
        }
    }
}