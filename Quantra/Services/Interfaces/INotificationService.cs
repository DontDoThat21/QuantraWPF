using MaterialDesignThemes.Wpf;
using System.Windows.Media;
using Quantra.Models;

namespace Quantra.Services.Interfaces
{
    public interface INotificationService
    {
        void ShowNotification(string message, PackIconKind icon, Color iconColor);
        void ShowInfo(string message);
        void ShowSuccess(string message);
        void ShowWarning(string message);
        void ShowError(string message);
        
        /// <summary>
        /// Shows a trade execution notification with detailed information about the trade
        /// </summary>
        /// <param name="trade">The trade record containing details about the executed trade</param>
        void ShowTradeNotification(TradeRecord trade);
        
        /// <summary>
        /// Shows an alert notification with custom visual indicator and plays sound if enabled
        /// </summary>
        /// <param name="alert">The alert that was triggered</param>
        void ShowAlertNotification(AlertModel alert);
    }
}
