using MaterialDesignThemes.Wpf;
using Quantra.DAL.Notifications;
using Quantra.Models;
using System.Windows.Media;

namespace Quantra.DAL.Services.Interfaces
{
    public interface INotificationService
    {
        // Decoupled from WPF/MaterialDesign: use a DAL enum and hex color string
        void ShowNotification(string message, PackIconKind icon, Color color);
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
