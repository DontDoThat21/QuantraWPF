using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for handling push notifications to mobile devices
    /// </summary>
    public interface IPushNotificationService
    {
        /// <summary>
        /// Registers a device token for push notifications
        /// </summary>
        /// <param name="deviceToken">The device token to register</param>
        /// <param name="deviceType">The type of device (iOS, Android)</param>
        /// <param name="userId">The user ID associated with the device</param>
        Task RegisterDeviceAsync(string deviceToken, string deviceType, string userId);
        
        /// <summary>
        /// Unregisters a device token to stop receiving push notifications
        /// </summary>
        /// <param name="deviceToken">The device token to unregister</param>
        Task UnregisterDeviceAsync(string deviceToken);
        
        /// <summary>
        /// Sends a push notification to a specific device
        /// </summary>
        /// <param name="deviceToken">The device token to send the notification to</param>
        /// <param name="title">Notification title</param>
        /// <param name="message">Notification message</param>
        /// <param name="payload">Optional additional data payload</param>
        Task SendNotificationAsync(string deviceToken, string title, string message, object payload = null);
        
        /// <summary>
        /// Sends a push notification to a user's registered devices
        /// </summary>
        /// <param name="userId">The user ID to send the notification to</param>
        /// <param name="title">Notification title</param>
        /// <param name="message">Notification message</param>
        /// <param name="payload">Optional additional data payload</param>
        Task SendNotificationToUserAsync(string userId, string title, string message, object payload = null);
        
        /// <summary>
        /// Sends an alert as a push notification
        /// </summary>
        /// <param name="alert">The alert to send</param>
        /// <param name="userId">The user ID to send the notification to</param>
        Task SendAlertNotificationAsync(AlertModel alert, string userId);
    }
}