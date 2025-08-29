using System.ComponentModel.DataAnnotations;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// Notification configuration
    /// </summary>
    public class NotificationConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable price alerts
        /// </summary>
        public bool EnablePriceAlerts
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable trade notifications
        /// </summary>
        public bool EnableTradeNotifications
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Email notification settings
        /// </summary>
        public EmailNotificationConfig Email
        {
            get => Get<EmailNotificationConfig>(new EmailNotificationConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// SMS notification settings
        /// </summary>
        public SmsNotificationConfig SMS
        {
            get => Get<SmsNotificationConfig>(new SmsNotificationConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// Push notification settings
        /// </summary>
        public PushNotificationConfig Push
        {
            get => Get<PushNotificationConfig>(new PushNotificationConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// Sound notification settings
        /// </summary>
        public SoundNotificationConfig Sound
        {
            get => Get<SoundNotificationConfig>(new SoundNotificationConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// Visual notification settings
        /// </summary>
        public VisualNotificationConfig Visual
        {
            get => Get<VisualNotificationConfig>(new VisualNotificationConfig());
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Email notification settings
    /// </summary>
    public class EmailNotificationConfig : ConfigModelBase
    {
        /// <summary>
        /// Default email recipient
        /// </summary>
        [EmailAddress]
        public string DefaultRecipient
        {
            get => Get(string.Empty);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable email alerts
        /// </summary>
        public bool EnableEmailAlerts
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable standard alert emails
        /// </summary>
        public bool EnableStandardAlertEmails
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable opportunity alert emails
        /// </summary>
        public bool EnableOpportunityAlertEmails
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable prediction alert emails
        /// </summary>
        public bool EnablePredictionAlertEmails
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable global alert emails
        /// </summary>
        public bool EnableGlobalAlertEmails
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable system health alert emails
        /// </summary>
        public bool EnableSystemHealthAlertEmails
        {
            get => Get(false);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// SMS notification settings
    /// </summary>
    public class SmsNotificationConfig : ConfigModelBase
    {
        /// <summary>
        /// Default phone number
        /// </summary>
        [Phone]
        public string DefaultRecipient
        {
            get => Get(string.Empty);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable SMS alerts
        /// </summary>
        public bool EnableSmsAlerts
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable standard alert SMS
        /// </summary>
        public bool EnableStandardAlertSms
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable opportunity alert SMS
        /// </summary>
        public bool EnableOpportunityAlertSms
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable prediction alert SMS
        /// </summary>
        public bool EnablePredictionAlertSms
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable global alert SMS
        /// </summary>
        public bool EnableGlobalAlertSms
        {
            get => Get(false);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Push notification settings
    /// </summary>
    public class PushNotificationConfig : ConfigModelBase
    {
        /// <summary>
        /// User ID for push notifications
        /// </summary>
        public string UserId
        {
            get => Get(string.Empty);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable push notifications
        /// </summary>
        public bool EnablePushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable standard push notifications
        /// </summary>
        public bool EnableStandardAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable opportunity push notifications
        /// </summary>
        public bool EnableOpportunityAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable prediction push notifications
        /// </summary>
        public bool EnablePredictionAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable global push notifications
        /// </summary>
        public bool EnableGlobalAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable technical indicator push notifications
        /// </summary>
        public bool EnableTechnicalIndicatorAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable sentiment shift push notifications
        /// </summary>
        public bool EnableSentimentShiftAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable system health push notifications
        /// </summary>
        public bool EnableSystemHealthAlertPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable trade execution push notifications
        /// </summary>
        public bool EnableTradeExecutionPushNotifications
        {
            get => Get(false);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Sound notification settings
    /// </summary>
    public class SoundNotificationConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable alert sounds
        /// </summary>
        public bool EnableAlertSounds
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Default alert sound
        /// </summary>
        public string DefaultAlertSound
        {
            get => Get("alert.wav");
            set => Set(value);
        }
        
        /// <summary>
        /// Default opportunity sound
        /// </summary>
        public string DefaultOpportunitySound
        {
            get => Get("opportunity.wav");
            set => Set(value);
        }
        
        /// <summary>
        /// Default prediction sound
        /// </summary>
        public string DefaultPredictionSound
        {
            get => Get("prediction.wav");
            set => Set(value);
        }
        
        /// <summary>
        /// Default technical indicator sound
        /// </summary>
        public string DefaultTechnicalIndicatorSound
        {
            get => Get("indicator.wav");
            set => Set(value);
        }
        
        /// <summary>
        /// Alert volume (0-100)
        /// </summary>
        public int AlertVolume
        {
            get => Get(80);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Visual notification settings
    /// </summary>
    public class VisualNotificationConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable visual indicators
        /// </summary>
        public bool EnableVisualIndicators
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Default visual indicator type (Toast, Banner, Popup, Flashcard)
        /// </summary>
        public string DefaultVisualIndicatorType
        {
            get => Get("Toast");
            set => Set(value);
        }
        
        /// <summary>
        /// Default visual indicator color
        /// </summary>
        public string DefaultVisualIndicatorColor
        {
            get => Get("#FFFF00");
            set => Set(value);
        }
        
        /// <summary>
        /// Visual indicator duration in seconds
        /// </summary>
        public int VisualIndicatorDuration
        {
            get => Get(5);
            set => Set(value);
        }
    }
}