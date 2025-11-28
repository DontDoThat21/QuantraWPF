using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a settings profile
    /// </summary>
    [Table("SettingsProfiles")]
    public class SettingsProfile
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(100)]
        public string Name { get; set; }

        [MaxLength(500)]
        public string? Description { get; set; }

        public bool IsDefault { get; set; }

        public bool EnableApiModalChecks { get; set; }
        public int ApiTimeoutSeconds { get; set; }
        public int CacheDurationMinutes { get; set; }
        public bool EnableHistoricalDataCache { get; set; }
        public bool EnableDarkMode { get; set; }
        public int ChartUpdateIntervalSeconds { get; set; }
        public bool EnablePriceAlerts { get; set; }
        public bool EnableTradeNotifications { get; set; }
        public bool EnablePaperTrading { get; set; }
        
        [MaxLength(50)]
        public string? RiskLevel { get; set; }
     
        public int DefaultGridRows { get; set; }
        public int DefaultGridColumns { get; set; }
        
        [MaxLength(20)]
        public string? GridBorderColor { get; set; }

        // Email alert settings
        [MaxLength(255)]
        public string? AlertEmail { get; set; }
        public bool EnableEmailAlerts { get; set; }
        public bool EnableStandardAlertEmails { get; set; }
        public bool EnableOpportunityAlertEmails { get; set; }
        public bool EnablePredictionAlertEmails { get; set; }
        public bool EnableGlobalAlertEmails { get; set; }
        public bool EnableSystemHealthAlertEmails { get; set; }

        // SMS alert settings
        [MaxLength(50)]
        public string? AlertPhoneNumber { get; set; }
        public bool EnableSmsAlerts { get; set; }
        public bool EnableStandardAlertSms { get; set; }
        public bool EnableOpportunityAlertSms { get; set; }
        public bool EnablePredictionAlertSms { get; set; }
        public bool EnableGlobalAlertSms { get; set; }

        // Push notification settings
        [MaxLength(255)]
        public string? PushNotificationUserId { get; set; }
        public bool EnablePushNotifications { get; set; }
        public bool EnableStandardAlertPushNotifications { get; set; }
        public bool EnableOpportunityAlertPushNotifications { get; set; }
        public bool EnablePredictionAlertPushNotifications { get; set; }
        public bool EnableGlobalAlertPushNotifications { get; set; }
        public bool EnableTechnicalIndicatorAlertPushNotifications { get; set; }
        public bool EnableSentimentShiftAlertPushNotifications { get; set; }
        public bool EnableSystemHealthAlertPushNotifications { get; set; }
        public bool EnableTradeExecutionPushNotifications { get; set; }

        // Alert sound settings
        public bool EnableAlertSounds { get; set; }
        [MaxLength(255)]
        public string? DefaultAlertSound { get; set; }
        [MaxLength(255)]
        public string? DefaultOpportunitySound { get; set; }
        [MaxLength(255)]
        public string? DefaultPredictionSound { get; set; }
        [MaxLength(255)]
        public string? DefaultTechnicalIndicatorSound { get; set; }
        public int AlertVolume { get; set; }

        // Visual indicator settings
        public bool EnableVisualIndicators { get; set; }
        [MaxLength(50)]
        public string? DefaultVisualIndicatorType { get; set; }
        [MaxLength(20)]
        public string? DefaultVisualIndicatorColor { get; set; }
        public int VisualIndicatorDuration { get; set; }

        // Risk management settings
        public decimal AccountSize { get; set; }
        public decimal BaseRiskPercentage { get; set; }
        [MaxLength(50)]
        public string? PositionSizingMethod { get; set; }
        public decimal MaxPositionSizePercent { get; set; }
        public decimal FixedTradeAmount { get; set; }
        public bool UseVolatilityBasedSizing { get; set; }
        public decimal ATRMultiple { get; set; }
        public bool UseKellyCriterion { get; set; }
        public decimal HistoricalWinRate { get; set; }
        public decimal HistoricalRewardRiskRatio { get; set; }
        public decimal KellyFractionMultiplier { get; set; }

        // News sentiment settings
        public bool EnableNewsSentimentAnalysis { get; set; }
        public int NewsArticleRefreshIntervalMinutes { get; set; }
        public int MaxNewsArticlesPerSymbol { get; set; }
        public bool EnableNewsSourceFiltering { get; set; }
        [MaxLength(1000)]
        public string? EnabledNewsSources { get; set; }

        // Analyst ratings settings
        public bool EnableAnalystRatings { get; set; }
        public int RatingsCacheExpiryHours { get; set; }
        public bool EnableRatingChangeAlerts { get; set; }
        public bool EnableConsensusChangeAlerts { get; set; }
        public decimal AnalystRatingSentimentWeight { get; set; }

        // Insider trading settings
        public bool EnableInsiderTradingAnalysis { get; set; }
        public int InsiderDataRefreshIntervalMinutes { get; set; }
        public bool EnableInsiderTradingAlerts { get; set; }
        public bool TrackNotableInsiders { get; set; }
        public decimal InsiderTradingSentimentWeight { get; set; }
        public bool HighlightCEOTransactions { get; set; }
        public bool HighlightOptionsActivity { get; set; }
        public bool EnableInsiderTransactionNotifications { get; set; }

        // VIX monitoring
        public bool EnableVixMonitoring { get; set; }

        // API Keys
        [MaxLength(255)]
        public string? AlphaVantageApiKey { get; set; }

        [Column("CreatedDate")]
        public DateTime CreatedAt { get; set; }
        
        [Column("ModifiedDate")]
        public DateTime LastModified { get; set; }
    }
}
