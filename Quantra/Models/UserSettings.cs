using System.Collections.Generic;

namespace Quantra.Models
{
    public class UserSettings
    {
        // Existing properties
    public bool EnableApiModalChecks { get; set; } = true;
    public int ApiTimeoutSeconds { get; set; } = 30;
    public int CacheDurationMinutes { get; set; } = 15;
    public bool EnableHistoricalDataCache { get; set; } = true;
    public bool EnableDarkMode { get; set; } = true;
    public int ChartUpdateIntervalSeconds { get; set; } = 2;
    public bool EnablePriceAlerts { get; set; } = true;
    public bool EnableTradeNotifications { get; set; } = true;
    public bool EnablePaperTrading { get; set; } = true;
    public string RiskLevel { get; set; } = "Low";
    
    // Grid settings
    public int DefaultGridRows { get; set; } = 4;
    public int DefaultGridColumns { get; set; } = 4;
    
    // Position sizing and risk management settings
    public double AccountSize { get; set; } = 100000.0;
    public double BaseRiskPercentage { get; set; } = 0.01; // Default 1%
    public string PositionSizingMethod { get; set; } = "FixedRisk";
    public double MaxPositionSizePercent { get; set; } = 0.10; // Default 10%
    public double FixedTradeAmount { get; set; } = 5000.0; // Default $5,000 per trade
    public bool UseVolatilityBasedSizing { get; set; } = false;
    public double ATRMultiple { get; set; } = 2.0;
    public bool UseKellyCriterion { get; set; } = false;
    public double HistoricalWinRate { get; set; } = 0.55;
    public double HistoricalRewardRiskRatio { get; set; } = 2.0;
    public double KellyFractionMultiplier { get; set; } = 0.5; // Half-Kelly by default
    
    // Grid settings continued
    public string GridBorderColor { get; set; } = "#FF00FFFF"; // Default Cyan
    
    // Email alert settings
    public string AlertEmail { get; set; } = "tylortrub@gmail.com";
    public bool EnableEmailAlerts { get; set; } = false;
    public bool EnableStandardAlertEmails { get; set; } = false;
    public bool EnableOpportunityAlertEmails { get; set; } = false;
    public bool EnablePredictionAlertEmails { get; set; } = false;
    public bool EnableGlobalAlertEmails { get; set; } = false;
    public bool EnableSystemHealthAlertEmails { get; set; } = false;
    
    // SMS alert settings
    public string AlertPhoneNumber { get; set; } = "";
    public bool EnableSmsAlerts { get; set; } = false;
    public bool EnableStandardAlertSms { get; set; } = false;
    public bool EnableOpportunityAlertSms { get; set; } = false;
    public bool EnablePredictionAlertSms { get; set; } = false;
    public bool EnableGlobalAlertSms { get; set; } = false;
    
    // Push notification alert settings
    public string PushNotificationUserId { get; set; } = "";
    public bool EnablePushNotifications { get; set; } = false;
    public bool EnableStandardAlertPushNotifications { get; set; } = false;
    public bool EnableOpportunityAlertPushNotifications { get; set; } = false;
    public bool EnablePredictionAlertPushNotifications { get; set; } = false;
    public bool EnableGlobalAlertPushNotifications { get; set; } = false;
    public bool EnableTechnicalIndicatorAlertPushNotifications { get; set; } = false;
    public bool EnableSentimentShiftAlertPushNotifications { get; set; } = false;
    public bool EnableSystemHealthAlertPushNotifications { get; set; } = false;
    public bool EnableTradeExecutionPushNotifications { get; set; } = false;
    
    // Sound alert settings
    public bool EnableAlertSounds { get; set; } = true;
    public string DefaultAlertSound { get; set; } = "alert.wav";
    public string DefaultOpportunitySound { get; set; } = "opportunity.wav";
    public string DefaultPredictionSound { get; set; } = "prediction.wav";
    public string DefaultTechnicalIndicatorSound { get; set; } = "indicator.wav";
    public int AlertVolume { get; set; } = 80; // 0-100 percentage
    
    // Active benchmark selection settings
    public string ActiveBenchmarkId { get; set; } = null; // ID of the active custom benchmark, if any
    public string ActiveBenchmarkType { get; set; } = "SPY"; // Type: "SPY", "QQQ", "IWM", "DIA", or "CUSTOM"
    
    // Visual indicator settings
    public bool EnableVisualIndicators { get; set; } = true;
    public string DefaultVisualIndicatorType { get; set; } = "Toast"; // Toast, Banner, Popup, Flashcard
    public string DefaultVisualIndicatorColor { get; set; } = "#FFFF00"; // Yellow
    public int VisualIndicatorDuration { get; set; } = 5; // Duration in seconds
    
    // Window state settings
    public bool RememberWindowState { get; set; } = true;
    public int LastWindowState { get; set; } = 0; // 0:Normal, 1:Minimized, 2:Maximized
    
    // News sentiment analysis settings
    public bool EnableNewsSentimentAnalysis { get; set; } = true;
    public int NewsArticleRefreshIntervalMinutes { get; set; } = 30;
    public int MaxNewsArticlesPerSymbol { get; set; } = 15;
    public bool EnableNewsSourceFiltering { get; set; } = true;
    public Dictionary<string, bool> EnabledNewsSources { get; set; } = new Dictionary<string, bool>
    {
        { "bloomberg.com", true },
        { "cnbc.com", true },
        { "wsj.com", true },
        { "reuters.com", true },
        { "marketwatch.com", true },
        { "finance.yahoo.com", true },
        { "ft.com", true }
    };
    
    // Analyst rating settings
    public bool EnableAnalystRatings { get; set; } = true;
    public int? RatingsCacheExpiryHours { get; set; } = 24;
    public bool EnableRatingChangeAlerts { get; set; } = true;
    public bool EnableConsensusChangeAlerts { get; set; } = true;
    public double AnalystRatingSentimentWeight { get; set; } = 2.0; // Weight in combined sentiment
    
    // Insider trading settings
    public bool EnableInsiderTradingAnalysis { get; set; } = true;
    public int InsiderDataRefreshIntervalMinutes { get; set; } = 120; // Refresh every 2 hours
    public bool EnableInsiderTradingAlerts { get; set; } = true;
    public bool TrackNotableInsiders { get; set; } = true;
    public List<string> NotableInsiderSymbols { get; set; } = new List<string>(); // Symbols to track for notable insider activity
    public double InsiderTradingSentimentWeight { get; set; } = 2.5; // Weight in combined sentiment (high due to significance)
    public bool HighlightCEOTransactions { get; set; } = true;
    public bool HighlightOptionsActivity { get; set; } = true;
        public bool EnableInsiderTransactionNotifications { get; set; } = true;
    
    // VIX monitoring settings
    public bool EnableVixMonitoring { get; set; } = true;
    }
}