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
        public string AlertEmail { get; set; } = "test@gmail.com";
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

        // Stock Explorer settings
        public bool EnableStockExplorerAutoRefresh { get; set; } = false;

        // API Keys settings
        public string AlphaVantageApiKey { get; set; } = "";

        // Alpha Vantage API Plan Settings
        public int AlphaVantageApiCallsPerMinute { get; set; } = 75; // Default to 75 calls/minute (standard plan)

        // Chart refresh settings
        public int ChartRefreshIntervalSeconds { get; set; } = 15; // Default 15 seconds

        // CandlestickChartModal window settings
        public double CandlestickWindowWidth { get; set; } = 1000; // Default width
        public double CandlestickWindowHeight { get; set; } = 700; // Default height
        public double CandlestickWindowLeft { get; set; } = double.NaN; // Default centered
        public double CandlestickWindowTop { get; set; } = double.NaN; // Default centered
        
        // Candlestick Chart Auto-Refresh Settings
        public bool CandlestickAutoRefreshDefault { get; set; } = true; // Start with auto-refresh ON by default
        
        // Favorite Refresh Intervals (JSON serialized list of favorite intervals in seconds)
        public string FavoriteRefreshIntervals { get; set; } = "[15, 30, 60]"; // Default favorites: 15s, 30s, 60s
        
        // Symbol Watchlist (JSON serialized list of favorite symbols)
        public string SymbolWatchlist { get; set; } = "[]"; // Empty by default
        
        // Last Viewed Symbols (JSON serialized list - tracks recent symbol history)
        public string LastViewedSymbols { get; set; } = "[]"; // Empty by default
        public int MaxLastViewedSymbols { get; set; } = 10; // Keep last 10 viewed symbols
        
        // Chart Layout Presets (JSON serialized dictionary of layout names to settings)
        public string ChartLayoutPresets { get; set; } = "{}"; // Empty by default
        public string ActiveChartLayoutPreset { get; set; } = "Default"; // Active preset name
        
        // Last Known Good Data Settings
        public bool EnableLastKnownGoodFallback { get; set; } = true; // Use cached data on API failure
        public int LastKnownGoodDataExpiryHours { get; set; } = 24; // Keep fallback data for 24 hours
        
        // API Error Circuit Breaker Settings
        public bool EnableApiCircuitBreaker { get; set; } = true; // Enable circuit breaker pattern
        public int CircuitBreakerFailureThreshold { get; set; } = 5; // Open circuit after 5 failures
        public int CircuitBreakerTimeoutSeconds { get; set; } = 60; // Reset circuit after 60 seconds
        public int CircuitBreakerHalfOpenRetries { get; set; } = 3; // Allow 3 retries in half-open state
    }
}