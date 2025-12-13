using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data.Entities;
using Quantra.Models;
using System;
using System.ComponentModel.DataAnnotations;

namespace Quantra.DAL.Data
{
    /// <summary>
    /// Entity Framework Core DbContext for the Quantra trading platform.
    /// Provides access to all database tables and handles database operations using EF Core.
    /// </summary>
    /// <remarks>
    /// This DbContext replaces the monolithic DatabaseMonolith pattern with a proper ORM approach.
    /// Uses SQL Server LocalDB for better performance and compatibility.
    /// 
    /// Key features:
    /// - Automatic change tracking
    /// - LINQ query support
    /// - Migration support for schema changes
    /// - Transaction management
    /// - Connection pooling
    /// </remarks>
    public class QuantraDbContext : DbContext
    {
        // Logging and Monitoring
        public DbSet<LogEntry> Logs { get; set; }

        // User Settings and Configuration
        public DbSet<UserAppSetting> UserAppSettings { get; set; }
        public DbSet<UserCredential> UserCredentials { get; set; }
        public DbSet<UserPreference> UserPreferences { get; set; }
        public DbSet<TabConfig> TabConfigs { get; set; }
        public DbSet<SettingsEntity> Settings { get; set; }
        public DbSet<SettingsProfile> SettingsProfiles { get; set; }
        public DbSet<IndicatorSettingsEntity> IndicatorSettings { get; set; }
        public DbSet<SavedFilter> SavedFilters { get; set; }

        // Stock and Market Data
        public DbSet<StockSymbolEntity> StockSymbols { get; set; }
        public DbSet<StockDataCache> StockDataCache { get; set; }
        public DbSet<FundamentalDataCache> FundamentalDataCache { get; set; }
        public DbSet<StockConfigurationEntity> StockConfigurations { get; set; }
        public DbSet<StockExplorerDataEntity> StockExplorerData { get; set; }

        // Trading Operations
        public DbSet<OrderHistoryEntity> OrderHistory { get; set; }
        public DbSet<TradeRecordEntity> TradeRecords { get; set; }
        public DbSet<TradingRuleEntity> TradingRules { get; set; }
        public DbSet<BacktestResultEntity> BacktestResults { get; set; }

        // Predictions and Analysis
        public DbSet<StockPredictionEntity> StockPredictions { get; set; }
        public DbSet<PredictionIndicatorEntity> PredictionIndicators { get; set; }
        public DbSet<PredictionCacheEntity> PredictionCache { get; set; }

        // Multi-Horizon TFT Predictions
        public DbSet<StockPredictionHorizonEntity> StockPredictionHorizons { get; set; }
        public DbSet<PredictionFeatureImportanceEntity> PredictionFeatureImportances { get; set; }
        public DbSet<PredictionTemporalAttentionEntity> PredictionTemporalAttentions { get; set; }

        // Chat History
        public DbSet<ChatHistoryEntity> ChatHistory { get; set; }

        // Query History (MarketChat story 5 - Natural Language Query Audit Trail)
        public DbSet<QueryHistoryEntity> QueryHistory { get; set; }

        // Analyst Ratings
        public DbSet<AnalystRatingEntity> AnalystRatings { get; set; }
        public DbSet<ConsensusHistoryEntity> ConsensusHistory { get; set; }

        // API Usage Tracking
        public DbSet<AlphaVantageApiUsage> AlphaVantageApiUsage { get; set; }

        // Insider Transactions
        public DbSet<InsiderTransactionEntity> InsiderTransactions { get; set; }

        // Model Training History
        public DbSet<ModelTrainingHistory> ModelTrainingHistory { get; set; }
        public DbSet<SymbolTrainingResult> SymbolTrainingResults { get; set; }

        // Paper Trading Persistence
        public DbSet<PaperTradingSessionEntity> PaperTradingSessions { get; set; }
        public DbSet<PaperTradingPositionEntity> PaperTradingPositions { get; set; }
        public DbSet<PaperTradingOrderEntity> PaperTradingOrders { get; set; }
        public DbSet<PaperTradingFillEntity> PaperTradingFills { get; set; }

        // Earnings Calendar (TFT known future inputs)
        public DbSet<EarningsCalendarEntity> EarningsCalendar { get; set; }

        public QuantraDbContext(DbContextOptions<QuantraDbContext> options) : base(options)
        {
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!optionsBuilder.IsConfigured)
            {
                // Default configuration if not provided via DI
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString,
            sqlServerOptions =>
            {
                sqlServerOptions.CommandTimeout(30);
            });
            }

            base.OnConfiguring(optionsBuilder);
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Apply all entity configurations from the assembly
            modelBuilder.ApplyConfigurationsFromAssembly(typeof(QuantraDbContext).Assembly);

            // Additional global configurations
            ConfigureConventions(modelBuilder);
        }

        private void ConfigureConventions(ModelBuilder modelBuilder)
        {
            // Set default string length for all string properties without explicit MaxLength
            foreach (var entityType in modelBuilder.Model.GetEntityTypes())
            {
                foreach (var property in entityType.GetProperties())
                {
                    if (property.ClrType == typeof(string))
                    {
                        var maxLength = property.GetMaxLength();
                        
                        // Only set default if no MaxLength was specified at all
                        // If MaxLength is null, check if the property has [MaxLength] attribute
                        // which indicates it should be NVARCHAR(MAX)
                        if (maxLength == null)
                        {
                            var memberInfo = property.PropertyInfo ?? (System.Reflection.MemberInfo)property.FieldInfo;
                            var hasMaxLengthAttribute = memberInfo != null && 
                                memberInfo.GetCustomAttributes(typeof(MaxLengthAttribute), false).Length > 0;
                            
                            // If [MaxLength] attribute exists without value, it means unlimited (NVARCHAR(MAX))
                            // Don't set a default length for these
                            if (!hasMaxLengthAttribute)
                            {
                                property.SetMaxLength(500);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Initializes the database and applies any pending migrations
        /// </summary>
        public void Initialize()
        {
            Database.EnsureCreated();

            // Fix UserPreferences.Value column if needed (for SQL Server)
            FixUserPreferencesValueColumnIfNeeded();
        }

        /// <summary>
        /// Fixes the UserPreferences.Value column to NVARCHAR(MAX) if it's currently truncated
        /// </summary>
        private void FixUserPreferencesValueColumnIfNeeded()
        {
            try
            {
                using (var connection = Database.GetDbConnection())
                {
                    connection.Open();
                    
                    // Check if column needs to be altered
                    var checkSql = @"
                        SELECT c.max_length
                        FROM sys.columns c
                        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
                        WHERE c.object_id = OBJECT_ID('dbo.UserPreferences')
                        AND c.name = 'Value'
                        AND t.name LIKE '%varchar%'";

                    using (var command = connection.CreateCommand())
                    {
                        command.CommandText = checkSql;
                        var maxLength = command.ExecuteScalar();

                        // If max_length is -1, it's already NVARCHAR(MAX)
                        // If it's anything else (like 1000 or 500), we need to alter it
                        if (maxLength != null && Convert.ToInt32(maxLength) != -1)
                        {
                            Console.WriteLine($"Fixing UserPreferences.Value column from VARCHAR({maxLength}) to NVARCHAR(MAX)");

                            // Alter the column to NVARCHAR(MAX)
                            var alterSql = "ALTER TABLE dbo.UserPreferences ALTER COLUMN [Value] NVARCHAR(MAX) NULL";
                            
                            using (var alterCommand = connection.CreateCommand())
                            {
                                alterCommand.CommandText = alterSql;
                                alterCommand.ExecuteNonQuery();
                            }

                            Console.WriteLine("Successfully updated UserPreferences.Value column to NVARCHAR(MAX)");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                // Don't throw - this is not critical for application startup
                // Just log to console as a fallback
                Console.WriteLine($"Warning: Could not alter UserPreferences.Value column: {ex.Message}");
            }
        }
    }
}
