using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data.Entities;
using Quantra.Models;
using System;

namespace Quantra.DAL.Data
{
    /// <summary>
    /// Entity Framework Core DbContext for the Quantra trading platform.
    /// Provides access to all database tables and handles database operations using EF Core.
    /// </summary>
    /// <remarks>
    /// This DbContext replaces the monolithic DatabaseMonolith pattern with a proper ORM approach.
    /// Uses SQLite with WAL mode for better concurrency and performance.
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

        // Stock and Market Data
 public DbSet<StockSymbolEntity> StockSymbols { get; set; }
        public DbSet<StockDataCache> StockDataCache { get; set; }
        public DbSet<FundamentalDataCache> FundamentalDataCache { get; set; }

        // Trading Operations
      public DbSet<OrderHistoryEntity> OrderHistory { get; set; }
        public DbSet<TradeRecordEntity> TradeRecords { get; set; }
        public DbSet<TradingRuleEntity> TradingRules { get; set; }

// Predictions and Analysis
        public DbSet<StockPredictionEntity> StockPredictions { get; set; }
        public DbSet<PredictionIndicatorEntity> PredictionIndicators { get; set; }
        public DbSet<PredictionCacheEntity> PredictionCache { get; set; }

     // Analyst Ratings
        public DbSet<AnalystRatingEntity> AnalystRatings { get; set; }
        public DbSet<ConsensusHistoryEntity> ConsensusHistory { get; set; }

        // API Usage Tracking
        public DbSet<AlphaVantageApiUsage> AlphaVantageApiUsage { get; set; }

  public QuantraDbContext(DbContextOptions<QuantraDbContext> options) : base(options)
        {
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
        if (!optionsBuilder.IsConfigured)
        {
      // Default configuration if not provided via DI
optionsBuilder.UseSqlite("Data Source=Quantra.db",
  sqliteOptions =>
            {
      sqliteOptions.CommandTimeout(30);
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
       // Set default string length for all string properties (SQLite doesn't enforce, but good for documentation)
      foreach (var entityType in modelBuilder.Model.GetEntityTypes())
            {
         foreach (var property in entityType.GetProperties())
         {
              if (property.ClrType == typeof(string) && property.GetMaxLength() == null)
                  {
                    property.SetMaxLength(500);
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
            
         // For SQLite, configure WAL mode
            Database.ExecuteSqlRaw("PRAGMA journal_mode=WAL;");
Database.ExecuteSqlRaw("PRAGMA synchronous=NORMAL;");
 Database.ExecuteSqlRaw("PRAGMA busy_timeout=30000;");
        }
    }
}
