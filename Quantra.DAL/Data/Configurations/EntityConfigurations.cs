using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Data.Configurations
{
    public class FundamentalDataCacheConfiguration : IEntityTypeConfiguration<FundamentalDataCache>
    {
        public void Configure(EntityTypeBuilder<FundamentalDataCache> builder)
        {
            // Composite primary key
            builder.HasKey(f => new { f.Symbol, f.DataType });

            // Indexes for performance
            builder.HasIndex(f => f.Symbol);
            builder.HasIndex(f => f.CacheTime);
        }
    }

    public class AnalystRatingConfiguration : IEntityTypeConfiguration<AnalystRatingEntity>
    {
        public void Configure(EntityTypeBuilder<AnalystRatingEntity> builder)
        {
            // Unique constraint
            builder.HasIndex(a => new { a.Symbol, a.AnalystName, a.RatingDate })
          .IsUnique();

            // Indexes
            builder.HasIndex(a => a.Symbol);
            builder.HasIndex(a => a.RatingDate);
        }
    }

    public class ConsensusHistoryConfiguration : IEntityTypeConfiguration<ConsensusHistoryEntity>
    {
        public void Configure(EntityTypeBuilder<ConsensusHistoryEntity> builder)
        {
            // Indexes
            builder.HasIndex(c => new { c.Symbol, c.SnapshotDate });
        }
    }

    public class StockPredictionConfiguration : IEntityTypeConfiguration<StockPredictionEntity>
    {
        public void Configure(EntityTypeBuilder<StockPredictionEntity> builder)
        {
            // Indexes for performance on GroupBy and Max queries
            builder.HasIndex(s => s.Symbol);
            builder.HasIndex(s => s.CreatedDate);
            builder.HasIndex(s => new { s.Symbol, s.CreatedDate });
            builder.HasIndex(s => s.Confidence);

            // Index for ChatHistoryId for querying predictions by chat history record
            builder.HasIndex(s => s.ChatHistoryId);

            // Relationship with ChatHistory
            builder.HasOne(s => s.ChatHistory)
                   .WithMany(c => c.Predictions)
                   .HasForeignKey(s => s.ChatHistoryId)
                   .OnDelete(DeleteBehavior.SetNull);

            // Configure double properties to use REAL (float) type to match existing database schema
            // This prevents casting errors when database uses FLOAT (System.Single)
            builder.Property(s => s.Confidence).HasColumnType("real");
            builder.Property(s => s.CurrentPrice).HasColumnType("real");
            builder.Property(s => s.TargetPrice).HasColumnType("real");
            builder.Property(s => s.PotentialReturn).HasColumnType("real");
        }
    }

    public class PredictionIndicatorConfiguration : IEntityTypeConfiguration<PredictionIndicatorEntity>
    {
        public void Configure(EntityTypeBuilder<PredictionIndicatorEntity> builder)
        {
            // Composite primary key
            builder.HasKey(p => new { p.PredictionId, p.IndicatorName });

            // Relationship
            builder.HasOne(p => p.Prediction)
                 .WithMany(s => s.Indicators)
             .HasForeignKey(p => p.PredictionId)
                  .OnDelete(DeleteBehavior.Cascade);

            // Configure double property to use REAL (float) type to match existing database schema
            builder.Property(p => p.IndicatorValue).HasColumnType("real");
        }
    }

    public class StockDataCacheConfiguration : IEntityTypeConfiguration<StockDataCache>
    {
        public void Configure(EntityTypeBuilder<StockDataCache> builder)
        {
            // Composite primary key matching database schema
            builder.HasKey(s => new { s.Symbol, s.TimeRange, s.Interval });

            // Indexes
            builder.HasIndex(s => new { s.Symbol, s.TimeRange });
            builder.HasIndex(s => s.CachedAt);
        }
    }

    public class IndicatorSettingsConfiguration : IEntityTypeConfiguration<IndicatorSettingsEntity>
    {
        public void Configure(EntityTypeBuilder<IndicatorSettingsEntity> builder)
        {
            // Unique constraint on ControlId and IndicatorName
            builder.HasIndex(i => new { i.ControlId, i.IndicatorName })
          .IsUnique();

            // Index for performance
            builder.HasIndex(i => i.ControlId);
        }
    }

    public class PredictionCacheConfiguration : IEntityTypeConfiguration<PredictionCacheEntity>
    {
        public void Configure(EntityTypeBuilder<PredictionCacheEntity> builder)
        {
            // Index for symbol lookups
            builder.HasIndex(p => p.Symbol);

            // Index for identifying stale entries
            builder.HasIndex(p => p.LastAccessedAt);

            // Composite index for cache hits
            builder.HasIndex(p => new { p.Symbol, p.ModelVersion, p.InputDataHash });

            // Set default values
            builder.Property(p => p.AccessCount).HasDefaultValue(0);
        }
    }

    public class ChatHistoryConfiguration : IEntityTypeConfiguration<ChatHistoryEntity>
    {
        public void Configure(EntityTypeBuilder<ChatHistoryEntity> builder)
        {
            // Index for session lookups
            builder.HasIndex(c => c.SessionId);

            // Index for timestamp-based queries
            builder.HasIndex(c => c.Timestamp);

            // Composite index for session and timestamp
            builder.HasIndex(c => new { c.SessionId, c.Timestamp });

            // Index for user queries
            builder.HasIndex(c => c.UserId);

            // Index for symbol-specific chat history
            builder.HasIndex(c => c.Symbol);

            // Configure Content as NVARCHAR(MAX) for large chat messages
            builder.Property(c => c.Content)
                   .HasColumnType("NVARCHAR(MAX)");
        }
    }

    public class SettingsProfileConfiguration : IEntityTypeConfiguration<SettingsProfile>
    {
        public void Configure(EntityTypeBuilder<SettingsProfile> builder)
        {
            // Tell EF Core that this table has triggers to prevent OUTPUT clause usage
            builder.ToTable(tb => tb.HasTrigger("TR_SettingsProfiles_Update"));

            // Configure relationship with UserCredential
            builder.HasOne(p => p.User)
                   .WithMany()
                   .HasForeignKey(p => p.UserId)
                   .OnDelete(DeleteBehavior.Cascade);

            // Index for efficient lookup by user
            builder.HasIndex(p => p.UserId);

            // Unique constraint on UserId and Name combination
            builder.HasIndex(p => new { p.UserId, p.Name })
                   .IsUnique()
                   .HasDatabaseName("UC_SettingsProfiles_UserId_Name");

            // Create a converter for double to decimal conversion
            var doubleToDecimalConverter = new ValueConverter<decimal, double>(
                v => (double)v,           // decimal to double for writing to DB
                v => (decimal)v           // double to decimal for reading from DB
            );

            // Configure decimal properties to convert from database double values
            builder.Property(p => p.AccountSize)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.BaseRiskPercentage)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.MaxPositionSizePercent)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.FixedTradeAmount)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.ATRMultiple)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.HistoricalWinRate)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.HistoricalRewardRiskRatio)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.KellyFractionMultiplier)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.AnalystRatingSentimentWeight)
                .HasConversion(doubleToDecimalConverter);

            builder.Property(p => p.InsiderTradingSentimentWeight)
                .HasConversion(doubleToDecimalConverter);
        }
    }

    public class BacktestResultConfiguration : IEntityTypeConfiguration<BacktestResultEntity>
    {
        public void Configure(EntityTypeBuilder<BacktestResultEntity> builder)
        {
            // Indexes for efficient querying
            builder.HasIndex(b => b.Symbol);
            builder.HasIndex(b => b.StrategyName);
            builder.HasIndex(b => b.CreatedAt);
            builder.HasIndex(b => new { b.Symbol, b.StrategyName });
            builder.HasIndex(b => new { b.Symbol, b.CreatedAt });

            // Configure JSON columns as NVARCHAR(MAX)
            builder.Property(b => b.EquityCurveJson)
                   .HasColumnType("NVARCHAR(MAX)");
            
            builder.Property(b => b.TradesJson)
                   .HasColumnType("NVARCHAR(MAX)");
            
            builder.Property(b => b.StrategyParametersJson)
                   .HasColumnType("NVARCHAR(MAX)");

            // Configure double properties to use FLOAT type (SQL Server float(53)) instead of REAL
            // FLOAT can store the full double range including double.MaxValue used for ProfitFactor
            // when there are no losing trades
            builder.Property(b => b.TotalReturn).HasColumnType("float");
            builder.Property(b => b.MaxDrawdown).HasColumnType("float");
            builder.Property(b => b.WinRate).HasColumnType("float");
            builder.Property(b => b.SharpeRatio).HasColumnType("float");
            builder.Property(b => b.SortinoRatio).HasColumnType("float");
            builder.Property(b => b.CAGR).HasColumnType("float");
            builder.Property(b => b.CalmarRatio).HasColumnType("float");
            builder.Property(b => b.ProfitFactor).HasColumnType("float");
            builder.Property(b => b.InformationRatio).HasColumnType("float");
            builder.Property(b => b.TotalTransactionCosts).HasColumnType("float");
            builder.Property(b => b.GrossReturn).HasColumnType("float");
            builder.Property(b => b.InitialCapital).HasColumnType("float");
            builder.Property(b => b.FinalEquity).HasColumnType("float");
        }
    }

    public class InsiderTransactionConfiguration : IEntityTypeConfiguration<InsiderTransactionEntity>
    {
        public void Configure(EntityTypeBuilder<InsiderTransactionEntity> builder)
        {
            // Unique constraint to prevent duplicate transactions
            builder.HasIndex(i => new { i.Symbol, i.FilingDate, i.TransactionDate, i.OwnerCik })
                   .IsUnique()
                   .HasDatabaseName("UQ_InsiderTransactions");

            // Additional indexes for performance
            builder.HasIndex(i => i.Symbol);
            builder.HasIndex(i => i.TransactionDate);
            builder.HasIndex(i => i.LastUpdated);
        }
    }

    public class PaperTradingSessionConfiguration : IEntityTypeConfiguration<PaperTradingSessionEntity>
    {
        public void Configure(EntityTypeBuilder<PaperTradingSessionEntity> builder)
        {
            // Unique constraint on SessionId
            builder.HasIndex(s => s.SessionId).IsUnique();

            // Index for active sessions
            builder.HasIndex(s => s.IsActive);
            builder.HasIndex(s => s.StartedAt);

            // Configure cascade delete for positions and orders
            builder.HasMany(s => s.Positions)
                   .WithOne(p => p.Session)
                   .HasForeignKey(p => p.SessionId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasMany(s => s.Orders)
                   .WithOne(o => o.Session)
                   .HasForeignKey(o => o.SessionId)
                   .OnDelete(DeleteBehavior.Cascade);
        }
    }

    public class PaperTradingPositionConfiguration : IEntityTypeConfiguration<PaperTradingPositionEntity>
    {
        public void Configure(EntityTypeBuilder<PaperTradingPositionEntity> builder)
        {
            // Indexes for efficient querying
            builder.HasIndex(p => p.SessionId);
            builder.HasIndex(p => p.Symbol);
            builder.HasIndex(p => new { p.SessionId, p.Symbol });
            builder.HasIndex(p => p.IsClosed);
            builder.HasIndex(p => new { p.SessionId, p.IsClosed });

            // Configure relationship with fills
            builder.HasMany(p => p.Fills)
                   .WithOne(f => f.Position)
                   .HasForeignKey(f => f.PositionEntityId)
                   .OnDelete(DeleteBehavior.SetNull);
        }
    }

    public class PaperTradingOrderConfiguration : IEntityTypeConfiguration<PaperTradingOrderEntity>
    {
        public void Configure(EntityTypeBuilder<PaperTradingOrderEntity> builder)
        {
            // Unique constraint on OrderId (Guid string)
            builder.HasIndex(o => o.OrderId).IsUnique();

            // Indexes for efficient querying
            builder.HasIndex(o => o.SessionId);
            builder.HasIndex(o => o.Symbol);
            builder.HasIndex(o => o.State);
            builder.HasIndex(o => new { o.SessionId, o.Symbol });
            builder.HasIndex(o => new { o.SessionId, o.State });
            builder.HasIndex(o => o.CreatedAt);

            // Configure cascade delete for fills
            builder.HasMany(o => o.Fills)
                   .WithOne(f => f.Order)
                   .HasForeignKey(f => f.OrderEntityId)
                   .OnDelete(DeleteBehavior.Cascade);
        }
    }

    public class PaperTradingFillConfiguration : IEntityTypeConfiguration<PaperTradingFillEntity>
    {
        public void Configure(EntityTypeBuilder<PaperTradingFillEntity> builder)
        {
            // Unique constraint on FillId (Guid string)
            builder.HasIndex(f => f.FillId).IsUnique();

            // Indexes for efficient querying
            builder.HasIndex(f => f.OrderEntityId);
            builder.HasIndex(f => f.PositionEntityId);
            builder.HasIndex(f => f.Symbol);
            builder.HasIndex(f => f.FillTime);
        }
    }

    public class StockPredictionHorizonConfiguration : IEntityTypeConfiguration<StockPredictionHorizonEntity>
    {
        public void Configure(EntityTypeBuilder<StockPredictionHorizonEntity> builder)
        {
            // Indexes for efficient querying
            builder.HasIndex(h => h.PredictionId);
            builder.HasIndex(h => h.Horizon);
            builder.HasIndex(h => h.ExpectedFruitionDate);

            // Relationship with StockPrediction (cascade delete)
            builder.HasOne(h => h.Prediction)
                   .WithMany(p => p.PredictionHorizons)
                   .HasForeignKey(h => h.PredictionId)
                   .OnDelete(DeleteBehavior.Cascade);

            // Configure double properties to use FLOAT type for SQL Server compatibility
            builder.Property(h => h.TargetPrice).HasColumnType("float");
            builder.Property(h => h.LowerBound).HasColumnType("float");
            builder.Property(h => h.UpperBound).HasColumnType("float");
            builder.Property(h => h.Confidence).HasColumnType("float");
            builder.Property(h => h.ActualPrice).HasColumnType("float");
            builder.Property(h => h.ActualReturn).HasColumnType("float");
            builder.Property(h => h.ErrorPct).HasColumnType("float");
        }
    }

    public class PredictionFeatureImportanceConfiguration : IEntityTypeConfiguration<PredictionFeatureImportanceEntity>
    {
        public void Configure(EntityTypeBuilder<PredictionFeatureImportanceEntity> builder)
        {
            // Indexes for efficient querying
            builder.HasIndex(f => f.PredictionId);
            builder.HasIndex(f => f.FeatureName);

            // Relationship with StockPrediction (cascade delete)
            builder.HasOne(f => f.Prediction)
                   .WithMany(p => p.FeatureImportances)
                   .HasForeignKey(f => f.PredictionId)
                   .OnDelete(DeleteBehavior.Cascade);

            // Configure double property to use FLOAT type
            builder.Property(f => f.ImportanceScore).HasColumnType("float");
        }
    }

    public class PredictionTemporalAttentionConfiguration : IEntityTypeConfiguration<PredictionTemporalAttentionEntity>
    {
        public void Configure(EntityTypeBuilder<PredictionTemporalAttentionEntity> builder)
        {
            // Index for efficient querying
            builder.HasIndex(t => t.PredictionId);

            // Relationship with StockPrediction (cascade delete)
            builder.HasOne(t => t.Prediction)
                   .WithMany(p => p.TemporalAttentions)
                   .HasForeignKey(t => t.PredictionId)
                   .OnDelete(DeleteBehavior.Cascade);

            // Configure double property to use FLOAT type
            builder.Property(t => t.AttentionWeight).HasColumnType("float");
        }
    }
}
