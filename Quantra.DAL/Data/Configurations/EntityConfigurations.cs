using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
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
        }
    }

    public class StockDataCacheConfiguration : IEntityTypeConfiguration<StockDataCache>
    {
        public void Configure(EntityTypeBuilder<StockDataCache> builder)
        {
            // Indexes
            builder.HasIndex(s => new { s.Symbol, s.TimeRange });
            builder.HasIndex(s => s.ExpiresAt);
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
}
