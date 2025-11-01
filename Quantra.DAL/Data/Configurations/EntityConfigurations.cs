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
}
