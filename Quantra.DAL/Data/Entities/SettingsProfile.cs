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
        public string Description { get; set; }

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
        public string RiskLevel { get; set; }
     
        public int DefaultGridRows { get; set; }
        public int DefaultGridColumns { get; set; }
        
        [MaxLength(20)]
        public string GridBorderColor { get; set; }

        [Column("CreatedDate")]
        public DateTime CreatedAt { get; set; }
        
        [Column("ModifiedDate")]
        public DateTime LastModified { get; set; }
    }
}
