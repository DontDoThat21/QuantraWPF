using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a log entry in the database
    /// </summary>
    [Table("Logs")]
    public class LogEntry
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public DateTime Timestamp { get; set; }

        [Required]
        [MaxLength(50)]
        public string Level { get; set; }

        [Required]
        [MaxLength(1000)]
        public string Message { get; set; }

        public string Details { get; set; }
    }

    /// <summary>
    /// Entity representing user app settings for tabs and layouts
    /// </summary>
    [Table("UserAppSettings")]
    public class UserAppSetting
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(200)]
        public string TabName { get; set; }

        public int TabOrder { get; set; }

        public string? CardPositions { get; set; }

        public string? ControlsConfig { get; set; }

        public string? ToolsConfig { get; set; }

        public int GridRows { get; set; } = 4;

        public int GridColumns { get; set; } = 4;

        public string? DataGridConfig { get; set; }
    }

    /// <summary>
    /// Entity representing user credentials for trading accounts
    /// </summary>
    [Table("UserCredentials")]
    public class UserCredential
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(200)]
        public string Username { get; set; }

        [Required]
        [MaxLength(500)]
        public string Password { get; set; }

        [MaxLength(50)]
        public string Pin { get; set; }

        public DateTime? LastLoginDate { get; set; }
    }

    /// <summary>
    /// Entity representing user preferences (key-value pairs)
    /// </summary>
    [Table("UserPreferences")]
    public class UserPreference
    {
        [Key]
        [MaxLength(200)]
        public string Key { get; set; }

        [MaxLength]
        public string Value { get; set; }

        [Required]
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// Entity representing tab configurations
    /// </summary>
    [Table("TabConfigs")]
    public class TabConfig
    {
        [Key]
        [MaxLength(200)]
        public string TabName { get; set; }

        public string ToolsConfig { get; set; }
    }

    /// <summary>
    /// Entity representing application settings
    /// </summary>
    [Table("Settings")]
    public class SettingsEntity
    {
        [Key]
        public int ID { get; set; }

        public bool EnableApiModalChecks { get; set; }

        public int ApiTimeoutSeconds { get; set; } = 30;

        public int CacheDurationMinutes { get; set; } = 15;

        public bool EnableHistoricalDataCache { get; set; } = true;

        public bool EnableDarkMode { get; set; } = true;

        public int ChartUpdateIntervalSeconds { get; set; } = 2;

        public bool EnablePriceAlerts { get; set; } = true;

        public bool EnableTradeNotifications { get; set; } = true;

        public bool EnablePaperTrading { get; set; } = true;

        [MaxLength(50)]
        public string RiskLevel { get; set; } = "Low";

        public int DefaultGridRows { get; set; } = 4;

        public int DefaultGridColumns { get; set; } = 4;

        [MaxLength(20)]
        public string GridBorderColor { get; set; } = "#FF00FFFF";

        [MaxLength(200)]
        public string AlertEmail { get; set; } = "test@gmail.com";
    }

    /// <summary>
    /// Entity representing indicator settings for controls
    /// </summary>
    [Table("IndicatorSettings")]
    public class IndicatorSettingsEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        public int ControlId { get; set; }

        [Required]
        [MaxLength(100)]
        public string IndicatorName { get; set; }

        [Required]
        public bool IsEnabled { get; set; }

        [Required]
        public DateTime LastUpdated { get; set; }
    }
}
