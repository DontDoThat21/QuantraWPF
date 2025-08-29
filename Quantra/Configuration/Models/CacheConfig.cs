using System.ComponentModel.DataAnnotations;
using Quantra.Configuration.Validation;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// Cache configuration
    /// </summary>
    public class CacheConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable historical data cache
        /// </summary>
        public bool EnableHistoricalDataCache
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Cache duration in minutes
        /// </summary>
        [ConfigurationRange(1, 1440)]
        public int CacheDurationMinutes
        {
            get => Get(15);
            set => Set(value);
        }
    }
}