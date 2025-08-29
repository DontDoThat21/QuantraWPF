using System.ComponentModel.DataAnnotations;
using Quantra.Configuration.Validation;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// UI configuration
    /// </summary>
    public class UIConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable dark mode
        /// </summary>
        public bool EnableDarkMode
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Chart update interval in seconds
        /// </summary>
        [ConfigurationRange(1, 60)]
        public int ChartUpdateIntervalSeconds
        {
            get => Get(2);
            set => Set(value);
        }
        
        /// <summary>
        /// Default grid rows
        /// </summary>
        [ConfigurationRange(1, 10)]
        public int DefaultGridRows
        {
            get => Get(4);
            set => Set(value);
        }
        
        /// <summary>
        /// Default grid columns
        /// </summary>
        [ConfigurationRange(1, 10)]
        public int DefaultGridColumns
        {
            get => Get(4);
            set => Set(value);
        }
        
        /// <summary>
        /// Grid border color
        /// </summary>
        [Required]
        public string GridBorderColor
        {
            get => Get("#FF00FFFF");
            set => Set(value);
        }
        
        /// <summary>
        /// Chart layout configuration JSON
        /// </summary>
        public string ChartLayoutConfig
        {
            get => Get(string.Empty);
            set => Set(value);
        }
    }
}