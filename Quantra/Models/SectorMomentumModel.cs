using System;

namespace Quantra.Models
{
    public class SectorMomentumModel
    {
        /// <summary>
        /// Full name of the sector/subsector
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// Stock ticker symbol or sector code
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// The momentum value as a percentage (-1.0 to 1.0)
        /// Represents the rate of change in price
        /// </summary>
        public double MomentumValue { get; set; }
        
        /// <summary>
        /// Trading volume
        /// </summary>
        public long Volume { get; set; }
        
        /// <summary>
        /// Optional timestamp for when this data was measured
        /// </summary>
        public DateTime? Timestamp { get; set; } = DateTime.Now;
    }
}
