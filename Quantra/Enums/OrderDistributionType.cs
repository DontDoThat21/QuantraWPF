using System;

namespace Quantra.Enums
{
    /// <summary>
    /// Defines how order quantity should be distributed across chunks when splitting large orders
    /// </summary>
    public enum OrderDistributionType
    {
        /// <summary>
        /// Equal distribution of shares across all chunks (default)
        /// </summary>
        Equal = 0,
        
        /// <summary>
        /// Front-loaded distribution with larger chunks at the beginning
        /// </summary>
        FrontLoaded = 1,
        
        /// <summary>
        /// Back-loaded distribution with larger chunks at the end
        /// </summary>
        BackLoaded = 2,
        
        /// <summary>
        /// Normal (bell curve) distribution with larger chunks in the middle
        /// </summary>
        Normal = 3
    }
}