using System;

namespace Quantra.Models
{
    /// <summary>
    /// Class for displaying performance comparison metrics between strategy and benchmark
    /// </summary>
    public class PerformanceComparisonMetric
    {
        /// <summary>
        /// Name of the metric being compared
        /// </summary>
        public string Metric { get; set; }
        
        /// <summary>
        /// Value for the strategy
        /// </summary>
        public string Strategy { get; set; }
        
        /// <summary>
        /// Value for the benchmark
        /// </summary>
        public string Benchmark { get; set; }
        
        /// <summary>
        /// Difference between strategy and benchmark (may include icons, colors via formatting)
        /// </summary>
        public string Difference { get; set; }
        
        /// <summary>
        /// Flag indicating if the strategy outperforms the benchmark on this metric
        /// </summary>
        public bool IsOutperforming { get; set; }
        
        /// <summary>
        /// Raw numerical difference between strategy and benchmark values
        /// </summary>
        public double NumericDifference { get; set; }
    }
}