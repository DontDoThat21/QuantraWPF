using System;

namespace Quantra.Configuration
{
    /// <summary>
    /// Configuration settings for concurrent task throttling
    /// </summary>
    public class ThrottlingConfiguration
    {
        /// <summary>
        /// Maximum degree of parallelism for technical indicator operations (default: 6)
        /// </summary>
        public int TechnicalIndicatorMaxDegreeOfParallelism { get; set; } = 6;

        /// <summary>
        /// Maximum degree of parallelism for API batching operations (default: 4)
        /// </summary>
        public int ApiBatchingMaxDegreeOfParallelism { get; set; } = 4;

        /// <summary>
        /// Maximum degree of parallelism for sentiment analysis operations (default: 4)
        /// </summary>
        public int SentimentAnalysisMaxDegreeOfParallelism { get; set; } = 4;

        /// <summary>
        /// Maximum degree of parallelism for alert checking operations (default: 4)
        /// </summary>
        public int AlertCheckingMaxDegreeOfParallelism { get; set; } = 4;

        /// <summary>
        /// Default configuration with recommended settings
        /// </summary>
        public static ThrottlingConfiguration Default => new ThrottlingConfiguration();

        /// <summary>
        /// Conservative configuration for low-end systems
        /// </summary>
        public static ThrottlingConfiguration Conservative => new ThrottlingConfiguration
        {
            TechnicalIndicatorMaxDegreeOfParallelism = 3,
            ApiBatchingMaxDegreeOfParallelism = 2,
            SentimentAnalysisMaxDegreeOfParallelism = 2,
            AlertCheckingMaxDegreeOfParallelism = 2
        };

        /// <summary>
        /// Aggressive configuration for high-end systems
        /// </summary>
        public static ThrottlingConfiguration Aggressive => new ThrottlingConfiguration
        {
            TechnicalIndicatorMaxDegreeOfParallelism = 10,
            ApiBatchingMaxDegreeOfParallelism = 8,
            SentimentAnalysisMaxDegreeOfParallelism = 6,
            AlertCheckingMaxDegreeOfParallelism = 6
        };

        /// <summary>
        /// Validates the configuration settings
        /// </summary>
        public void Validate()
        {
            if (TechnicalIndicatorMaxDegreeOfParallelism <= 0)
                throw new ArgumentException("TechnicalIndicatorMaxDegreeOfParallelism must be greater than zero");
            
            if (ApiBatchingMaxDegreeOfParallelism <= 0)
                throw new ArgumentException("ApiBatchingMaxDegreeOfParallelism must be greater than zero");
            
            if (SentimentAnalysisMaxDegreeOfParallelism <= 0)
                throw new ArgumentException("SentimentAnalysisMaxDegreeOfParallelism must be greater than zero");
            
            if (AlertCheckingMaxDegreeOfParallelism <= 0)
                throw new ArgumentException("AlertCheckingMaxDegreeOfParallelism must be greater than zero");

            // Warn about potentially high values
            if (TechnicalIndicatorMaxDegreeOfParallelism > 20)
                DatabaseMonolith.Log("Warning", "TechnicalIndicatorMaxDegreeOfParallelism is very high (>20), may cause thread pool exhaustion", "");
            
            if (ApiBatchingMaxDegreeOfParallelism > 16)
                DatabaseMonolith.Log("Warning", "ApiBatchingMaxDegreeOfParallelism is very high (>16), may exceed API rate limits", "");
        }
    }
}