using System;
using Quantra.Models;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Represents the result of a prediction context lookup, including cache metadata
    /// for Market Chat cache integration (MarketChat story 3).
    /// </summary>
    public class PredictionContextResult
    {
        /// <summary>
        /// The formatted prediction context string for AI prompts
        /// </summary>
        public string Context { get; set; }

        /// <summary>
        /// Whether this prediction was retrieved from cache (true) or freshly generated (false)
        /// </summary>
        public bool IsCached { get; set; }

        /// <summary>
        /// The age of the cached prediction. Null if the prediction was freshly generated.
        /// </summary>
        public TimeSpan? CacheAge { get; set; }

        /// <summary>
        /// The timestamp when the prediction was originally created
        /// </summary>
        public DateTime? PredictionTimestamp { get; set; }

        /// <summary>
        /// The model version used to generate the prediction
        /// </summary>
        public string ModelVersion { get; set; }

        /// <summary>
        /// The actual prediction result, if available (used by ChartGenerationService)
        /// </summary>
        public PredictionResult Prediction { get; set; }

        /// <summary>
        /// Gets a human-readable description of the cache status for display in chat
        /// </summary>
        public string CacheStatusDisplay
        {
            get
            {
                if (!IsCached || !CacheAge.HasValue)
                {
                    return "Freshly generated prediction";
                }

                var age = CacheAge.Value;
                if (age.TotalMinutes < 1)
                {
                    return "Based on prediction from just now";
                }
                else if (age.TotalMinutes < 60)
                {
                    var minutes = (int)age.TotalMinutes;
                    return $"Based on prediction from {minutes} minute{(minutes != 1 ? "s" : "")} ago";
                }
                else if (age.TotalHours < 24)
                {
                    var hours = (int)age.TotalHours;
                    return $"Based on prediction from {hours} hour{(hours != 1 ? "s" : "")} ago";
                }
                else
                {
                    var days = (int)age.TotalDays;
                    return $"Based on prediction from {days} day{(days != 1 ? "s" : "")} ago";
                }
            }
        }

        /// <summary>
        /// Creates an empty result indicating no prediction was found
        /// </summary>
        public static PredictionContextResult Empty => new PredictionContextResult
        {
            Context = null,
            IsCached = false,
            CacheAge = null,
            PredictionTimestamp = null,
            ModelVersion = null,
            Prediction = null
        };
    }
}
