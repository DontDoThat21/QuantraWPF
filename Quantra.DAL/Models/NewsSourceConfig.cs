using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Configuration for news sources used in sentiment analysis
    /// </summary>
    public class NewsSourceConfig
    {
        /// <summary>
        /// Domain of the news source (e.g., "bloomberg.com")
        /// </summary>
        public string Domain { get; set; }

        /// <summary>
        /// Display name of the news source (e.g., "Bloomberg")
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Weight assigned to this news source for sentiment aggregation (higher = more influential)
        /// </summary>
        public double Weight { get; set; } = 1.0;

        /// <summary>
        /// Whether this source is enabled for fetching
        /// </summary>
        public bool IsEnabled { get; set; } = true;

        /// <summary>
        /// Specific API endpoint for this source (if applicable, otherwise null)
        /// </summary>
        public string ApiEndpoint { get; set; }

        /// <summary>
        /// Additional keywords to help match relevant articles for a stock symbol
        /// </summary>
        public List<string> RelevanceKeywords { get; set; } = new List<string>();
    }
}