using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Model for storing earnings call transcript analysis results including sentiment, entities, and topics
    /// </summary>
    public class EarningsTranscriptAnalysisResult
    {
        /// <summary>
        /// The stock symbol this transcript is for
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// The date of the earnings call
        /// </summary>
        public DateTime EarningsDate { get; set; }

        /// <summary>
        /// Quarter and year of the earnings report (e.g., "Q1 2023")
        /// </summary>
        public string Quarter { get; set; }

        /// <summary>
        /// Overall sentiment score from the transcript (-1.0 to 1.0)
        /// </summary>
        public double SentimentScore { get; set; }

        /// <summary>
        /// Distribution of sentiment across the transcript (percentage positive, negative, neutral)
        /// </summary>
        public Dictionary<string, double> SentimentDistribution { get; set; }

        /// <summary>
        /// Key topics identified in the transcript
        /// </summary>
        public List<string> KeyTopics { get; set; }

        /// <summary>
        /// Named entities extracted from the transcript, categorized by entity type
        /// </summary>
        public Dictionary<string, List<string>> NamedEntities { get; set; }

        /// <summary>
        /// URL to the source transcript if available
        /// </summary>
        public string TranscriptUrl { get; set; }

        /// <summary>
        /// Timestamp when this analysis was performed
        /// </summary>
        public DateTime AnalysisTimestamp { get; set; }

        /// <summary>
        /// Default constructor
        /// </summary>
        public EarningsTranscriptAnalysisResult()
        {
            SentimentDistribution = new Dictionary<string, double>();
            KeyTopics = new List<string>();
            NamedEntities = new Dictionary<string, List<string>>();
            AnalysisTimestamp = DateTime.UtcNow;
        }
    }
}