using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Model representing sentiment data for a specific market sector
    /// </summary>
    public class SectorSentimentModel
    {
        /// <summary>
        /// The name of the market sector
        /// </summary>
        public string SectorName { get; set; }

        /// <summary>
        /// The overall sentiment score for the sector (-1.0 to 1.0)
        /// </summary>
        public double SentimentScore { get; set; }

        /// <summary>
        /// Sentiment breakdown by news sources in this sector
        /// </summary>
        public Dictionary<string, double> SourceSentiment { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// The most influential news articles for this sector
        /// </summary>
        public List<NewsArticle> KeyArticles { get; set; } = new List<NewsArticle>();

        /// <summary>
        /// Sentiment trend over time (most recent first)
        /// </summary>
        public List<(DateTime Date, double Sentiment)> SentimentTrend { get; set; } = new List<(DateTime, double)>();

        /// <summary>
        /// Companies in this sector with the highest sentiment
        /// </summary>
        public Dictionary<string, double> TopCompanies { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Companies in this sector with the lowest sentiment
        /// </summary>
        public Dictionary<string, double> BottomCompanies { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// The key themes or topics identified in this sector's sentiment analysis
        /// </summary>
        public List<string> KeyThemes { get; set; } = new List<string>();

        /// <summary>
        /// The 7-day sentiment trend direction
        /// </summary>
        public string TrendDirection
        {
            get
            {
                if (SentimentTrend.Count < 7)
                    return "Neutral";

                // Get start and end points of the trend (7 days)
                var startSentiment = SentimentTrend[Math.Min(SentimentTrend.Count - 1, 6)].Sentiment;
                var endSentiment = SentimentTrend[0].Sentiment;
                var change = endSentiment - startSentiment;

                if (change >= 0.15)
                    return "Strong Upward";
                else if (change >= 0.05)
                    return "Upward";
                else if (change <= -0.15)
                    return "Strong Downward";
                else if (change <= -0.05)
                    return "Downward";
                else
                    return "Neutral";
            }
        }

        /// <summary>
        /// Simple textual description of the current sentiment
        /// </summary>
        public string SentimentDescription
        {
            get
            {
                if (SentimentScore >= 0.6)
                    return "Very Positive";
                else if (SentimentScore >= 0.2)
                    return "Positive";
                else if (SentimentScore >= -0.2)
                    return "Neutral";
                else if (SentimentScore >= -0.6)
                    return "Negative";
                else
                    return "Very Negative";
            }
        }
    }
}