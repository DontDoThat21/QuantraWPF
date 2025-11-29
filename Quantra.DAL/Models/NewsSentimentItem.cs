using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a news item from the Alpha Vantage NEWS_SENTIMENT API
    /// </summary>
    public class NewsSentimentItem
    {
        /// <summary>
        /// Article title
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// URL to the full article
        /// </summary>
        public string Url { get; set; }

        /// <summary>
        /// Publication time in ISO format
        /// </summary>
        public DateTime TimePublished { get; set; }

        /// <summary>
        /// Authors of the article
        /// </summary>
        public List<string> Authors { get; set; } = new List<string>();

        /// <summary>
        /// Article summary
        /// </summary>
        public string Summary { get; set; }

        /// <summary>
        /// Banner image URL
        /// </summary>
        public string BannerImage { get; set; }

        /// <summary>
        /// Source domain (e.g., "www.reuters.com")
        /// </summary>
        public string Source { get; set; }

        /// <summary>
        /// Category within the source
        /// </summary>
        public string CategoryWithinSource { get; set; }

        /// <summary>
        /// Topics related to the article
        /// </summary>
        public List<TopicInfo> Topics { get; set; } = new List<TopicInfo>();

        /// <summary>
        /// Overall sentiment score (-1.0 to 1.0)
        /// </summary>
        public double OverallSentimentScore { get; set; }

        /// <summary>
        /// Overall sentiment label (Bullish, Bearish, Neutral, Somewhat-Bullish, Somewhat-Bearish)
        /// </summary>
        public string OverallSentimentLabel { get; set; }

        /// <summary>
        /// Ticker-specific sentiment data
        /// </summary>
        public List<TickerSentiment> TickerSentiments { get; set; } = new List<TickerSentiment>();

        /// <summary>
        /// Gets the formatted time ago string
        /// </summary>
        public string TimeAgo
        {
            get
            {
                var timeSpan = DateTime.Now - TimePublished;
                if (timeSpan.TotalMinutes < 60)
                    return $"{(int)timeSpan.TotalMinutes}m ago";
                if (timeSpan.TotalHours < 24)
                    return $"{(int)timeSpan.TotalHours}h ago";
                if (timeSpan.TotalDays < 7)
                    return $"{(int)timeSpan.TotalDays}d ago";
                return TimePublished.ToString("MMM dd");
            }
        }
    }

    /// <summary>
    /// Topic information for a news article
    /// </summary>
    public class TopicInfo
    {
        /// <summary>
        /// Topic name
        /// </summary>
        public string Topic { get; set; }

        /// <summary>
        /// Relevance score (0.0 to 1.0)
        /// </summary>
        public double RelevanceScore { get; set; }
    }

    /// <summary>
    /// Ticker-specific sentiment information
    /// </summary>
    public class TickerSentiment
    {
        /// <summary>
        /// Stock ticker symbol
        /// </summary>
        public string Ticker { get; set; }

        /// <summary>
        /// Relevance score (0.0 to 1.0)
        /// </summary>
        public double RelevanceScore { get; set; }

        /// <summary>
        /// Ticker sentiment score (-1.0 to 1.0)
        /// </summary>
        public double SentimentScore { get; set; }

        /// <summary>
        /// Ticker sentiment label
        /// </summary>
        public string SentimentLabel { get; set; }
    }

    /// <summary>
    /// Response container for news sentiment API
    /// </summary>
    public class NewsSentimentResponse
    {
        /// <summary>
        /// Number of items returned
        /// </summary>
        public int ItemsCount { get; set; }

        /// <summary>
        /// Sentiment score definition
        /// </summary>
        public string SentimentScoreDefinition { get; set; }

        /// <summary>
        /// Relevance score definition
        /// </summary>
        public string RelevanceScoreDefinition { get; set; }

        /// <summary>
        /// List of news feed items
        /// </summary>
        public List<NewsSentimentItem> Feed { get; set; } = new List<NewsSentimentItem>();
    }
}
