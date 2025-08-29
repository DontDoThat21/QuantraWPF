using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a financial news article with sentiment data
    /// </summary>
    public class NewsArticle
    {
        /// <summary>
        /// Unique identifier for the article
        /// </summary>
        public string Id { get; set; }
        
        /// <summary>
        /// Article title
        /// </summary>
        public string Title { get; set; }
        
        /// <summary>
        /// Article description/summary
        /// </summary>
        public string Description { get; set; }
        
        /// <summary>
        /// Article content snippet
        /// </summary>
        public string Content { get; set; }
        
        /// <summary>
        /// Full URL to the article
        /// </summary>
        public string Url { get; set; }
        
        /// <summary>
        /// Domain name of the source
        /// </summary>
        public string SourceDomain { get; set; }
        
        /// <summary>
        /// Display name of the source
        /// </summary>
        public string SourceName { get; set; }
        
        /// <summary>
        /// Publication date of the article
        /// </summary>
        public DateTime PublishedAt { get; set; }
        
        /// <summary>
        /// Sentiment score for the article (-1.0 to 1.0)
        /// </summary>
        public double SentimentScore { get; set; }
        
        /// <summary>
        /// Relevance score for the article relative to a stock symbol (0.0 to 1.0)
        /// </summary>
        public double RelevanceScore { get; set; }
        
        /// <summary>
        /// Relevance score for the article relative to a market sector (0.0 to 1.0)
        /// </summary>
        public double SectorRelevance { get; set; }
        
        /// <summary>
        /// The primary market sector that the article relates to (if identifiable)
        /// </summary>
        public string PrimarySector { get; set; }
        
        /// <summary>
        /// Creates the combined text content for sentiment analysis
        /// </summary>
        public string GetCombinedContent()
        {
            return $"{Title}. {Description} {Content}".Trim();
        }
        
        /// <summary>
        /// Checks if this article is likely a duplicate of another
        /// </summary>
        public bool IsSimilarTo(NewsArticle other)
        {
            if (other == null) return false;
            
            // Simple title similarity check
            if (Title != null && other.Title != null)
            {
                var similarity = CalculateTextSimilarity(Title, other.Title);
                if (similarity > 0.8) return true;
            }
            
            return false;
        }
        
        /// <summary>
        /// Calculate simple text similarity ratio
        /// </summary>
        private double CalculateTextSimilarity(string text1, string text2)
        {
            if (string.IsNullOrEmpty(text1) || string.IsNullOrEmpty(text2)) return 0;
            
            // Simple word overlap ratio calculation
            var words1 = text1.ToLowerInvariant().Split(new[] { ' ', ',', '.', '!', '?' }, 
                StringSplitOptions.RemoveEmptyEntries);
            var words2 = text2.ToLowerInvariant().Split(new[] { ' ', ',', '.', '!', '?' }, 
                StringSplitOptions.RemoveEmptyEntries);
            
            int matches = 0;
            foreach (var word in words1)
            {
                if (word.Length > 3 && Array.IndexOf(words2, word) >= 0)
                {
                    matches++;
                }
            }
            
            return words1.Length > 0 ? (double)matches / words1.Length : 0;
        }
    }
}