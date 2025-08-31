using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for services that provide sentiment analysis from social media platforms.
    /// </summary>
    public interface ISocialMediaSentimentService
    {
        /// <summary>
        /// High-level method: fetches social media content and returns average sentiment for a symbol.
        /// </summary>
        Task<double> GetSymbolSentimentAsync(string symbol);

        /// <summary>
        /// Analyzes sentiment from a list of text content.
        /// </summary>
        Task<double> AnalyzeSentimentAsync(List<string> textContent);
        
        /// <summary>
        /// Gets detailed sentiment data for a symbol by source.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Dictionary with source keys and sentiment scores as values</returns>
        Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol);
        
        /// <summary>
        /// Gets recent content (articles, posts, etc.) for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="count">Maximum number of content items to fetch</param>
        /// <returns>List of content items</returns>
        Task<List<string>> FetchRecentContentAsync(string symbol, int count = 10);
    }
}