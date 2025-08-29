using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.Services.Interfaces
{
    /// <summary>
    /// Interface for services that provide analyst rating data and analysis
    /// </summary>
    public interface IAnalystRatingService
    {
        /// <summary>
        /// Retrieves recent analyst ratings for a stock symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="count">Maximum number of ratings to retrieve</param>
        /// <returns>List of analyst ratings</returns>
        Task<List<AnalystRating>> GetRecentRatingsAsync(string symbol, int count = 20);

        /// <summary>
        /// Gets aggregated analyst rating data for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Aggregated rating data</returns>
        Task<AnalystRatingAggregate> GetAggregatedRatingsAsync(string symbol);

        /// <summary>
        /// Gets a sentiment score based on analyst ratings for a symbol (-1.0 to 1.0)
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Sentiment score</returns>
        Task<double> GetRatingSentimentAsync(string symbol);

        /// <summary>
        /// Detects changes in analyst ratings since a specified date
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="since">Date to check changes from</param>
        /// <returns>List of changed ratings</returns>
        Task<List<AnalystRating>> GetRatingChangesAsync(string symbol, DateTime since);

        /// <summary>
        /// Refreshes rating data for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>True if successful</returns>
        Task<bool> RefreshRatingsAsync(string symbol);
        
        /// <summary>
        /// Clears cached rating data
        /// </summary>
        /// <param name="symbol">Optional symbol to clear (null clears all)</param>
        void ClearCache(string symbol = null);
        
        /// <summary>
        /// Analyzes historical consensus trends for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="days">Number of days to analyze</param>
        /// <returns>Analysis of consensus trends</returns>
        Task<AnalystRatingAggregate> AnalyzeConsensusHistoryAsync(string symbol, int days = 30);
        
        /// <summary>
        /// Gets historical analyst consensus data for trend analysis
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="days">Number of days of history to retrieve</param>
        /// <returns>List of historical consensus data points</returns>
        Task<List<AnalystRatingAggregate>> GetConsensusHistoryAsync(string symbol, int days = 90);

        /// <summary>
        /// Gets AI-powered analyst sentiment using ChatGPT/AI models for enhanced analysis
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>AI-generated sentiment score (-1.0 to 1.0)</returns>
        Task<double> GetAnalystSentimentAsync(string symbol);
    }
}