using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for services that provide insider trading activity data and sentiment analysis.
    /// </summary>
    public interface IInsiderTradingService
    {
        /// <summary>
        /// High-level method: fetches recent insider transactions and returns sentiment score for a symbol.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Sentiment score based on insider trading activity (-1 to 1)</returns>
        Task<double> GetInsiderSentimentAsync(string symbol);

        /// <summary>
        /// Gets detailed insider trading activity for a symbol.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>List of recent insider transactions</returns>
        Task<List<InsiderTransaction>> GetInsiderTransactionsAsync(string symbol);

        /// <summary>
        /// Gets notable insider trading activity from influential figures.
        /// </summary>
        /// <param name="symbol">Stock symbol (optional - if null, returns all notable transactions)</param>
        /// <returns>List of notable insider transactions</returns>
        Task<List<InsiderTransaction>> GetNotableInsiderTransactionsAsync(string symbol = null);

        /// <summary>
        /// Gets aggregate insider trading metrics for a symbol.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Dictionary with insider metrics as keys and values</returns>
        Task<Dictionary<string, double>> GetInsiderMetricsAsync(string symbol);

        /// <summary>
        /// Gets aggregate insider sentiment grouped by notable individuals.
        /// </summary>
        /// <param name="symbol">Stock symbol (optional)</param>
        /// <returns>Dictionary with individual names as keys and sentiment scores as values</returns>
        Task<Dictionary<string, double>> GetNotableIndividualSentimentAsync(string symbol = null);
    }
}