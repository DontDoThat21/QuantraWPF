using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for querying ML prediction data from the database
    /// Used by Market Chat to provide AI-generated forecast context in conversations
    /// </summary>
    public interface IPredictionDataService
    {
        /// <summary>
        /// Gets prediction context for a specific symbol to include in Market Chat conversations
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL", "MSFT")</param>
        /// <returns>Formatted prediction context string for AI prompts, or null if no predictions exist</returns>
        Task<string> GetPredictionContextAsync(string symbol);
    }
}
