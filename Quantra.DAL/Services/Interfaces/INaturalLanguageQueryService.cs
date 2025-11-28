using System.Threading.Tasks;
using Quantra.DAL.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for translating natural language queries to SQL (MarketChat story 5).
    /// Uses OpenAI function calling to generate parameterized SQL queries.
    /// </summary>
    public interface INaturalLanguageQueryService
    {
        /// <summary>
        /// Translates a natural language query to SQL and executes it safely.
        /// </summary>
        /// <param name="naturalLanguageQuery">The user's natural language query (e.g., "Show me all stocks with predictions above 80% confidence")</param>
        /// <returns>The query result containing the data and metadata</returns>
        Task<NaturalLanguageQueryResult> ProcessQueryAsync(string naturalLanguageQuery);

        /// <summary>
        /// Determines if a user message is likely a database query request
        /// </summary>
        /// <param name="message">The user's message</param>
        /// <returns>True if the message appears to be a database query request</returns>
        bool IsQueryRequest(string message);

        /// <summary>
        /// Gets the schema information for allowed tables as context for AI
        /// </summary>
        /// <returns>Schema description string for AI context</returns>
        string GetSchemaContext();
    }
}
