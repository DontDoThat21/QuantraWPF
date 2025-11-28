using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for safely executing database queries from natural language input (MarketChat story 5).
    /// Implements a whitelist-based approach to prevent destructive operations and unauthorized data access.
    /// </summary>
    public interface ISafeQueryExecutor
    {
        /// <summary>
        /// Gets the list of allowed tables for querying
        /// </summary>
        IReadOnlyList<string> AllowedTables { get; }

        /// <summary>
        /// Gets the list of allowed SQL operations (SELECT only for safety)
        /// </summary>
        IReadOnlyList<string> AllowedOperations { get; }

        /// <summary>
        /// Executes a parameterized SQL query against the database.
        /// Only SELECT operations are allowed on whitelisted tables.
        /// </summary>
        /// <param name="sqlQuery">The SQL query to execute</param>
        /// <param name="parameters">Optional parameters for the query</param>
        /// <returns>Query result containing data and metadata</returns>
        Task<NaturalLanguageQueryResult> ExecuteQueryAsync(string sqlQuery, Dictionary<string, object> parameters = null);

        /// <summary>
        /// Validates if a SQL query is safe to execute.
        /// Checks for destructive operations and unauthorized table access.
        /// </summary>
        /// <param name="sqlQuery">The SQL query to validate</param>
        /// <returns>Tuple of (isValid, reason)</returns>
        (bool IsValid, string Reason) ValidateQuery(string sqlQuery);

        /// <summary>
        /// Extracts table names from a SQL query
        /// </summary>
        /// <param name="sqlQuery">The SQL query to analyze</param>
        /// <returns>List of table names found in the query</returns>
        List<string> ExtractTableNames(string sqlQuery);
    }
}
