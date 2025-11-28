using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for safely executing database queries from natural language input (MarketChat story 5).
    /// Implements a whitelist-based approach to prevent destructive operations and unauthorized data access.
    /// </summary>
    public class SafeQueryExecutor : ISafeQueryExecutor
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<SafeQueryExecutor> _logger;

        // Whitelist of tables allowed for querying
        private static readonly List<string> _allowedTables = new List<string>
        {
            "StockPredictions",
            "PredictionCache", 
            "PredictionIndicators",
            "StockSymbols",
            "StockDataCache",
            "FundamentalDataCache",
            "AnalystRatings",
            "ConsensusHistory"
        };

        // Only SELECT operations are allowed for safety
        private static readonly List<string> _allowedOperations = new List<string>
        {
            "SELECT"
        };

        // Patterns for detecting dangerous SQL operations
        private static readonly Regex DestructivePattern = new Regex(
            @"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE|GRANT|REVOKE|DENY|xp_|sp_|OPENQUERY|OPENDATASOURCE|OPENROWSET|BULK)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        // Pattern for extracting table names from FROM/JOIN clauses
        private static readonly Regex TableNamePattern = new Regex(
            @"(?:FROM|JOIN)\s+(?:\[?dbo\]?\.?)?\[?(\w+)\]?",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        // Pattern for detecting SELECT statements
        private static readonly Regex SelectPattern = new Regex(
            @"^\s*SELECT\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        // Maximum rows to return for safety
        private const int MaxRowsLimit = 1000;

        /// <inheritdoc/>
        public IReadOnlyList<string> AllowedTables => _allowedTables.AsReadOnly();

        /// <inheritdoc/>
        public IReadOnlyList<string> AllowedOperations => _allowedOperations.AsReadOnly();

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public SafeQueryExecutor(QuantraDbContext context, ILogger<SafeQueryExecutor> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Parameterless constructor for backward compatibility
        /// </summary>
        public SafeQueryExecutor()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString, sqlServerOptions =>
            {
                sqlServerOptions.CommandTimeout(30);
            });
            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        /// <inheritdoc/>
        public async Task<NaturalLanguageQueryResult> ExecuteQueryAsync(string sqlQuery, Dictionary<string, object> parameters = null)
        {
            var stopwatch = Stopwatch.StartNew();

            try
            {
                _logger?.LogInformation("Executing safe query: {Query}", sqlQuery);

                // Validate the query first
                var (isValid, reason) = ValidateQuery(sqlQuery);
                if (!isValid)
                {
                    _logger?.LogWarning("Query blocked: {Reason}", reason);
                    var blockedResult = NaturalLanguageQueryResult.CreateBlocked(sqlQuery, reason);
                    await LogQueryHistoryAsync(blockedResult);
                    return blockedResult;
                }

                // Extract table names for logging
                var tables = ExtractTableNames(sqlQuery);

                // Ensure query has a TOP clause to limit results
                var limitedQuery = EnsureRowLimit(sqlQuery);

                // Execute the query using ADO.NET for flexibility
                var columns = new List<string>();
                var rows = new List<List<object>>();

                using (var connection = _context.Database.GetDbConnection())
                {
                    await connection.OpenAsync();

                    using (var command = connection.CreateCommand())
                    {
                        command.CommandText = limitedQuery;
                        command.CommandType = CommandType.Text;
                        command.CommandTimeout = 30;

                        // Add parameters if provided
                        if (parameters != null)
                        {
                            foreach (var param in parameters)
                            {
                                var sqlParam = command.CreateParameter();
                                sqlParam.ParameterName = param.Key.StartsWith("@") ? param.Key : "@" + param.Key;
                                sqlParam.Value = param.Value ?? DBNull.Value;
                                command.Parameters.Add(sqlParam);
                            }
                        }

                        using (var reader = await command.ExecuteReaderAsync())
                        {
                            // Get column names
                            for (int i = 0; i < reader.FieldCount; i++)
                            {
                                columns.Add(reader.GetName(i));
                            }

                            // Read data rows
                            while (await reader.ReadAsync())
                            {
                                var row = new List<object>();
                                for (int i = 0; i < reader.FieldCount; i++)
                                {
                                    row.Add(reader.GetValue(i));
                                }
                                rows.Add(row);
                            }
                        }
                    }
                }

                stopwatch.Stop();

                var result = NaturalLanguageQueryResult.CreateSuccess(
                    sqlQuery,
                    limitedQuery,
                    columns,
                    rows,
                    tables,
                    stopwatch.ElapsedMilliseconds);

                _logger?.LogInformation("Query executed successfully: {RowCount} rows in {TimeMs}ms", 
                    result.RowCount, stopwatch.ElapsedMilliseconds);

                // Log to query history
                await LogQueryHistoryAsync(result);

                return result;
            }
            catch (Exception ex)
            {
                stopwatch.Stop();
                _logger?.LogError(ex, "Error executing query: {Query}", sqlQuery);
                
                var failedResult = NaturalLanguageQueryResult.CreateFailure(sqlQuery, ex.Message);
                failedResult.ExecutionTimeMs = stopwatch.ElapsedMilliseconds;
                
                await LogQueryHistoryAsync(failedResult);
                
                return failedResult;
            }
        }

        /// <inheritdoc/>
        public (bool IsValid, string Reason) ValidateQuery(string sqlQuery)
        {
            if (string.IsNullOrWhiteSpace(sqlQuery))
            {
                return (false, "Query cannot be empty");
            }

            // Check for destructive operations
            if (DestructivePattern.IsMatch(sqlQuery))
            {
                return (false, "Query contains forbidden operations. Only SELECT queries are allowed.");
            }

            // Ensure it's a SELECT query
            if (!SelectPattern.IsMatch(sqlQuery))
            {
                return (false, "Only SELECT queries are allowed for safety.");
            }

            // Check that only allowed tables are accessed
            var tables = ExtractTableNames(sqlQuery);
            foreach (var table in tables)
            {
                if (!_allowedTables.Contains(table, StringComparer.OrdinalIgnoreCase))
                {
                    return (false, $"Access to table '{table}' is not allowed. Allowed tables: {string.Join(", ", _allowedTables)}");
                }
            }

            // Check for SQL injection patterns
            if (ContainsSqlInjectionPatterns(sqlQuery))
            {
                return (false, "Query contains suspicious patterns that could indicate SQL injection.");
            }

            return (true, null);
        }

        /// <inheritdoc/>
        public List<string> ExtractTableNames(string sqlQuery)
        {
            var tables = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            
            var matches = TableNamePattern.Matches(sqlQuery);
            foreach (Match match in matches)
            {
                if (match.Groups.Count > 1)
                {
                    tables.Add(match.Groups[1].Value);
                }
            }

            return tables.ToList();
        }

        /// <summary>
        /// Ensures the query has a TOP clause to limit results
        /// </summary>
        private string EnsureRowLimit(string sqlQuery)
        {
            // Check if query already has TOP or LIMIT
            if (Regex.IsMatch(sqlQuery, @"\bTOP\s*\(?\s*\d+\s*\)?", RegexOptions.IgnoreCase) ||
                Regex.IsMatch(sqlQuery, @"\bLIMIT\s+\d+", RegexOptions.IgnoreCase))
            {
                return sqlQuery;
            }

            // Add TOP clause after SELECT
            var modifiedQuery = SelectPattern.Replace(sqlQuery, $"SELECT TOP {MaxRowsLimit} ", 1);
            return modifiedQuery;
        }

        /// <summary>
        /// Checks for common SQL injection patterns
        /// </summary>
        private bool ContainsSqlInjectionPatterns(string query)
        {
            // Check for comment sequences that could hide malicious code
            if (query.Contains("--") || query.Contains("/*") || query.Contains("*/"))
            {
                return true;
            }

            // Check for multiple statements (semicolons)
            if (query.Contains(";") && query.IndexOf(";") != query.LastIndexOf(";"))
            {
                return true;
            }

            // Check for UNION-based attacks
            if (Regex.IsMatch(query, @"\bUNION\s+ALL\s+SELECT\b", RegexOptions.IgnoreCase))
            {
                return true;
            }

            // Check for common SQL injection payload patterns
            if (Regex.IsMatch(query, @"'\s*OR\s+'\d+'\s*=\s*'\d+'", RegexOptions.IgnoreCase) ||
                Regex.IsMatch(query, @"'\s*OR\s+1\s*=\s*1", RegexOptions.IgnoreCase))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// Logs the query execution to the QueryHistory table
        /// </summary>
        private async Task LogQueryHistoryAsync(NaturalLanguageQueryResult result)
        {
            try
            {
                var historyEntry = new QueryHistoryEntity
                {
                    OriginalQuery = result.OriginalQuery?.Substring(0, Math.Min(result.OriginalQuery?.Length ?? 0, 2000)),
                    TranslatedSql = result.TranslatedSql,
                    QueriedTables = result.QueriedTables != null ? string.Join(",", result.QueriedTables) : null,
                    Success = result.Success,
                    RowCount = result.RowCount,
                    ErrorMessage = result.ErrorMessage?.Substring(0, Math.Min(result.ErrorMessage?.Length ?? 0, 2000)),
                    WasBlocked = result.WasBlocked,
                    BlockedReason = result.BlockedReason?.Substring(0, Math.Min(result.BlockedReason?.Length ?? 0, 500)),
                    ExecutionTimeMs = result.ExecutionTimeMs,
                    ExecutedAt = result.ExecutedAt
                };

                _context.Set<QueryHistoryEntity>().Add(historyEntry);
                await _context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                // Non-critical operation - just log and continue
                _logger?.LogWarning(ex, "Failed to log query history");
            }
        }
    }
}
