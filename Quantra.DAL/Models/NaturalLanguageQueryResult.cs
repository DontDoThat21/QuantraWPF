using System;
using System.Collections.Generic;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Represents the result of a natural language query execution (MarketChat story 5).
    /// Contains the query results in a format suitable for display in Market Chat.
    /// </summary>
    public class NaturalLanguageQueryResult
    {
        /// <summary>
        /// Whether the query was executed successfully
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// The original natural language query from the user
        /// </summary>
        public string OriginalQuery { get; set; }

        /// <summary>
        /// The translated SQL query that was executed
        /// </summary>
        public string TranslatedSql { get; set; }

        /// <summary>
        /// The table(s) that were queried
        /// </summary>
        public List<string> QueriedTables { get; set; } = new List<string>();

        /// <summary>
        /// The column names in the result set
        /// </summary>
        public List<string> Columns { get; set; } = new List<string>();

        /// <summary>
        /// The data rows returned by the query (list of column values)
        /// </summary>
        public List<List<object>> Rows { get; set; } = new List<List<object>>();

        /// <summary>
        /// Total number of rows returned
        /// </summary>
        public int RowCount { get; set; }

        /// <summary>
        /// Error message if the query failed
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// Timestamp when the query was executed
        /// </summary>
        public DateTime ExecutedAt { get; set; } = DateTime.Now;

        /// <summary>
        /// Time taken to execute the query in milliseconds
        /// </summary>
        public long ExecutionTimeMs { get; set; }

        /// <summary>
        /// Whether the query was blocked for safety reasons
        /// </summary>
        public bool WasBlocked { get; set; }

        /// <summary>
        /// Reason the query was blocked, if applicable
        /// </summary>
        public string BlockedReason { get; set; }

        /// <summary>
        /// Formats the result as a Markdown table for display in Market Chat
        /// </summary>
        public string ToMarkdownTable()
        {
            if (!Success || Columns.Count == 0 || Rows.Count == 0)
            {
                if (WasBlocked)
                {
                    return $"⚠️ **Query blocked**: {BlockedReason}";
                }
                if (!Success)
                {
                    return $"❌ **Query failed**: {ErrorMessage}";
                }
                return "No results found.";
            }

            var lines = new List<string>();

            // Header row
            lines.Add("| " + string.Join(" | ", Columns) + " |");

            // Separator row
            lines.Add("| " + string.Join(" | ", Columns.ConvertAll(_ => "---")) + " |");

            // Data rows (limit to 50 for readability)
            int displayCount = Math.Min(Rows.Count, 50);
            for (int i = 0; i < displayCount; i++)
            {
                var row = Rows[i];
                var formattedValues = new List<string>();
                for (int j = 0; j < row.Count; j++)
                {
                    formattedValues.Add(FormatValue(row[j]));
                }
                lines.Add("| " + string.Join(" | ", formattedValues) + " |");
            }

            // Add summary if results were truncated
            if (Rows.Count > 50)
            {
                lines.Add("");
                lines.Add($"*Showing 50 of {Rows.Count} results*");
            }

            return string.Join(Environment.NewLine, lines);
        }

        /// <summary>
        /// Formats a single value for display in the Markdown table
        /// </summary>
        private static string FormatValue(object value)
        {
            if (value == null || value == DBNull.Value)
            {
                return "-";
            }

            if (value is DateTime dt)
            {
                return dt.ToString("yyyy-MM-dd HH:mm");
            }

            if (value is double d)
            {
                // Format as percentage if value looks like a confidence/percentage
                if (d >= 0 && d <= 1)
                {
                    return $"{d:P1}";
                }
                // Format as currency if it looks like a price
                if (d > 1 && d < 100000)
                {
                    return $"${d:F2}";
                }
                return $"{d:F4}";
            }

            if (value is decimal dec)
            {
                return $"${dec:F2}";
            }

            if (value is long l)
            {
                // Format large numbers with commas
                return l.ToString("N0");
            }

            if (value is int i)
            {
                return i.ToString("N0");
            }

            return value.ToString()?.Replace("|", "\\|") ?? "-";
        }

        /// <summary>
        /// Creates a successful result
        /// </summary>
        public static NaturalLanguageQueryResult CreateSuccess(
            string originalQuery,
            string translatedSql,
            List<string> columns,
            List<List<object>> rows,
            List<string> tables,
            long executionTimeMs)
        {
            return new NaturalLanguageQueryResult
            {
                Success = true,
                OriginalQuery = originalQuery,
                TranslatedSql = translatedSql,
                Columns = columns,
                Rows = rows,
                RowCount = rows.Count,
                QueriedTables = tables,
                ExecutionTimeMs = executionTimeMs
            };
        }

        /// <summary>
        /// Creates a failed result
        /// </summary>
        public static NaturalLanguageQueryResult CreateFailure(string originalQuery, string errorMessage)
        {
            return new NaturalLanguageQueryResult
            {
                Success = false,
                OriginalQuery = originalQuery,
                ErrorMessage = errorMessage
            };
        }

        /// <summary>
        /// Creates a blocked result
        /// </summary>
        public static NaturalLanguageQueryResult CreateBlocked(string originalQuery, string reason)
        {
            return new NaturalLanguageQueryResult
            {
                Success = false,
                OriginalQuery = originalQuery,
                WasBlocked = true,
                BlockedReason = reason
            };
        }
    }
}
