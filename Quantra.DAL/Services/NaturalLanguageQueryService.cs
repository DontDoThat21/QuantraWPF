using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for translating natural language queries to SQL using OpenAI (MarketChat story 5).
    /// Uses function calling to generate parameterized SQL queries that are then executed safely.
    /// </summary>
    public class NaturalLanguageQueryService : INaturalLanguageQueryService
    {
        private readonly ISafeQueryExecutor _safeQueryExecutor;
        private readonly ILogger<NaturalLanguageQueryService> _logger;
        private readonly HttpClient _httpClient;
        private const string OpenAiBaseUrl = "https://api.openai.com";
        private const string OpenAiModel = "gpt-3.5-turbo";
        private const double OpenAiTemperature = 0.1; // Low temperature for deterministic SQL generation

        // Keywords that indicate a database query request
        private static readonly HashSet<string> QueryKeywords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "show", "list", "find", "get", "display", "query", "search", "retrieve",
            "count", "how many", "total", "average", "sum", "filter", "where",
            "predictions", "stocks", "symbols", "confidence", "price", "rating",
            "above", "below", "greater", "less", "between", "highest", "lowest",
            "top", "bottom", "recent", "latest", "oldest"
        };

        // Regex patterns to detect query intent
        private static readonly Regex QueryPatterns = new Regex(
            @"\b(show\s+me|list|find|get|display|query|search|retrieve|how\s+many|count)\b.*\b(predictions?|stocks?|symbols?|confidence|ratings?|prices?)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        /// <summary>
        /// Schema information for allowed tables
        /// </summary>
        private static readonly string SchemaDescription = @"
Available tables and their columns:

1. StockPredictions - Stock prediction records
   - Id (int): Primary key
   - Symbol (varchar(20)): Stock symbol (e.g., 'AAPL', 'MSFT')
   - PredictedAction (varchar(20)): Predicted action ('BUY', 'SELL', 'HOLD')
   - Confidence (float): Prediction confidence (0.0 to 1.0)
   - CurrentPrice (float): Current stock price
   - TargetPrice (float): Predicted target price
   - PotentialReturn (float): Expected return percentage
   - CreatedDate (datetime): When the prediction was created

2. PredictionCache - Cached ML prediction results
   - Id (int): Primary key
   - Symbol (varchar(20)): Stock symbol
   - ModelVersion (varchar(50)): ML model version
   - PredictedPrice (float): Predicted price
   - PredictedAction (varchar(20)): Predicted action
   - Confidence (float): Prediction confidence (0.0 to 1.0)
   - CreatedAt (datetime): Cache entry creation time
   - AccessCount (int): Number of times accessed

3. PredictionIndicators - Technical indicators used in predictions
   - PredictionId (int): Foreign key to StockPredictions
   - IndicatorName (varchar(100)): Indicator name (e.g., 'RSI', 'MACD')
   - IndicatorValue (float): Indicator value

4. StockSymbols - Stock symbol master data
   - Symbol (varchar(20)): Stock symbol (primary key)
   - Name (varchar(500)): Company name
   - Sector (varchar(200)): Market sector
   - Industry (varchar(200)): Industry classification
   - LastUpdated (datetime): Last update time

5. StockDataCache - Cached stock price data
   - Id (int): Primary key
   - Symbol (varchar(20)): Stock symbol
   - TimeRange (varchar(50)): Time range for the data
   - Data (text): JSON serialized price data
   - CachedAt (datetime): When the data was cached

6. FundamentalDataCache - Cached fundamental data
   - Symbol (varchar(20)): Stock symbol
   - DataType (varchar(100)): Type of data (e.g., 'PERatio', 'MarketCap')
   - Value (float): The cached value
   - CacheTime (datetime): When the data was cached

7. AnalystRatings - Analyst rating records
   - Id (int): Primary key
   - Symbol (varchar(20)): Stock symbol
   - Rating (varchar(50)): Rating value
   - PriceTarget (float): Analyst price target
   - Analyst (varchar(200)): Analyst name
   - Firm (varchar(200)): Analyst firm
   - Date (datetime): Rating date

8. ConsensusHistory - Consensus rating history
   - Id (int): Primary key
   - Symbol (varchar(20)): Stock symbol
   - ConsensusRating (varchar(50)): Consensus rating
   - AverageTarget (float): Average price target
   - NumberOfAnalysts (int): Number of contributing analysts
   - Date (datetime): Consensus date
";

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public NaturalLanguageQueryService(ISafeQueryExecutor safeQueryExecutor, ILogger<NaturalLanguageQueryService> logger)
        {
            _safeQueryExecutor = safeQueryExecutor ?? throw new ArgumentNullException(nameof(safeQueryExecutor));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            _httpClient = new HttpClient();
            try
            {
                var apiKey = GetOpenAiApiKey();
                _httpClient.BaseAddress = new Uri(OpenAiBaseUrl);
                _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
                _httpClient.Timeout = TimeSpan.FromSeconds(60);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to initialize NaturalLanguageQueryService with OpenAI API key");
                throw new InvalidOperationException("OpenAI API key not configured", ex);
            }
        }

        /// <inheritdoc/>
        public async Task<NaturalLanguageQueryResult> ProcessQueryAsync(string naturalLanguageQuery)
        {
            if (string.IsNullOrWhiteSpace(naturalLanguageQuery))
            {
                return NaturalLanguageQueryResult.CreateFailure(naturalLanguageQuery, "Query cannot be empty");
            }

            try
            {
                _logger?.LogInformation("Processing natural language query: {Query}", naturalLanguageQuery);

                // Translate natural language to SQL using OpenAI
                var sqlQuery = await TranslateToSqlAsync(naturalLanguageQuery);

                if (string.IsNullOrWhiteSpace(sqlQuery))
                {
                    return NaturalLanguageQueryResult.CreateFailure(
                        naturalLanguageQuery, 
                        "Could not translate the query to SQL. Please try rephrasing your question.");
                }

                _logger?.LogInformation("Translated to SQL: {Sql}", sqlQuery);

                // Execute the query safely
                var result = await _safeQueryExecutor.ExecuteQueryAsync(sqlQuery);
                result.OriginalQuery = naturalLanguageQuery;

                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error processing natural language query: {Query}", naturalLanguageQuery);
                return NaturalLanguageQueryResult.CreateFailure(naturalLanguageQuery, $"Error processing query: {ex.Message}");
            }
        }

        /// <inheritdoc/>
        public bool IsQueryRequest(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return false;
            }

            // Check for query pattern matches
            if (QueryPatterns.IsMatch(message))
            {
                return true;
            }

            // Count keyword matches
            var words = message.ToLower().Split(new[] { ' ', ',', '.', '?', '!' }, StringSplitOptions.RemoveEmptyEntries);
            var keywordMatches = words.Count(w => QueryKeywords.Contains(w));

            // If at least 2 query keywords are present, it's likely a query
            return keywordMatches >= 2;
        }

        /// <inheritdoc/>
        public string GetSchemaContext()
        {
            return SchemaDescription;
        }

        /// <summary>
        /// Translates a natural language query to SQL using OpenAI
        /// </summary>
        private async Task<string> TranslateToSqlAsync(string naturalLanguageQuery)
        {
            var systemPrompt = BuildSystemPrompt();
            var userPrompt = BuildUserPrompt(naturalLanguageQuery);

            var messages = new[]
            {
                new { role = "system", content = systemPrompt },
                new { role = "user", content = userPrompt }
            };

            try
            {
                var response = await ResilienceHelper.ExternalApiCallAsync("OpenAI", async () =>
                {
                    var requestBody = new
                    {
                        model = OpenAiModel,
                        messages,
                        temperature = OpenAiTemperature,
                        max_tokens = 500
                    };

                    var content = new StringContent(
                        JsonSerializer.Serialize(requestBody),
                        Encoding.UTF8,
                        "application/json"
                    );

                    var httpResponse = await _httpClient.PostAsync("/v1/chat/completions", content);
                    httpResponse.EnsureSuccessStatusCode();

                    var responseString = await httpResponse.Content.ReadAsStringAsync();
                    return JsonSerializer.Deserialize<OpenAIResponse>(responseString, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                });

                var sqlQuery = response?.Choices?.FirstOrDefault()?.Message?.Content?.Trim();

                // Clean up the response - remove markdown code blocks if present
                if (!string.IsNullOrWhiteSpace(sqlQuery))
                {
                    sqlQuery = CleanSqlResponse(sqlQuery);
                }

                return sqlQuery;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error calling OpenAI for SQL translation");
                throw;
            }
        }

        /// <summary>
        /// Builds the system prompt for SQL translation
        /// </summary>
        private string BuildSystemPrompt()
        {
            return $@"You are a SQL query generator for a stock trading platform database.
Your task is to translate natural language questions about stocks and predictions into SQL Server SELECT queries.

IMPORTANT RULES:
1. Generate ONLY SELECT queries - never INSERT, UPDATE, DELETE, DROP, or any other destructive operations
2. Only query the following tables: StockPredictions, PredictionCache, PredictionIndicators, StockSymbols, StockDataCache, FundamentalDataCache, AnalystRatings, ConsensusHistory
3. Always use proper SQL Server syntax
4. Use proper column names as specified in the schema
5. Add appropriate WHERE clauses based on the user's criteria
6. Use ORDER BY for sorting when relevant
7. Limit results with TOP when appropriate
8. Return ONLY the SQL query, no explanations or markdown formatting

{SchemaDescription}

Common query patterns:
- ""Show stocks with high confidence"" → SELECT * FROM StockPredictions WHERE Confidence > 0.8 ORDER BY Confidence DESC
- ""List predictions for AAPL"" → SELECT * FROM StockPredictions WHERE Symbol = 'AAPL' ORDER BY CreatedDate DESC
- ""Find BUY predictions"" → SELECT * FROM StockPredictions WHERE PredictedAction = 'BUY'
- ""Count predictions by symbol"" → SELECT Symbol, COUNT(*) as PredictionCount FROM StockPredictions GROUP BY Symbol
- ""Top 10 highest confidence predictions"" → SELECT TOP 10 * FROM StockPredictions ORDER BY Confidence DESC";
        }

        /// <summary>
        /// Builds the user prompt for SQL translation
        /// </summary>
        private string BuildUserPrompt(string naturalLanguageQuery)
        {
            return $"Translate this question to a SQL query: {naturalLanguageQuery}";
        }

        /// <summary>
        /// Cleans the SQL response from OpenAI (removes markdown, extra whitespace)
        /// </summary>
        private string CleanSqlResponse(string response)
        {
            // Remove markdown code blocks
            response = Regex.Replace(response, @"^```sql\s*", "", RegexOptions.IgnoreCase | RegexOptions.Multiline);
            response = Regex.Replace(response, @"^```\s*", "", RegexOptions.Multiline);
            response = Regex.Replace(response, @"\s*```$", "", RegexOptions.Multiline);

            // Remove any explanatory text before or after the query
            var lines = response.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            var sqlLines = lines.Where(l => 
                !l.TrimStart().StartsWith("--") && 
                !l.TrimStart().StartsWith("Here") &&
                !l.TrimStart().StartsWith("This") &&
                !l.TrimStart().StartsWith("Note") &&
                !string.IsNullOrWhiteSpace(l));

            response = string.Join(" ", sqlLines);

            // Normalize whitespace
            response = Regex.Replace(response, @"\s+", " ").Trim();

            return response;
        }

        /// <summary>
        /// Retrieves the OpenAI API key from environment or settings file.
        /// </summary>
        private static string GetOpenAiApiKey()
        {
            // Prefer environment variable if available
            var envKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            if (!string.IsNullOrWhiteSpace(envKey))
            {
                return envKey;
            }

            // Fallback to local settings file
            const string settingsFile = "alphaVantageSettings.json";
            const string openAiApiKeyProperty = "OpenAiApiKey";

            if (!File.Exists(settingsFile))
            {
                throw new FileNotFoundException($"Settings file '{settingsFile}' not found.");
            }

            var json = File.ReadAllText(settingsFile);
            using (var doc = JsonDocument.Parse(json))
            {
                if (doc.RootElement.TryGetProperty(openAiApiKeyProperty, out var apiKeyElement))
                {
                    var key = apiKeyElement.GetString();
                    if (string.IsNullOrWhiteSpace(key))
                    {
                        throw new InvalidOperationException($"'{openAiApiKeyProperty}' is empty in settings file.");
                    }
                    return key;
                }
            }

            throw new KeyNotFoundException($"'{openAiApiKeyProperty}' not found in settings file.");
        }
    }

    /// <summary>
    /// Response model for OpenAI API (reusing existing model structure)
    /// </summary>
    internal class OpenAIResponse
    {
        public List<OpenAIChoice> Choices { get; set; }
    }

    internal class OpenAIChoice
    {
        public OpenAIMessage Message { get; set; }
    }

    internal class OpenAIMessage
    {
        public string Role { get; set; }
        public string Content { get; set; }
    }
}
