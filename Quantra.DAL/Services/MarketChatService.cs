using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for handling ChatGPT/OpenAI API calls for market analysis conversations.
    /// Integrates with PredictionCacheService to leverage cached predictions (MarketChat story 3).
    /// Integrates with ModelExplainerService to provide plain English explanations of ML predictions (MarketChat story 4).
    /// Integrates with NaturalLanguageQueryService for SQL table queries via natural language (MarketChat story 5).
    /// Integrates with SentimentPriceCorrelationAnalysis for sentiment-price correlation context (MarketChat story 6).
    /// Integrates with MultiSymbolAnalyzer for multi-symbol comparative analysis (MarketChat story 7).
    /// Integrates with ChartGenerationService for chart visualization in chat (MarketChat story 8).
    /// Supports Python script orchestration for on-demand predictions (MarketChat story 9).
    /// </summary>
    public class MarketChatService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<MarketChatService> _logger;
        private readonly IMarketDataEnrichmentService _marketDataEnrichmentService;
        private readonly IPredictionDataService _predictionDataService;
        private readonly IModelExplainerService _modelExplainerService;
        private readonly INaturalLanguageQueryService _naturalLanguageQueryService;
        private readonly SentimentPriceCorrelationAnalysis _sentimentCorrelationAnalysis;
        private readonly IMultiSymbolAnalyzer _multiSymbolAnalyzer;
        private readonly IChartGenerationService _chartGenerationService;
        private readonly RealTimeInferenceService _realTimeInferenceService;
        private readonly PredictionCacheService _predictionCacheService;
        private readonly List<MarketChatMessage> _conversationHistory;
        private readonly Dictionary<string, string> _enrichedContextHistory;
        private readonly Dictionary<string, string> _predictionContextHistory;
        private readonly Dictionary<string, string> _sentimentCorrelationContextHistory;
        private readonly Dictionary<string, PredictionContextResult> _predictionCacheResults;
        private readonly Dictionary<string, PredictionResult> _lastPredictionResults;
        private readonly Dictionary<string, ProjectionChartData> _chartDataCache;
        private const string OpenAiBaseUrl = "https://api.openai.com";
        private const string OpenAiModel = "gpt-3.5-turbo";
        private const double OpenAiTemperature = 0.3;
        private const int OpenAiTimeout = 60;

        // Compiled regex patterns for symbol extraction (performance optimization)
        private static readonly Regex DollarSymbolPattern = new Regex(@"\$([A-Z]{1,5})\b", RegexOptions.Compiled);
        private static readonly Regex StandaloneSymbolPattern = new Regex(@"\b([A-Z]{1,5})\b(?=\s|$|[,.\)])", RegexOptions.Compiled);

        // Common words that should not be treated as stock symbols
        private static readonly HashSet<string> CommonWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "I", "A", "AN", "THE", "IN", "ON", "AT", "TO", "FOR", "OF", "AND", "OR", "IS", "IT",
            "BE", "AS", "BY", "IF", "DO", "GO", "SO", "NO", "UP", "MY", "ME", "WE", "US", "AM",
            "CAN", "ALL", "NEW", "ONE", "TWO", "NOW", "HOW", "WHY", "WHAT", "WHEN", "WHO",
            "NOT", "BUT", "OUT", "HAS", "HAD", "GET", "GOT", "MAY", "SAY", "SEE", "SET",
            "RSI", "EMA", "SMA", "MACD", "ATR", "ADX", "PLAN", "RISK", "BUY", "SELL", "HIGH", "LOW"
        };

        /// <summary>
        /// Gets the most recent prediction cache result for determining cache hit/miss status.
        /// This is populated after each prediction context fetch and can be used by the ViewModel
        /// to display cache status information in the UI (MarketChat story 3).
        /// </summary>
        public PredictionContextResult LastPredictionCacheResult { get; private set; }

        /// <summary>
        /// Gets the most recent natural language query result (MarketChat story 5).
        /// </summary>
        public NaturalLanguageQueryResult LastQueryResult { get; private set; }

        /// <summary>
        /// Gets the most recent multi-symbol comparison result (MarketChat story 7).
        /// </summary>
        public MultiSymbolComparisonResult LastComparisonResult { get; private set; }

        /// <summary>
        /// Gets the most recent chart data generated for visualization (MarketChat story 8).
        /// </summary>
        public ProjectionChartData LastChartData { get; private set; }

        /// <summary>
        /// Gets the most recent Python prediction execution result (MarketChat story 9).
        /// </summary>
        public PythonPredictionExecutionResult LastPythonPredictionResult { get; private set; }

        /// <summary>
        /// Constructor for MarketChatService
        /// </summary>
        public MarketChatService(
            ILogger<MarketChatService> logger,
            IMarketDataEnrichmentService marketDataEnrichmentService = null,
            IPredictionDataService predictionDataService = null,
            IModelExplainerService modelExplainerService = null,
            INaturalLanguageQueryService naturalLanguageQueryService = null,
            SentimentPriceCorrelationAnalysis sentimentCorrelationAnalysis = null,
            IMultiSymbolAnalyzer multiSymbolAnalyzer = null,
            IConfigurationManager configManager = null,
            IChartGenerationService chartGenerationService = null,
            RealTimeInferenceService realTimeInferenceService = null,
            PredictionCacheService predictionCacheService = null)
        {
            _logger = logger;
            _marketDataEnrichmentService = marketDataEnrichmentService;
            _predictionDataService = predictionDataService;
            _modelExplainerService = modelExplainerService ?? new ModelExplainerService();
            _naturalLanguageQueryService = naturalLanguageQueryService;
            _sentimentCorrelationAnalysis = sentimentCorrelationAnalysis ?? new SentimentPriceCorrelationAnalysis();
            _multiSymbolAnalyzer = multiSymbolAnalyzer;
            _chartGenerationService = chartGenerationService ?? new ChartGenerationService(null, predictionDataService);
            _realTimeInferenceService = realTimeInferenceService ?? new RealTimeInferenceService();
            _predictionCacheService = predictionCacheService;
            _httpClient = new HttpClient();
            _conversationHistory = new List<MarketChatMessage>();
            _enrichedContextHistory = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            _predictionContextHistory = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            _sentimentCorrelationContextHistory = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            _predictionCacheResults = new Dictionary<string, PredictionContextResult>(StringComparer.OrdinalIgnoreCase);
            _lastPredictionResults = new Dictionary<string, PredictionResult>(StringComparer.OrdinalIgnoreCase);
            _chartDataCache = new Dictionary<string, ProjectionChartData>(StringComparer.OrdinalIgnoreCase);

            try
            {
                // Set up the HTTP client with API key from settings
                var apiKey = GetOpenAiApiKey();
                _httpClient.BaseAddress = new Uri(OpenAiBaseUrl);
                _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
                _httpClient.Timeout = TimeSpan.FromSeconds(OpenAiTimeout);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize MarketChatService with OpenAI API key");
                throw new InvalidOperationException("OpenAI API key not configured. Please add 'OpenAiApiKey' to alphaVantageSettings.json", ex);
            }
        }

        /// <summary>
        /// Sends a market analysis question to ChatGPT and returns the response.
        /// If the question is detected as a database query, it will be processed
        /// by the NaturalLanguageQueryService instead (MarketChat story 5).
        /// If the question is detected as a multi-symbol comparison request,
        /// it will be processed by the MultiSymbolAnalyzer (MarketChat story 7).
        /// If the question is detected as a chart request, it will generate
        /// chart data for visualization (MarketChat story 8).
        /// If the question is detected as a Python prediction request, it will
        /// invoke stock_predictor.py and stream progress updates (MarketChat story 9).
        /// </summary>
        /// <param name="userQuestion">The user's market analysis question</param>
        /// <param name="includeContext">Whether to include recent market data as context</param>
        /// <returns>The ChatGPT response</returns>
        public async Task<string> SendMarketAnalysisRequestAsync(string userQuestion, bool includeContext = true)
        {
            try
            {
                _logger.LogInformation($"Processing market analysis request: {userQuestion}");

                // Check if this is a database query request (MarketChat story 5)
                if (_naturalLanguageQueryService != null && _naturalLanguageQueryService.IsQueryRequest(userQuestion))
                {
                    _logger.LogInformation("Detected database query request, processing with NaturalLanguageQueryService");
                    return await ProcessDatabaseQueryAsync(userQuestion);
                }

                // Check if this is a multi-symbol comparison request (MarketChat story 7)
                if (IsMultiSymbolComparisonRequest(userQuestion))
                {
                    _logger.LogInformation("Detected multi-symbol comparison request, processing with MultiSymbolAnalyzer");
                    return await ProcessMultiSymbolComparisonAsync(userQuestion);
                }

                // Check if this is a chart visualization request (MarketChat story 8)
                if (IsChartRequest(userQuestion))
                {
                    _logger.LogInformation("Detected chart visualization request, processing with ChartGenerationService");
                    return await ProcessChartRequestAsync(userQuestion);
                }

                // Build the system prompt
                string systemPrompt = BuildSystemPrompt();

                // Build the user prompt with optional context
                string enhancedPrompt = await BuildEnhancedPromptAsync(userQuestion, includeContext);

                // Prepare conversation messages
                var messages = BuildConversationMessages(systemPrompt, enhancedPrompt);

                // Call the OpenAI API
                var response = await ResilienceHelper.ExternalApiCallAsync("OpenAI", async () =>
                {
                    var requestBody = new
                    {
                        model = OpenAiModel,
                        messages,
                        temperature = OpenAiTemperature,
                        max_tokens = 1000 // Sufficient for detailed market analysis
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

                string assistantResponse = response.Choices[0].Message.Content;

                // Store the conversation for context in future messages
                StoreConversationTurn(userQuestion, assistantResponse);

                _logger.LogInformation("Market analysis request processed successfully");
                return assistantResponse;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing market analysis request");
                return "I apologize, but I'm currently unable to process your market analysis request due to a technical issue. Please try again in a moment.";
            }
        }

        /// <summary>
        /// Determines if the user's message is a Python prediction request (MarketChat story 9).
        /// Detects commands like "Run new prediction for TSLA" or "Run LSTM model for AAPL"
        /// or "Generate fresh prediction for MSFT using transformer".
        /// </summary>
        /// <param name="message">The user's message</param>
        /// <returns>True if the message is a prediction execution request</returns>
        public bool IsPredictionRequest(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return false;
            }

            var lowerMessage = message.ToLowerInvariant();

            // Check for prediction execution keywords
            bool hasPredictionKeyword =
                lowerMessage.Contains("run prediction") ||
                lowerMessage.Contains("run new prediction") ||
                lowerMessage.Contains("generate prediction") ||
                lowerMessage.Contains("generate new prediction") ||
                lowerMessage.Contains("run lstm") ||
                lowerMessage.Contains("run gru") ||
                lowerMessage.Contains("run transformer") ||
                lowerMessage.Contains("run random forest") ||
                lowerMessage.Contains("execute prediction") ||
                lowerMessage.Contains("fresh prediction") ||
                lowerMessage.Contains("new ml prediction") ||
                lowerMessage.Contains("run ml model") ||
                (lowerMessage.Contains("run") && lowerMessage.Contains("model")) ||
                (lowerMessage.Contains("predict") && (lowerMessage.Contains("using") || lowerMessage.Contains("with")));

            if (!hasPredictionKeyword)
            {
                return false;
            }

            // Must have at least one symbol
            var symbols = ExtractSymbolsFromQuestion(message);
            return symbols.Count >= 1;
        }

        /// <summary>
        /// Extracts the model type from a prediction request message (MarketChat story 9).
        /// </summary>
        /// <param name="message">The user's message</param>
        /// <returns>The model type (lstm, gru, transformer, random_forest, or auto)</returns>
        public string ExtractModelTypeFromRequest(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return "auto";
            }

            var lowerMessage = message.ToLowerInvariant();

            if (lowerMessage.Contains("lstm"))
            {
                return "lstm";
            }
            if (lowerMessage.Contains("gru"))
            {
                return "gru";
            }
            if (lowerMessage.Contains("transformer"))
            {
                return "transformer";
            }
            if (lowerMessage.Contains("random forest") || lowerMessage.Contains("rf model"))
            {
                return "random_forest";
            }

            return "auto";
        }

        /// <summary>
        /// Processes a Python prediction request by invoking stock_predictor.py (MarketChat story 9).
        /// </summary>
        /// <param name="userQuestion">The user's prediction request</param>
        /// <param name="progressCallback">Optional callback for streaming progress updates</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Formatted response with prediction results</returns>
        public async Task<string> ProcessPredictionRequestAsync(
            string userQuestion,
            PredictionProgressCallback progressCallback = null,
            CancellationToken cancellationToken = default)
        {
            try
            {
                // Extract symbol and model type
                var symbols = ExtractSymbolsFromQuestion(userQuestion);
                if (symbols.Count == 0)
                {
                    return "Please specify a stock symbol for the prediction. For example: 'Run new prediction for TSLA' or 'Run LSTM model for AAPL'.";
                }

                var symbol = symbols.First();
                var modelType = ExtractModelTypeFromRequest(userQuestion);

                _logger.LogInformation("Processing Python prediction request for {Symbol} with model type {ModelType}", symbol, modelType);

                // Execute the Python prediction
                var result = await _realTimeInferenceService.ExecutePythonPredictionAsync(
                    symbol,
                    modelType,
                    progressCallback: progressCallback,
                    cancellationToken: cancellationToken);

                LastPythonPredictionResult = result;

                if (!result.Success)
                {
                    _logger.LogWarning("Python prediction failed for {Symbol}: {Error}", symbol, result.ErrorMessage);
                    return $"❌ **Prediction failed for {symbol}**\n\n{result.ErrorMessage}\n\nPlease try again or check that Python and the required ML libraries are installed.";
                }

                // Cache the prediction result (MarketChat story 9)
                if (_predictionCacheService != null && result.Prediction != null)
                {
                    try
                    {
                        var inputHash = _predictionCacheService.GenerateInputDataHash(new Dictionary<string, double>
                        {
                            ["symbol_hash"] = symbol.GetHashCode(),
                            ["model_type_hash"] = modelType.GetHashCode(),
                            ["timestamp"] = DateTime.Now.Ticks / 10000000.0 // Seconds since epoch, rounded
                        });

                        _predictionCacheService.CachePrediction(symbol, result.ModelType ?? modelType, inputHash, result.Prediction);
                        _logger.LogInformation("Cached prediction for {Symbol} with model {ModelType}", symbol, result.ModelType);
                    }
                    catch (Exception cacheEx)
                    {
                        _logger.LogWarning(cacheEx, "Failed to cache prediction for {Symbol}", symbol);
                    }
                }

                // Store prediction for follow-up questions
                _lastPredictionResults[symbol.ToUpperInvariant()] = result.Prediction;

                // Build the response
                var response = FormatPredictionResponse(symbol, result);

                // Store conversation
                StoreConversationTurn(userQuestion, response);

                _logger.LogInformation("Successfully completed Python prediction for {Symbol} in {Time:F0}ms", symbol, result.ExecutionTimeMs);
                return response;
            }
            catch (OperationCanceledException)
            {
                return "The prediction was cancelled.";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing Python prediction request");
                return "I apologize, but I encountered an error while running the prediction. Please try again.";
            }
        }

        /// <summary>
        /// Formats a Python prediction result into a user-friendly response (MarketChat story 9).
        /// </summary>
        private string FormatPredictionResponse(string symbol, PythonPredictionExecutionResult result)
        {
            var prediction = result.Prediction;
            var responseBuilder = new StringBuilder();

            // Header with prediction action
            var actionEmoji = prediction.Action?.ToUpperInvariant() switch
            {
                "BUY" => "📈",
                "SELL" => "📉",
                _ => "⏸️"
            };

            responseBuilder.AppendLine($"{actionEmoji} **ML Prediction for {symbol}**");
            responseBuilder.AppendLine();

            // Main prediction details
            responseBuilder.AppendLine($"**Action:** {prediction.Action ?? "HOLD"}");
            responseBuilder.AppendLine($"**Confidence:** {prediction.Confidence:P0}");
            responseBuilder.AppendLine($"**Target Price:** ${prediction.TargetPrice:F2}");
            responseBuilder.AppendLine();

            // Model information
            responseBuilder.AppendLine($"**Model:** {GetModelDisplayName(result.ModelType)}");
            responseBuilder.AppendLine($"**Execution Time:** {result.ExecutionTimeMs:F0}ms");
            responseBuilder.AppendLine();

            // Risk metrics if available
            if (prediction.RiskMetrics != null || prediction.RiskScore > 0)
            {
                responseBuilder.AppendLine("**Risk Assessment:**");
                responseBuilder.AppendLine($"- Risk Score: {prediction.RiskScore:P0}");
                if (prediction.ValueAtRisk > 0)
                {
                    responseBuilder.AppendLine($"- Value at Risk (95%): ${prediction.ValueAtRisk:F2}");
                }
                if (prediction.MaxDrawdown > 0)
                {
                    responseBuilder.AppendLine($"- Max Drawdown: {prediction.MaxDrawdown:P1}");
                }
                if (prediction.SharpeRatio != 0)
                {
                    responseBuilder.AppendLine($"- Sharpe Ratio: {prediction.SharpeRatio:F2}");
                }
                responseBuilder.AppendLine();
            }

            // Time series predictions if available
            if (prediction.TimeSeries?.PricePredictions?.Count > 0)
            {
                responseBuilder.AppendLine("**Price Projections:**");
                for (int i = 0; i < Math.Min(5, prediction.TimeSeries.PricePredictions.Count); i++)
                {
                    var date = prediction.TimeSeries.TimePoints?.Count > i
                        ? prediction.TimeSeries.TimePoints[i].ToString("MMM dd")
                        : $"Day {i + 1}";
                    responseBuilder.AppendLine($"- {date}: ${prediction.TimeSeries.PricePredictions[i]:F2}");
                }
                responseBuilder.AppendLine();
            }

            // Technical patterns if available
            if (prediction.DetectedPatterns?.Count > 0)
            {
                responseBuilder.AppendLine("**Detected Patterns:**");
                foreach (var pattern in prediction.DetectedPatterns.Take(3))
                {
                    responseBuilder.AppendLine($"- {pattern.PatternName}: {pattern.ExpectedOutcome} ({pattern.PatternStrength:P0} strength)");
                }
                responseBuilder.AppendLine();
            }

            // Feature weights if available (top 5)
            if (prediction.FeatureWeights?.Count > 0)
            {
                responseBuilder.AppendLine("**Key Factors:**");
                var topWeights = prediction.FeatureWeights
                    .OrderByDescending(w => Math.Abs(w.Value))
                    .Take(5);
                foreach (var weight in topWeights)
                {
                    var direction = weight.Value > 0 ? "↑" : "↓";
                    responseBuilder.AppendLine($"- {FormatFeatureName(weight.Key)}: {direction} {Math.Abs(weight.Value):P0}");
                }
                responseBuilder.AppendLine();
            }

            // Footer
            responseBuilder.AppendLine("_This prediction was generated by the Quantra ML engine. Past performance does not guarantee future results._");

            return responseBuilder.ToString();
        }

        /// <summary>
        /// Gets a user-friendly display name for the model type.
        /// </summary>
        private static string GetModelDisplayName(string modelType)
        {
            return modelType?.ToLowerInvariant() switch
            {
                "lstm" => "LSTM Neural Network",
                "gru" => "GRU Neural Network",
                "transformer" => "Transformer",
                "random_forest" => "Random Forest",
                "pytorch" => "PyTorch LSTM",
                "tensorflow" => "TensorFlow LSTM",
                "auto" => "Auto-selected ML Model",
                _ => modelType ?? "ML Model"
            };
        }

        /// <summary>
        /// Formats a feature name into a user-friendly display string.
        /// </summary>
        private static string FormatFeatureName(string featureName)
        {
            if (string.IsNullOrWhiteSpace(featureName))
            {
                return "Unknown";
            }

            // Handle common technical indicator names
            var mappings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["rsi"] = "RSI (Relative Strength Index)",
                ["macd"] = "MACD",
                ["sma"] = "Simple Moving Average",
                ["ema"] = "Exponential Moving Average",
                ["volume"] = "Trading Volume",
                ["momentum"] = "Price Momentum",
                ["volatility"] = "Volatility",
                ["atr"] = "Average True Range",
                ["bb_upper"] = "Bollinger Upper Band",
                ["bb_lower"] = "Bollinger Lower Band",
                ["bb_width"] = "Bollinger Band Width",
                ["roc"] = "Rate of Change",
                ["returns"] = "Price Returns"
            };

            // Check for exact match first
            if (mappings.TryGetValue(featureName, out var mapped))
            {
                return mapped;
            }

            // Check for partial matches
            foreach (var kvp in mappings)
            {
                if (featureName.ToLowerInvariant().Contains(kvp.Key))
                {
                    return kvp.Value;
                }
            }

            // Default: Title case the feature name
            return System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase(
                featureName.Replace("_", " ").ToLowerInvariant());
        }

        /// <summary>
        /// Determines if the user's message is a multi-symbol comparison request (MarketChat story 7).
        /// Detects queries like "Compare predictions for AAPL, MSFT, and GOOGL" or
        /// "Compare AAPL vs MSFT vs GOOGL" or "Which is better, AAPL or MSFT?"
        /// </summary>
        /// <param name="message">The user's message</param>
        /// <returns>True if the message appears to be a multi-symbol comparison request</returns>
        public bool IsMultiSymbolComparisonRequest(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return false;
            }

            var lowerMessage = message.ToLowerInvariant();

            // Check for explicit comparison keywords
            bool hasCompareKeyword = lowerMessage.Contains("compare") ||
                                     lowerMessage.Contains("comparison") ||
                                     lowerMessage.Contains(" vs ") ||
                                     lowerMessage.Contains("versus") ||
                                     lowerMessage.Contains("which is better") ||
                                     lowerMessage.Contains("which one") ||
                                     lowerMessage.Contains("difference between") ||
                                     lowerMessage.Contains("side by side") ||
                                     lowerMessage.Contains("side-by-side") ||
                                     lowerMessage.Contains("relative value");

            if (!hasCompareKeyword)
            {
                return false;
            }

            // Extract symbols and check if there are multiple
            var symbols = ExtractSymbolsFromQuestion(message);
            return symbols.Count >= 2;
        }

        /// <summary>
        /// Processes a multi-symbol comparison request (MarketChat story 7).
        /// </summary>
        /// <param name="userQuestion">The user's comparison question</param>
        /// <returns>Formatted response with comparison analysis and AI recommendations</returns>
        private async Task<string> ProcessMultiSymbolComparisonAsync(string userQuestion)
        {
            try
            {
                // Extract symbols from the question
                var symbols = ExtractSymbolsFromQuestion(userQuestion);
                if (symbols.Count < 2)
                {
                    return "I need at least two stock symbols to compare. Please specify the symbols you'd like to compare (e.g., 'Compare predictions for AAPL, MSFT, and GOOGL').";
                }

                _logger.LogInformation("Processing multi-symbol comparison for: {Symbols}", string.Join(", ", symbols));

                // Use MultiSymbolAnalyzer if available, otherwise create one
                var analyzer = _multiSymbolAnalyzer ?? new MultiSymbolAnalyzer(
                    null,
                    _predictionDataService,
                    _marketDataEnrichmentService,
                    null);

                // Perform the comparison
                var comparisonResult = await analyzer.CompareSymbolsAsync(symbols, includeHistoricalContext: true);
                LastComparisonResult = comparisonResult;

                if (!comparisonResult.IsSuccessful)
                {
                    var errorMessage = comparisonResult.Errors.Count > 0
                        ? string.Join("; ", comparisonResult.Errors)
                        : "Unable to retrieve data for the specified symbols.";
                    return $"I couldn't complete the comparison: {errorMessage}";
                }

                // Build the response with comparison tables and AI analysis
                var responseBuilder = new StringBuilder();

                // Add formatted comparison tables
                responseBuilder.AppendLine(analyzer.FormatComparisonAsMarkdown(comparisonResult));
                responseBuilder.AppendLine();

                // Add allocation recommendations based on detected risk profile
                var riskProfile = ExtractRiskProfile(userQuestion);
                responseBuilder.AppendLine(analyzer.GenerateAllocationRecommendations(comparisonResult, riskProfile));
                responseBuilder.AppendLine();

                // Now get AI-enhanced analysis for portfolio optimization
                var aiAnalysis = await GetAiComparisonAnalysisAsync(comparisonResult, userQuestion);
                if (!string.IsNullOrEmpty(aiAnalysis))
                {
                    responseBuilder.AppendLine("### AI Analysis & Recommendations");
                    responseBuilder.AppendLine(aiAnalysis);
                }

                var response = responseBuilder.ToString();

                // Store the conversation
                StoreConversationTurn(userQuestion, response);

                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing multi-symbol comparison");
                return "I apologize, but I encountered an error while comparing the symbols. Please try again.";
            }
        }

        /// <summary>
        /// Gets AI-enhanced analysis for a multi-symbol comparison (MarketChat story 7).
        /// Uses OpenAI to provide portfolio optimization recommendations based on the comparison data.
        /// </summary>
        private async Task<string> GetAiComparisonAnalysisAsync(MultiSymbolComparisonResult comparisonResult, string originalQuestion)
        {
            try
            {
                // Build a comparison-focused system prompt
                var systemPrompt = BuildComparisonSystemPrompt();

                // Build the context from comparison data
                var analyzer = _multiSymbolAnalyzer ?? new MultiSymbolAnalyzer(null, _predictionDataService, _marketDataEnrichmentService, null);
                var comparisonContext = analyzer.BuildComparisonContext(comparisonResult);

                var userPrompt = new StringBuilder();
                userPrompt.AppendLine(comparisonContext);
                userPrompt.AppendLine();
                userPrompt.AppendLine($"Original question: {originalQuestion}");
                userPrompt.AppendLine();
                userPrompt.AppendLine("Based on the comparison data above, provide:");
                userPrompt.AppendLine("1. A summary of the key differences between these symbols");
                userPrompt.AppendLine("2. Which symbol(s) present the best risk-adjusted opportunity");
                userPrompt.AppendLine("3. Any portfolio allocation suggestions");
                userPrompt.AppendLine("4. Key risks to consider for each symbol");

                var messages = new List<object>
                {
                    new { role = "system", content = systemPrompt },
                    new { role = "user", content = userPrompt.ToString() }
                };

                var response = await ResilienceHelper.ExternalApiCallAsync("OpenAI", async () =>
                {
                    var requestBody = new
                    {
                        model = OpenAiModel,
                        messages = messages.ToArray(),
                        temperature = 0.4, // Slightly higher for portfolio recommendations
                        max_tokens = 1200
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

                return response.Choices[0].Message.Content;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get AI comparison analysis");
                return null;
            }
        }

        /// <summary>
        /// Builds the system prompt specifically for multi-symbol comparison analysis (MarketChat story 7).
        /// </summary>
        private string BuildComparisonSystemPrompt()
        {
            return "You are a professional portfolio analyst for Quantra, an advanced algorithmic trading platform. " +
                   "You are analyzing a multi-symbol comparison and providing portfolio allocation recommendations. " +
                   "Focus on relative value analysis - comparing the strengths and weaknesses of each symbol against the others. " +
                   "When discussing predictions, reference the specific confidence levels and composite scores. " +
                   "When discussing risk, reference the specific risk scores and volatility metrics. " +
                   "Provide clear, actionable recommendations for portfolio allocation based on the comparison data. " +
                   "Consider both return potential and risk when making recommendations. " +
                   "If there are clear winners or losers in the comparison, highlight them with specific reasons. " +
                   "Format your response with clear sections and bullet points for easy reading. " +
                   "Always mention relevant risks and avoid giving direct investment advice. " +
                   "Use professional yet accessible language appropriate for experienced traders.";
        }

        /// <summary>
        /// Extracts the risk profile from a message for multi-symbol comparison (MarketChat story 7).
        /// </summary>
        private string ExtractRiskProfile(string message)
        {
            string loweredMessage = message.ToLower();

            if (loweredMessage.Contains("conservative") || loweredMessage.Contains("low risk") || loweredMessage.Contains("safe"))
                return "conservative";
            if (loweredMessage.Contains("aggressive") || loweredMessage.Contains("high risk") || loweredMessage.Contains("risky"))
                return "aggressive";

            return "moderate"; // Default risk profile
        }

        /// <summary>
        /// Processes a database query request using natural language (MarketChat story 5).
        /// </summary>
        /// <param name="userQuestion">The user's natural language database query</param>
        /// <returns>Formatted response with query results</returns>
        private async Task<string> ProcessDatabaseQueryAsync(string userQuestion)
        {
            try
            {
                var result = await _naturalLanguageQueryService.ProcessQueryAsync(userQuestion);
                LastQueryResult = result;

                if (!result.Success)
                {
                    return result.WasBlocked
                        ? $"⚠️ **Query blocked for safety**: {result.BlockedReason}\n\nPlease rephrase your query or ask a different question."
                        : $"❌ **Query failed**: {result.ErrorMessage}\n\nPlease try rephrasing your question.";
                }

                // Format the response with markdown table
                var responseBuilder = new StringBuilder();
                responseBuilder.AppendLine($"📊 **Database Query Results** ({result.RowCount} rows, {result.ExecutionTimeMs}ms)");
                responseBuilder.AppendLine();
                responseBuilder.AppendLine(result.ToMarkdownTable());

                if (!string.IsNullOrEmpty(result.TranslatedSql))
                {
                    responseBuilder.AppendLine();
                    responseBuilder.AppendLine("<details>");
                    responseBuilder.AppendLine("<summary>SQL Query (click to expand)</summary>");
                    responseBuilder.AppendLine();
                    responseBuilder.AppendLine($"```sql");
                    responseBuilder.AppendLine(result.TranslatedSql);
                    responseBuilder.AppendLine($"```");
                    responseBuilder.AppendLine("</details>");
                }

                // Store the conversation
                StoreConversationTurn(userQuestion, responseBuilder.ToString());

                return responseBuilder.ToString();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing database query");
                return "I apologize, but I couldn't process your database query. Please try rephrasing your question.";
            }
        }

        /// <summary>
        /// Determines if the user's message is a database query request (MarketChat story 5).
        /// </summary>
        /// <param name="message">The user's message</param>
        /// <returns>True if the message appears to be a database query request</returns>
        public bool IsQueryRequest(string message)
        {
            return _naturalLanguageQueryService?.IsQueryRequest(message) ?? false;
        }

        /// <summary>
        /// Determines if the user's message is a chart visualization request (MarketChat story 8).
        /// </summary>
        /// <param name="message">The user's message</param>
        /// <returns>True if the message appears to be a chart request</returns>
        public bool IsChartRequest(string message)
        {
            return _chartGenerationService?.IsChartRequest(message) ?? false;
        }

        /// <summary>
        /// Processes a chart visualization request (MarketChat story 8).
        /// Generates chart data for historical prices and ML projections.
        /// </summary>
        /// <param name="userQuestion">The user's chart request</param>
        /// <returns>Response with chart description and chart data set in LastChartData</returns>
        private async Task<string> ProcessChartRequestAsync(string userQuestion)
        {
            try
            {
                var parameters = _chartGenerationService.ExtractChartParameters(userQuestion);
                if (parameters.Symbols.Count == 0)
                {
                    return "Please specify a stock symbol for the chart. For example: 'Show me a chart for AAPL with 30-day projections'";
                }

                var symbol = parameters.Symbols.First();
                var startDate = parameters.StartDate ?? DateTime.Now.AddDays(-parameters.HistoricalDays);

                _logger.LogInformation("Generating projection chart for {Symbol} from {StartDate} with {ForecastDays} day forecast",
                    symbol, startDate, parameters.ForecastDays);

                var chartData = await _chartGenerationService.GenerateProjectionChartAsync(symbol, startDate, parameters.ForecastDays);
                LastChartData = chartData;

                // Cache the chart data for multi-turn conversations (normalized to uppercase)
                _chartDataCache[symbol.ToUpperInvariant()] = chartData;

                if (!chartData.IsValid)
                {
                    return $"Unable to generate chart for {symbol}: {chartData.ErrorMessage}";
                }

                // Build response with chart context
                var responseBuilder = new StringBuilder();
                responseBuilder.AppendLine($"📈 **{chartData.ChartTitle}**");
                responseBuilder.AppendLine();
                responseBuilder.AppendLine($"**Current Price:** ${chartData.CurrentPrice:F2}");
                responseBuilder.AppendLine($"**Target Price:** ${chartData.TargetPrice:F2} ({(chartData.TargetPrice > chartData.CurrentPrice ? "+" : "")}{((chartData.TargetPrice - chartData.CurrentPrice) / chartData.CurrentPrice):P1})");
                responseBuilder.AppendLine($"**Prediction:** {chartData.PredictedAction} ({chartData.Confidence:P0} confidence)");
                responseBuilder.AppendLine();

                // Bollinger Bands summary
                if (chartData.BollingerUpper?.Count > 0)
                {
                    var lastUpper = chartData.BollingerUpper.LastOrDefault(v => !double.IsNaN(v));
                    var lastMiddle = chartData.BollingerMiddle.LastOrDefault(v => !double.IsNaN(v));
                    var lastLower = chartData.BollingerLower.LastOrDefault(v => !double.IsNaN(v));
                    
                    if (lastMiddle > 0)
                    {
                        responseBuilder.AppendLine("**Bollinger Bands (20-day):**");
                        responseBuilder.AppendLine($"- Upper: ${lastUpper:F2}");
                        responseBuilder.AppendLine($"- Middle (SMA): ${lastMiddle:F2}");
                        responseBuilder.AppendLine($"- Lower: ${lastLower:F2}");
                        
                        // Position analysis
                        if (chartData.CurrentPrice > lastUpper)
                        {
                            responseBuilder.AppendLine($"- *Price is above upper band (potentially overbought)*");
                        }
                        else if (chartData.CurrentPrice < lastLower)
                        {
                            responseBuilder.AppendLine($"- *Price is below lower band (potentially oversold)*");
                        }
                        responseBuilder.AppendLine();
                    }
                }

                // Support/Resistance summary
                if (chartData.SupportLevels?.Count > 0 || chartData.ResistanceLevels?.Count > 0)
                {
                    responseBuilder.AppendLine("**Key Levels:**");
                    if (chartData.ResistanceLevels?.Count > 0)
                    {
                        responseBuilder.AppendLine($"- Resistance: {string.Join(", ", chartData.ResistanceLevels.Select(r => $"${r:F2}"))}");
                    }
                    if (chartData.SupportLevels?.Count > 0)
                    {
                        responseBuilder.AppendLine($"- Support: {string.Join(", ", chartData.SupportLevels.Select(s => $"${s:F2}"))}");
                    }
                    responseBuilder.AppendLine();
                }

                responseBuilder.AppendLine($"_Chart shows {chartData.HistoricalPrices.Count} days of historical data and {chartData.PredictionPrices.Count} days of ML projections._");
                responseBuilder.AppendLine();
                responseBuilder.AppendLine("*The chart is displayed above this message.*");

                var response = responseBuilder.ToString();
                StoreConversationTurn(userQuestion, response);

                _logger.LogInformation("Successfully generated chart for {Symbol} with {Historical} historical points and {Prediction} prediction points",
                    symbol, chartData.HistoricalPrices.Count, chartData.PredictionPrices.Count);

                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing chart request");
                LastChartData = null;
                return "I apologize, but I couldn't generate the chart. Please try again with a valid stock symbol.";
            }
        }

        /// <summary>
        /// Gets cached chart data for a symbol if available (MarketChat story 8).
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Cached chart data or null</returns>
        public ProjectionChartData GetCachedChartData(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return null;
            }

            _chartDataCache.TryGetValue(symbol.ToUpperInvariant(), out var chartData);
            return chartData;
        }

        /// <summary>
        /// Generate a trading plan for a specific ticker and market conditions
        /// </summary>
        /// <param name="ticker">The ticker symbol</param>
        /// <param name="timeframe">The trading timeframe (e.g., "next week", "next month")</param>
        /// <param name="marketContext">Optional market context information</param>
        /// <param name="riskProfile">The user's risk tolerance (e.g., "conservative", "moderate", "aggressive")</param>
        /// <returns>A detailed trading plan</returns>
        public async Task<string> GetTradingPlanAsync(string ticker, string timeframe = "next month", string marketContext = null, string riskProfile = "moderate")
        {
            try
            {
                _logger.LogInformation($"Generating trading plan for {ticker}, timeframe: {timeframe}, risk profile: {riskProfile}");

                // Build the system prompt specific to trading plans
                string systemPrompt = BuildTradingPlanSystemPrompt();

                // Build the user prompt
                var promptBuilder = new StringBuilder();

                // Add historical data context for the ticker
                if (_marketDataEnrichmentService != null && !string.IsNullOrWhiteSpace(ticker))
                {
                    try
                    {
                        var historicalContext = await _marketDataEnrichmentService.GetHistoricalContextAsync(ticker.ToUpperInvariant(), 60);
                        if (!string.IsNullOrEmpty(historicalContext))
                        {
                            promptBuilder.AppendLine(historicalContext);
                            promptBuilder.AppendLine();

                            // Store in enriched context history for follow-up questions
                            _enrichedContextHistory[ticker.ToUpperInvariant()] = historicalContext;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to fetch historical context for trading plan ticker {Ticker}", ticker);
                    }
                }

                // Add ML prediction context for the ticker
                if (_predictionDataService != null && !string.IsNullOrWhiteSpace(ticker))
                {
                    try
                    {
                        var predictionContext = await _predictionDataService.GetPredictionContextAsync(ticker.ToUpperInvariant());
                        if (!string.IsNullOrEmpty(predictionContext))
                        {
                            promptBuilder.AppendLine(predictionContext);
                            promptBuilder.AppendLine();

                            // Store in prediction context history for follow-up questions
                            _predictionContextHistory[ticker.ToUpperInvariant()] = predictionContext;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to fetch prediction context for trading plan ticker {Ticker}", ticker);
                    }
                }

                promptBuilder.AppendLine($"Suggest a detailed trading plan for {ticker} for the {timeframe}.");
                promptBuilder.AppendLine($"Include entry and exit criteria, position sizing, stop-loss, and a brief rationale.");
                promptBuilder.AppendLine($"Assume {riskProfile} risk tolerance.");

                if (!string.IsNullOrEmpty(marketContext))
                {
                    promptBuilder.AppendLine($"Market context: {marketContext}");
                }

                promptBuilder.AppendLine($"Current market session: {GetCurrentMarketSession()}");
                promptBuilder.AppendLine($"Current timestamp: {DateTime.Now:yyyy-MM-dd HH:mm:ss} UTC");

                string userPrompt = promptBuilder.ToString();

                // Prepare conversation messages with specialized system prompt
                var messages = new List<object>
                {
                    new { role = "system", content = systemPrompt },
                    new { role = "user", content = userPrompt }
                };

                // Call the OpenAI API
                var response = await ResilienceHelper.ExternalApiCallAsync("OpenAI", async () =>
                {
                    var requestBody = new
                    {
                        model = OpenAiModel,
                        messages = messages.ToArray(),
                        temperature = 0.4, // Slightly higher than standard for creative trading plans
                        max_tokens = 1500 // More tokens for detailed plans
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

                string tradingPlan = response.Choices[0].Message.Content;
                _logger.LogInformation($"Successfully generated trading plan for {ticker}");

                return tradingPlan;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating trading plan for {ticker}");
                return "I apologize, but I'm currently unable to generate a trading plan due to a technical issue. Please try again in a moment.";
            }
        }

        /// <summary>
        /// Gets recent conversation history for context
        /// </summary>
        public List<MarketChatMessage> GetConversationHistory()
        {
            return new List<MarketChatMessage>(_conversationHistory);
        }

        /// <summary>
        /// Clears the conversation history and enriched context cache
        /// </summary>
        public void ClearConversationHistory()
        {
            _conversationHistory.Clear();
            _enrichedContextHistory.Clear();
            _predictionContextHistory.Clear();
            _sentimentCorrelationContextHistory.Clear();
            _predictionCacheResults.Clear();
            _lastPredictionResults.Clear();
            _chartDataCache.Clear();
            LastPredictionCacheResult = null;
            LastChartData = null;
            _logger.LogInformation("Conversation history and enriched context cleared");
        }

        /// <summary>
        /// Gets prediction context for a specific stock symbol.
        /// Returns null if no predictions exist or the prediction service is not configured.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL", "MSFT")</param>
        /// <returns>Formatted prediction context string for display, or null if unavailable</returns>
        public async Task<string> GetPredictionContext(string symbol)
        {
            if (_predictionDataService == null)
            {
                _logger.LogWarning("Prediction data service is not configured");
                return null;
            }

            if (string.IsNullOrWhiteSpace(symbol))
            {
                return null;
            }

            try
            {
                return await _predictionDataService.GetPredictionContextAsync(symbol);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching prediction context for {Symbol}", symbol);
                return null;
            }
        }

        /// <summary>
        /// Gets prediction context with cache metadata for a specific stock symbol (MarketChat story 3).
        /// Updates LastPredictionCacheResult property with the cache status.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL", "MSFT")</param>
        /// <returns>PredictionContextResult containing the context and cache metadata</returns>
        public async Task<PredictionContextResult> GetPredictionContextWithCacheAsync(string symbol)
        {
            if (_predictionDataService == null)
            {
                _logger.LogWarning("Prediction data service is not configured");
                return PredictionContextResult.Empty;
            }

            if (string.IsNullOrWhiteSpace(symbol))
            {
                return PredictionContextResult.Empty;
            }

            try
            {
                var result = await _predictionDataService.GetPredictionContextWithCacheAsync(symbol);
                LastPredictionCacheResult = result;
                _predictionCacheResults[symbol.ToUpperInvariant()] = result;

                if (result?.IsCached == true)
                {
                    _logger.LogInformation("Retrieved cached prediction for {Symbol} ({CacheAge} old)", 
                        symbol, result.CacheAge);
                }
                else if (result?.Context != null)
                {
                    _logger.LogInformation("Retrieved fresh prediction for {Symbol}", symbol);
                }

                return result ?? PredictionContextResult.Empty;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching prediction context with cache for {Symbol}", symbol);
                return PredictionContextResult.Empty;
            }
        }

        /// <summary>
        /// Warms the prediction cache for popular symbols during market hours (MarketChat story 3).
        /// </summary>
        /// <param name="symbols">Optional list of symbols to warm. Uses popular symbols if null.</param>
        /// <returns>Number of symbols successfully warmed</returns>
        public async Task<int> WarmPredictionCacheAsync(IEnumerable<string> symbols = null)
        {
            if (_predictionDataService == null)
            {
                _logger.LogWarning("Prediction data service is not configured - cannot warm cache");
                return 0;
            }

            try
            {
                return await _predictionDataService.WarmCacheForSymbolsAsync(symbols);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error warming prediction cache");
                return 0;
            }
        }

        /// <summary>
        /// Gets the cache result for a specific symbol if available (MarketChat story 3).
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>The cached result or null if not available</returns>
        public PredictionContextResult GetCachedPredictionResult(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return null;
            }

            _predictionCacheResults.TryGetValue(symbol.ToUpperInvariant(), out var result);
            return result;
        }

        /// <summary>
        /// Generates a plain English explanation of the factors driving a prediction (MarketChat story 4).
        /// </summary>
        /// <param name="prediction">The prediction result to explain</param>
        /// <returns>A detailed explanation of the prediction factors</returns>
        public string ExplainPredictionFactors(PredictionResult prediction)
        {
            if (_modelExplainerService == null)
            {
                _logger.LogWarning("Model explainer service is not configured");
                return "Prediction explanation service is not available.";
            }

            try
            {
                return _modelExplainerService.ExplainPredictionFactors(prediction);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating prediction explanation");
                return "Unable to generate prediction explanation at this time.";
            }
        }

        /// <summary>
        /// Gets a stored prediction result for a symbol (MarketChat story 4).
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>The prediction result or null if not available</returns>
        public PredictionResult GetStoredPredictionResult(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return null;
            }

            _lastPredictionResults.TryGetValue(symbol.ToUpperInvariant(), out var result);
            return result;
        }

        /// <summary>
        /// Stores a prediction result for later explanation (MarketChat story 4).
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="prediction">The prediction result to store</param>
        public void StorePredictionResult(string symbol, PredictionResult prediction)
        {
            if (string.IsNullOrWhiteSpace(symbol) || prediction == null)
            {
                return;
            }

            _lastPredictionResults[symbol.ToUpperInvariant()] = prediction;
            _logger.LogDebug("Stored prediction result for {Symbol}", symbol);
        }

        /// <summary>
        /// Compares predictions from different model types for a symbol (MarketChat story 4).
        /// </summary>
        /// <param name="predictions">Dictionary of model type to prediction result</param>
        /// <returns>A plain English comparison of the predictions</returns>
        public string CompareModelPredictions(Dictionary<string, PredictionResult> predictions)
        {
            if (_modelExplainerService == null)
            {
                _logger.LogWarning("Model explainer service is not configured");
                return "Model comparison service is not available.";
            }

            try
            {
                return _modelExplainerService.CompareModelPredictions(predictions);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error comparing model predictions");
                return "Unable to compare model predictions at this time.";
            }
        }

        /// <summary>
        /// Gets sentiment-price correlation context for a specific symbol (MarketChat story 6).
        /// Returns historical analysis of how sentiment shifts have correlated with price movements.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL", "MSFT")</param>
        /// <param name="days">Number of days to analyze (default 30)</param>
        /// <returns>Formatted context string with correlation data, or null if unavailable</returns>
        public async Task<string> GetSentimentCorrelationContext(string symbol, int days = 30)
        {
            if (_sentimentCorrelationAnalysis == null)
            {
                _logger.LogWarning("Sentiment correlation analysis service is not configured");
                return null;
            }

            if (string.IsNullOrWhiteSpace(symbol))
            {
                return null;
            }

            try
            {
                // Check if we have cached context for this symbol
                if (_sentimentCorrelationContextHistory.TryGetValue(symbol.ToUpperInvariant(), out var cachedContext))
                {
                    return cachedContext;
                }

                // Fetch fresh sentiment correlation context
                var context = await _sentimentCorrelationAnalysis.GetHistoricalSentimentContext(symbol, days);
                
                if (!string.IsNullOrEmpty(context))
                {
                    // Cache the context for future use
                    _sentimentCorrelationContextHistory[symbol.ToUpperInvariant()] = context;
                    _logger.LogInformation("Retrieved sentiment-price correlation context for {Symbol}", symbol);
                }

                return context;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching sentiment correlation context for {Symbol}", symbol);
                return null;
            }
        }

        /// <summary>
        /// Generates a detailed feature weight context for AI prompts (MarketChat story 4).
        /// </summary>
        /// <param name="prediction">The prediction result</param>
        /// <returns>Formatted context string with feature weights and explanations</returns>
        private string BuildFeatureWeightContext(PredictionResult prediction)
        {
            if (prediction?.FeatureWeights == null || prediction.FeatureWeights.Count == 0)
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            builder.AppendLine();
            builder.AppendLine("Feature Weight Analysis (factors driving this prediction):");
            
            // Add top influential factors
            var topFactors = _modelExplainerService?.GetTopInfluentialFactors(prediction.FeatureWeights, 5);
            if (!string.IsNullOrEmpty(topFactors))
            {
                builder.AppendLine(topFactors);
            }

            // Add confidence explanation
            if (prediction.Confidence > 0 && _modelExplainerService != null)
            {
                builder.AppendLine(_modelExplainerService.ExplainConfidenceScore(prediction.Confidence, prediction.ModelType));
            }

            // Add risk metrics if available
            if (prediction.RiskMetrics != null && _modelExplainerService != null)
            {
                builder.AppendLine();
                builder.AppendLine("Risk Assessment:");
                builder.AppendLine(_modelExplainerService.ExplainRiskMetrics(prediction.RiskMetrics));
            }

            return builder.ToString();
        }

        /// <summary>
        /// Builds the system prompt for market analysis
        /// </summary>
        private string BuildSystemPrompt()
        {
            return "You are a professional financial analyst assistant for Quantra, an advanced algorithmic trading platform. " +
                   "Provide clear, concise, and actionable market analysis to help traders make informed decisions. " +
                   "Focus on technical analysis, market sentiment, risk factors, and actionable insights. " +
                   "When historical data context is provided, reference specific price levels, moving averages, volatility metrics, and volume patterns in your analysis. " +
                   "When ML prediction data is provided, discuss the AI-generated forecast including the predicted action (BUY/SELL/HOLD), target price, and confidence level. " +
                   "When feature weights are provided, explain which technical indicators drove the prediction the most, " +
                   "translating terms like RSI, MACD, VWAP into plain English (e.g., 'The RSI at 28 indicates oversold conditions, meaning the stock may be undervalued'). " +
                   "When risk metrics are provided (VaR, Sharpe ratio, max drawdown), explain what they mean for the trader's risk exposure. " +
                   "When sentiment-price correlation data is provided, explain the historical relationship between sentiment shifts and price movements. " +
                   "Reference the correlation coefficients from different sentiment sources (Twitter, News, Analyst Ratings, etc.) and explain what they mean. " +
                   "If sentiment leads price (positive lead/lag), mention that sentiment shifts have historically preceded price movements. " +
                   "Explain the predictive accuracy score in context - for example: 'Twitter sentiment for NVDA showed a +0.65 correlation with next-day price moves over the past 30 days, " +
                   "with sentiment shifts correctly predicting price direction 68% of the time.' " +
                   "When recent sentiment shift events are provided, discuss specific instances where sentiment predicted (or failed to predict) price movements. " +
                   "If the user asks about factors driving a prediction, focus on the top 3-5 most influential features and their practical meaning. " +
                   "For example: 'The 0.85 confidence score is driven primarily by RSI (weight: 0.32) indicating oversold conditions, " +
                   "combined with MACD momentum (weight: 0.24) showing bullish divergence...' " +
                   "Always mention relevant risks and avoid giving direct investment advice. " +
                   "Use professional yet accessible language appropriate for experienced traders.";
        }

        /// <summary>
        /// Builds the system prompt specifically for trading plan requests
        /// </summary>
        private string BuildTradingPlanSystemPrompt()
        {
            return "You are a professional trading coach for Quantra, an advanced algorithmic trading platform. " +
                   "Generate detailed, actionable trading plans based on technical analysis, market conditions, and risk management. " +
                   "When historical data context is provided, use actual price levels, moving averages, volatility, and volume patterns to inform your recommendations. " +
                   "When ML prediction data is provided, incorporate the AI-generated forecast into your trading plan, explaining how the predicted action, target price, and confidence level inform entry/exit strategies. " +
                   "When feature weights are provided, explain which technical indicators influenced the prediction most and how they should factor into the trading plan. " +
                   "For risk metrics (VaR, Sharpe ratio, max drawdown), incorporate them into position sizing and stop-loss recommendations. " +
                   "When sentiment-price correlation data is provided, incorporate the sentiment analysis into the trading plan. " +
                   "Explain how the historical correlation between sentiment and price movements for this symbol affects timing and conviction. " +
                   "If sentiment has strong predictive accuracy, consider it as a signal for entry/exit timing. " +
                   "Reference specific sentiment shift events that have historically preceded price movements. " +
                   "Structure your responses with clear sections for Entry Criteria, Exit Criteria, Position Sizing, " +
                   "Stop-Loss Strategy, Take-Profit Targets, and Rationale. " +
                   "Always consider risk management as a priority and explain your reasoning. " +
                   "Use professional yet accessible language appropriate for experienced traders.";
        }

        /// <summary>
        /// Builds an enhanced prompt with market context and historical data
        /// </summary>
        private async Task<string> BuildEnhancedPromptAsync(string userQuestion, bool includeContext)
        {
            var promptBuilder = new StringBuilder();

            if (includeContext)
            {
                // Add current market context
                promptBuilder.AppendLine("Current Market Context:");
                promptBuilder.AppendLine($"- Timestamp: {DateTime.Now:yyyy-MM-dd HH:mm:ss} UTC");
                promptBuilder.AppendLine("- Market Session: " + GetCurrentMarketSession());
                promptBuilder.AppendLine();

                // Extract symbol(s) from the user question and add historical context
                var symbols = ExtractSymbolsFromQuestion(userQuestion);
                if (symbols.Count > 0)
                {
                    foreach (var symbol in symbols)
                    {
                        // Add historical market data context
                        if (_marketDataEnrichmentService != null)
                        {
                            try
                            {
                                // Check if we have enriched context in history for follow-up questions
                                if (_enrichedContextHistory.TryGetValue(symbol, out var cachedContext))
                                {
                                    promptBuilder.AppendLine(cachedContext);
                                    promptBuilder.AppendLine();
                                }
                                else
                                {
                                    // Fetch fresh historical context
                                    var historicalContext = await _marketDataEnrichmentService.GetHistoricalContextAsync(symbol, 60);
                                    if (!string.IsNullOrEmpty(historicalContext))
                                    {
                                        promptBuilder.AppendLine(historicalContext);
                                        promptBuilder.AppendLine();

                                        // Store in enriched context history for follow-up questions
                                        _enrichedContextHistory[symbol] = historicalContext;
                                        _logger.LogInformation("Added historical context for {Symbol} to conversation", symbol);
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "Failed to fetch historical context for {Symbol}", symbol);
                            }
                        }

                        // Add ML prediction context using cache-aware method (MarketChat story 3)
                        if (_predictionDataService != null)
                        {
                            try
                            {
                                // Check if we have prediction context in history for follow-up questions
                                if (_predictionContextHistory.TryGetValue(symbol, out var cachedPrediction))
                                {
                                    promptBuilder.AppendLine(cachedPrediction);
                                    promptBuilder.AppendLine();
                                }
                                else
                                {
                                    // Fetch prediction context using cache-aware method
                                    var predictionResult = await _predictionDataService.GetPredictionContextWithCacheAsync(symbol);
                                    if (predictionResult != null && !string.IsNullOrEmpty(predictionResult.Context))
                                    {
                                        promptBuilder.AppendLine(predictionResult.Context);

                                        // Add cache status information to the context
                                        if (predictionResult.IsCached)
                                        {
                                            promptBuilder.AppendLine($"- Data Source: Cached ({predictionResult.CacheStatusDisplay})");
                                        }
                                        else
                                        {
                                            promptBuilder.AppendLine("- Data Source: Freshly generated prediction");
                                        }
                                        promptBuilder.AppendLine();

                                        // Store in prediction context history for follow-up questions
                                        _predictionContextHistory[symbol] = predictionResult.Context;
                                        _predictionCacheResults[symbol] = predictionResult;
                                        LastPredictionCacheResult = predictionResult;

                                        _logger.LogInformation(
                                            "Added ML prediction context for {Symbol} to conversation (cached: {IsCached})", 
                                            symbol, predictionResult.IsCached);
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "Failed to fetch prediction context for {Symbol}", symbol);
                            }
                        }

                        // Add sentiment-price correlation context (MarketChat story 6)
                        if (_sentimentCorrelationAnalysis != null)
                        {
                            try
                            {
                                // Check if we have sentiment correlation context in history for follow-up questions
                                if (_sentimentCorrelationContextHistory.TryGetValue(symbol, out var cachedSentimentCorrelation))
                                {
                                    promptBuilder.AppendLine(cachedSentimentCorrelation);
                                    promptBuilder.AppendLine();
                                }
                                else
                                {
                                    // Fetch sentiment-price correlation context
                                    var sentimentCorrelationContext = await _sentimentCorrelationAnalysis.GetHistoricalSentimentContext(symbol, 30);
                                    if (!string.IsNullOrEmpty(sentimentCorrelationContext))
                                    {
                                        promptBuilder.AppendLine(sentimentCorrelationContext);
                                        promptBuilder.AppendLine();

                                        // Store in sentiment correlation context history for follow-up questions
                                        _sentimentCorrelationContextHistory[symbol] = sentimentCorrelationContext;
                                        _logger.LogInformation("Added sentiment-price correlation context for {Symbol} to conversation", symbol);
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "Failed to fetch sentiment-price correlation context for {Symbol}", symbol);
                            }
                        }
                    }
                }
            }

            promptBuilder.AppendLine("User Question:");
            promptBuilder.AppendLine(userQuestion);

            return promptBuilder.ToString();
        }

        /// <summary>
        /// Extracts stock symbols from a user question
        /// </summary>
        private List<string> ExtractSymbolsFromQuestion(string question)
        {
            var symbols = new List<string>();
            
            if (string.IsNullOrWhiteSpace(question))
            {
                return symbols;
            }

            // Use compiled regex patterns for performance
            var compiledPatterns = new[] { DollarSymbolPattern, StandaloneSymbolPattern };

            foreach (var pattern in compiledPatterns)
            {
                var matches = pattern.Matches(question);
                foreach (Match match in matches)
                {
                    var symbol = match.Groups[1].Value;
                    
                    // Filter out common words that might match the pattern
                    if (!IsCommonWord(symbol) && !symbols.Contains(symbol))
                    {
                        symbols.Add(symbol);
                    }
                }
            }

            // Limit to first 3 symbols to avoid overwhelming the context
            return symbols.Take(3).ToList();
        }

        /// <summary>
        /// Checks if a word is a common English word (not a stock symbol)
        /// </summary>
        private static bool IsCommonWord(string word)
        {
            return CommonWords.Contains(word);
        }

        /// <summary>
        /// Builds conversation messages including history for context
        /// </summary>
        private object[] BuildConversationMessages(string systemPrompt, string userPrompt)
        {
            var messages = new List<object>
            {
                new { role = "system", content = systemPrompt }
            };

            // Add recent conversation history (last 4 messages to stay within token limits)
            var recentHistory = _conversationHistory.Count > 4
                ? _conversationHistory.GetRange(_conversationHistory.Count - 4, 4)
                : _conversationHistory;

            foreach (var historyMessage in recentHistory)
            {
                if (historyMessage.IsFromUser)
                {
                    messages.Add(new { role = "user", content = historyMessage.Content });
                }
                else
                {
                    messages.Add(new { role = "assistant", content = historyMessage.Content });
                }
            }

            // Add the current user message
            messages.Add(new { role = "user", content = userPrompt });

            return messages.ToArray();
        }

        /// <summary>
        /// Stores a conversation turn for future context
        /// </summary>
        private void StoreConversationTurn(string userQuestion, string assistantResponse)
        {
            var timestamp = DateTime.UtcNow;

            _conversationHistory.Add(new MarketChatMessage
            {
                Content = userQuestion,
                IsFromUser = true,
                Timestamp = timestamp,
                MessageType = MessageType.UserQuestion
            });

            _conversationHistory.Add(new MarketChatMessage
            {
                Content = assistantResponse,
                IsFromUser = false,
                Timestamp = timestamp.AddSeconds(1),
                MessageType = MessageType.AssistantResponse
            });

            // Keep only the last 20 messages to manage memory
            if (_conversationHistory.Count > 20)
            {
                _conversationHistory.RemoveRange(0, _conversationHistory.Count - 20);
            }
        }

        /// <summary>
        /// Gets the current market session for context
        /// </summary>
        private string GetCurrentMarketSession()
        {
            var currentTime = DateTime.Now.TimeOfDay;
            var currentDay = DateTime.Now.DayOfWeek;

            // US Market hours (9:30 AM - 4:00 PM ET)
            if (currentDay >= DayOfWeek.Monday && currentDay <= DayOfWeek.Friday)
            {
                if (currentTime >= new TimeSpan(9, 30, 0) && currentTime <= new TimeSpan(16, 0, 0))
                {
                    return "US Market Open";
                }
                else if (currentTime >= new TimeSpan(4, 0, 0) && currentTime <= new TimeSpan(20, 0, 0))
                {
                    return "After Hours Trading";
                }
                else
                {
                    return "Pre-Market";
                }
            }

            return "Market Closed (Weekend)";
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
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty(openAiApiKeyProperty, out var apiKeyElement))
            {
                var key = apiKeyElement.GetString();
                if (string.IsNullOrWhiteSpace(key))
                {
                    throw new InvalidOperationException($"'{openAiApiKeyProperty}' is empty in settings file.");
                }
                return key;
            }

            throw new KeyNotFoundException($"'{openAiApiKeyProperty}' not found in settings file.");
        }
    }
}