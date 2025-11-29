using System;
using System.Collections.ObjectModel;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Extensions.Logging;
using Quantra.Commands;
using Quantra.Models;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels.Base;
using Quantra.DAL.Services;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Market Chat interface.
    /// Supports PredictionCache integration for displaying cache hit/miss status (MarketChat story 3).
    /// </summary>
    public class MarketChatViewModel : ViewModelBase
    {
        private readonly MarketChatService _marketChatService;
        private readonly ILogger<MarketChatViewModel> _logger;

        private string _currentMessage;
        private bool _isProcessing;
        private string _statusMessage;
        private ObservableCollection<MarketChatMessage> _messages;
        private bool _isTradingPlanRequest;

        /// <summary>
        /// Constructor with dependencies
        /// </summary>
        public MarketChatViewModel(MarketChatService marketChatService, ILogger<MarketChatViewModel> logger)
        {
            _marketChatService = marketChatService ?? throw new ArgumentNullException(nameof(marketChatService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            // Initialize collections
            Messages = new ObservableCollection<MarketChatMessage>();

            // Initialize commands
            SendMessageCommand = new RelayCommand(_ => ExecuteSendMessage(), _ => CanSendMessage());
            ClearHistoryCommand = new RelayCommand(_ => ExecuteClearHistory());
            RequestTradingPlanCommand = new RelayCommand(_ => ExecuteRequestTradingPlan(), _ => !IsProcessing);
            WarmCacheCommand = new RelayCommand(async _ => await ExecuteWarmCacheAsync(), _ => !IsProcessing);

            // Set initial status
            StatusMessage = "Ready to analyze market questions";

            // Add welcome message
            AddWelcomeMessage();
        }

        #region Properties

        /// <summary>
        /// The current message being typed by the user
        /// </summary>
        public string CurrentMessage
        {
            get => _currentMessage;
            set
            {
                SetProperty(ref _currentMessage, value);
                // Update command availability when message changes
                ((RelayCommand)SendMessageCommand).RaiseCanExecuteChanged();
            }
        }

        /// <summary>
        /// Whether the system is currently processing a request
        /// </summary>
        public bool IsProcessing
        {
            get => _isProcessing;
            set
            {
                SetProperty(ref _isProcessing, value);
                ((RelayCommand)SendMessageCommand).RaiseCanExecuteChanged();
            }
        }

        /// <summary>
        /// Status message to display to the user
        /// </summary>
        public string StatusMessage
        {
            get => _statusMessage;
            set => SetProperty(ref _statusMessage, value);
        }

        /// <summary>
        /// Collection of chat messages
        /// </summary>
        public ObservableCollection<MarketChatMessage> Messages
        {
            get => _messages;
            set => SetProperty(ref _messages, value);
        }

        /// <summary>
        /// Indicates if the latest user message is a trading plan request
        /// </summary>
        public bool IsTradingPlanRequest
        {
            get => _isTradingPlanRequest;
            set => SetProperty(ref _isTradingPlanRequest, value);
        }

        #endregion

        #region Commands

        /// <summary>
        /// Command to send a message
        /// </summary>
        public ICommand SendMessageCommand { get; }

        /// <summary>
        /// Command to clear chat history
        /// </summary>
        public ICommand ClearHistoryCommand { get; }

        /// <summary>
        /// Command to explicitly request a trading plan
        /// </summary>
        public ICommand RequestTradingPlanCommand { get; }

        /// <summary>
        /// Command to warm the prediction cache for popular symbols (MarketChat story 3)
        /// </summary>
        public ICommand WarmCacheCommand { get; }

        #endregion

        #region Command Methods

        /// <summary>
        /// Determines if a message can be sent
        /// </summary>
        private bool CanSendMessage()
        {
            return !IsProcessing && !string.IsNullOrWhiteSpace(CurrentMessage);
        }

        /// <summary>
        /// Executes the send message command
        /// </summary>
        private async void ExecuteSendMessage()
        {
            if (!CanSendMessage()) return;

            var userMessage = CurrentMessage.Trim();
            CurrentMessage = string.Empty;

            try
            {
                // Add user message to chat
                var userChatMessage = new MarketChatMessage
                {
                    Content = userMessage,
                    IsFromUser = true,
                    Timestamp = DateTime.Now,
                    MessageType = MessageType.UserQuestion
                };
                Messages.Add(userChatMessage);

                // Show processing state
                IsProcessing = true;

                // Check if this is a trading plan request, database query, multi-symbol comparison, chart request, Python prediction request, or cache management request
                bool isTradingPlan = IsTradingPlanRequestMessage(userMessage);
                bool isQueryRequest = _marketChatService.IsQueryRequest(userMessage);
                bool isComparisonRequest = _marketChatService.IsMultiSymbolComparisonRequest(userMessage);
                bool isChartRequest = _marketChatService.IsChartRequest(userMessage);
                bool isPredictionRequest = _marketChatService.IsPredictionRequest(userMessage);
                bool isCacheManagementRequest = _marketChatService.IsCacheManagementRequest(userMessage);

                if (isCacheManagementRequest)
                {
                    StatusMessage = "Managing cache...";
                }
                else if (isPredictionRequest)
                {
                    StatusMessage = "Running ML prediction...";
                }
                else if (isQueryRequest)
                {
                    StatusMessage = "Querying database...";
                }
                else if (isComparisonRequest)
                {
                    StatusMessage = "Comparing symbols...";
                }
                else if (isChartRequest)
                {
                    StatusMessage = "Generating chart...";
                }
                else if (isTradingPlan)
                {
                    StatusMessage = "Creating trading plan...";
                }
                else
                {
                    StatusMessage = "Analyzing your question...";
                }

                // Add loading message (for prediction requests, we'll update it with progress)
                string loadingContent = isCacheManagementRequest ? "Processing cache request..." :
                                        (isPredictionRequest ? "Starting prediction..." :
                                        (isQueryRequest ? "Querying database..." : 
                                        (isComparisonRequest ? "Comparing symbols..." :
                                         (isChartRequest ? "Generating chart..." :
                                          (isTradingPlan ? "Creating trading plan..." : "Thinking...")))));
                var loadingMessage = new MarketChatMessage
                {
                    Content = loadingContent,
                    IsFromUser = false,
                    Timestamp = DateTime.Now,
                    MessageType = isPredictionRequest ? MessageType.PredictionProgress : MessageType.LoadingMessage,
                    IsLoading = true
                };
                Messages.Add(loadingMessage);

                // Get response based on message type
                string response;

                if (isPredictionRequest)
                {
                    // Process Python prediction request with progress updates (MarketChat story 9)
                    var truncatedMessage = userMessage.Length > 50 ? userMessage.Substring(0, 50) : userMessage;
                    _logger.LogInformation("Detected Python prediction request: {Message}", truncatedMessage);
                    
                    // Create progress callback that updates the loading message
                    PredictionProgressCallback progressCallback = (progressMessage) =>
                    {
                        // Update on UI thread
                        try
                        {
                            if (loadingMessage != null)
                            {
                                loadingMessage.Content = progressMessage;
                                loadingMessage.PredictionProgressMessage = progressMessage;
                                StatusMessage = progressMessage;
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Error updating prediction progress");
                        }
                    };

                    response = await _marketChatService.ProcessPredictionRequestAsync(userMessage, progressCallback);
                }
                else if (isTradingPlan)
                {
                    // Extract parameters from the message
                    string ticker = ExtractTickerSymbol(userMessage);
                    string timeframe = ExtractTimeframe(userMessage);
                    string riskProfile = ExtractRiskProfile(userMessage);
                    string marketContext = ExtractMarketContext(userMessage);

                    _logger.LogInformation($"Detected trading plan request for ticker: {ticker}, timeframe: {timeframe}, risk: {riskProfile}");

                    // Generate trading plan
                    response = await _marketChatService.GetTradingPlanAsync(ticker, timeframe, marketContext, riskProfile);
                }
                else
                {
                    // Standard market analysis (will also handle database queries via MarketChat story 5
                    // and multi-symbol comparisons via MarketChat story 7)
                    response = await _marketChatService.SendMarketAnalysisRequestAsync(userMessage, includeContext: true);
                }

                // Remove loading message
                Messages.Remove(loadingMessage);

                // Get cache metadata from the last prediction lookup (MarketChat story 3)
                var cacheResult = _marketChatService.LastPredictionCacheResult;

                // Get query result metadata (MarketChat story 5)
                var queryResult = _marketChatService.LastQueryResult;

                // Get comparison result metadata (MarketChat story 7)
                var comparisonResult = _marketChatService.LastComparisonResult;
                bool isComparisonResponse = comparisonResult != null && 
                                           comparisonResult.IsSuccessful && 
                                           _marketChatService.IsMultiSymbolComparisonRequest(userMessage);

                // Get chart data metadata (MarketChat story 8)
                var chartData = _marketChatService.LastChartData;
                bool isChartResponse = chartData != null && chartData.IsValid && isChartRequest;

                // Get Python prediction result metadata (MarketChat story 9)
                var pythonPredictionResult = _marketChatService.LastPythonPredictionResult;
                bool isPredictionResponse = pythonPredictionResult != null && isPredictionRequest;

                // Get cache management result metadata (MarketChat story 10)
                var cacheManagementResult = _marketChatService.LastCacheManagementResult;
                bool isCacheManagementResponse = cacheManagementResult != null && isCacheManagementRequest;

                // Determine message type
                MessageType messageType = MessageType.AssistantResponse;
                if (isCacheManagementResponse)
                {
                    messageType = MessageType.CacheManagementResult;
                }
                else if (isPredictionResponse)
                {
                    messageType = pythonPredictionResult.Success ? MessageType.AssistantResponse : MessageType.SystemMessage;
                }
                else if (queryResult?.Success == true)
                {
                    messageType = MessageType.QueryResult;
                }
                else if (isComparisonResponse)
                {
                    messageType = MessageType.ComparisonResult;
                }
                else if (isChartResponse)
                {
                    messageType = MessageType.ChartMessage;
                }

                // Add assistant response with cache/query/comparison/chart/prediction/cacheManagement metadata
                var assistantMessage = new MarketChatMessage
                {
                    Content = response,
                    IsFromUser = false,
                    Timestamp = DateTime.Now,
                    MessageType = messageType,
                    UsesCachedData = cacheResult?.IsCached ?? false,
                    CacheStatusDisplay = cacheResult?.CacheStatusDisplay,
                    CacheAge = cacheResult?.CacheAge,
                    IsQueryResult = queryResult?.Success == true,
                    QueryRowCount = queryResult?.RowCount ?? 0,
                    QueryExecutionTimeMs = queryResult?.ExecutionTimeMs ?? 0,
                    IsComparisonResult = isComparisonResponse,
                    ComparisonSymbolCount = comparisonResult?.Symbols?.Count ?? 0,
                    ChartData = isChartResponse ? chartData : null,
                    // Python prediction result metadata (MarketChat story 9)
                    IsPredictionResult = isPredictionResponse && pythonPredictionResult.Success,
                    PredictionSymbol = isPredictionResponse && pythonPredictionResult.Success ? pythonPredictionResult.Prediction?.Symbol : null,
                    PredictionModelType = isPredictionResponse ? pythonPredictionResult.ModelType : null,
                    PredictionExecutionTimeMs = isPredictionResponse ? pythonPredictionResult.ExecutionTimeMs : 0,
                    // Cache management result metadata (MarketChat story 10)
                    IsCacheManagementResult = isCacheManagementResponse,
                    CacheOperationType = isCacheManagementResponse ? cacheManagementResult.OperationType : null,
                    CacheEntriesAffected = isCacheManagementResponse ? cacheManagementResult.EntriesAffected : 0,
                    CacheRecommendation = isCacheManagementResponse ? cacheManagementResult.Recommendation : null
                };
                Messages.Add(assistantMessage);

                // Update status to show appropriate info
                if (isCacheManagementResponse)
                {
                    var statusText = cacheManagementResult.Success
                        ? $"Cache operation complete | {cacheManagementResult.OperationType}"
                        : "Cache operation requires confirmation";
                    if (cacheManagementResult.EntriesAffected > 0)
                    {
                        statusText += $" ({cacheManagementResult.EntriesAffected} entries)";
                    }
                    StatusMessage = statusText;
                    _logger.LogInformation("Cache management operation: {Operation}, Success: {Success}, Entries: {Entries}",
                        cacheManagementResult.OperationType, cacheManagementResult.Success, cacheManagementResult.EntriesAffected);
                }
                else if (isPredictionResponse && pythonPredictionResult.Success)
                {
                    StatusMessage = $"Prediction complete | {pythonPredictionResult.Prediction?.Symbol} - {pythonPredictionResult.Prediction?.Action} ({pythonPredictionResult.ExecutionTimeMs:F0}ms)";
                    _logger.LogInformation("Python prediction completed for {Symbol} with action {Action} in {Time:F0}ms",
                        pythonPredictionResult.Prediction?.Symbol, pythonPredictionResult.Prediction?.Action, pythonPredictionResult.ExecutionTimeMs);
                }
                else if (isPredictionResponse && !pythonPredictionResult.Success)
                {
                    StatusMessage = "Prediction failed - see response for details";
                    _logger.LogWarning("Python prediction failed: {Error}", pythonPredictionResult.ErrorMessage);
                }
                else if (queryResult?.Success == true)
                {
                    StatusMessage = $"Query complete | {queryResult.RowCount} rows in {queryResult.ExecutionTimeMs}ms";
                    _logger.LogInformation("Database query completed: {RowCount} rows in {TimeMs}ms", queryResult.RowCount, queryResult.ExecutionTimeMs);
                }
                else if (isComparisonResponse)
                {
                    StatusMessage = $"Comparison complete | {comparisonResult.Symbols.Count} symbols analyzed";
                    _logger.LogInformation("Multi-symbol comparison completed: {Count} symbols", comparisonResult.Symbols.Count);
                }
                else if (isChartResponse)
                {
                    StatusMessage = $"Chart ready | {chartData.Symbol} - {chartData.PredictedAction}";
                    _logger.LogInformation("Chart generated for {Symbol} with {Historical} historical points and {Prediction} prediction points",
                        chartData.Symbol, chartData.HistoricalPrices?.Count ?? 0, chartData.PredictionPrices?.Count ?? 0);
                }
                else if (cacheResult?.IsCached == true)
                {
                    StatusMessage = $"Ready | {cacheResult.CacheStatusDisplay}";
                    _logger.LogInformation("Response used cached prediction data ({Age} old)", cacheResult.CacheAge);
                }
                else
                {
                    StatusMessage = "Ready for your next question";
                }

                var logTruncatedMsg = userMessage.Length > 50 ? userMessage.Substring(0, 50) : userMessage;
                _logger.LogInformation($"Successfully processed {(isCacheManagementRequest ? "cache management" : isPredictionRequest ? "Python prediction" : isTradingPlan ? "trading plan" : isChartResponse ? "chart" : isComparisonResponse ? "comparison" : "market analysis")} question: {logTruncatedMsg}...");
            }
            catch (Exception ex)
            {
                // Remove loading message if it exists
                var loadingMessage = Messages.Count > 0 && Messages[Messages.Count - 1].IsLoading
                    ? Messages[Messages.Count - 1]
                    : null;
                if (loadingMessage != null)
                {
                    Messages.Remove(loadingMessage);
                }

                // Add error message
                var errorMessage = new MarketChatMessage
                {
                    Content = "I apologize, but I encountered an error while processing your request. Please try again.",
                    IsFromUser = false,
                    Timestamp = DateTime.Now,
                    MessageType = MessageType.SystemMessage
                };
                Messages.Add(errorMessage);

                StatusMessage = "Error occurred - please try again";
                _logger.LogError(ex, $"Error processing market analysis question: {userMessage}");
            }
            finally
            {
                IsProcessing = false;
            }
        }

        /// <summary>
        /// Executes the clear history command
        /// </summary>
        private void ExecuteClearHistory()
        {
            try
            {
                Messages.Clear();
                _marketChatService.ClearConversationHistory();
                AddWelcomeMessage();
                StatusMessage = "Chat history cleared";
                _logger.LogInformation("Chat history cleared by user");
            }
            catch (Exception ex)
            {
                StatusMessage = "Error clearing history";
                _logger.LogError(ex, "Error clearing chat history");
            }
        }

        /// <summary>
        /// Executes the request trading plan command
        /// </summary>
        private void ExecuteRequestTradingPlan()
        {
            // Set placeholder text to guide user for a trading plan request
            CurrentMessage = "Suggest a trading plan for TICKER for the next month";
        }

        /// <summary>
        /// Executes the warm cache command for popular symbols (MarketChat story 3)
        /// </summary>
        private async Task ExecuteWarmCacheAsync()
        {
            try
            {
                IsProcessing = true;
                StatusMessage = "Warming prediction cache for popular symbols...";
                _logger.LogInformation("Starting cache warming operation");

                var warmedCount = await _marketChatService.WarmPredictionCacheAsync();

                StatusMessage = $"Cache warmed for {warmedCount} symbols";
                _logger.LogInformation("Cache warming completed: {Count} symbols warmed", warmedCount);

                // Add system message to show cache warming complete
                var systemMessage = new MarketChatMessage
                {
                    Content = $"Prediction cache warmed for {warmedCount} popular symbols. Future queries for these symbols will have faster response times.",
                    IsFromUser = false,
                    Timestamp = DateTime.Now,
                    MessageType = MessageType.SystemMessage
                };
                Messages.Add(systemMessage);
            }
            catch (Exception ex)
            {
                StatusMessage = "Error warming cache";
                _logger.LogError(ex, "Error during cache warming operation");
            }
            finally
            {
                IsProcessing = false;
            }
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Adds a welcome message to start the conversation
        /// </summary>
        private void AddWelcomeMessage()
        {
            var welcomeMessage = new MarketChatMessage
            {
                Content = "Hello! I'm your AI market analysis assistant. Ask me anything about market conditions, stock analysis, trading strategies, or financial insights. For example:\n\n" +
                         "**Market Analysis:**\n" +
                         "• \"What's the current sentiment around AAPL?\"\n" +
                         "• \"Analyze the risks for tech stocks this quarter\"\n" +
                         "• \"Explain the recent volatility in the market\"\n\n" +
                         "**Trading Plans:**\n" +
                         "• \"Suggest a trading plan for MSFT for the next month\"\n" +
                         "• \"Give me a trading plan for TSLA in a volatile market\"\n\n" +
                         "**Run ML Predictions (MarketChat story 9):**\n" +
                         "• \"Run new prediction for TSLA\"\n" +
                         "• \"Run LSTM model for AAPL\"\n" +
                         "• \"Generate fresh prediction for NVDA using transformer\"\n" +
                         "• \"Run GRU prediction for MSFT\"\n" +
                         "• \"Execute random forest model for GOOGL\"\n\n" +
                         "**Cache Management (MarketChat story 10):**\n" +
                         "• \"Show cache statistics\"\n" +
                         "• \"Clear cache for AAPL\"\n" +
                         "• \"Show cache info for TSLA\"\n" +
                         "• \"Clear expired cache\"\n" +
                         "• \"Clear all cache\" (requires confirmation)\n\n" +
                         "**Multi-Symbol Comparison:**\n" +
                         "• \"Compare predictions for AAPL, MSFT, and GOOGL\"\n" +
                         "• \"Which is better: NVDA vs AMD vs INTC?\"\n" +
                         "• \"Side-by-side comparison of TSLA and F\"\n" +
                         "• \"Compare risk and return for tech giants\"\n\n" +
                         "**Sentiment Correlation Analysis:**\n" +
                         "• \"How does sentiment correlate with price for NVDA?\"\n" +
                         "• \"Show historical sentiment-price correlation for AAPL\"\n" +
                         "• \"Has Twitter sentiment predicted price movements for TSLA?\"\n\n" +
                         "**Database Queries:**\n" +
                         "• \"Show me all stocks with predictions above 80% confidence\"\n" +
                         "• \"List recent predictions for AAPL\"\n" +
                         "• \"Find all BUY predictions with high confidence\"\n" +
                         "• \"Count predictions by stock symbol\"",
                IsFromUser = false,
                Timestamp = DateTime.Now,
                MessageType = MessageType.SystemMessage
            };
            Messages.Add(welcomeMessage);
        }

        /// <summary>
        /// Determines if the message is requesting a trading plan
        /// </summary>
        private bool IsTradingPlanRequestMessage(string message)
        {
            string loweredMessage = message.ToLower().Trim();

            // Check for trading plan keywords
            bool containsTradingPlan = loweredMessage.Contains("trading plan") ||
                                       loweredMessage.Contains("trade plan") ||
                                       (loweredMessage.Contains("plan") &&
                                        (loweredMessage.Contains("trade") || loweredMessage.Contains("trading")));

            // Must contain a ticker symbol pattern (look for uppercase 1-5 letters that could be a ticker)
            bool containsTickerPattern = System.Text.RegularExpressions.Regex.IsMatch(message, @"\b[A-Z]{1,5}\b");

            return containsTradingPlan && containsTickerPattern;
        }

        /// <summary>
        /// Extracts ticker symbol from a message
        /// </summary>
        private string ExtractTickerSymbol(string message)
        {
            // Look for uppercase 1-5 letters that could be a ticker
            var match = System.Text.RegularExpressions.Regex.Match(message, @"\b[A-Z]{1,5}\b");
            return match.Success ? match.Value : string.Empty;
        }

        /// <summary>
        /// Extracts timeframe from a message
        /// </summary>
        private string ExtractTimeframe(string message)
        {
            string loweredMessage = message.ToLower();

            if (loweredMessage.Contains("next month") || loweredMessage.Contains("month"))
                return "next month";
            if (loweredMessage.Contains("next week") || loweredMessage.Contains("week"))
                return "next week";
            if (loweredMessage.Contains("next day") || loweredMessage.Contains("tomorrow"))
                return "next day";
            if (loweredMessage.Contains("long term") || loweredMessage.Contains("long-term"))
                return "long-term";
            if (loweredMessage.Contains("short term") || loweredMessage.Contains("short-term"))
                return "short-term";

            return "next month"; // Default timeframe
        }

        /// <summary>
        /// Extracts risk profile from a message
        /// </summary>
        private string ExtractRiskProfile(string message)
        {
            string loweredMessage = message.ToLower();

            if (loweredMessage.Contains("conservative") || loweredMessage.Contains("low risk"))
                return "conservative";
            if (loweredMessage.Contains("aggressive") || loweredMessage.Contains("high risk"))
                return "aggressive";

            return "moderate"; // Default risk profile
        }

        /// <summary>
        /// Extracts market context from a message
        /// </summary>
        private string ExtractMarketContext(string message)
        {
            string loweredMessage = message.ToLower();
            StringBuilder contextBuilder = new StringBuilder();

            if (loweredMessage.Contains("bull") || loweredMessage.Contains("bullish"))
                contextBuilder.Append("in a bullish market ");
            else if (loweredMessage.Contains("bear") || loweredMessage.Contains("bearish"))
                contextBuilder.Append("in a bearish market ");

            if (loweredMessage.Contains("volatile") || loweredMessage.Contains("volatility"))
                contextBuilder.Append("with high volatility ");
            else if (loweredMessage.Contains("stable"))
                contextBuilder.Append("with low volatility ");

            return contextBuilder.ToString().Trim();
        }

        #endregion
    }
}