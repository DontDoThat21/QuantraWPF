using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.Configuration;
using Quantra.Configuration.Models;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Helpers;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for handling ChatGPT/OpenAI API calls for market analysis conversations
    /// </summary>
    public class MarketChatService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<MarketChatService> _logger;
        private readonly List<MarketChatMessage> _conversationHistory;
        private const string OpenAiBaseUrl = "https://api.openai.com";
        private const string OpenAiModel = "gpt-3.5-turbo";
        private const double OpenAiTemperature = 0.3;
        private const int OpenAiTimeout = 60;

        /// <summary>
        /// Constructor for MarketChatService
        /// </summary>
        public MarketChatService(
            ILogger<MarketChatService> logger,
            IConfigurationManager configManager = null)
        {
            _logger = logger;
            _httpClient = new HttpClient();
            _conversationHistory = new List<MarketChatMessage>();
            
            try
            {
                // Set up the HTTP client with API key from settings
                var apiKey = ApiKeyHelper.GetOpenAiApiKey();
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
        /// Sends a market analysis question to ChatGPT and returns the response
        /// </summary>
        /// <param name="userQuestion">The user's market analysis question</param>
        /// <param name="includeContext">Whether to include recent market data as context</param>
        /// <returns>The ChatGPT response</returns>
        public async Task<string> SendMarketAnalysisRequestAsync(string userQuestion, bool includeContext = true)
        {
            try
            {
                _logger.LogInformation($"Processing market analysis request: {userQuestion}");

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
        /// Clears the conversation history
        /// </summary>
        public void ClearConversationHistory()
        {
            _conversationHistory.Clear();
            _logger.LogInformation("Conversation history cleared");
        }

        /// <summary>
        /// Builds the system prompt for market analysis
        /// </summary>
        private string BuildSystemPrompt()
        {
            return "You are a professional financial analyst assistant for Quantra, an advanced algorithmic trading platform. " +
                   "Provide clear, concise, and actionable market analysis to help traders make informed decisions. " +
                   "Focus on technical analysis, market sentiment, risk factors, and actionable insights. " +
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
                   "Structure your responses with clear sections for Entry Criteria, Exit Criteria, Position Sizing, " +
                   "Stop-Loss Strategy, Take-Profit Targets, and Rationale. " +
                   "Always consider risk management as a priority and explain your reasoning. " +
                   "Use professional yet accessible language appropriate for experienced traders.";
        }

        /// <summary>
        /// Builds an enhanced prompt with market context
        /// </summary>
        private async Task<string> BuildEnhancedPromptAsync(string userQuestion, bool includeContext)
        {
            var promptBuilder = new StringBuilder();
            
            if (includeContext)
            {
                // Add current market context (simplified for now)
                promptBuilder.AppendLine("Current Market Context:");
                promptBuilder.AppendLine($"- Timestamp: {DateTime.Now:yyyy-MM-dd HH:mm:ss} UTC");
                promptBuilder.AppendLine("- Market Session: " + GetCurrentMarketSession());
                promptBuilder.AppendLine();
            }

            promptBuilder.AppendLine("User Question:");
            promptBuilder.AppendLine(userQuestion);

            return promptBuilder.ToString();
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
    }
}