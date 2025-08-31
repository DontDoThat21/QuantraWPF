using System;
using System.Collections.ObjectModel;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Extensions.Logging;
using Quantra.Commands;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Market Chat interface
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
                
                // Check if this is a trading plan request
                bool isTradingPlan = IsTradingPlanRequestMessage(userMessage);
                StatusMessage = isTradingPlan ? "Creating trading plan..." : "Analyzing your question...";

                // Add loading message
                var loadingMessage = new MarketChatMessage
                {
                    Content = isTradingPlan ? "Creating trading plan..." : "Thinking...",
                    IsFromUser = false,
                    Timestamp = DateTime.Now,
                    MessageType = MessageType.LoadingMessage,
                    IsLoading = true
                };
                Messages.Add(loadingMessage);

                // Get response based on message type
                string response;
                
                if (isTradingPlan)
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
                    // Standard market analysis
                    response = await _marketChatService.SendMarketAnalysisRequestAsync(userMessage, includeContext: true);
                }

                // Remove loading message
                Messages.Remove(loadingMessage);

                // Add assistant response
                var assistantMessage = new MarketChatMessage
                {
                    Content = response,
                    IsFromUser = false,
                    Timestamp = DateTime.Now,
                    MessageType = MessageType.AssistantResponse
                };
                Messages.Add(assistantMessage);

                StatusMessage = "Ready for your next question";
                _logger.LogInformation($"Successfully processed {(isTradingPlan ? "trading plan" : "market analysis")} question: {userMessage.Substring(0, Math.Min(50, userMessage.Length))}...");
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
                         "• \"What's the current sentiment around AAPL?\"\n" +
                         "• \"Analyze the risks for tech stocks this quarter\"\n" +
                         "• \"Explain the recent volatility in the market\"\n" +
                         "• \"Suggest a trading plan for MSFT for the next month\"\n" +
                         "• \"Give me a trading plan for TSLA in a volatile market\"",
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