using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Quantra.DAL.Models;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a message in the market chat interface
    /// </summary>
    public class MarketChatMessage : INotifyPropertyChanged
    {
        private string _content;
        private bool _isFromUser;
        private DateTime _timestamp;
        private bool _isLoading;
        private MessageType _messageType;
        private bool _usesCachedData;
        private string _cacheStatusDisplay;
        private TimeSpan? _cacheAge;
        private bool _isQueryResult;
        private int _queryRowCount;
        private long _queryExecutionTimeMs;
        private bool _isComparisonResult;
        private int _comparisonSymbolCount;
        private ProjectionChartData _chartData;

        /// <summary>
        /// The content of the message
        /// </summary>
        public string Content
        {
            get => _content;
            set => SetProperty(ref _content, value);
        }

        /// <summary>
        /// Whether this message is from the user (true) or from the AI assistant (false)
        /// </summary>
        public bool IsFromUser
        {
            get => _isFromUser;
            set => SetProperty(ref _isFromUser, value);
        }

        /// <summary>
        /// When the message was created
        /// </summary>
        public DateTime Timestamp
        {
            get => _timestamp;
            set => SetProperty(ref _timestamp, value);
        }

        /// <summary>
        /// Whether this message is currently being loaded/generated
        /// </summary>
        public bool IsLoading
        {
            get => _isLoading;
            set => SetProperty(ref _isLoading, value);
        }

        /// <summary>
        /// The type of message for styling purposes
        /// </summary>
        public MessageType MessageType
        {
            get => _messageType;
            set => SetProperty(ref _messageType, value);
        }

        /// <summary>
        /// Indicates whether this response uses cached prediction data (MarketChat story 3)
        /// </summary>
        public bool UsesCachedData
        {
            get => _usesCachedData;
            set => SetProperty(ref _usesCachedData, value);
        }

        /// <summary>
        /// Human-readable description of the cache status for display in chat (MarketChat story 3)
        /// Example: "Based on prediction from 23 minutes ago..."
        /// </summary>
        public string CacheStatusDisplay
        {
            get => _cacheStatusDisplay;
            set => SetProperty(ref _cacheStatusDisplay, value);
        }

        /// <summary>
        /// The age of the cached prediction data, if applicable (MarketChat story 3)
        /// </summary>
        public TimeSpan? CacheAge
        {
            get => _cacheAge;
            set => SetProperty(ref _cacheAge, value);
        }

        /// <summary>
        /// Indicates whether this response is a database query result (MarketChat story 5)
        /// </summary>
        public bool IsQueryResult
        {
            get => _isQueryResult;
            set => SetProperty(ref _isQueryResult, value);
        }

        /// <summary>
        /// Number of rows returned from a database query (MarketChat story 5)
        /// </summary>
        public int QueryRowCount
        {
            get => _queryRowCount;
            set => SetProperty(ref _queryRowCount, value);
        }

        /// <summary>
        /// Execution time of the database query in milliseconds (MarketChat story 5)
        /// </summary>
        public long QueryExecutionTimeMs
        {
            get => _queryExecutionTimeMs;
            set => SetProperty(ref _queryExecutionTimeMs, value);
        }

        /// <summary>
        /// Indicates whether this response is a multi-symbol comparison result (MarketChat story 7)
        /// </summary>
        public bool IsComparisonResult
        {
            get => _isComparisonResult;
            set => SetProperty(ref _isComparisonResult, value);
        }

        /// <summary>
        /// Number of symbols in the comparison (MarketChat story 7)
        /// </summary>
        public int ComparisonSymbolCount
        {
            get => _comparisonSymbolCount;
            set => SetProperty(ref _comparisonSymbolCount, value);
        }

        /// <summary>
        /// Formatted timestamp for display
        /// </summary>
        public string TimestampDisplay => Timestamp.ToString("HH:mm:ss");

        /// <summary>
        /// Indicates whether cache status should be displayed (non-empty and response is from assistant)
        /// </summary>
        public bool ShowCacheStatus => !IsFromUser && !IsLoading && !string.IsNullOrEmpty(CacheStatusDisplay);

        /// <summary>
        /// Indicates whether query result status should be displayed (MarketChat story 5)
        /// </summary>
        public bool ShowQueryStatus => !IsFromUser && !IsLoading && IsQueryResult;

        /// <summary>
        /// Indicates whether comparison result status should be displayed (MarketChat story 7)
        /// </summary>
        public bool ShowComparisonStatus => !IsFromUser && !IsLoading && IsComparisonResult;

        /// <summary>
        /// Gets a display string for query result status (MarketChat story 5)
        /// </summary>
        public string QueryStatusDisplay => IsQueryResult ? $"{QueryRowCount} rows in {QueryExecutionTimeMs}ms" : null;

        /// <summary>
        /// Gets a display string for comparison result status (MarketChat story 7)
        /// </summary>
        public string ComparisonStatusDisplay => IsComparisonResult ? $"Comparing {ComparisonSymbolCount} symbols" : null;

        /// <summary>
        /// Chart data for displaying historical + projection chart (MarketChat story 8)
        /// </summary>
        public ProjectionChartData ChartData
        {
            get => _chartData;
            set
            {
                if (SetProperty(ref _chartData, value))
                {
                    OnPropertyChanged(nameof(HasChartData));
                }
            }
        }

        /// <summary>
        /// Indicates whether this message has chart data to display (MarketChat story 8)
        /// </summary>
        public bool HasChartData => _chartData?.IsValid ?? false;

        /// <summary>
        /// Indicates whether chart should be displayed (MarketChat story 8)
        /// </summary>
        public bool ShowChart => !IsFromUser && !IsLoading && HasChartData;

        /// <summary>
        /// Gets a display string for chart status (MarketChat story 8)
        /// </summary>
        public string ChartStatusDisplay => HasChartData ? $"{ChartData?.Symbol} - {ChartData?.PredictedAction} ({ChartData?.Confidence:P0})" : null;

        /// <summary>
        /// Progress message for Python prediction execution (MarketChat story 9).
        /// Updated in real-time as the prediction progresses.
        /// </summary>
        private string _predictionProgressMessage;
        public string PredictionProgressMessage
        {
            get => _predictionProgressMessage;
            set
            {
                if (SetProperty(ref _predictionProgressMessage, value))
                {
                    OnPropertyChanged(nameof(ShowPredictionProgress));
                }
            }
        }

        /// <summary>
        /// Indicates whether this message is a Python prediction request result (MarketChat story 9).
        /// </summary>
        private bool _isPredictionResult;
        public bool IsPredictionResult
        {
            get => _isPredictionResult;
            set => SetProperty(ref _isPredictionResult, value);
        }

        /// <summary>
        /// The symbol for which the prediction was run (MarketChat story 9).
        /// </summary>
        private string _predictionSymbol;
        public string PredictionSymbol
        {
            get => _predictionSymbol;
            set => SetProperty(ref _predictionSymbol, value);
        }

        /// <summary>
        /// The model type used for the prediction (MarketChat story 9).
        /// </summary>
        private string _predictionModelType;
        public string PredictionModelType
        {
            get => _predictionModelType;
            set => SetProperty(ref _predictionModelType, value);
        }

        /// <summary>
        /// Execution time for the Python prediction in milliseconds (MarketChat story 9).
        /// </summary>
        private double _predictionExecutionTimeMs;
        public double PredictionExecutionTimeMs
        {
            get => _predictionExecutionTimeMs;
            set => SetProperty(ref _predictionExecutionTimeMs, value);
        }

        /// <summary>
        /// Indicates whether prediction progress should be displayed (MarketChat story 9).
        /// </summary>
        public bool ShowPredictionProgress => !IsFromUser && !string.IsNullOrEmpty(PredictionProgressMessage);

        /// <summary>
        /// Gets a display string for prediction result status (MarketChat story 9).
        /// </summary>
        public string PredictionStatusDisplay => IsPredictionResult ? $"{PredictionSymbol} - {PredictionModelType} ({PredictionExecutionTimeMs:F0}ms)" : null;

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string propertyName = null)
        {
            if (Equals(storage, value))
            {
                return false;
            }

            storage = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }

    /// <summary>
    /// Types of chat messages for styling and categorization
    /// </summary>
    public enum MessageType
    {
        /// <summary>
        /// Regular user question
        /// </summary>
        UserQuestion,

        /// <summary>
        /// AI assistant response
        /// </summary>
        AssistantResponse,

        /// <summary>
        /// System message (errors, status updates)
        /// </summary>
        SystemMessage,

        /// <summary>
        /// Loading/thinking indicator
        /// </summary>
        LoadingMessage,

        /// <summary>
        /// Database query result (MarketChat story 5)
        /// </summary>
        QueryResult,

        /// <summary>
        /// Multi-symbol comparison result (MarketChat story 7)
        /// </summary>
        ComparisonResult,

        /// <summary>
        /// Chart visualization message with historical + projection data (MarketChat story 8)
        /// </summary>
        ChartMessage,

        /// <summary>
        /// Python prediction progress message with streaming updates (MarketChat story 9)
        /// </summary>
        PredictionProgress
    }
}