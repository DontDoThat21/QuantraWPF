using System;

namespace Quantra.Exceptions
{
    /// <summary>
    /// Base exception for candlestick chart operations
    /// </summary>
    public class CandlestickChartException : Exception
    {
        public string Symbol { get; }
        public string Interval { get; }

        public CandlestickChartException(string message) : base(message)
        {
        }

        public CandlestickChartException(string message, Exception innerException) 
            : base(message, innerException)
        {
        }

        public CandlestickChartException(string message, string symbol, string interval)
            : base(message)
        {
            Symbol = symbol;
            Interval = interval;
        }

        public CandlestickChartException(string message, string symbol, string interval, Exception innerException)
            : base(message, innerException)
        {
            Symbol = symbol;
            Interval = interval;
        }
    }

    /// <summary>
    /// Exception thrown when no data is available for the requested symbol/interval
    /// </summary>
    public class NoDataAvailableException : CandlestickChartException
    {
        public NoDataAvailableException(string symbol, string interval)
            : base($"No data available for {symbol} with interval {interval}", symbol, interval)
        {
        }
    }

    /// <summary>
    /// Exception thrown when API rate limit is exceeded
    /// </summary>
    public class ApiRateLimitExceededException : CandlestickChartException
    {
        public int CurrentApiCalls { get; }
        public int MaxApiCalls { get; }

        public ApiRateLimitExceededException(int currentCalls, int maxCalls)
            : base($"API rate limit exceeded: {currentCalls}/{maxCalls} calls")
        {
            CurrentApiCalls = currentCalls;
            MaxApiCalls = maxCalls;
        }
    }

    /// <summary>
    /// Exception thrown when data loading fails
    /// </summary>
    public class DataLoadException : CandlestickChartException
    {
        public DataLoadException(string message, string symbol, string interval, Exception innerException)
            : base(message, symbol, interval, innerException)
        {
        }
    }
}
