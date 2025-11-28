using System;
using System.Collections.Generic;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Contains all data needed to render a projection chart in the chat UI.
    /// Designed to be LiveCharts-compatible for WPF integration (MarketChat story 8).
    /// </summary>
    public class ProjectionChartData
    {
        /// <summary>
        /// Stock symbol this chart represents
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Title for the chart
        /// </summary>
        public string ChartTitle { get; set; }

        /// <summary>
        /// Historical price values (close prices)
        /// </summary>
        public List<double> HistoricalPrices { get; set; } = new List<double>();

        /// <summary>
        /// Dates corresponding to historical prices
        /// </summary>
        public List<DateTime> HistoricalDates { get; set; } = new List<DateTime>();

        /// <summary>
        /// ML prediction price values (forecasted prices)
        /// </summary>
        public List<double> PredictionPrices { get; set; } = new List<double>();

        /// <summary>
        /// Dates corresponding to prediction prices
        /// </summary>
        public List<DateTime> PredictionDates { get; set; } = new List<DateTime>();

        /// <summary>
        /// Upper Bollinger Band values
        /// </summary>
        public List<double> BollingerUpper { get; set; } = new List<double>();

        /// <summary>
        /// Middle Bollinger Band values (20-day SMA)
        /// </summary>
        public List<double> BollingerMiddle { get; set; } = new List<double>();

        /// <summary>
        /// Lower Bollinger Band values
        /// </summary>
        public List<double> BollingerLower { get; set; } = new List<double>();

        /// <summary>
        /// Support level price(s) identified from historical data
        /// </summary>
        public List<double> SupportLevels { get; set; } = new List<double>();

        /// <summary>
        /// Resistance level price(s) identified from historical data
        /// </summary>
        public List<double> ResistanceLevels { get; set; } = new List<double>();

        /// <summary>
        /// Current price of the stock
        /// </summary>
        public double CurrentPrice { get; set; }

        /// <summary>
        /// Target price from ML prediction
        /// </summary>
        public double TargetPrice { get; set; }

        /// <summary>
        /// Prediction confidence level (0-1)
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Predicted action (BUY, SELL, HOLD)
        /// </summary>
        public string PredictedAction { get; set; }

        /// <summary>
        /// Indicates whether the chart data is valid and ready for display
        /// </summary>
        public bool IsValid => HistoricalPrices?.Count > 0;

        /// <summary>
        /// Error message if chart generation failed
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// Timestamp when the chart data was generated
        /// </summary>
        public DateTime GeneratedAt { get; set; } = DateTime.Now;

        /// <summary>
        /// Labels for X-axis (formatted dates)
        /// </summary>
        public List<string> XAxisLabels { get; set; } = new List<string>();

        /// <summary>
        /// Combined values for historical + prediction for continuous line display
        /// </summary>
        public List<double> CombinedPrices { get; set; } = new List<double>();

        /// <summary>
        /// Combined dates for historical + prediction
        /// </summary>
        public List<DateTime> CombinedDates { get; set; } = new List<DateTime>();

        /// <summary>
        /// Index where predictions start in the combined series (for styling the prediction portion differently)
        /// </summary>
        public int PredictionStartIndex { get; set; }
    }

    /// <summary>
    /// Parameters extracted from a chart request message (MarketChat story 8)
    /// </summary>
    public class ChartRequestParameters
    {
        /// <summary>
        /// Stock symbol(s) to chart
        /// </summary>
        public List<string> Symbols { get; set; } = new List<string>();

        /// <summary>
        /// Start date for historical data (null means default lookback)
        /// </summary>
        public DateTime? StartDate { get; set; }

        /// <summary>
        /// Number of days to forecast
        /// </summary>
        public int ForecastDays { get; set; } = 30;

        /// <summary>
        /// Whether to include Bollinger Bands
        /// </summary>
        public bool IncludeBollingerBands { get; set; } = true;

        /// <summary>
        /// Whether to include support/resistance levels
        /// </summary>
        public bool IncludeSupportResistance { get; set; } = true;

        /// <summary>
        /// Days of historical data to include (default 60)
        /// </summary>
        public int HistoricalDays { get; set; } = 60;
    }
}
