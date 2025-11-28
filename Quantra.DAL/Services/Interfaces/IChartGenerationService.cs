using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Models;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for chart generation service that creates LiveCharts-compatible data
    /// for historical prices and ML projections (MarketChat story 8).
    /// </summary>
    public interface IChartGenerationService
    {
        /// <summary>
        /// Generates projection chart data combining historical prices and ML predictions.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="startDate">Start date for historical data</param>
        /// <param name="forecastDays">Number of days to forecast</param>
        /// <returns>Chart data result containing LiveCharts-compatible series data</returns>
        Task<ProjectionChartData> GenerateProjectionChartAsync(string symbol, DateTime startDate, int forecastDays);

        /// <summary>
        /// Generates chart data from existing historical data and prediction result.
        /// </summary>
        /// <param name="historicalData">Historical price data</param>
        /// <param name="prediction">Prediction result with time series data</param>
        /// <param name="symbol">Stock symbol for display purposes</param>
        /// <returns>Chart data result</returns>
        ProjectionChartData GenerateChartFromData(List<HistoricalPrice> historicalData, PredictionResult prediction, string symbol);

        /// <summary>
        /// Determines if a user message is requesting a chart or visualization.
        /// </summary>
        /// <param name="message">User's message</param>
        /// <returns>True if the message is a chart request</returns>
        bool IsChartRequest(string message);

        /// <summary>
        /// Extracts chart parameters from a user's message.
        /// </summary>
        /// <param name="message">User's message</param>
        /// <returns>Extracted chart request parameters</returns>
        ChartRequestParameters ExtractChartParameters(string message);
    }
}
