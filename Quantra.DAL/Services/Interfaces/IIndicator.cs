using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for all technical indicators in the Quantra platform
    /// </summary>
    public interface IIndicator
    {
        /// <summary>
        /// Unique identifier for the indicator
        /// </summary>
        string Id { get; }

        /// <summary>
        /// Display name for the indicator
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// Description of what the indicator measures and how it should be interpreted
        /// </summary>
        string Description { get; set; }

        /// <summary>
        /// Category of the indicator (e.g., Momentum, Volume, Trend, Volatility)
        /// </summary>
        string Category { get; set; }

        /// <summary>
        /// List of configurable parameters for this indicator
        /// </summary>
        Dictionary<string, IndicatorParameter> Parameters { get; }

        /// <summary>
        /// Calculate the indicator value based on historical price data
        /// </summary>
        /// <param name="historicalData">Historical price data for calculation</param>
        /// <returns>Dictionary of output values (allows multi-value indicators like MACD)</returns>
        Task<Dictionary<string, double>> CalculateAsync(List<HistoricalPrice> historicalData);

        /// <summary>
        /// Whether this indicator can be used as input for other indicators
        /// </summary>
        bool IsComposable { get; }

        /// <summary>
        /// Get the input dependencies required by this indicator
        /// </summary>
        /// <returns>List of indicator IDs this indicator depends on</returns>
        IEnumerable<string> GetDependencies();
    }
}