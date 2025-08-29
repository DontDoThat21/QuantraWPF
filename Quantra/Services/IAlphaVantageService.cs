using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

public interface IAlphaVantageService
{
    void LogApiUsage();
    void LogApiUsage(string operation, string details = null);
    int GetCurrentDbApiCallCount();

    /// <summary>
    /// Fetches all technical indicator data for the given stock symbol in a single API call if possible.
    /// </summary>
    /// <param name="symbol">The stock symbol.</param>
    /// <returns>A dictionary of indicator name to value.</returns>
    Task<Dictionary<string, double>> GetAllTechnicalIndicatorsAsync(string symbol);

    /// <summary>
    /// Fetches indicator data for the given stock symbol.
    /// </summary>
    /// <param name="symbol">The stock symbol.</param>
    /// <returns>A list of indicator data.</returns>
    Task<List<StockIndicator>> GetIndicatorsAsync(string symbol);
}
