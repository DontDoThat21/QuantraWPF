using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

// NOTE: This interface is deprecated and should be consolidated with
// Quantra.DAL.Services.Interfaces.IAlphaVantageService
// Kept for backward compatibility with existing code
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
    
    /// <summary>
    /// Adds an order to the order history database.
    /// </summary>
    /// <param name="order">The order to add to history.</param>
    void AddOrderToHistory(OrderModel order);
}
