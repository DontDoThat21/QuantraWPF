namespace Quantra.Services
{
    public interface IHistoricalDataService
    {
        /// <summary>
        /// Gets historical price data for a symbol.
        /// </summary>
        /// <param name="symbol">The stock symbol.</param>
        /// <param name="range">The range (e.g., "1mo").</param>
        /// <param name="interval">The interval (e.g., "1d").</param>
        /// <returns>List of HistoricalPrice objects.</returns>
        Task<List<Quantra.Models.HistoricalPrice>> GetHistoricalPrices(string symbol, string range, string interval);
    }
}