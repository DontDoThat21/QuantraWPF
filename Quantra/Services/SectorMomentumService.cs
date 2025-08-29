using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Quantra;
using Quantra.Models;

namespace Quantra.Services
{
    public class SectorMomentumService
    {
        private static readonly Dictionary<string, DateTime> _lastUpdateTimes = new Dictionary<string, DateTime>();
        private static readonly Dictionary<string, Dictionary<string, List<SectorMomentumModel>>> _cachedData = 
            new Dictionary<string, Dictionary<string, List<SectorMomentumModel>>>();
        
        private static readonly HttpClient _httpClient = new HttpClient();
        private static readonly TimeSpan _cacheExpiration = TimeSpan.FromMinutes(30);
        private readonly AlphaVantageService _alphaVantageService;
        
        // Comprehensive sector-to-symbol mapping for real market data
        private static readonly Dictionary<string, string[]> _sectorSymbols = new Dictionary<string, string[]>
        {
            ["Technology"] = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "CRM" },
            ["Financial"] = new[] { "JPM", "BAC", "GS", "WFC", "MS", "C", "AXP" },
            ["Energy"] = new[] { "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "HAL" },
            ["Healthcare"] = new[] { "JNJ", "PFE", "UNH", "ABT", "MRK", "TMO", "MDT" },
            ["Industrial"] = new[] { "BA", "CAT", "GE", "HON", "LMT", "MMM", "RTX" },
            ["Materials"] = new[] { "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "DOW" },
            ["Consumer Discretionary"] = new[] { "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX" },
            ["Consumer Staples"] = new[] { "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB" },
            ["Utilities"] = new[] { "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "D" },
            ["Real Estate"] = new[] { "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "SPG" }
        };
        
        public SectorMomentumService()
        {
            // Initialize the HTTP client if needed
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "Quantra/1.0");
            _httpClient.Timeout = TimeSpan.FromSeconds(30);
            _alphaVantageService = new AlphaVantageService();
        }
        
        public Dictionary<string, List<SectorMomentumModel>> GetSectorMomentumData(string timeframe, bool forceRefresh = false)
        {
            try
            {
                // Check if we have cached data and it's not expired
                if (!forceRefresh && 
                    _cachedData.ContainsKey(timeframe) && 
                    _lastUpdateTimes.ContainsKey(timeframe) && 
                    DateTime.Now - _lastUpdateTimes[timeframe] < _cacheExpiration)
                {
                    return _cachedData[timeframe];
                }
                
                // If we reach here, we need to fetch or generate new data
                Dictionary<string, List<SectorMomentumModel>> result;
                
                // Fetch real data only
                result = FetchRealSectorMomentumData(timeframe);
                DatabaseMonolith.Log("Info", $"Successfully fetched real sector momentum data for timeframe {timeframe}");
                
                // Cache the new data
                _cachedData[timeframe] = result;
                _lastUpdateTimes[timeframe] = DateTime.Now;
                
                return result;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting sector momentum data for timeframe {timeframe}", ex.ToString());
                
                // Return cached data if available
                if (_cachedData.ContainsKey(timeframe))
                {
                    return _cachedData[timeframe];
                }
                
                // If no cached data, return empty result
                return new Dictionary<string, List<SectorMomentumModel>>();
            }
        }
        
        public async Task<Dictionary<string, List<SectorMomentumModel>>> GetSectorMomentumDataAsync(string timeframe, bool forceRefresh = false)
        {
            try
            {
                // Check if we have cached data and it's not expired
                if (!forceRefresh && 
                    _cachedData.ContainsKey(timeframe) && 
                    _lastUpdateTimes.ContainsKey(timeframe) && 
                    DateTime.Now - _lastUpdateTimes[timeframe] < _cacheExpiration)
                {
                    return _cachedData[timeframe];
                }
                
                // In a real implementation, this would fetch data from an API using await
                // For this example, just return the synchronous method result
                return await Task.Run(() => GetSectorMomentumData(timeframe, forceRefresh));
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting sector momentum data async for timeframe {timeframe}", ex.ToString());
                throw;
            }
        }
        
        
        private Dictionary<string, List<SectorMomentumModel>> FetchRealSectorMomentumData(string timeframe)
        {
            var result = new Dictionary<string, List<SectorMomentumModel>>();
            
            foreach (var sector in _sectorSymbols)
            {
                var sectorName = sector.Key;
                var symbols = sector.Value;
                var sectorStocks = new List<SectorMomentumModel>();
                
                foreach (var symbol in symbols)
                {
                    try
                    {
                        // Get real momentum data from Alpha Vantage
                        var momentum = CalculateStockMomentum(symbol, timeframe);
                        var volume = GetStockVolume(symbol);
                        
                        sectorStocks.Add(new SectorMomentumModel
                        {
                            Name = GetCompanyName(symbol),
                            Symbol = symbol,
                            MomentumValue = momentum,
                            Volume = volume,
                            Timestamp = DateTime.Now
                        });
                    }
                    catch (Exception ex)
                    {
                        DatabaseMonolith.Log("Warning", $"Failed to fetch data for symbol {symbol}", ex.ToString());
                        // Continue with other symbols even if one fails
                    }
                }
                
                if (sectorStocks.Any())
                {
                    // Sort by momentum value (descending)
                    result[sectorName] = sectorStocks.OrderByDescending(s => s.MomentumValue).ToList();
                }
            }
            
            // If no real data was fetched, throw exception to trigger fallback
            if (!result.Any())
            {
                throw new InvalidOperationException("No real sector data could be fetched from Alpha Vantage");
            }
            
            return result;
        }
        
        private double CalculateStockMomentum(string symbol, string timeframe)
        {
            try
            {
                // For momentum calculation, we need historical price data
                // We'll use the Alpha Vantage TIME_SERIES_DAILY function to get price data
                // and calculate momentum as the percentage change over the specified timeframe
                
                var currentPrice = GetCurrentPrice(symbol);
                var historicalPrice = GetHistoricalPrice(symbol, timeframe);
                
                if (currentPrice > 0 && historicalPrice > 0)
                {
                    var momentum = (currentPrice - historicalPrice) / historicalPrice;
                    // Clamp to reasonable bounds
                    return Math.Max(-0.75, Math.Min(0.75, momentum));
                }
                
                return 0.0; // Neutral if we can't calculate
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Failed to calculate momentum for {symbol}", ex.ToString());
                return 0.0; // Return neutral momentum if calculation fails
            }
        }
        
        private double GetCurrentPrice(string symbol)
        {
            try
            {
                // Use the existing AlphaVantage service to get current price
                var quote = _alphaVantageService.GetQuoteData(symbol).Result;
                return quote;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Failed to get current price for {symbol}", ex.ToString());
                return 0.0;
            }
        }
        
        private double GetHistoricalPrice(string symbol, string timeframe)
        {
            try
            {
                // Calculate how many days back to look based on timeframe
                int daysBack = timeframe switch
                {
                    "1d" => 1,
                    "1w" => 7,
                    "1m" => 30,
                    "3m" => 90,
                    "ytd" => (DateTime.Now.DayOfYear - 1), // Days since start of year
                    _ => 30 // Default to 30 days (1 month)
                };
                
                // TODO: Implement real historical price fetching from Alpha Vantage TIME_SERIES_DAILY
                // For now, return 0 to indicate we need real historical data implementation
                DatabaseMonolith.Log("Warning", $"Historical price fetching not yet implemented for {symbol}, timeframe {timeframe}");
                return 0.0;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Failed to get historical price for {symbol}", ex.ToString());
                return 0.0;
            }
        }
        
        private long GetStockVolume(string symbol)
        {
            try
            {
                // TODO: Implement real volume fetching from Alpha Vantage quote data
                // For now, return 0 to indicate we need real volume data implementation
                DatabaseMonolith.Log("Warning", $"Volume fetching not yet implemented for {symbol}");
                return 0;
            }
            catch
            {
                return 0; // Default volume
            }
        }
        
        private string GetCompanyName(string symbol)
        {
            // Simple mapping of symbols to company names
            var companyNames = new Dictionary<string, string>
            {
                ["AAPL"] = "Apple Inc.",
                ["MSFT"] = "Microsoft Corp.",
                ["GOOGL"] = "Alphabet Inc.",
                ["AMZN"] = "Amazon.com Inc.",
                ["META"] = "Meta Platforms Inc.",
                ["NVDA"] = "NVIDIA Corp.",
                ["CRM"] = "Salesforce Inc.",
                ["JPM"] = "JPMorgan Chase",
                ["BAC"] = "Bank of America",
                ["GS"] = "Goldman Sachs",
                ["WFC"] = "Wells Fargo",
                ["MS"] = "Morgan Stanley",
                ["C"] = "Citigroup Inc.",
                ["AXP"] = "American Express",
                ["XOM"] = "Exxon Mobil",
                ["CVX"] = "Chevron Corp.",
                ["COP"] = "ConocoPhillips",
                ["EOG"] = "EOG Resources",
                ["SLB"] = "Schlumberger",
                ["OXY"] = "Occidental Petroleum",
                ["HAL"] = "Halliburton",
                // Add more as needed
            };
            
            return companyNames.TryGetValue(symbol, out var name) ? name : symbol;
        }

    }
}
