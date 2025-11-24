using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to analyze sentiment trends within specific market sectors.
    /// Aggregates sentiment data from individual stocks within a sector and provides
    /// comparative analysis across different sectors.
    /// </summary>
    public class SectorSentimentAnalysisService : IDisposable
    {
        private readonly FinancialNewsSentimentService _newsSentimentService;
        private readonly ISocialMediaSentimentService _twitterSentimentService;
        private readonly UserSettings _userSettings;
        private readonly Dictionary<string, DateTime> _sectorCacheTimestamps = new Dictionary<string, DateTime>();
        private readonly Dictionary<string, Dictionary<string, double>> _sectorSentimentCache = new Dictionary<string, Dictionary<string, double>>();
        private readonly TimeSpan _cacheExpiration = TimeSpan.FromMinutes(30);
        private readonly ConcurrentTaskThrottler _taskThrottler;
        private bool _disposed = false;

        // Stock symbol to sector mapping cache
        private readonly Dictionary<string, string> _symbolSectorCache = new Dictionary<string, string>();

        // Default sectors for classification
        private static readonly HashSet<string> StandardSectors = new HashSet<string>
        {
            "Technology",
            "Financial",
            "Energy",
            "Healthcare",
            "Industrial",
            "Materials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Utilities",
            "Real Estate",
            "Communication",
            "Other"
        };

        // Sector-specific keywords for relevance detection
        private static readonly Dictionary<string, List<string>> SectorKeywords = new Dictionary<string, List<string>>
        {
            ["Technology"] = new List<string> {
                "tech", "software", "hardware", "digital", "cloud", "artificial intelligence", "ai", "machine learning",
                "semiconductor", "cybersecurity", "internet", "computing", "data", "automation", "blockchain"
            },
            ["Financial"] = new List<string> {
                "bank", "finance", "insurance", "investment", "credit", "lending", "mortgage", "fintech",
                "wealth management", "capital", "trading", "stock market", "interest rates"
            },
            ["Energy"] = new List<string> {
                "oil", "gas", "electricity", "renewable", "solar", "wind", "power", "petroleum", "drilling",
                "energy", "coal", "nuclear", "fracking", "utilities"
            },
            ["Healthcare"] = new List<string> {
                "health", "medical", "pharmaceutical", "biotech", "drug", "hospital", "therapy", "diagnostic",
                "clinical", "patient", "medicine", "healthcare", "vaccine", "treatment"
            },
            ["Industrial"] = new List<string> {
                "manufacturing", "industrial", "machinery", "aerospace", "defense", "logistics", "transportation",
                "construction", "engineering", "automotive", "aviation", "factory", "supply chain"
            },
            ["Materials"] = new List<string> {
                "materials", "chemical", "mining", "metal", "steel", "commodity", "minerals", "lumber", "paper",
                "packaging", "raw materials", "construction materials", "aluminum"
            },
            ["Consumer Discretionary"] = new List<string> {
                "retail", "consumer", "luxury", "apparel", "entertainment", "travel", "restaurant", "automotive",
                "hospitality", "leisure", "e-commerce", "discretionary", "shopping"
            },
            ["Consumer Staples"] = new List<string> {
                "food", "beverage", "tobacco", "household", "personal care", "staples", "grocery",
                "supermarket", "consumer goods", "toiletries", "non-discretionary"
            },
            ["Utilities"] = new List<string> {
                "utility", "water", "electric", "gas", "power", "waste management", "sewage", "electricity distribution"
            },
            ["Real Estate"] = new List<string> {
                "real estate", "property", "reit", "commercial property", "residential", "mortgage", "leasing",
                "development", "construction", "housing", "office", "rental"
            },
            ["Communication"] = new List<string> {
                "telecom", "media", "communication", "wireless", "broadband", "television", "publishing",
                "advertising", "entertainment", "social media", "streaming", "network"
            }
        };

        public SectorSentimentAnalysisService(UserSettings userSettings = null)
        {
            _userSettings = userSettings ?? new UserSettings();
            _newsSentimentService = new FinancialNewsSentimentService(userSettings);
            _twitterSentimentService = new TwitterSentimentService();
            _taskThrottler = new ConcurrentTaskThrottler(4); // Throttle sentiment API calls
        }

        /// <summary>
        /// Gets the overall sentiment for a specific market sector by analyzing
        /// individual stocks within that sector.
        /// </summary>
        /// <param name="sector">The market sector to analyze</param>
        /// <param name="forceRefresh">Whether to force a refresh of cached data</param>
        /// <returns>Sentiment score from -1.0 (negative) to 1.0 (positive)</returns>
        public async Task<double> GetSectorSentimentAsync(string sector, bool forceRefresh = false)
        {
            var sectorSentiment = await GetDetailedSectorSentimentAsync(sector, forceRefresh);

            // Calculate weighted average sentiment across all stocks in the sector
            if (sectorSentiment.Count == 0)
                return 0.0;

            return sectorSentiment.Values.Average();
        }

        /// <summary>
        /// Gets detailed sentiment for all stocks within a sector
        /// </summary>
        /// <param name="sector">The market sector to analyze</param>
        /// <param name="forceRefresh">Whether to force a refresh of cached data</param>
        /// <returns>Dictionary mapping stock symbols to sentiment scores</returns>
        public async Task<Dictionary<string, double>> GetDetailedSectorSentimentAsync(string sector, bool forceRefresh = false)
        {
            // Check cache
            string cacheKey = $"sector_{sector}";
            if (!forceRefresh &&
                _sectorSentimentCache.ContainsKey(cacheKey) &&
                _sectorCacheTimestamps.ContainsKey(cacheKey) &&
                DateTime.Now - _sectorCacheTimestamps[cacheKey] < _cacheExpiration)
            {
                return _sectorSentimentCache[cacheKey];
            }

            // Get all symbols for this sector
            List<string> sectorSymbols = await GetSymbolsForSectorAsync(sector);
            var sectorSentiment = new Dictionary<string, double>();

            // Process each stock with throttling to prevent API overload
            var taskFactories = sectorSymbols.Select(symbol =>
                new Func<Task<(string Symbol, double Sentiment)>>(async () =>
                {
                    try
                    {
                        double sentiment = await _newsSentimentService.GetSymbolSentimentAsync(symbol);
                        return (Symbol: symbol, Sentiment: sentiment);
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Warning", $"Error getting sentiment for {symbol}", ex.ToString());
                        return (Symbol: symbol, Sentiment: 0.0);
                    }
                }));

            var results = await _taskThrottler.ExecuteThrottledAsync(taskFactories);
            foreach (var result in results)
            {
                if (!sectorSentiment.ContainsKey(result.Symbol))
                {
                    sectorSentiment[result.Symbol] = result.Sentiment;
                }
            }

            // Cache the results
            _sectorSentimentCache[cacheKey] = sectorSentiment;
            _sectorCacheTimestamps[cacheKey] = DateTime.Now;

            return sectorSentiment;
        }

        /// <summary>
        /// Gets sentiment data for all sectors for comparative analysis
        /// </summary>
        /// <param name="forceRefresh">Whether to force a refresh of cached data</param>
        /// <returns>Dictionary mapping sectors to sentiment scores</returns>
        public async Task<Dictionary<string, double>> GetAllSectorsSentimentAsync(bool forceRefresh = false)
        {
            var result = new Dictionary<string, double>();

            foreach (var sector in StandardSectors)
            {
                try
                {
                    double sectorSentiment = await GetSectorSentimentAsync(sector, forceRefresh);
                    result[sector] = sectorSentiment;
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", $"Error getting sentiment for sector {sector}", ex.ToString());
                    result[sector] = 0.0;
                }
            }

            return result;
        }

        /// <summary>
        /// Gets detailed news sentiment data for a specific sector, including source-specific sentiment
        /// </summary>
        /// <param name="sector">The market sector to analyze</param>
        /// <returns>Tuple containing sentiment by source and relevant news articles</returns>
        public async Task<(Dictionary<string, double> SectorSentimentBySource, List<NewsArticle> SectorArticles)>
            GetSectorNewsAnalysisAsync(string sector)
        {
            var sectorArticles = new List<NewsArticle>();
            var sentimentBySource = new Dictionary<string, double>();
            var symbolsBySource = new Dictionary<string, HashSet<string>>();

            // Get all symbols for this sector
            List<string> sectorSymbols = await GetSymbolsForSectorAsync(sector);

            // Limit number of symbols to process for performance
            int maxSymbols = Math.Min(sectorSymbols.Count, 5);
            var symbolsToProcess = sectorSymbols.Take(maxSymbols).ToList();

            // Get articles for each symbol in the sector
            foreach (var symbol in symbolsToProcess)
            {
                try
                {
                    var (symbolSentiment, symbolArticles) =
                        await _newsSentimentService.GetDetailedNewsAnalysisAsync(symbol);

                    // Filter articles for sector relevance
                    foreach (var article in symbolArticles)
                    {
                        // Calculate sector relevance
                        double sectorRelevance = CalculateSectorRelevance(article.GetCombinedContent(), sector);

                        // Only include articles with sufficient sector relevance
                        if (sectorRelevance >= 0.3)
                        {
                            article.SectorRelevance = sectorRelevance;

                            // Avoid duplicates
                            if (!sectorArticles.Any(a => a.IsSimilarTo(article)))
                            {
                                sectorArticles.Add(article);

                                // Track which sources have which symbols
                                string source = article.SourceDomain;
                                if (!symbolsBySource.ContainsKey(source))
                                {
                                    symbolsBySource[source] = new HashSet<string>();
                                }
                                symbolsBySource[source].Add(symbol);
                            }
                        }
                    }

                    // Merge sentiment data by source
                    foreach (var kvp in symbolSentiment)
                    {
                        string source = kvp.Key;
                        double score = kvp.Value;

                        if (sentimentBySource.ContainsKey(source))
                        {
                            sentimentBySource[source] = (sentimentBySource[source] + score) / 2.0;
                        }
                        else
                        {
                            sentimentBySource[source] = score;
                        }
                    }
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", $"Error getting news analysis for {symbol} in sector {sector}", ex.ToString());
                }
            }

            // Sort articles by sector relevance and sentiment
            sectorArticles = sectorArticles
                .OrderByDescending(a => a.SectorRelevance * 0.7 + Math.Abs(a.SentimentScore) * 0.3)
                .ToList();

            return (sentimentBySource, sectorArticles);
        }

        /// <summary>
        /// Gets sector-specific sentiment trends over a specified time period
        /// </summary>
        /// <param name="sector">The market sector to analyze</param>
        /// <param name="days">Number of days to include in the trend analysis</param>
        /// <returns>Dictionary with dates as keys and sentiment scores as values</returns>
        public async Task<List<(DateTime Date, double Sentiment)>> GetSectorSentimentTrendAsync(
            string sector,
            int days = 30)
        {
            var result = new List<(DateTime, double)>();
            var startDate = DateTime.Now.AddDays(-days);

            // In a real implementation, this would query a database of historical sentiment data
            // For this example, we'll generate synthetic data based on the current sentiment

            // Get current sentiment as a baseline
            double currentSentiment = await GetSectorSentimentAsync(sector);

            // Generate synthetic history using a random walk with a bias toward the current sentiment
            var random = new Random(sector.GetHashCode()); // Use sector name as seed for consistent results

            for (int i = 0; i < days; i++)
            {
                DateTime date = startDate.AddDays(i);

                // Calculate a randomized sentiment that tends toward the current value
                // The closer to present day, the closer to current sentiment
                double dayFactor = (double)i / days; // 0.0 to 1.0
                double randomFactor = random.NextDouble() * 0.6 - 0.3; // -0.3 to +0.3

                // Blend between random initial value and current sentiment based on how close we are to present
                double historicalSentiment = (1 - dayFactor) * randomFactor + dayFactor * currentSentiment;

                // Apply small random variations
                historicalSentiment += random.NextDouble() * 0.1 - 0.05;

                // Ensure within bounds
                historicalSentiment = Math.Max(-1.0, Math.Min(1.0, historicalSentiment));

                result.Add((date, historicalSentiment));
            }

            return result;
        }

        /// <summary>
        /// Compares sentiment trends across multiple sectors over a specified time period
        /// </summary>
        /// <param name="sectors">List of sectors to compare</param>
        /// <param name="days">Number of days to include</param>
        /// <returns>Dictionary mapping sectors to their sentiment trends</returns>
        public async Task<Dictionary<string, List<(DateTime Date, double Sentiment)>>> CompareSectorSentimentTrendsAsync(
            List<string> sectors,
            int days = 30)
        {
            var result = new Dictionary<string, List<(DateTime, double)>>();

            foreach (var sector in sectors)
            {
                try
                {
                    var trend = await GetSectorSentimentTrendAsync(sector, days);
                    result[sector] = trend;
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", $"Error getting sentiment trend for sector {sector}", ex.ToString());
                }
            }

            return result;
        }

        #region Helper Methods

        /// <summary>
        /// Gets a list of stock symbols that belong to a specific sector
        /// </summary>
        private async Task<List<string>> GetSymbolsForSectorAsync(string sector)
        {
            // In a real implementation, this would query a database of stocks by sector
            // For this example, we'll use some predefined mappings

            // Common symbols for major sectors (simplified mapping for the example)
            var sectorSymbols = new Dictionary<string, List<string>>
            {
                ["Technology"] = new List<string> { "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "ADBE", "CRM" },
                ["Financial"] = new List<string> { "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK" },
                ["Healthcare"] = new List<string> { "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "ABT", "TMO", "BMY", "AMGN" },
                ["Consumer Discretionary"] = new List<string> { "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "TJX", "MAR" },
                ["Communication"] = new List<string> { "NFLX", "CMCSA", "VZ", "T", "TMUS", "DIS", "CHTR", "ATVI", "EA", "TTWO" },
                ["Industrial"] = new List<string> { "HON", "UNP", "UPS", "BA", "CAT", "DE", "LMT", "GE", "MMM", "RTX" },
                ["Energy"] = new List<string> { "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "KMI" },
                ["Materials"] = new List<string> { "LIN", "APD", "ECL", "NEM", "FCX", "DD", "DOW", "NUE", "VMC", "BLL" },
                ["Consumer Staples"] = new List<string> { "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "EL", "CL", "GIS" },
                ["Utilities"] = new List<string> { "NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "ED", "WEC", "ES" },
                ["Real Estate"] = new List<string> { "AMT", "PLD", "CCI", "PSA", "EQIX", "DLR", "O", "SBAC", "WELL", "AVB" }
            };

            // For "Other" sector or if sector not found
            if (!sectorSymbols.ContainsKey(sector))
            {
                // Return a small set of miscellaneous stocks
                return new List<string> { "PG", "KO", "WMT", "JNJ", "XOM" };
            }

            return sectorSymbols[sector];
        }

        /// <summary>
        /// Calculates how relevant content is to a specific sector
        /// </summary>
        /// <param name="content">The text content to analyze</param>
        /// <param name="sector">The sector to check relevance for</param>
        /// <returns>Relevance score between 0.0 and 1.0</returns>
        private double CalculateSectorRelevance(string content, string sector)
        {
            if (string.IsNullOrEmpty(content) || string.IsNullOrEmpty(sector))
                return 0.0;

            // Convert to lowercase for case-insensitive matching
            string normalizedContent = content.ToLowerInvariant();

            double score = 0.0;

            // Check for sector name
            if (normalizedContent.Contains(sector.ToLowerInvariant()))
                score += 0.4; // Direct mention of the sector

            // Check for sector-specific keywords
            if (SectorKeywords.TryGetValue(sector, out List<string> keywords))
            {
                foreach (var keyword in keywords)
                {
                    if (normalizedContent.Contains(keyword.ToLowerInvariant()))
                    {
                        score += 0.1;
                        // Cap at 0.6 from keywords
                        if (score >= 0.6)
                            break;
                    }
                }
            }

            // Check for symbols from this sector
            var sectorSymbols = GetSymbolsForSectorAsync(sector).GetAwaiter().GetResult();
            foreach (var symbol in sectorSymbols)
            {
                if (normalizedContent.Contains(symbol.ToLowerInvariant()))
                {
                    score += 0.2;
                    break;
                }
            }

            // Cap at 1.0
            return Math.Min(1.0, score);
        }

        /// <summary>
        /// Clears the sector sentiment cache
        /// </summary>
        public void ClearCache(string sector = null)
        {
            if (string.IsNullOrEmpty(sector))
            {
                _sectorSentimentCache.Clear();
                _sectorCacheTimestamps.Clear();
            }
            else
            {
                string cacheKey = $"sector_{sector}";
                _sectorSentimentCache.Remove(cacheKey);
                _sectorCacheTimestamps.Remove(cacheKey);
            }
        }

        /// <summary>
        /// Disposes the service and releases resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the service and releases resources
        /// </summary>
        /// <param name="disposing">True if disposing managed resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _taskThrottler?.Dispose();
                _disposed = true;
            }
        }

        #endregion
    }
}