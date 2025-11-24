using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to monitor and analyze insider trading activity, including transactions by notable figures
    /// </summary>
    public class InsiderTradingService : IInsiderTradingService
    {
        private readonly HttpClient _client;
        private readonly UserSettings _userSettings;
        private readonly string _apiKey;

        // Cache for insider transactions to reduce API calls
        private readonly ConcurrentDictionary<string, (List<InsiderTransaction> Transactions, DateTime Timestamp)> _transactionsCache =
            new ConcurrentDictionary<string, (List<InsiderTransaction>, DateTime)>();

        // List of notable insiders we're tracking
        private readonly List<InsiderProfile> _notableInsiders;

        public InsiderTradingService(UserSettings userSettings = null)
        {
            _client = new HttpClient();
            _userSettings = userSettings ?? new UserSettings();
            _apiKey = GetInsiderTradingApiKey();

            // Initialize notable insider profiles
            _notableInsiders = InitializeNotableInsiders();
        }

        /// <summary>
        /// Gets the insider trading API key from settings or environment
        /// </summary>
        private string GetInsiderTradingApiKey()
        {
            try
            {
                // First try environment variable
                string key = Environment.GetEnvironmentVariable("INSIDER_API_KEY");
                if (!string.IsNullOrEmpty(key))
                    return key;

                // Fallback to AlphaVantage key since we're simulating for now
                return Quantra.DAL.Utilities.Utilities.GetAlphaVantageApiKey();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to get insider trading API key", ex.ToString());
                return "YOUR_API_KEY_HERE"; // Fallback
            }
        }

        /// <summary>
        /// Initialize a list of notable insiders we're tracking
        /// </summary>
        private List<InsiderProfile> InitializeNotableInsiders()
        {
            var insiders = new List<InsiderProfile>();

            // Political figures
            insiders.Add(new InsiderProfile
            {
                Name = "Nancy Pelosi",
                Title = "Former Speaker of the House",
                Organization = "U.S. House of Representatives",
                Category = NotableFigureCategory.PoliticalFigure,
                InfluenceScore = 90,
                IsPriority = true,
                PerformanceMetrics = new InsiderPerformanceMetrics
                {
                    TotalTransactions = 47,
                    AverageBuyReturn = 15.3,
                    AverageSellReturn = 4.8,
                    SuccessRate = 68.5,
                    AverageTransactionValue = 850000
                },
                Notes = "Spouse (Paul Pelosi) makes most trades. Strong tech sector performance."
            });

            // Fund managers
            insiders.Add(new InsiderProfile
            {
                Name = "Cathie Wood",
                Title = "CEO",
                Organization = "ARK Investment Management LLC",
                Category = NotableFigureCategory.FundManager,
                InfluenceScore = 85,
                IsPriority = true,
                PerformanceMetrics = new InsiderPerformanceMetrics
                {
                    TotalTransactions = 126,
                    AverageBuyReturn = 22.7,
                    AverageSellReturn = -4.2,
                    SuccessRate = 59.8,
                    AverageTransactionValue = 4750000
                },
                Notes = "Focus on disruptive innovation. Significant influence on market for certain stocks."
            });

            // Corporate executives
            insiders.Add(new InsiderProfile
            {
                Name = "Satya Nadella",
                Title = "CEO",
                Organization = "Microsoft",
                Category = NotableFigureCategory.CorporateExecutive,
                InfluenceScore = 80,
                PerformanceMetrics = new InsiderPerformanceMetrics
                {
                    TotalTransactions = 18,
                    AverageBuyReturn = 12.1,
                    AverageSellReturn = 3.4,
                    SuccessRate = 72.2,
                    AverageTransactionValue = 2100000
                }
            });

            insiders.Add(new InsiderProfile
            {
                Name = "Elon Musk",
                Title = "CEO",
                Organization = "Tesla, SpaceX",
                Category = NotableFigureCategory.CorporateExecutive,
                InfluenceScore = 95,
                IsPriority = true,
                PerformanceMetrics = new InsiderPerformanceMetrics
                {
                    TotalTransactions = 32,
                    AverageBuyReturn = 41.3,
                    AverageSellReturn = 7.6,
                    SuccessRate = 65.1,
                    AverageTransactionValue = 8500000
                },
                Notes = "High market impact through social media presence and trading activity."
            });

            // Add more notable insiders as needed...
            return insiders;
        }

        /// <summary>
        /// High-level method: fetches recent insider transactions and returns sentiment score for a symbol
        /// </summary>
        public async Task<double> GetInsiderSentimentAsync(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return 0;

            try
            {
                var transactions = await GetInsiderTransactionsAsync(symbol);

                if (transactions == null || transactions.Count == 0)
                    return 0;

                // Get sentiment for each transaction
                List<double> sentimentScores = new List<double>();
                double totalValue = 0;

                // Calculate total value for weighting
                foreach (var transaction in transactions)
                {
                    totalValue += Math.Abs(transaction.Value);
                }

                // If no transactions or all have zero value, return neutral
                if (totalValue == 0)
                    return 0;

                // Calculate weighted sentiment score
                double weightedSentiment = 0;

                foreach (var transaction in transactions)
                {
                    double transactionSentiment = transaction.GetSentimentScore();
                    double weight = Math.Abs(transaction.Value) / totalValue;
                    weightedSentiment += transactionSentiment * weight;
                }

                // Cap at -1.0 to 1.0
                return Math.Max(-1.0, Math.Min(1.0, weightedSentiment));
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error calculating insider sentiment for {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Gets detailed insider trading activity for a symbol
        /// </summary>
        public async Task<List<InsiderTransaction>> GetInsiderTransactionsAsync(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return new List<InsiderTransaction>();

            // Check cache first
            string cacheKey = $"insider_{symbol.ToLower()}";
            if (_transactionsCache.TryGetValue(cacheKey, out var cachedData))
            {
                // If cache is still valid (not older than configured interval)
                var cacheAgeMinutes = (DateTime.Now - cachedData.Timestamp).TotalMinutes;
                if (cacheAgeMinutes < _userSettings.InsiderDataRefreshIntervalMinutes)
                {
                    return cachedData.Transactions;
                }
            }

            // For now, we'll generate simulated data
            // In a production environment, this would call an API
            var transactions = GenerateSimulatedInsiderTransactions(symbol);

            // Cache the results
            _transactionsCache[cacheKey] = (transactions, DateTime.Now);

            return transactions;
        }

        /// <summary>
        /// Gets notable insider trading activity from influential figures
        /// </summary>
        public async Task<List<InsiderTransaction>> GetNotableInsiderTransactionsAsync(string symbol = null)
        {
            var allTransactions = new List<InsiderTransaction>();

            // If symbol is specified, get only transactions for that symbol
            if (!string.IsNullOrEmpty(symbol))
            {
                var transactions = await GetInsiderTransactionsAsync(symbol);
                return transactions.Where(t => t.IsNotableFigure).ToList();
            }

            // Otherwise, get all notable transactions across symbols
            if (_userSettings.NotableInsiderSymbols == null || _userSettings.NotableInsiderSymbols.Count == 0)
            {
                // Default to S&P 500 top 10 stocks if no symbols are configured
                string[] defaultSymbols = { "AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "META", "TSLA", "BRK.B", "UNH", "XOM" };

                foreach (var sym in defaultSymbols)
                {
                    try
                    {
                        var transactions = await GetInsiderTransactionsAsync(sym);
                        allTransactions.AddRange(transactions.Where(t => t.IsNotableFigure));
                    }
                    catch
                    {
                        // Continue if one fails
                        continue;
                    }
                }
            }
            else
            {
                // Use configured symbols
                foreach (var sym in _userSettings.NotableInsiderSymbols)
                {
                    try
                    {
                        var transactions = await GetInsiderTransactionsAsync(sym);
                        allTransactions.AddRange(transactions.Where(t => t.IsNotableFigure));
                    }
                    catch
                    {
                        // Continue if one fails
                        continue;
                    }
                }
            }

            // Sort by transaction date (most recent first)
            return allTransactions.OrderByDescending(t => t.TransactionDate).ToList();
        }

        /// <summary>
        /// Gets aggregate insider trading metrics for a symbol
        /// </summary>
        public async Task<Dictionary<string, double>> GetInsiderMetricsAsync(string symbol)
        {
            var metrics = new Dictionary<string, double>();

            try
            {
                var transactions = await GetInsiderTransactionsAsync(symbol);

                if (transactions == null || transactions.Count == 0)
                    return metrics;

                // Calculate aggregate metrics
                double totalBuyValue = 0;
                double totalSellValue = 0;
                int buyCount = 0;
                int sellCount = 0;
                int optionsCount = 0;
                double ceoTransactionsValue = 0;
                double notableFigureValue = 0;
                double netInsiderValue = 0;

                foreach (var transaction in transactions)
                {
                    switch (transaction.TransactionType)
                    {
                        case InsiderTransactionType.Purchase:
                            totalBuyValue += transaction.Value;
                            buyCount++;
                            netInsiderValue += transaction.Value;
                            break;
                        case InsiderTransactionType.Sale:
                            totalSellValue += transaction.Value;
                            sellCount++;
                            netInsiderValue -= transaction.Value;
                            break;
                        case InsiderTransactionType.OptionExercise:
                            optionsCount++;
                            break;
                    }

                    // Track CEO transactions
                    if (transaction.InsiderTitle?.Contains("CEO") == true ||
                        transaction.InsiderTitle?.Contains("Chief Executive") == true)
                    {
                        if (transaction.TransactionType == InsiderTransactionType.Purchase)
                            ceoTransactionsValue += transaction.Value;
                        else if (transaction.TransactionType == InsiderTransactionType.Sale)
                            ceoTransactionsValue -= transaction.Value;
                    }

                    // Track notable figure transactions
                    if (transaction.IsNotableFigure)
                    {
                        if (transaction.TransactionType == InsiderTransactionType.Purchase)
                            notableFigureValue += transaction.Value;
                        else if (transaction.TransactionType == InsiderTransactionType.Sale)
                            notableFigureValue -= transaction.Value;
                    }
                }

                // Add metrics to dictionary
                metrics["InsiderBuyValue"] = totalBuyValue;
                metrics["InsiderSellValue"] = totalSellValue;
                metrics["InsiderBuyCount"] = buyCount;
                metrics["InsiderSellCount"] = sellCount;
                metrics["InsiderOptionsCount"] = optionsCount;
                metrics["CEOTransactionValue"] = ceoTransactionsValue;
                metrics["NotableFigureValue"] = notableFigureValue;
                metrics["NetInsiderValue"] = netInsiderValue;

                // Calculate buy/sell ratio (avoid division by zero)
                metrics["BuySellRatio"] = sellCount > 0 ? (double)buyCount / sellCount : buyCount;

                // Calculate percentage of transactions by notable figures
                metrics["NotableFigurePercentage"] = (double)transactions.Count(t => t.IsNotableFigure) / transactions.Count;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error calculating insider metrics for {symbol}", ex.ToString());
            }

            return metrics;
        }

        /// <summary>
        /// Gets aggregate insider sentiment grouped by notable individuals
        /// </summary>
        public async Task<Dictionary<string, double>> GetNotableIndividualSentimentAsync(string symbol = null)
        {
            var sentimentByIndividual = new Dictionary<string, double>();

            try
            {
                List<InsiderTransaction> transactions;

                if (!string.IsNullOrEmpty(symbol))
                {
                    transactions = await GetInsiderTransactionsAsync(symbol);
                    transactions = transactions.Where(t => t.IsNotableFigure).ToList();
                }
                else
                {
                    transactions = await GetNotableInsiderTransactionsAsync();
                }

                // Group by insider name
                var groupedTransactions = transactions
                    .GroupBy(t => t.InsiderName)
                    .Where(g => g.Count() > 0); // Must have at least one transaction

                foreach (var group in groupedTransactions)
                {
                    double totalValue = group.Sum(t => Math.Abs(t.Value));

                    // Calculate weighted sentiment for this individual
                    if (totalValue > 0)
                    {
                        double weightedSentiment = 0;
                        foreach (var transaction in group)
                        {
                            double weight = Math.Abs(transaction.Value) / totalValue;
                            weightedSentiment += transaction.GetSentimentScore() * weight;
                        }

                        sentimentByIndividual[group.Key] = weightedSentiment;
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error calculating notable individual sentiment", ex.ToString());
            }

            return sentimentByIndividual;
        }

        /// <summary>
        /// Generate simulated insider transaction data for a symbol
        /// In a production environment, this would be replaced with API calls
        /// </summary>
        private List<InsiderTransaction> GenerateSimulatedInsiderTransactions(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return new List<InsiderTransaction>();

            List<InsiderTransaction> transactions = new List<InsiderTransaction>();
            DateTime now = DateTime.Now;

            // Use the symbol as a seed for deterministic randomization
            int seed = string.IsNullOrEmpty(symbol) ? 0 : symbol.GetHashCode();
            Random random = new Random(seed);

            // Some constants for simulation
            string[] executiveTitles = {
                "CEO", "CFO", "COO", "CTO", "President",
                "EVP", "SVP", "Director", "Board Member", "Chairman"
            };

            string[] executiveNames = {
                "John Smith", "Jane Doe", "Robert Johnson", "Maria Garcia", "David Williams",
                "Lisa Brown", "James Wilson", "Jennifer Jones", "Michael Lee", "Patricia Taylor"
            };

            // Determine if we should include notable figures based on symbol
            bool includeNotable = true;

            // Generate between 5 and 15 transactions
            int numTransactions = random.Next(5, 16);
            for (int i = 0; i < numTransactions; i++)
            {
                InsiderTransaction transaction = new InsiderTransaction();
                transaction.Symbol = symbol;

                // Set dates within the last 90 days
                int daysAgo = random.Next(1, 91);
                transaction.TransactionDate = now.AddDays(-daysAgo);
                transaction.FilingDate = transaction.TransactionDate.AddDays(random.Next(1, 5));

                // Randomly determine if this is a notable figure transaction
                bool isNotable = random.NextDouble() < 0.2 && includeNotable;

                if (isNotable)
                {
                    // Select a notable insider
                    var notableInsider = _notableInsiders[random.Next(_notableInsiders.Count)];
                    transaction.InsiderName = notableInsider.Name;
                    transaction.InsiderTitle = notableInsider.Title;
                    transaction.IsNotableFigure = true;
                    transaction.NotableCategory = notableInsider.Category;

                    // For notable figures, transactions tend to be larger
                    transaction.Quantity = random.Next(1000, 100001);
                }
                else
                {
                    // Regular insider
                    transaction.InsiderName = executiveNames[random.Next(executiveNames.Length)];
                    transaction.InsiderTitle = executiveTitles[random.Next(executiveTitles.Length)];
                    transaction.IsNotableFigure = false;

                    // Regular insiders have smaller transactions
                    transaction.Quantity = random.Next(100, 10001);
                }

                // Transaction type
                double typeRandom = random.NextDouble();
                if (typeRandom < 0.4)
                {
                    transaction.TransactionType = InsiderTransactionType.Purchase;
                }
                else if (typeRandom < 0.8)
                {
                    transaction.TransactionType = InsiderTransactionType.Sale;
                }
                else if (typeRandom < 0.95)
                {
                    transaction.TransactionType = InsiderTransactionType.OptionExercise;

                    // Add option details
                    transaction.StrikePrice = random.Next(5, 501);
                    transaction.ExpirationDate = now.AddMonths(random.Next(1, 37));
                }
                else
                {
                    transaction.TransactionType = InsiderTransactionType.GrantAward;
                }

                // Generate a realistic price based on the symbol's hash
                double basePrice = 10 + Math.Abs(symbol.GetHashCode() % 1000) / 10.0;
                transaction.Price = Math.Round(basePrice * (0.9 + random.NextDouble() * 0.2), 2);

                transactions.Add(transaction);
            }

            // Always include the CEO for more realistic data
            InsiderTransaction ceoTransaction = new InsiderTransaction
            {
                Symbol = symbol,
                InsiderName = executiveNames[0],
                InsiderTitle = "CEO",
                TransactionDate = now.AddDays(-random.Next(1, 30)),
                IsNotableFigure = false
            };

            ceoTransaction.FilingDate = ceoTransaction.TransactionDate.AddDays(random.Next(1, 5));

            // CEO transactions are often larger
            ceoTransaction.Quantity = random.Next(5000, 50001);

            // CEO more likely to buy than sell if stock is performing well
            if (random.NextDouble() < 0.6)
            {
                ceoTransaction.TransactionType = InsiderTransactionType.Purchase;
            }
            else
            {
                ceoTransaction.TransactionType = InsiderTransactionType.Sale;
            }

            // Similar price logic
            double ceoBasePrice = 10 + Math.Abs(symbol.GetHashCode() % 1000) / 10.0;
            ceoTransaction.Price = Math.Round(ceoBasePrice * (0.95 + random.NextDouble() * 0.1), 2);

            transactions.Add(ceoTransaction);

            // Sort by transaction date (most recent first)
            return transactions.OrderByDescending(t => t.TransactionDate).ToList();
        }

        /// <summary>
        /// Clears the insider transactions cache for a symbol or all symbols
        /// </summary>
        public void ClearCache(string symbol = null)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                _transactionsCache.Clear();
            }
            else
            {
                string cacheKey = $"insider_{symbol.ToLower()}";
                _transactionsCache.TryRemove(cacheKey, out _);
            }
        }
    }
}