using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing custom benchmarks
    /// </summary>
    public class CustomBenchmarkService
    {
        private readonly string _customBenchmarksFilePath;
        private readonly HistoricalDataService _historicalDataService;
        private List<CustomBenchmark> _customBenchmarks;
        
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="historicalDataService">Historical data service for loading price data</param>
        public CustomBenchmarkService(HistoricalDataService historicalDataService = null)
        {
            _historicalDataService = historicalDataService ?? new HistoricalDataService();
            
            // Define path for storing custom benchmarks
            string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string appFolder = Path.Combine(appDataPath, "Quantra");
            
            // Create directory if it doesn't exist
            if (!Directory.Exists(appFolder))
                Directory.CreateDirectory(appFolder);
                
            _customBenchmarksFilePath = Path.Combine(appFolder, "custom_benchmarks.json");
            
            // Initialize benchmarks list
            _customBenchmarks = new List<CustomBenchmark>();
            
            // Load any existing custom benchmarks
            LoadCustomBenchmarks();
        }
        
        /// <summary>
        /// Get all custom benchmarks
        /// </summary>
        /// <returns>List of custom benchmarks</returns>
        public List<CustomBenchmark> GetCustomBenchmarks()
        {
            return _customBenchmarks;
        }
        
        /// <summary>
        /// Get a custom benchmark by ID
        /// </summary>
        /// <param name="id">ID of the benchmark</param>
        /// <returns>The custom benchmark, or null if not found</returns>
        public CustomBenchmark GetCustomBenchmark(string id)
        {
            return _customBenchmarks.FirstOrDefault(b => b.Id == id);
        }
        
        /// <summary>
        /// Save a custom benchmark
        /// </summary>
        /// <param name="benchmark">The benchmark to save</param>
        /// <returns>True if saved successfully, false otherwise</returns>
        public bool SaveCustomBenchmark(CustomBenchmark benchmark)
        {
            try
            {
                if (benchmark == null)
                    return false;
                    
                // Validate benchmark
                if (!benchmark.Validate(out string errorMessage))
                {
                    //DatabaseMonolith.Log("Error", $"Invalid custom benchmark: {errorMessage}");
                    return false;
                }
                
                // Update modified date
                benchmark.ModifiedDate = DateTime.Now;
                
                // Check if this is an update to an existing benchmark
                int existingIndex = _customBenchmarks.FindIndex(b => b.Id == benchmark.Id);
                if (existingIndex >= 0)
                {
                    _customBenchmarks[existingIndex] = benchmark;
                }
                else
                {
                    _customBenchmarks.Add(benchmark);
                }
                
                // Save to file
                return SaveCustomBenchmarksToFile();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to save custom benchmark", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Delete a custom benchmark
        /// </summary>
        /// <param name="id">ID of the benchmark to delete</param>
        /// <returns>True if deleted successfully, false otherwise</returns>
        public bool DeleteCustomBenchmark(string id)
        {
            try
            {
                int initialCount = _customBenchmarks.Count;
                _customBenchmarks.RemoveAll(b => b.Id == id);
                
                if (_customBenchmarks.Count == initialCount)
                    return false; // Nothing was removed
                
                // Save to file
                return SaveCustomBenchmarksToFile();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to delete custom benchmark", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Calculate historical data for a custom benchmark
        /// </summary>
        /// <param name="benchmark">The custom benchmark</param>
        /// <param name="startDate">Start date for the data</param>
        /// <param name="endDate">End date for the data</param>
        /// <returns>BenchmarkComparisonData object</returns>
        public async Task<BenchmarkComparisonData> CalculateCustomBenchmarkData(
            CustomBenchmark benchmark, DateTime startDate, DateTime endDate)
        {
            if (benchmark == null || benchmark.Components.Count == 0)
                return null;
                
            try
            {
                // Dictionary to store component historical data
                var componentData = new Dictionary<string, List<HistoricalPrice>>();
                
                // Load historical data for each component
                foreach (var component in benchmark.Components)
                {
                    var data = await _historicalDataService.GetComprehensiveHistoricalData(component.Symbol);
                    
                    // Filter to the required date range
                    var filteredData = data
                        .Where(h => h.Date >= startDate && h.Date <= endDate)
                        .OrderBy(h => h.Date)
                        .ToList();
                        
                    if (filteredData.Count > 0)
                    {
                        componentData[component.Symbol] = filteredData;
                    }
                    else
                    {
                        //DatabaseMonolith.Log("Warning", $"No historical data for {component.Symbol} in date range {startDate} to {endDate}");
                    }
                }
                
                // If no data was loaded, return null
                if (componentData.Count == 0)
                    return null;
                    
                // Find the date range where all components have data
                DateTime commonStartDate = componentData.Values.Max(data => data.First().Date);
                DateTime commonEndDate = componentData.Values.Min(data => data.Last().Date);
                
                // If the common date range is invalid, return null
                if (commonStartDate > commonEndDate)
                {
                    //DatabaseMonolith.Log("Warning", "No overlapping date range for custom benchmark components");
                    return null;
                }
                
                // Create a list of dates in the common range
                List<DateTime> commonDates = new List<DateTime>();
                for (DateTime date = commonStartDate; date <= commonEndDate; date = date.AddDays(1))
                {
                    // Skip weekends
                    if (date.DayOfWeek != DayOfWeek.Saturday && date.DayOfWeek != DayOfWeek.Sunday)
                    {
                        commonDates.Add(date);
                    }
                }
                
                // Filter all component data to only include the common dates
                var alignedData = new Dictionary<string, Dictionary<DateTime, HistoricalPrice>>();
                foreach (var kvp in componentData)
                {
                    alignedData[kvp.Key] = kvp.Value.ToDictionary(h => h.Date.Date, h => h);
                }
                
                // Create the composite historical data
                var compositeData = new List<HistoricalPrice>();
                foreach (DateTime date in commonDates)
                {
                    // Check if all components have data for this date
                    bool allComponentsHaveData = benchmark.Components
                        .All(c => componentData.ContainsKey(c.Symbol) && 
                                 alignedData[c.Symbol].ContainsKey(date));
                    
                    if (allComponentsHaveData)
                    {
                        // Create a weighted average of component data for this date
                        double weightedOpen = 0;
                        double weightedHigh = 0;
                        double weightedLow = 0;
                        double weightedClose = 0;
                        long weightedVolume = 0;
                        
                        foreach (var component in benchmark.Components)
                        {
                            var price = alignedData[component.Symbol][date];
                            weightedOpen += price.Open * component.Weight;
                            weightedHigh += price.High * component.Weight;
                            weightedLow += price.Low * component.Weight;
                            weightedClose += price.Close * component.Weight;
                            weightedVolume += (long)(price.Volume * component.Weight);
                        }
                        
                        compositeData.Add(new HistoricalPrice
                        {
                            Date = date,
                            Open = weightedOpen,
                            High = weightedHigh,
                            Low = weightedLow,
                            Close = weightedClose,
                            Volume = weightedVolume,
                            AdjClose = weightedClose
                        });
                    }
                }
                
                // If no composite data could be created, return null
                if (compositeData.Count == 0)
                    return null;
                
                // Create benchmark comparison data
                var benchmarkData = new BenchmarkComparisonData
                {
                    Symbol = benchmark.DisplaySymbol,
                    Name = benchmark.Name,
                    HistoricalData = compositeData,
                    Dates = compositeData.Select(h => h.Date).ToList()
                };
                
                // Calculate normalized returns (starting at 1.0)
                double initialPrice = compositeData.First().Close;
                benchmarkData.NormalizedReturns = compositeData
                    .Select(h => h.Close / initialPrice)
                    .ToList();
                
                // Calculate performance metrics
                CalculatePerformanceMetrics(benchmarkData, compositeData);
                
                return benchmarkData;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate custom benchmark data for {benchmark.Name}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Calculate performance metrics for a benchmark
        /// </summary>
        private void CalculatePerformanceMetrics(BenchmarkComparisonData benchmarkData, List<HistoricalPrice> historicalData)
        {
            // Calculate total return
            benchmarkData.TotalReturn = historicalData.Last().Close / historicalData.First().Close - 1;
            
            // Calculate drawdown
            double peak = historicalData.First().Close;
            double maxDrawdown = 0;
            List<double> dailyReturns = new List<double>();
            List<double> downsideReturns = new List<double>();
            
            for (int i = 1; i < historicalData.Count; i++)
            {
                if (historicalData[i].Close > peak)
                {
                    peak = historicalData[i].Close;
                }
                
                double drawdown = (peak - historicalData[i].Close) / peak;
                if (drawdown > maxDrawdown)
                {
                    maxDrawdown = drawdown;
                }
                
                // Calculate daily returns
                double dailyReturn = (historicalData[i].Close - historicalData[i - 1].Close) / historicalData[i - 1].Close;
                dailyReturns.Add(dailyReturn);
                
                if (dailyReturn < 0)
                {
                    downsideReturns.Add(dailyReturn);
                }
            }
            
            benchmarkData.MaxDrawdown = maxDrawdown;
            benchmarkData.Volatility = CalculateStandardDeviation(dailyReturns);
            
            // Calculate CAGR
            double totalDays = (historicalData.Last().Date - historicalData.First().Date).TotalDays;
            if (totalDays > 0)
            {
                benchmarkData.CAGR = Math.Pow(1 + benchmarkData.TotalReturn, 365.0 / totalDays) - 1;
            }
            
            // Calculate Sharpe, Sortino, and Calmar ratios
            double averageReturn = dailyReturns.Count > 0 ? dailyReturns.Average() : 0;
            double riskFreeRate = 0.0; // Simplified assumption
            
            benchmarkData.SharpeRatio = benchmarkData.Volatility > 0 ? 
                (averageReturn - riskFreeRate) / benchmarkData.Volatility * Math.Sqrt(252) : 0;
            
            double downsideDeviation = CalculateStandardDeviation(downsideReturns);
            benchmarkData.SortinoRatio = downsideDeviation > 0 ? 
                (averageReturn - riskFreeRate) / downsideDeviation * Math.Sqrt(252) : 0;
            
            benchmarkData.CalmarRatio = benchmarkData.MaxDrawdown > 0 ? 
                benchmarkData.CAGR / benchmarkData.MaxDrawdown : 0;
                
            benchmarkData.InformationRatio = benchmarkData.Volatility > 0 ?
                (averageReturn - riskFreeRate) / benchmarkData.Volatility * Math.Sqrt(252) : 0;
        }
        
        /// <summary>
        /// Calculate standard deviation of a list of values
        /// </summary>
        private double CalculateStandardDeviation(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0;
                
            double avg = values.Average();
            double sumOfSquaresOfDifferences = values.Sum(val => Math.Pow(val - avg, 2));
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
        }
        
        /// <summary>
        /// Load custom benchmarks from file
        /// </summary>
        private void LoadCustomBenchmarks()
        {
            try
            {
                if (File.Exists(_customBenchmarksFilePath))
                {
                    string json = File.ReadAllText(_customBenchmarksFilePath);
                    var benchmarks = JsonConvert.DeserializeObject<List<CustomBenchmark>>(json);
                    if (benchmarks != null)
                    {
                        _customBenchmarks = benchmarks;
                        //DatabaseMonolith.Log("Info", $"Loaded {_customBenchmarks.Count} custom benchmarks from file");
                    }
                }
                else
                {
                    //DatabaseMonolith.Log("Info", "No custom benchmarks file found, creating sample benchmarks");
                    CreateSampleBenchmarks();
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to load custom benchmarks", ex.ToString());
                _customBenchmarks = new List<CustomBenchmark>();
                
                // Create sample benchmarks if loading failed
                CreateSampleBenchmarks();
            }
        }
        
        /// <summary>
        /// Create sample benchmarks for testing
        /// </summary>
        private void CreateSampleBenchmarks()
        {
            try
            {
                // Create a 60/40 Stock/Bond portfolio
                var balanced = new CustomBenchmark("Balanced Portfolio", "60% SPY / 40% IEF balanced portfolio");
                balanced.AddComponent("SPY", "S&P 500", 0.6);
                balanced.AddComponent("IEF", "7-10 Year Treasury Bond", 0.4);
                
                // Validate balanced benchmark before saving
                if (!balanced.Validate(out string balancedError))
                {
                    //DatabaseMonolith.Log("Error", $"Balanced benchmark validation failed: {balancedError}");
                }
                else
                {
                    //DatabaseMonolith.Log("Debug", $"Created balanced benchmark: Name='{balanced.Name}', DisplaySymbol='{balanced.DisplaySymbol}'");
                }
                
                // Create a Technology focused portfolio
                var tech = new CustomBenchmark("Tech Portfolio", "Technology-focused portfolio");
                tech.AddComponent("QQQ", "NASDAQ-100", 0.5);
                tech.AddComponent("XLK", "Technology Sector SPDR", 0.3);
                tech.AddComponent("VGT", "Vanguard Information Technology", 0.2);
                
                // Validate tech benchmark before saving
                if (!tech.Validate(out string techError))
                {
                    //DatabaseMonolith.Log("Error", $"Tech benchmark validation failed: {techError}");
                }
                else
                {
                    //DatabaseMonolith.Log("Debug", $"Created tech benchmark: Name='{tech.Name}', DisplaySymbol='{tech.DisplaySymbol}'");
                }
                
                // Save sample benchmarks
                SaveCustomBenchmark(balanced);
                SaveCustomBenchmark(tech);
                
                //DatabaseMonolith.Log("Info", "Created sample custom benchmarks");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to create sample benchmarks", ex.ToString());
            }
        }
        
        /// <summary>
        /// Save custom benchmarks to file
        /// </summary>
        private bool SaveCustomBenchmarksToFile()
        {
            try
            {
                string json = JsonConvert.SerializeObject(_customBenchmarks, Formatting.Indented);
                File.WriteAllText(_customBenchmarksFilePath, json);
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to save custom benchmarks to file", ex.ToString());
                return false;
            }
        }
    }
}