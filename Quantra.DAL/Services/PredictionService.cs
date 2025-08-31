using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;
using System.Diagnostics;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class PredictionService
    {
        private const string PythonScript = "python/stock_predictor.py";
        private const string PythonExecutable = "python";
        
        /// <summary>
        /// Predicts future stock price movement using random forest model
        /// </summary>
        public async Task<PredictionModel> PredictStockMovement(string symbol, List<StockDataPoint> historicalData)
        {
            try
            {
                // Prepare input data
                var inputData = historicalData.Select(h => new {
                    date = h.Date.ToString("yyyy-MM-dd"),
                    open = h.Open,
                    high = h.High,
                    low = h.Low,
                    close = h.Close,
                    volume = h.Volume
                }).ToList();
                
                // Create temporary files for input/output
                string tempInput = Path.GetTempFileName();
                string tempOutput = Path.GetTempFileName();
                
                try
                {
                    // Write input data to temp file
                    await File.WriteAllTextAsync(tempInput, JsonSerializer.Serialize(inputData));
                    
                    // Create process to run Python script
                    var startInfo = new ProcessStartInfo
                    {
                        FileName = PythonExecutable,
                        Arguments = $"\"{PythonScript}\" \"{tempInput}\" \"{tempOutput}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };
                    
                    // Run prediction script
                    using var process = Process.Start(startInfo);
                    string output = await process.StandardOutput.ReadToEndAsync();
                    string error = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();
                    
                    if (process.ExitCode != 0)
                    {
                        throw new Exception($"Python prediction failed: {error}");
                    }
                    
                    // Read prediction results
                    var jsonResult = await File.ReadAllTextAsync(tempOutput);
                    var result = JsonSerializer.Deserialize<PredictionResult>(jsonResult);
                    
                    // Convert to PredictionModel
                    return new PredictionModel
                    {
                        Symbol = symbol,
                        PredictedAction = result.action,
                        Confidence = result.confidence,
                        CurrentPrice = result.currentPrice,
                        TargetPrice = result.targetPrice,
                        PredictionDate = DateTime.Now,
                        PotentialReturn = (result.targetPrice - result.currentPrice) / result.currentPrice,
                        // Add feature importances to indicators
                        Indicators = result.featureWeights.ToDictionary(
                            kv => kv.Key,
                            kv => (double)kv.Value)
                    };
                }
                finally
                {
                    // Cleanup temp files
                    try
                    {
                        File.Delete(tempInput);
                        File.Delete(tempOutput);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to predict stock movement for {symbol}", ex.ToString());
                throw;
            }
        }
        
        private class PredictionResult
        {
            public string action { get; set; }
            public double confidence { get; set; }
            public double targetPrice { get; set; }
            public double currentPrice { get; set; }
            public double predictedPrice { get; set; }
            public double priceChangePct { get; set; }
            public Dictionary<string, double> featureWeights { get; set; }
        }
    }
    
    public class StockDataPoint
    {
        public DateTime Date { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public double Volume { get; set; }
    }
}