using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace Quantra.Tests
{
    /// <summary>
    /// Simple test runner for sentiment-price correlation analysis
    /// </summary>
    public class TestRunner
    {
        public static async Task RunAsync(string[] args)
        {
            Console.WriteLine("Running Sentiment-Price Correlation Analysis Test");
            Console.WriteLine("================================================");
            
            var results = await SentimentPriceCorrelationTests.RunTestAnalysisAsync();
            
            Console.WriteLine("Test Results:");
            Console.WriteLine("-------------");
            
            // Check if there was an error
            if (results.ContainsKey("Error"))
            {
                Console.WriteLine($"ERROR: {results["Error"]}");
                Console.WriteLine($"Stack Trace: {results["StackTrace"]}");
                return;
            }
            
            // Print results
            foreach (var key in results.Keys)
            {
                if (key == "SourceCorrelations" || key == "SampleEvents")
                    continue;
                    
                Console.WriteLine($"{key}: {results[key]}");
            }
            
            // Print source correlations
            Console.WriteLine("\nSource Correlations:");
            var sourceCorrelations = (JsonElement)results["SourceCorrelations"];
            foreach (var property in sourceCorrelations.EnumerateObject())
            {
                Console.WriteLine($"  {property.Name}: {property.Value}");
            }
            
            // Print sample events
            Console.WriteLine("\nSample Sentiment Shift Events:");
            var events = (JsonElement)results["SampleEvents"];
            foreach (var evnt in events.EnumerateArray())
            {
                Console.WriteLine($"  {evnt.GetString()}");
            }
            
            // Save results to file
            string json = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText("sentiment_price_correlation_results.json", json);
            Console.WriteLine("\nResults saved to sentiment_price_correlation_results.json");
        }
    }
}