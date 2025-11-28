using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Quantra.Models;
using Quantra.Modules;

namespace Quantra.Tests
{
    /// <summary>
    /// Test utility for sentiment-price correlation analysis
    /// </summary>
    public static class SentimentPriceCorrelationTests
    {
        /// <summary>
        /// Run test analysis on sample data
        /// </summary>
        public static async Task<Dictionary<string, object>> RunTestAnalysisAsync()
        {
            try
            {
                var results = new Dictionary<string, object>();

                // Create the analyzer with default settings
                var analyzer = new SentimentPriceCorrelationAnalysis();

                // Test with a popular stock symbol
                string symbol = "AAPL";
                int lookbackDays = 30;

                // Run analysis
                var correlation = await analyzer.AnalyzeSentimentPriceCorrelation(
                    symbol, lookbackDays);

                // Get visualization data
                var visualData = await analyzer.GetVisualizationData(
                    symbol, lookbackDays);

                // Store results
                results["OverallCorrelation"] = correlation.OverallCorrelation.ToString("F4");
                results["LeadLagDays"] = correlation.LeadLagRelationship.ToString("F2");
                results["PredictiveAccuracy"] = correlation.PredictiveAccuracy.ToString("P2");
                results["SourcesAnalyzed"] = string.Join(", ", correlation.SourceCorrelations.Keys);
                results["ShiftEventsCount"] = correlation.SentimentShiftEvents.Count;

                // Source correlations
                var sourceCorrelations = new Dictionary<string, string>();
                foreach (var source in correlation.SourceCorrelations.Keys)
                {
                    sourceCorrelations[source] = correlation.SourceCorrelations[source].ToString("F4");
                }
                results["SourceCorrelations"] = sourceCorrelations;

                // Sample events
                var events = new List<string>();
                foreach (var shiftEvent in correlation.SentimentShiftEvents.Take(3))
                {
                    events.Add($"{shiftEvent.Date:MM/dd}: {shiftEvent.Source} {shiftEvent.SentimentShift:+0.00;-0.00} -> Price {shiftEvent.SubsequentPriceChange:+0.00;-0.00}%");
                }
                results["SampleEvents"] = events;

                return results;
            }
            catch (Exception ex)
            {
                return new Dictionary<string, object>
                {
                    ["Error"] = ex.Message,
                    ["StackTrace"] = ex.StackTrace
                };
            }
        }
    }
}