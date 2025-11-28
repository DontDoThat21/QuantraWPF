using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for multi-symbol comparative analysis.
    /// Used by Market Chat to handle queries like "Compare predictions for AAPL, MSFT, and GOOGL".
    /// (MarketChat story 7)
    /// </summary>
    public interface IMultiSymbolAnalyzer
    {
        /// <summary>
        /// Analyzes and compares multiple symbols, returning batched predictions, indicators, and risk metrics.
        /// </summary>
        /// <param name="symbols">List of stock symbols to compare (e.g., ["AAPL", "MSFT", "GOOGL"])</param>
        /// <param name="includeHistoricalContext">Whether to include historical price context in the analysis</param>
        /// <returns>MultiSymbolComparisonResult containing comparative analysis data</returns>
        Task<MultiSymbolComparisonResult> CompareSymbolsAsync(IEnumerable<string> symbols, bool includeHistoricalContext = true);

        /// <summary>
        /// Generates a Markdown-formatted comparison table for the given symbols.
        /// </summary>
        /// <param name="comparisonResult">The comparison result to format</param>
        /// <returns>Markdown-formatted string with comparison tables</returns>
        string FormatComparisonAsMarkdown(MultiSymbolComparisonResult comparisonResult);

        /// <summary>
        /// Calculates a composite score for each symbol based on weighted factors (confidence, risk, momentum).
        /// </summary>
        /// <param name="comparisonResult">The comparison result containing symbol data</param>
        /// <returns>Dictionary of symbol to composite score (0-100)</returns>
        Dictionary<string, double> CalculateCompositeScores(MultiSymbolComparisonResult comparisonResult);

        /// <summary>
        /// Generates portfolio allocation recommendations based on comparative analysis.
        /// </summary>
        /// <param name="comparisonResult">The comparison result to base recommendations on</param>
        /// <param name="riskProfile">Risk tolerance profile ("conservative", "moderate", "aggressive")</param>
        /// <returns>Portfolio allocation suggestions as formatted string</returns>
        string GenerateAllocationRecommendations(MultiSymbolComparisonResult comparisonResult, string riskProfile = "moderate");

        /// <summary>
        /// Identifies the strongest and weakest signals across the compared symbols.
        /// </summary>
        /// <param name="comparisonResult">The comparison result to analyze</param>
        /// <returns>SignalHighlights containing strongest/weakest signal information</returns>
        SignalHighlights IdentifySignalHighlights(MultiSymbolComparisonResult comparisonResult);

        /// <summary>
        /// Builds an enhanced prompt context for OpenAI to provide portfolio optimization recommendations.
        /// </summary>
        /// <param name="comparisonResult">The comparison result to include in the prompt</param>
        /// <returns>Formatted context string for AI prompt engineering</returns>
        string BuildComparisonContext(MultiSymbolComparisonResult comparisonResult);
    }
}
