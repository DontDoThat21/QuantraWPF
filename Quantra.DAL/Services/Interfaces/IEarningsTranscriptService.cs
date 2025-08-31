using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for services that provide analysis of earnings call transcripts
    /// </summary>
    public interface IEarningsTranscriptService : ISocialMediaSentimentService
    {
        /// <summary>
        /// Fetches the most recent earnings call transcript for a stock symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Transcript text</returns>
        Task<string> FetchLatestEarningsTranscriptAsync(string symbol);
        
        /// <summary>
        /// Analyzes an earnings call transcript using NLP techniques
        /// </summary>
        /// <param name="transcript">The transcript text</param>
        /// <returns>Analysis result with sentiment, entities, and topics</returns>
        Task<EarningsTranscriptAnalysisResult> AnalyzeEarningsTranscriptAsync(string transcript);
        
        /// <summary>
        /// Fetches and analyzes the most recent earnings call transcript for a stock symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Analysis result with sentiment, entities, and topics</returns>
        Task<EarningsTranscriptAnalysisResult> GetEarningsTranscriptAnalysisAsync(string symbol);
        
        /// <summary>
        /// Gets historical earnings call transcript analysis for a stock
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="quarters">Number of past quarters to retrieve</param>
        /// <returns>List of earnings call transcript analyses</returns>
        Task<List<EarningsTranscriptAnalysisResult>> GetHistoricalEarningsAnalysisAsync(string symbol, int quarters = 4);
    }
}