using System;
using System.Threading.Tasks;
using Quantra.Services;

namespace Quantra.Services.Interfaces
{
    /// <summary>
    /// Interface for services that generate analyst consensus reports
    /// </summary>
    public interface IAnalystConsensusReportService
    {
        /// <summary>
        /// Generates a detailed consensus report for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="historyDays">Number of days of history to include</param>
        /// <returns>Consensus report with historical trend analysis</returns>
        Task<ConsensusReport> GenerateConsensusReportAsync(string symbol, int historyDays = 90);
    }
}