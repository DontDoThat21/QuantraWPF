using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for trade record management operations
    /// </summary>
    public interface ITradeRecordService
    {
     /// <summary>
      /// Saves a trade record synchronously
        /// </summary>
        /// <param name="trade">The trade record to save</param>
  void SaveTradeRecord(TradeRecord trade);

        /// <summary>
  /// Saves a trade record asynchronously
        /// </summary>
  /// <param name="trade">The trade record to save</param>
        Task SaveTradeRecordAsync(TradeRecord trade);

        /// <summary>
        /// Gets trade records, optionally filtered by symbol
      /// </summary>
    /// <param name="symbol">Optional stock symbol to filter records (null for all records)</param>
    /// <returns>List of trade records matching the criteria</returns>
        Task<List<TradeRecord>> GetTradeRecordsAsync(string symbol = null);

   /// <summary>
     /// Gets trade records within a specific date range
 /// </summary>
   /// <param name="startDate">Start date of the range</param>
        /// <param name="endDate">End date of the range</param>
        /// <returns>List of trade records in the date range</returns>
        Task<List<TradeRecord>> GetTradeRecordsByDateRangeAsync(DateTime startDate, DateTime endDate);

        /// <summary>
        /// Gets a specific trade record by ID
        /// </summary>
        /// <param name="id">The ID of the trade record to retrieve</param>
        /// <returns>The trade record or null if not found</returns>
   Task<TradeRecord> GetTradeRecordByIdAsync(int id);

        /// <summary>
  /// Deletes a trade record by ID
        /// </summary>
        /// <param name="id">The ID of the trade record to delete</param>
   /// <returns>True if deleted successfully, false if record not found</returns>
        Task<bool> DeleteTradeRecordAsync(int id);
    }
}
