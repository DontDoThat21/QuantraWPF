using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public static class QuoteDataService
    {
        public static async Task<QuoteData> GetLatestQuoteData(string symbol)
        {
            return await AlphaVantageService.GetLatestQuoteData(symbol);
        }

        public static async Task<List<QuoteData>> GetLatestQuoteData(IEnumerable<string> symbols)
        {
            return await DatabaseMonolith.GetLatestQuoteData(symbols);
        }

        public static async Task<(QuoteData, DateTime?)> GetLatestQuoteDataWithTimestamp(string symbol)
        {
            return await DatabaseMonolith.GetLatestQuoteDataWithTimestamp(symbol);
        }
    }
}
