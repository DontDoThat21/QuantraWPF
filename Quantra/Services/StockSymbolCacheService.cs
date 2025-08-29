using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Quantra.Models;
using System.Data.SQLite;
using Newtonsoft.Json;

namespace Quantra.Services
{
    public static class StockSymbolCacheService
    {
        public static void CacheStockSymbols(List<StockSymbol> symbols)
        {
            DatabaseMonolith.CacheStockSymbols(symbols);
        }

        public static ObservableCollection<StockSymbol> GetAllStockSymbols()
        {
            return DatabaseMonolith.GetAllStockSymbols();
        }

        public static StockSymbol GetStockSymbol(string symbol)
        {
            return DatabaseMonolith.GetStockSymbol(symbol);
        }

        public static bool IsSymbolCacheValid(int maxAgeDays = 7)
        {
            return DatabaseMonolith.IsSymbolCacheValid(maxAgeDays);
        }

        public static int RefreshSymbolCache()
        {
            return DatabaseMonolith.RefreshSymbolCache();
        }

        public static List<StockSymbol> SearchSymbols(string searchTerm)
        {
            return DatabaseMonolith.SearchSymbols(searchTerm);
        }
    }
}
