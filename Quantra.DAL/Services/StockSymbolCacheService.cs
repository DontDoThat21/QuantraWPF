using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Newtonsoft.Json;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing stock symbol caching using Entity Framework Core
    /// </summary>
    public class StockSymbolCacheService
    {
        private static LoggingService _loggingService;

        public StockSymbolCacheService(LoggingService loggingService)
        {
            _loggingService = loggingService;
        }

        private DbContextOptions<QuantraDbContext> CreateOptions()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            return optionsBuilder.Options;
        }

        /// <summary>
        /// Caches stock symbols in the database
        /// </summary>
        /// <param name="symbols">List of stock symbols to cache</param>
  public void CacheStockSymbols(List<StockSymbol> symbols)
        {
  if (symbols == null || !symbols.Any())
            {
          _loggingService.Log("Warning", "Attempted to cache empty symbol list");
 return;
          }

            try
          {
       using (var context = new QuantraDbContext(CreateOptions()))
    {
          foreach (var symbol in symbols)
        {
      var entity = new StockSymbolEntity
           {
 Symbol = symbol.Symbol,
              Name = symbol.Name ?? string.Empty,
            Sector = symbol.Sector ?? string.Empty,
      Industry = symbol.Industry ?? string.Empty,
           LastUpdated = DateTime.Now
       };

       // Check if symbol already exists
       var existing = context.StockSymbols.FirstOrDefault(s => s.Symbol == symbol.Symbol);
          if (existing != null)
     {
          // Update existing
       existing.Name = entity.Name;
 existing.Sector = entity.Sector;
           existing.Industry = entity.Industry;
              existing.LastUpdated = entity.LastUpdated;
          }
       else
                {
      // Add new
      context.StockSymbols.Add(entity);
}
  }

           context.SaveChanges();
     _loggingService.Log("Info", $"Successfully cached {symbols.Count} stock symbols");
         }
       }
       catch (Exception ex)
            {
     _loggingService.LogErrorWithContext(ex, "Failed to cache stock symbols");
      throw;
            }
    }

        /// <summary>
        /// Retrieves all cached stock symbols from the database.
      /// </summary>
        /// <returns>ObservableCollection of StockSymbol objects</returns>
      public ObservableCollection<StockSymbol> GetAllCachedStockSymbols()
        {
          var symbols = new ObservableCollection<StockSymbol>();
            
            try
   {
        using (var context = new QuantraDbContext(CreateOptions()))
           {
            var entities = context.StockSymbols
      .AsNoTracking()
  .OrderBy(s => s.Symbol)
            .ToList();

            foreach (var entity in entities)
          {
          symbols.Add(new StockSymbol
       {
            Symbol = entity.Symbol,
       Name = entity.Name ?? string.Empty,
           Sector = entity.Sector ?? string.Empty,
          Industry = entity.Industry ?? string.Empty,
  LastUpdated = entity.LastUpdated ?? DateTime.MinValue
            });
       }

    _loggingService.Log("Info", $"Retrieved {symbols.Count} cached stock symbols");
    }
      }
            catch (Exception ex)
          {
      _loggingService.LogErrorWithContext(ex, "Failed to retrieve stock symbols");
            }

return symbols;
      }

    /// <summary>
    /// Retrieves a specific stock symbol from the cache.
        /// </summary>
        /// <param name="symbol">The stock symbol to retrieve</param>
        /// <returns>StockSymbol object or null if not found</returns>
        public StockSymbol GetStockSymbol(string symbol)
        {
 if (string.IsNullOrWhiteSpace(symbol))
          {
          _loggingService.Log("Warning", "Attempted to retrieve empty symbol");
          return null;
    }

      try
 {
   using (var context = new QuantraDbContext(CreateOptions()))
          {
var entity = context.StockSymbols
         .AsNoTracking()
        .FirstOrDefault(s => s.Symbol == symbol.Trim().ToUpper());

         if (entity != null)
     {
  return new StockSymbol
    {
        Symbol = entity.Symbol,
        Name = entity.Name ?? string.Empty,
       Sector = entity.Sector ?? string.Empty,
       Industry = entity.Industry ?? string.Empty,
       LastUpdated = entity.LastUpdated ?? DateTime.MinValue
    };
      }
      }
            }
        catch (Exception ex)
            {
    _loggingService.LogErrorWithContext(ex, $"Failed to retrieve stock symbol {symbol}");
        }

     return null;
        }

        /// <summary>
        /// Deletes a stock symbol and all associated cached data from the database.
        /// </summary>
        /// <param name="symbol">The stock symbol to delete</param>
        /// <returns>True if deletion was successful, false otherwise</returns>
        public static bool DeleteStockSymbol(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                _loggingService.Log("Warning", "DeleteStockSymbol called with null or empty symbol");
                return false;
            }

            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    using (var transaction = context.Database.BeginTransaction())
                    {
                        try
                        {
                            var symbolToDelete = symbol.Trim().ToUpper();

                            // Delete from StockDataCache table first (due to potential foreign key constraints)
                            var cacheEntries = context.StockDataCache
                                .Where(c => c.Symbol == symbolToDelete)
                                .ToList();
                            
                            if (cacheEntries.Any())
                            {
                                context.StockDataCache.RemoveRange(cacheEntries);
                                _loggingService.Log("Info", $"Deleted {cacheEntries.Count} cache entries for symbol {symbolToDelete}");
                            }

                            // Delete from StockSymbols table
                            var symbolEntity = context.StockSymbols
                                .FirstOrDefault(s => s.Symbol == symbolToDelete);

                            if (symbolEntity != null)
                            {
                                context.StockSymbols.Remove(symbolEntity);
                                context.SaveChanges();
                                transaction.Commit();
                                
                                _loggingService.Log("Info", $"Successfully deleted stock symbol {symbolToDelete} from database");
                                return true;
                            }
                            else
                            {
                                transaction.Rollback();
                                _loggingService.Log("Warning", $"Stock symbol {symbolToDelete} was not found in database");
                                return false;
                            }
                        }
                        catch (Exception ex)
                        {
                            transaction.Rollback();
                            throw;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to delete stock symbol {symbol}");
                return false;
            }
        }

        /// <summary>
        /// Checks if the symbol cache is still valid based on the LastUpdated timestamp.
        /// </summary>
        /// <param name="maxAgeDays">Maximum age of the cache in days (default: 7)</param>
        /// <returns>True if cache is valid, false otherwise</returns>
        public bool IsSymbolCacheValid(int maxAgeDays = 7)
        {
     try
            {
             using (var context = new QuantraDbContext(CreateOptions()))
                {
     // First check if we have any symbols cached at all
     var count = context.StockSymbols.Count();
            if (count == 0)
       {
      return false; // No symbols in cache
            }

  // Check if the oldest entry is still valid
    var oldestUpdate = context.StockSymbols.Min(s => s.LastUpdated);
 if (!oldestUpdate.HasValue)
                 {
     return false;
 }

         var cacheAge = (DateTime.Now - oldestUpdate.Value).TotalDays;
      return cacheAge <= maxAgeDays;
  }
}
       catch (Exception ex)
        {
      _loggingService.LogErrorWithContext(ex, "Failed to validate symbol cache");
        return false;
            }
        }

        /// <summary>
      /// Forces a refresh of the symbol cache by updating the LastUpdated timestamp.
        /// </summary>
        /// <returns>Number of symbols refreshed</returns>
     public int RefreshSymbolCache()
   {
            try
        {
     using (var context = new QuantraDbContext(CreateOptions()))
       {
            var allSymbols = context.StockSymbols.ToList();
              var now = DateTime.Now;
     
  foreach (var symbol in allSymbols)
           {
                 symbol.LastUpdated = now;
                 }

          context.SaveChanges();
            var count = allSymbols.Count;
         _loggingService.Log("Info", $"Refreshed timestamps for {count} stock symbols");
              return count;
    }
            }
            catch (Exception ex)
            {
        _loggingService.LogErrorWithContext(ex, "Failed to refresh symbol cache");
  return 0;
 }
        }

     /// <summary>
        /// Searches for stock symbols by name or ticker.
        /// </summary>
        /// <param name="searchTerm">Search term to match against symbol or name</param>
        /// <returns>List of matching StockSymbol objects</returns>
 public List<StockSymbol> SearchSymbols(string searchTerm)
        {
 var results = new List<StockSymbol>();
       
   if (string.IsNullOrWhiteSpace(searchTerm))
            {
         return results;
      }

            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
         {
        var entities = context.StockSymbols
  .AsNoTracking()
              .Where(s => s.Symbol.Contains(searchTerm) || s.Name.Contains(searchTerm))
      .OrderBy(s => s.Symbol)
        .Take(100) // Limiting results for performance
     .ToList();

         foreach (var entity in entities)
        {
              results.Add(new StockSymbol
     {
              Symbol = entity.Symbol,
           Name = entity.Name ?? string.Empty,
          Sector = entity.Sector ?? string.Empty,
             Industry = entity.Industry ?? string.Empty,
    LastUpdated = entity.LastUpdated ?? DateTime.MinValue
  });
   }

   _loggingService.Log("Info", $"Found {results.Count} matching stock symbols for search term '{searchTerm}'");
      }
          }
   catch (Exception ex)
            {
      _loggingService.LogErrorWithContext(ex, $"Failed to search for stock symbols with term '{searchTerm}'");
       }

      return results;
        }

        /// <summary>
        /// Retrieves all stock symbols as a simple list of strings from the database.
        /// </summary>
        /// <returns>List of stock symbol strings</returns>
        public static List<string> GetAllSymbolsAsList()
        {
            var symbols = new List<string>();

            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    symbols = context.StockSymbols
                        .AsNoTracking()
                        .OrderBy(s => s.Symbol)
                        .Select(s => s.Symbol)
                        .ToList();

                    _loggingService.Log("Info", $"Retrieved {symbols.Count} stock symbols as list");
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Failed to retrieve stock symbols as list");
            }

            return symbols;
        }
    }
}
