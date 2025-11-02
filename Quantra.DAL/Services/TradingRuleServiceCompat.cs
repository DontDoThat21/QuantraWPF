using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Models;
using Quantra.DAL.Data;
using Microsoft.EntityFrameworkCore;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Temporary backward compatibility adapter for TradingRuleService.
    /// This allows existing code to continue working while migration is in progress.
    /// 
  /// ?? DEPRECATED: This class will be removed in a future version.
    /// New code should use ITradingRuleService with dependency injection.
    /// </summary>
    [Obsolete("Use ITradingRuleService with dependency injection instead. This compatibility layer will be removed in a future version.")]
 public static class TradingRuleServiceCompat
    {
   private static QuantraDbContext GetContext()
     {
     // Create a temporary context for backward compatibility
       // In a real application, you should use DI
     var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
 optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
return new QuantraDbContext(optionsBuilder.Options);
  }

      /// <summary>
 /// [DEPRECATED] Gets trading rules synchronously for backward compatibility.
        /// Use ITradingRuleService.GetTradingRulesAsync instead.
        /// </summary>
   [Obsolete("Use ITradingRuleService.GetTradingRulesAsync via dependency injection")]
     public static List<TradingRule> GetTradingRules(string symbol = null)
        {
          using var context = GetContext();
            var service = new TradingRuleService(context);
 
        // Synchronously wait for async operation (not ideal, but needed for compatibility)
            return service.GetTradingRulesAsync(symbol).GetAwaiter().GetResult();
        }

    /// <summary>
        /// [DEPRECATED] Saves a trading rule synchronously for backward compatibility.
        /// Use ITradingRuleService.SaveTradingRuleAsync instead.
  /// </summary>
        [Obsolete("Use ITradingRuleService.SaveTradingRuleAsync via dependency injection")]
        public static void SaveTradingRule(TradingRule rule)
{
          using var context = GetContext();
            var service = new TradingRuleService(context);
        
      // Synchronously wait for async operation (not ideal, but needed for compatibility)
  service.SaveTradingRuleAsync(rule).GetAwaiter().GetResult();
        }

        /// <summary>
        /// [DEPRECATED] Deletes a trading rule synchronously for backward compatibility.
     /// Use ITradingRuleService.DeleteTradingRuleAsync instead.
        /// </summary>
        [Obsolete("Use ITradingRuleService.DeleteTradingRuleAsync via dependency injection")]
        public static bool DeleteRule(int ruleId)
     {
            using var context = GetContext();
    var service = new TradingRuleService(context);
   
        // Synchronously wait for async operation (not ideal, but needed for compatibility)
       return service.DeleteTradingRuleAsync(ruleId).GetAwaiter().GetResult();
        }
    }
}
