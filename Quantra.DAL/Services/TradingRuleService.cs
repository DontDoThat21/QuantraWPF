using System;
using System.Collections.Generic;
using Quantra.Models;
//using System.Data.SQLite;

namespace Quantra.DAL.Services
{
    public static class TradingRuleService
    {
        public static List<TradingRule> GetTradingRules(string symbol = null)
        {
            return DatabaseMonolith.GetTradingRules(symbol);
        }

        public static void SaveTradingRule(TradingRule rule)
        {
            DatabaseMonolith.SaveTradingRule(rule);
        }
    }
}
