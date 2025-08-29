using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Quantra.Adapters
{
    /// <summary>
    /// Adapter class for PredictionModel utility methods and transformations.
    /// </summary>
    public static class PredictionModelAdapter
    {
        // All methods now use Quantra.Models.PredictionModel only.
        public static PredictionModel ToModelsPredictionModel(PredictionModel source)
        {
            if (source == null)
                return null;

            var result = new PredictionModel
            {
                Symbol = source.Symbol,
                PredictedAction = source.PredictedAction,
                Confidence = source.Confidence,
                CurrentPrice = source.CurrentPrice,
                TargetPrice = source.TargetPrice,
                TradingRule = source.TradingRule
            };

            if (source.Indicators != null && source.Indicators.Count > 0)
            {
                result.Indicators = new Dictionary<string, double>(source.Indicators);
            }

            return result;
        }

        public static PredictionModel ToControlsPredictionModel(PredictionModel source)
        {
            if (source == null)
                return null;

            var result = new PredictionModel
            {
                Symbol = source.Symbol,
                PredictedAction = source.PredictedAction,
                Confidence = source.Confidence,
                CurrentPrice = source.CurrentPrice,
                TargetPrice = source.TargetPrice,
                PredictionDate = source.PredictionDate,
                TradingRule = source.TradingRule,
                PotentialReturn = source.PotentialReturn
            };

            if (result.PotentialReturn == 0 && result.CurrentPrice > 0)
            {
                // Implement CalculatePotentialReturn if needed
            }

            if (source.Indicators != null && source.Indicators.Count > 0)
            {
                result.Indicators = new Dictionary<string, double>(source.Indicators);
            }

            if (source.MarketContext != null)
            {
                result.MarketContext = source.MarketContext; // Shallow copy
            }

            return result;
        }

        public static double CalculatePotentialReturn(string predictedAction, double currentPrice, double targetPrice)
        {
            if (predictedAction == "BUY")
                return (targetPrice - currentPrice) / currentPrice;
            else if (predictedAction == "SELL")
                return (currentPrice - targetPrice) / currentPrice;
            else
                return 0;
        }

        public static string GetExistingTradingRule(this PredictionModel model)
        {
            return model?.TradingRule ?? string.Empty;
        }

        public static void SetExistingTradingRule(this PredictionModel model, string value)
        {
            if (model != null)
            {
                model.TradingRule = value;
            }
        }

        // Add this method to convert a Quantra.Models.PredictionModel to the UI model if needed
        // For now, just return the same model (since the UI should use Quantra.Models.PredictionModel)
        public static Quantra.Models.PredictionModel ToPredictionAnalysisModel(Quantra.Models.PredictionModel model)
        {
            return model;
        }
    }
}
