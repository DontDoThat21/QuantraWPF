using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Data;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for querying ML prediction data from the database.
    /// Used by Market Chat to provide AI-generated forecast context in conversations.
    /// </summary>
    public class PredictionDataService : IPredictionDataService
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<PredictionDataService> _logger;

        /// <summary>
        /// Constructor for PredictionDataService with dependency injection
        /// </summary>
        public PredictionDataService(QuantraDbContext context, ILogger<PredictionDataService> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Parameterless constructor for backward compatibility
        /// </summary>
        public PredictionDataService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString, sqlServerOptions =>
            {
                sqlServerOptions.CommandTimeout(30);
            });
            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        /// <inheritdoc/>
        public async Task<string> GetPredictionContextAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return null;
            }

            try
            {
                symbol = symbol.ToUpperInvariant().Trim();
                _logger?.LogInformation("Fetching prediction context for {Symbol}", symbol);

                // Get the most recent prediction for the symbol
                var prediction = await _context.StockPredictions
                    .AsNoTracking()
                    .Where(p => p.Symbol == symbol)
                    .OrderByDescending(p => p.CreatedDate)
                    .FirstOrDefaultAsync();

                if (prediction == null)
                {
                    _logger?.LogInformation("No predictions found for {Symbol}", symbol);
                    return null;
                }

                // Get the indicators for this prediction
                var indicators = await _context.PredictionIndicators
                    .AsNoTracking()
                    .Where(i => i.PredictionId == prediction.Id)
                    .ToListAsync();

                // Build the prediction context string
                var contextBuilder = new StringBuilder();
                contextBuilder.AppendLine($"ML Prediction Data for {symbol}:");
                contextBuilder.AppendLine($"- Predicted Action: {prediction.PredictedAction}");
                contextBuilder.AppendLine($"- Confidence: {prediction.Confidence:P0}");
                contextBuilder.AppendLine($"- Current Price: ${prediction.CurrentPrice:F2}");
                contextBuilder.AppendLine($"- Target Price: ${prediction.TargetPrice:F2}");
                contextBuilder.AppendLine($"- Potential Return: {prediction.PotentialReturn:P2}");
                contextBuilder.AppendLine($"- Prediction Date: {prediction.CreatedDate:yyyy-MM-dd HH:mm}");

                // Add indicator rationale if available
                if (indicators.Any())
                {
                    contextBuilder.AppendLine();
                    contextBuilder.AppendLine("Technical Indicators Used:");
                    foreach (var indicator in indicators)
                    {
                        contextBuilder.AppendLine($"- {indicator.IndicatorName}: {indicator.IndicatorValue:F4}");
                    }

                    // Add indicator interpretation guidance
                    contextBuilder.AppendLine();
                    contextBuilder.AppendLine("Indicator Interpretation:");
                    AppendIndicatorInterpretation(contextBuilder, indicators, prediction.PredictedAction);
                }

                _logger?.LogInformation("Successfully built prediction context for {Symbol}", symbol);
                return contextBuilder.ToString();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error fetching prediction context for {Symbol}", symbol);
                return null;
            }
        }

        /// <summary>
        /// Appends interpretation guidance for common technical indicators
        /// </summary>
        private void AppendIndicatorInterpretation(StringBuilder builder, System.Collections.Generic.List<Data.Entities.PredictionIndicatorEntity> indicators, string predictedAction)
        {
            foreach (var indicator in indicators)
            {
                string interpretation = GetIndicatorInterpretation(indicator.IndicatorName, indicator.IndicatorValue, predictedAction);
                if (!string.IsNullOrEmpty(interpretation))
                {
                    builder.AppendLine($"  • {interpretation}");
                }
            }
        }

        /// <summary>
        /// Gets human-readable interpretation for a specific indicator
        /// </summary>
        private string GetIndicatorInterpretation(string indicatorName, double value, string predictedAction)
        {
            var name = indicatorName.ToUpperInvariant();

            if (name.Contains("RSI"))
            {
                if (value < 30) return $"RSI at {value:F1} indicates oversold conditions (bullish signal)";
                if (value > 70) return $"RSI at {value:F1} indicates overbought conditions (bearish signal)";
                return $"RSI at {value:F1} is in neutral territory";
            }

            if (name.Contains("MACD"))
            {
                if (value > 0) return $"MACD at {value:F4} is positive (bullish momentum)";
                if (value < 0) return $"MACD at {value:F4} is negative (bearish momentum)";
                return $"MACD at {value:F4} is near zero (neutral)";
            }

            if (name.Contains("ADX"))
            {
                if (value > 25) return $"ADX at {value:F1} indicates strong trend";
                return $"ADX at {value:F1} indicates weak trend";
            }

            if (name.Contains("BOLLINGER") || name.Contains("BB"))
            {
                return $"Bollinger Band position at {value:F2}";
            }

            if (name.Contains("VWAP"))
            {
                return $"VWAP at ${value:F2}";
            }

            if (name.Contains("EMA") || name.Contains("SMA"))
            {
                return $"{indicatorName} at ${value:F2}";
            }

            if (name.Contains("STOCH"))
            {
                if (value < 20) return $"Stochastic at {value:F1} indicates oversold";
                if (value > 80) return $"Stochastic at {value:F1} indicates overbought";
                return $"Stochastic at {value:F1}";
            }

            // Default: return the indicator name and value
            return null;
        }
    }
}
