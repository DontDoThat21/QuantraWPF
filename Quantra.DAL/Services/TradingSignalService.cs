using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing trading signals
    /// </summary>
    /// <remarks>
    /// TODO: Replace in-memory storage with database persistence using Entity Framework Core
    /// when TradingSignalEntity is added to QuantraDbContext. Current implementation is for
    /// MVP/prototype purposes and signals will be lost when the application restarts.
    /// </remarks>
    public class TradingSignalService : ITradingSignalService
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly ITechnicalIndicatorService _technicalIndicatorService;
        
        // TODO: Replace with database persistence - signals are currently stored in memory only
        private readonly List<TradingSignal> _signals = new List<TradingSignal>();
        private int _nextId = 1;

        public TradingSignalService(
            AlphaVantageService alphaVantageService,
            ITechnicalIndicatorService technicalIndicatorService)
        {
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _technicalIndicatorService = technicalIndicatorService ?? throw new ArgumentNullException(nameof(technicalIndicatorService));
        }

        /// <summary>
        /// Gets all trading signals
        /// </summary>
        public Task<List<TradingSignal>> GetAllSignalsAsync()
        {
            return Task.FromResult(_signals.ToList());
        }

        /// <summary>
        /// Gets a trading signal by ID
        /// </summary>
        public Task<TradingSignal> GetSignalByIdAsync(int signalId)
        {
            var signal = _signals.FirstOrDefault(s => s.Id == signalId);
            return Task.FromResult(signal);
        }

        /// <summary>
        /// Gets all enabled trading signals
        /// </summary>
        public Task<List<TradingSignal>> GetEnabledSignalsAsync()
        {
            var enabledSignals = _signals.Where(s => s.IsEnabled).ToList();
            return Task.FromResult(enabledSignals);
        }

        /// <summary>
        /// Saves a trading signal (creates new or updates existing)
        /// </summary>
        public Task<bool> SaveSignalAsync(TradingSignal signal)
        {
            if (signal == null)
            {
                return Task.FromResult(false);
            }

            try
            {
                if (signal.Id == 0)
                {
                    // Create new signal
                    signal.Id = _nextId++;
                    signal.CreatedDate = DateTime.Now;
                    signal.LastModified = DateTime.Now;
                    _signals.Add(signal);
                }
                else
                {
                    // Update existing signal
                    var existingIndex = _signals.FindIndex(s => s.Id == signal.Id);
                    if (existingIndex >= 0)
                    {
                        signal.LastModified = DateTime.Now;
                        _signals[existingIndex] = signal;
                    }
                    else
                    {
                        return Task.FromResult(false);
                    }
                }

                return Task.FromResult(true);
            }
            catch (Exception)
            {
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Deletes a trading signal by ID
        /// </summary>
        public Task<bool> DeleteSignalAsync(int signalId)
        {
            var signal = _signals.FirstOrDefault(s => s.Id == signalId);
            if (signal == null)
            {
                return Task.FromResult(false);
            }

            _signals.Remove(signal);
            return Task.FromResult(true);
        }

        /// <summary>
        /// Validates a stock symbol against the Alpha Vantage API
        /// </summary>
        public async Task<(bool IsValid, string Message)> ValidateSymbolAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return (false, "Symbol cannot be empty.");
            }

            try
            {
                // Search for the symbol using the Alpha Vantage service
                var searchResults = await _alphaVantageService.SearchSymbolsAsync(symbol.ToUpper());

                if (searchResults != null && searchResults.Any(r =>
                    string.Equals(r.Symbol, symbol, StringComparison.OrdinalIgnoreCase)))
                {
                    var match = searchResults.First(r =>
                        string.Equals(r.Symbol, symbol, StringComparison.OrdinalIgnoreCase));
                    return (true, $"Valid symbol: {match.Name}");
                }

                return (false, "Symbol not found in market data.");
            }
            catch (Exception ex)
            {
                return (false, $"Validation error: {ex.Message}");
            }
        }

        /// <summary>
        /// Evaluates a trading signal against current market data
        /// </summary>
        /// <remarks>
        /// Evaluation logic: Conditions are evaluated in order. The first condition's logical operator is ignored.
        /// For subsequent conditions, if the logical operator is AND, both the previous result and current condition
        /// must be true. If OR, either the previous result or current condition can be true.
        /// </remarks>
        public async Task<bool> EvaluateSignalAsync(TradingSignal signal)
        {
            if (signal == null || !signal.IsEnabled)
            {
                return false;
            }

            try
            {
                // Evaluate each symbol
                foreach (var symbolConfig in signal.Symbols.Where(s => s.IsActive))
                {
                    var indicatorValues = await _technicalIndicatorService.GetIndicatorsForPrediction(
                        symbolConfig.Symbol, "1day");

                    // Evaluate conditions in order with proper chaining
                    var orderedConditions = signal.Conditions.OrderBy(c => c.Order).ToList();
                    
                    if (!orderedConditions.Any())
                    {
                        continue;
                    }

                    // Evaluate first condition (its logical operator is ignored)
                    bool cumulativeResult = false;
                    bool firstCondition = true;

                    foreach (var condition in orderedConditions)
                    {
                        if (!indicatorValues.TryGetValue(condition.Indicator, out var value))
                        {
                            // If indicator not found, treat as false for this condition
                            if (firstCondition)
                            {
                                cumulativeResult = false;
                                firstCondition = false;
                            }
                            else if (condition.LogicalOperator == "AND")
                            {
                                cumulativeResult = false;
                            }
                            // For OR, keep the previous result
                            continue;
                        }

                        bool conditionMet = EvaluateCondition(value, condition.ComparisonOperator, condition.ThresholdValue);

                        if (firstCondition)
                        {
                            cumulativeResult = conditionMet;
                            firstCondition = false;
                        }
                        else
                        {
                            // Apply logical operator to chain with previous result
                            if (condition.LogicalOperator == "OR")
                            {
                                cumulativeResult = cumulativeResult || conditionMet;
                            }
                            else // AND (default)
                            {
                                cumulativeResult = cumulativeResult && conditionMet;
                            }
                        }
                    }

                    // If conditions are met for this symbol, return true
                    if (cumulativeResult)
                    {
                        return true;
                    }
                }

                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private bool EvaluateCondition(double value, string comparisonOperator, double threshold)
        {
            return comparisonOperator switch
            {
                ">" => value > threshold,
                "<" => value < threshold,
                ">=" => value >= threshold,
                "<=" => value <= threshold,
                "=" => Math.Abs(value - threshold) < 0.001,
                "!=" => Math.Abs(value - threshold) >= 0.001,
                _ => false
            };
        }
    }
}
