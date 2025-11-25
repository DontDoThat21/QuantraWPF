using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy that aggregates signals from multiple underlying strategies
    /// </summary>
    public class AggregatedStrategyProfile : TradingStrategyProfile
    {
        public enum AggregationMethod
        {
            MajorityVote,      // Simple majority of strategies determine the signal
            Consensus,         // Requires a certain percentage agreement to generate signal
            WeightedVote,      // Like majority vote but strategies have weights
            PriorityBased      // Higher priority strategies can override lower ones
        }

        private ObservableCollection<StrategyWeight> _strategies;
        private AggregationMethod _aggregationMethod;
        private double _consensusThreshold;

        /// <summary>
        /// Collection of strategies with their weights
        /// </summary>
        public ObservableCollection<StrategyWeight> Strategies
        {
            get => _strategies;
            set
            {
                _strategies = value;
                OnPropertyChanged(nameof(Strategies));
            }
        }

        /// <summary>
        /// Method used to aggregate signals from multiple strategies
        /// </summary>
        public AggregationMethod Method
        {
            get => _aggregationMethod;
            set
            {
                _aggregationMethod = value;
                OnPropertyChanged(nameof(Method));
            }
        }

        /// <summary>
        /// Threshold required for consensus (0.0-1.0, represents percentage agreement)
        /// </summary>
        public double ConsensusThreshold
        {
            get => _consensusThreshold;
            set
            {
                if (_consensusThreshold != value && value >= 0.0 && value <= 1.0)
                {
                    _consensusThreshold = value;
                    OnPropertyChanged(nameof(ConsensusThreshold));
                }
            }
        }

        /// <summary>
        /// Initializes a new instance of the AggregatedStrategyProfile
        /// </summary>
        public AggregatedStrategyProfile()
        {
            Name = "Aggregated Strategy";
            Description = "Combines signals from multiple strategies based on configurable aggregation methods.";
            _strategies = new ObservableCollection<StrategyWeight>();
            _aggregationMethod = AggregationMethod.MajorityVote;
            _consensusThreshold = 0.75;
            RiskLevel = 0.5;
            MinConfidence = 0.6;
        }

        /// <summary>
        /// Add a strategy to the aggregation with a specified weight
        /// </summary>
        /// <param name="strategy">The strategy to add</param>
        /// <param name="weight">Weight of this strategy (higher values have more influence)</param>
        public void AddStrategy(StrategyProfile strategy, double weight = 1.0)
        {
            if (strategy != null && !_strategies.Any(s => s.Strategy == strategy))
            {
                _strategies.Add(new StrategyWeight { Strategy = strategy, Weight = weight });
                OnPropertyChanged(nameof(Strategies));
            }
        }

        /// <summary>
        /// Remove a strategy from the aggregation
        /// </summary>
        /// <param name="strategy">The strategy to remove</param>
        /// <returns>True if removed, false if not found</returns>
        public bool RemoveStrategy(StrategyProfile strategy)
        {
            var strategyToRemove = _strategies.FirstOrDefault(s => s.Strategy == strategy);
            if (strategyToRemove != null)
            {
                bool result = _strategies.Remove(strategyToRemove);
                if (result)
                    OnPropertyChanged(nameof(Strategies));
                return result;
            }
            return false;
        }

        /// <summary>
        /// Set the weight for a specific strategy
        /// </summary>
        /// <param name="strategy">The strategy to update</param>
        /// <param name="weight">New weight value</param>
        /// <returns>True if updated, false if strategy not found</returns>
        public bool SetStrategyWeight(StrategyProfile strategy, double weight)
        {
            if (weight <= 0)
                return false;

            var strategyItem = _strategies.FirstOrDefault(s => s.Strategy == strategy);
            if (strategyItem != null)
            {
                strategyItem.Weight = weight;
                OnPropertyChanged(nameof(Strategies));
                return true;
            }
            return false;
        }

        /// <summary>
        /// Get all required indicators from all underlying strategies
        /// </summary>
        public override IEnumerable<string> RequiredIndicators =>
            _strategies.SelectMany(s => s.Strategy.RequiredIndicators).Distinct();

        /// <summary>
        /// Generate a trading signal by aggregating signals from all underlying strategies
        /// </summary>
        /// <param name="prices">Historical price data</param>
        /// <param name="index">Index to generate signal for (defaults to last candle)</param>
        /// <returns>"BUY", "SELL", null for no signal, or "EXIT" for exit signal</returns>
        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || !prices.Any() || _strategies.Count == 0)
                return null;

            // Collect signals from all strategies
            var signals = new Dictionary<StrategyProfile, string>();

            foreach (var strategyWeight in _strategies)
            {
                if (strategyWeight.Strategy.IsEnabled)
                {
                    string signal = strategyWeight.Strategy.GenerateSignal(prices, index);
                    if (signal != null)
                    {
                        signals[strategyWeight.Strategy] = signal;
                    }
                }
            }

            // If no strategies returned a signal, return null
            if (signals.Count == 0)
                return null;

            // Aggregate signals based on the selected method
            return AggregateSignals(signals);
        }

        /// <summary>
        /// Aggregate signals from multiple strategies based on the selected aggregation method
        /// </summary>
        /// <param name="signals">Dictionary of strategies and their signals</param>
        /// <returns>The aggregated signal</returns>
        private string AggregateSignals(Dictionary<StrategyProfile, string> signals)
        {
            switch (_aggregationMethod)
            {
                case AggregationMethod.MajorityVote:
                    return MajorityVoteAggregation(signals);

                case AggregationMethod.Consensus:
                    return ConsensusAggregation(signals);

                case AggregationMethod.WeightedVote:
                    return WeightedVoteAggregation(signals);

                case AggregationMethod.PriorityBased:
                    return PriorityBasedAggregation(signals);

                default:
                    return MajorityVoteAggregation(signals);
            }
        }

        private string MajorityVoteAggregation(Dictionary<StrategyProfile, string> signals)
        {
            var signalCounts = signals.Values
                .GroupBy(signal => signal)
                .ToDictionary(g => g.Key, g => g.Count());

            // Find the signal with the maximum count
            if (signalCounts.Any())
            {
                var maxSignal = signalCounts.OrderByDescending(kv => kv.Value).First();
                return maxSignal.Key;
            }

            return null;
        }

        private string ConsensusAggregation(Dictionary<StrategyProfile, string> signals)
        {
            var signalCounts = signals.Values
                .GroupBy(signal => signal)
                .ToDictionary(g => g.Key, g => g.Count());

            // Find the signal with the maximum count
            if (signalCounts.Any())
            {
                var maxSignal = signalCounts.OrderByDescending(kv => kv.Value).First();
                double consensusPercentage = (double)maxSignal.Value / signals.Count;

                // Only return the signal if it meets the consensus threshold
                if (consensusPercentage >= ConsensusThreshold)
                {
                    return maxSignal.Key;
                }
            }

            return null;
        }

        private string WeightedVoteAggregation(Dictionary<StrategyProfile, string> signals)
        {
            var weightedSignals = new Dictionary<string, double>();

            // Calculate weighted sum for each signal
            foreach (var signal in signals)
            {
                var strategyWeight = _strategies.FirstOrDefault(s => s.Strategy == signal.Key);
                double weight = strategyWeight?.Weight ?? 1.0;

                if (!weightedSignals.ContainsKey(signal.Value))
                {
                    weightedSignals[signal.Value] = 0;
                }
                weightedSignals[signal.Value] += weight;
            }

            // Find signal with highest weighted sum
            if (weightedSignals.Any())
            {
                return weightedSignals.OrderByDescending(kv => kv.Value).First().Key;
            }

            return null;
        }

        private string PriorityBasedAggregation(Dictionary<StrategyProfile, string> signals)
        {
            // Sort strategies by weight in descending order (higher weight = higher priority)
            var sortedStrategies = _strategies
                .OrderByDescending(s => s.Weight)
                .Select(s => s.Strategy);

            // Return the signal from highest priority strategy that generated a signal
            foreach (var strategy in sortedStrategies)
            {
                if (signals.TryGetValue(strategy, out string signal) && !string.IsNullOrEmpty(signal))
                {
                    return signal;
                }
            }

            return null;
        }

        /// <summary>
        /// Calculate a confidence score for the aggregated signal
        /// </summary>
        /// <param name="signals">Dictionary of strategies and their signals</param>
        /// <returns>Confidence score (0.0-1.0)</returns>
        public double CalculateConfidence(Dictionary<StrategyProfile, string> signals)
        {
            if (signals == null || !signals.Any())
                return 0.0;

            string aggregatedSignal = AggregateSignals(signals);
            if (aggregatedSignal == null)
                return 0.0;

            switch (_aggregationMethod)
            {
                case AggregationMethod.WeightedVote:
                    // Calculate weighted confidence
                    double totalWeight = _strategies.Sum(s => s.Strategy.IsEnabled ? s.Weight : 0);
                    if (totalWeight <= 0)
                        return 0.0;

                    double signalWeight = signals
                        .Where(s => s.Value == aggregatedSignal)
                        .Sum(s =>
                        {
                            var strategyWeight = _strategies.FirstOrDefault(sw => sw.Strategy == s.Key);
                            return strategyWeight?.Weight ?? 0;
                        });

                    return signalWeight / totalWeight;

                default:
                    // Calculate what percentage of strategies agree with the final signal
                    int matchCount = signals.Count(s => s.Value == aggregatedSignal);
                    return (double)matchCount / signals.Count;
            }
        }

        /// <summary>
        /// Validate if the current market conditions meet the strategy criteria
        /// </summary>
        /// <param name="indicators">Dictionary of indicator values</param>
        /// <returns>True if conditions are met, false otherwise</returns>
        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            // No strategies or no indicators means conditions aren't met
            if (_strategies.Count == 0 || indicators == null || !indicators.Any())
                return false;

            // Count how many strategies have their conditions met
            int validCount = 0;
            foreach (var strategyWeight in _strategies)
            {
                if (strategyWeight.Strategy.IsEnabled &&
                    strategyWeight.Strategy.ValidateConditions(indicators))
                {
                    validCount++;
                }
            }

            // Apply the same logic as with signal aggregation
            switch (_aggregationMethod)
            {
                case AggregationMethod.Consensus:
                    return (double)validCount / _strategies.Count >= ConsensusThreshold;

                case AggregationMethod.PriorityBased:
                    // If any strategy with above-average weight validates, return true
                    double avgWeight = _strategies.Average(s => s.Weight);
                    return _strategies
                        .Where(s => s.Weight > avgWeight && s.Strategy.IsEnabled)
                        .Any(s => s.Strategy.ValidateConditions(indicators));

                default:
                    // For majority and weighted voting, require at least half of strategies to validate
                    return validCount >= Math.Ceiling(_strategies.Count / 2.0);
            }
        }

        /// <summary>
        /// Dynamic stop loss based on aggregation of all strategy stop losses
        /// </summary>
        /// <returns>Aggregated stop loss percentage</returns>
        public override double GetStopLossPercentage()
        {
            if (_strategies.Count == 0)
                return base.GetStopLossPercentage();

            // Use the most conservative (highest) stop loss if we have buy signals
            // or the most aggressive (lowest) if we have sell signals
            double stopLoss = 0;

            switch (_aggregationMethod)
            {
                case AggregationMethod.WeightedVote:
                    // Weighted average of stop loss values
                    double totalWeight = _strategies.Sum(s => s.Strategy.IsEnabled ? s.Weight : 0);
                    if (totalWeight <= 0)
                        return base.GetStopLossPercentage();

                    stopLoss = _strategies.Sum(s =>
                        s.Strategy.IsEnabled ? s.Strategy.GetStopLossPercentage() * s.Weight : 0) / totalWeight;
                    break;

                default:
                    // Simple average
                    stopLoss = _strategies
                        .Where(s => s.Strategy.IsEnabled)
                        .Select(s => s.Strategy.GetStopLossPercentage())
                        .DefaultIfEmpty(base.GetStopLossPercentage())
                        .Average();
                    break;
            }

            // Apply risk level adjustment
            return stopLoss * (1 + (RiskLevel - 0.5) * 0.5);
        }

        /// <summary>
        /// Dynamic take profit based on aggregation of all strategy take profit values
        /// </summary>
        /// <returns>Aggregated take profit percentage</returns>
        public override double GetTakeProfitPercentage()
        {
            if (_strategies.Count == 0)
                return base.GetTakeProfitPercentage();

            double takeProfit;

            switch (_aggregationMethod)
            {
                case AggregationMethod.WeightedVote:
                    // Weighted average of take profit values
                    double totalWeight = _strategies.Sum(s => s.Strategy.IsEnabled ? s.Weight : 0);
                    if (totalWeight <= 0)
                        return base.GetTakeProfitPercentage();

                    takeProfit = _strategies.Sum(s =>
                        s.Strategy.IsEnabled ? s.Strategy.GetTakeProfitPercentage() * s.Weight : 0) / totalWeight;
                    break;

                default:
                    // Simple average
                    takeProfit = _strategies
                        .Where(s => s.Strategy.IsEnabled)
                        .Select(s => s.Strategy.GetTakeProfitPercentage())
                        .DefaultIfEmpty(base.GetTakeProfitPercentage())
                        .Average();
                    break;
            }

            // Apply risk level adjustment
            return takeProfit * (1 + (RiskLevel - 0.5) * 0.5);
        }
    }

    /// <summary>
    /// Represents a strategy with an associated weight for aggregation
    /// </summary>
    public class StrategyWeight : INotifyPropertyChanged
    {
        private StrategyProfile _strategy;
        private double _weight;

        /// <summary>
        /// The trading strategy
        /// </summary>
        public StrategyProfile Strategy
        {
            get => _strategy;
            set
            {
                _strategy = value;
                OnPropertyChanged(nameof(Strategy));
            }
        }

        /// <summary>
        /// Weight of this strategy in aggregation (higher values have more influence)
        /// </summary>
        public double Weight
        {
            get => _weight;
            set
            {
                if (value > 0 && _weight != value)
                {
                    _weight = value;
                    OnPropertyChanged(nameof(Weight));
                }
            }
        }

        public StrategyWeight()
        {
            _weight = 1.0;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}