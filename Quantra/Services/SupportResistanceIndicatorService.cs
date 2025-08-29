using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Media;
using Quantra.Models;
using Quantra.Services.Interfaces;

namespace Quantra.Services
{
    /// <summary>
    /// Support/Resistance level indicator for integration with the technical indicator system
    /// </summary>
    public class SupportResistanceIndicatorService : IIndicator
    {
        // todo what is a strategy and how is it different
        // from a service? doesnt this belong in services?
        private SupportResistanceStrategy _strategy;
        private string _name;
        private string _description;
        private string _category;
        private bool _isComposable;

        public SupportResistanceIndicatorService()
        {
            _strategy = new SupportResistanceStrategy();

            _name = "Support/Resistance Levels";
            _description = "Identifies key support and resistance levels using multiple techniques including price action, pivot points, Fibonacci retracements, and volume profile analysis.";
            _category = "Price Levels";
            _isComposable = true;

            // Initialize Parameters property
            Parameters = new Dictionary<string, IndicatorParameter>
            {
                { "UsePriceAction", new IndicatorParameter
                {
                    Name = "Use Price Action",
                    Description = "Detect levels using price swing highs/lows",
                    DefaultValue = true,
                    Value = true,
                    ParameterType = typeof(bool),
                    IsOptional = false
                }},
                { "UsePivotPoints", new IndicatorParameter
                {
                    Name = "Use Pivot Points",
                    Description = "Detect levels using pivot point calculations",
                    DefaultValue = true,
                    Value = true,
                    ParameterType = typeof(bool),
                    IsOptional = false
                }},
                { "UseFibonacciLevels", new IndicatorParameter
                {
                    Name = "Use Fibonacci Levels",
                    Description = "Detect levels using Fibonacci retracements",
                    DefaultValue = true,
                    Value = true,
                    ParameterType = typeof(bool),
                    IsOptional = false
                }},
                { "UseVolumeProfile", new IndicatorParameter
                {
                    Name = "Use Volume Profile",
                    Description = "Detect levels using volume profile analysis",
                    DefaultValue = true,
                    Value = true,
                    ParameterType = typeof(bool),
                    IsOptional = false
                }},
                { "LookbackPeriods", new IndicatorParameter
                {
                    Name = "Lookback Periods",
                    Description = "Number of historical bars to analyze",
                    DefaultValue = 100,
                    Value = 100,
                    ParameterType = typeof(int),
                    MinValue = 20,
                    MaxValue = 200,
                    IsOptional = false
                }},
                { "MinTouchesToConfirm", new IndicatorParameter
                {
                    Name = "Min Touches to Confirm",
                    Description = "Minimum touches required to confirm a level",
                    DefaultValue = 2,
                    Value = 2,
                    ParameterType = typeof(int),
                    MinValue = 1,
                    MaxValue = 5,
                    IsOptional = false
                }},
                { "LevelTolerance", new IndicatorParameter
                {
                    Name = "Level Tolerance (%)",
                    Description = "Percentage tolerance for grouping nearby price levels",
                    DefaultValue = 0.5,
                    Value = 0.5,
                    ParameterType = typeof(double),
                    MinValue = 0.1,
                    MaxValue = 5.0,
                    IsOptional = false
                }}
            };
        }

        #region IIndicator Implementation

        public string Id => "SupportResistanceLevels";

        public string Name
        {
            get => _name;
            set => _name = value;
        }

        public string Description
        {
            get => _description;
            set => _description = value;
        }

        public string Category
        {
            get => _category;
            set => _category = value;
        }

        public bool IsComposable => _isComposable;

        public Dictionary<string, IndicatorParameter> Parameters { get; }

        public async Task<Dictionary<string, double>> CalculateAsync(List<HistoricalPrice> historicalData)
        {
            if (historicalData == null || historicalData.Count < 10)
                return new Dictionary<string, double>();

            // Configure the strategy based on parameters
            UpdateStrategyFromParameters();

            // Generate signals using the strategy (this will detect levels)
            _strategy.GenerateSignal(historicalData);

            // Get detected levels
            var levels = _strategy.GetDetectedLevels();

            // Convert levels to indicator values
            var result = new Dictionary<string, double>();

            // Add the current price for reference
            double currentPrice = historicalData.Last().Close;
            result.Add("CurrentPrice", currentPrice);

            // Find nearest support below current price
            var nearestSupport = levels
                .Where(l => l.IsSupport && l.Price < currentPrice)
                .OrderByDescending(l => l.Price)
                .FirstOrDefault();

            // Find nearest resistance above current price
            var nearestResistance = levels
                .Where(l => l.IsResistance && l.Price > currentPrice)
                .OrderBy(l => l.Price)
                .FirstOrDefault();

            // Add nearest levels to result
            if (nearestSupport != null)
            {
                result.Add("NearestSupport", nearestSupport.Price);
                result.Add("SupportStrength", nearestSupport.Strength);
            }

            if (nearestResistance != null)
            {
                result.Add("NearestResistance", nearestResistance.Price);
                result.Add("ResistanceStrength", nearestResistance.Strength);
            }

            // Calculate distance to nearest levels
            if (nearestSupport != null)
            {
                double distanceToSupport = (currentPrice - nearestSupport.Price) / currentPrice * 100.0;
                result.Add("DistanceToSupport", distanceToSupport);
            }

            if (nearestResistance != null)
            {
                double distanceToResistance = (nearestResistance.Price - currentPrice) / currentPrice * 100.0;
                result.Add("DistanceToResistance", distanceToResistance);
            }

            // Calculate trading signal based on price position between support/resistance
            if (nearestSupport != null && nearestResistance != null)
            {
                // Calculate position ratio between nearest support and resistance (0-1)
                double range = nearestResistance.Price - nearestSupport.Price;
                double position = (currentPrice - nearestSupport.Price) / range;
                result.Add("RelativePosition", position);

                // Generate a signal value (-100 to +100)
                // Closer to support is more bullish, closer to resistance is more bearish
                double signal = 100 - position * 200;
                result.Add("Signal", signal);
            }

            return result;
        }

        public IEnumerable<string> GetDependencies()
        {
            // This indicator has no dependencies on other indicators
            return Enumerable.Empty<string>();
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Update strategy configuration from indicator parameters
        /// </summary>
        private void UpdateStrategyFromParameters()
        {
            // Apply parameters to strategy
            _strategy.UsePriceAction = (bool)Parameters["UsePriceAction"].Value;
            _strategy.UsePivotPoints = (bool)Parameters["UsePivotPoints"].Value;
            _strategy.UseFibonacciLevels = (bool)Parameters["UseFibonacciLevels"].Value;
            _strategy.UseVolumeProfile = (bool)Parameters["UseVolumeProfile"].Value;
            _strategy.LookbackPeriods = (int)Parameters["LookbackPeriods"].Value;
            _strategy.MinTouchesToConfirm = (int)Parameters["MinTouchesToConfirm"].Value;
            _strategy.LevelTolerance = (double)Parameters["LevelTolerance"].Value;
        }

        #endregion
    }
}