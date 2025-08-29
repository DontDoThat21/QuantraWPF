using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace Quantra.Models
{
    /// <summary>
    /// Base class for trading strategy profiles that can be applied to prediction analysis
    /// </summary>
    public abstract class StrategyProfile : INotifyPropertyChanged
    {
        private string _name;
        private bool _isEnabled = true;
        private double _minConfidence = 0.6;
        private double _riskLevel = 0.5;

        /// <summary>
        /// The display name of this strategy
        /// </summary>
        public string Name
        {
            get => _name;
            set
            {
                if (_name != value)
                {
                    _name = value;
                    OnPropertyChanged(nameof(Name));
                }
            }
        }

        /// <summary>
        /// Whether this strategy profile is enabled and available for use
        /// </summary>
        public bool IsEnabled
        {
            get => _isEnabled;
            set
            {
                if (_isEnabled != value)
                {
                    _isEnabled = value;
                    OnPropertyChanged(nameof(IsEnabled));
                }
            }
        }

        /// <summary>
        /// Minimum confidence level required for signals (0.0-1.0)
        /// </summary>
        public double MinConfidence
        {
            get => _minConfidence;
            set
            {
                if (_minConfidence != value && value >= 0.0 && value <= 1.0)
                {
                    _minConfidence = value;
                    OnPropertyChanged(nameof(MinConfidence));
                }
            }
        }

        /// <summary>
        /// Risk level for this strategy (0.0=Low, 1.0=High)
        /// </summary>
        public double RiskLevel
        {
            get => _riskLevel;
            set
            {
                if (_riskLevel != value && value >= 0.0 && value <= 1.0)
                {
                    _riskLevel = value;
                    OnPropertyChanged(nameof(RiskLevel));
                }
            }
        }

        /// <summary>
        /// Description of the strategy including key indicators and signals used
        /// </summary>
        public virtual string Description { get; set; }

        /// <summary>
        /// Generate a trading signal based on historical price data
        /// </summary>
        /// <param name="prices">List of historical price data</param>
        /// <param name="index">Index to generate signal for (defaults to last candle)</param>
        /// <returns>"BUY", "SELL", or null for no signal</returns>
        public abstract string GenerateSignal(List<HistoricalPrice> prices, int? index = null);

        /// <summary>
        /// Validate if the current market conditions meet the strategy criteria
        /// </summary>
        /// <param name="indicators">Dictionary of indicator values</param>
        /// <returns>True if conditions are met, false otherwise</returns>
        public abstract bool ValidateConditions(Dictionary<string, double> indicators);

        /// <summary>
        /// Get the recommended stop-loss percentage for this strategy
        /// </summary>
        public virtual double GetStopLossPercentage() => 0.02 * (1 + RiskLevel);

        /// <summary>
        /// Get the recommended take-profit percentage for this strategy
        /// </summary>
        public virtual double GetTakeProfitPercentage() => 0.04 * (1 + RiskLevel);

        /// <summary>
        /// Returns technical indicators required by this strategy
        /// </summary>
        public virtual IEnumerable<string> RequiredIndicators { get; } = new List<string>();

        #region INotifyPropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}