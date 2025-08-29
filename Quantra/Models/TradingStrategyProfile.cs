using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace Quantra.Models
{
    /// <summary>
    /// Base class for trading strategies with extended functionality
    /// </summary>
    public abstract class TradingStrategyProfile : StrategyProfile, INotifyPropertyChanged
    {
        private string _name;
        private bool _isEnabled = true;
        private double _minConfidence = 0.6;
        private double _riskLevel = 0.5;
        private string _description;

        /// <summary>
        /// The display name of this strategy
        /// </summary>
        public new string Name
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
        public new bool IsEnabled
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
        public new double MinConfidence
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
        public new double RiskLevel
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
        public override string Description
        {
            get => _description;
            set
            {
                if (_description != value)
                {
                    _description = value;
                    OnPropertyChanged(nameof(Description));
                }
            }
        }

        /// <summary>
        /// Generate a trading signal based on historical price data
        /// </summary>
        /// <param name="prices">List of historical price data</param>
        /// <param name="index">Index to generate signal for (defaults to last candle)</param>
        /// <returns>"BUY", "SELL", or null for no signal</returns>
        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (index.HasValue)
            {
                // Implement the logic to generate a signal based on the provided index
                return "BUY"; // Placeholder return value
            }
            return null;
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        
        protected override void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}