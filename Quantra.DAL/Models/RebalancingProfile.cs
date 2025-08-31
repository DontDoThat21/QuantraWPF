using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Profile for portfolio rebalancing with target allocations and rules
    /// </summary>
    public class RebalancingProfile : INotifyPropertyChanged
    {
        private string _profileId = Guid.NewGuid().ToString();
        private string _name = "Default Rebalancing Profile";
        private Dictionary<string, double> _targetAllocations = new Dictionary<string, double>();
        private double _tolerancePercentage = 0.02; // 2% default
        private RebalancingRiskLevel _riskLevel = RebalancingRiskLevel.Balanced;
        private RebalancingSchedule _schedule = RebalancingSchedule.Monthly;
        private DateTime? _lastRebalanceDate;
        private bool _enableMarketConditionAdjustments = true;
        private double _volatilityThreshold = 25.0; // VIX level above which to reduce risk assets
        private double _marketTrendSensitivity = 0.5; // 0-1 scale
        private double _maxDeviationInAdverseConditions = 0.1; // 10% max deviation
        private DateTime? _nextScheduledRebalance;

        /// <summary>
        /// Unique identifier for this rebalancing profile
        /// </summary>
        public string ProfileId
        {
            get => _profileId;
            set
            {
                if (_profileId != value)
                {
                    _profileId = value;
                    OnPropertyChanged(nameof(ProfileId));
                }
            }
        }

        /// <summary>
        /// Display name for this rebalancing profile
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
        /// Target asset allocations (symbol to percentage mapping, should sum to 1.0)
        /// </summary>
        public Dictionary<string, double> TargetAllocations
        {
            get => _targetAllocations;
            set
            {
                _targetAllocations = new Dictionary<string, double>(value);
                OnPropertyChanged(nameof(TargetAllocations));
            }
        }

        /// <summary>
        /// Tolerance percentage before rebalancing is triggered (0.02 = 2%)
        /// </summary>
        public double TolerancePercentage
        {
            get => _tolerancePercentage;
            set
            {
                if (_tolerancePercentage != value && value > 0)
                {
                    _tolerancePercentage = value;
                    OnPropertyChanged(nameof(TolerancePercentage));
                }
            }
        }

        /// <summary>
        /// Risk profile for this allocation strategy
        /// </summary>
        public RebalancingRiskLevel RiskLevel
        {
            get => _riskLevel;
            set
            {
                if (_riskLevel != value)
                {
                    _riskLevel = value;
                    OnPropertyChanged(nameof(RiskLevel));
                }
            }
        }

        /// <summary>
        /// Schedule for automatic portfolio rebalancing
        /// </summary>
        public RebalancingSchedule Schedule
        {
            get => _schedule;
            set
            {
                if (_schedule != value)
                {
                    _schedule = value;
                    OnPropertyChanged(nameof(Schedule));
                    CalculateNextRebalanceDate();
                }
            }
        }

        /// <summary>
        /// Date of the last portfolio rebalance
        /// </summary>
        public DateTime? LastRebalanceDate
        {
            get => _lastRebalanceDate;
            set
            {
                if (_lastRebalanceDate != value)
                {
                    _lastRebalanceDate = value;
                    OnPropertyChanged(nameof(LastRebalanceDate));
                    CalculateNextRebalanceDate();
                }
            }
        }

        /// <summary>
        /// Whether to adjust allocations based on market conditions
        /// </summary>
        public bool EnableMarketConditionAdjustments
        {
            get => _enableMarketConditionAdjustments;
            set
            {
                if (_enableMarketConditionAdjustments != value)
                {
                    _enableMarketConditionAdjustments = value;
                    OnPropertyChanged(nameof(EnableMarketConditionAdjustments));
                }
            }
        }

        /// <summary>
        /// Volatility index threshold above which to reduce risk assets
        /// </summary>
        public double VolatilityThreshold
        {
            get => _volatilityThreshold;
            set
            {
                if (_volatilityThreshold != value)
                {
                    _volatilityThreshold = value;
                    OnPropertyChanged(nameof(VolatilityThreshold));
                }
            }
        }

        /// <summary>
        /// Sensitivity to market trends (0-1 scale, 1 being most sensitive)
        /// </summary>
        public double MarketTrendSensitivity
        {
            get => _marketTrendSensitivity;
            set
            {
                if (_marketTrendSensitivity != value && value >= 0 && value <= 1)
                {
                    _marketTrendSensitivity = value;
                    OnPropertyChanged(nameof(MarketTrendSensitivity));
                }
            }
        }

        /// <summary>
        /// Maximum deviation allowed from target allocations during adverse market conditions
        /// </summary>
        public double MaxDeviationInAdverseConditions
        {
            get => _maxDeviationInAdverseConditions;
            set
            {
                if (_maxDeviationInAdverseConditions != value && value > 0)
                {
                    _maxDeviationInAdverseConditions = value;
                    OnPropertyChanged(nameof(MaxDeviationInAdverseConditions));
                }
            }
        }

        /// <summary>
        /// Next scheduled rebalance date based on last rebalance and schedule
        /// </summary>
        public DateTime? NextScheduledRebalance
        {
            get => _nextScheduledRebalance;
            private set
            {
                if (_nextScheduledRebalance != value)
                {
                    _nextScheduledRebalance = value;
                    OnPropertyChanged(nameof(NextScheduledRebalance));
                }
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        /// <summary>
        /// Creates a new rebalancing profile
        /// </summary>
        public RebalancingProfile() { }

        /// <summary>
        /// Creates a new rebalancing profile with specified name and allocations
        /// </summary>
        public RebalancingProfile(string name, Dictionary<string, double> allocations)
        {
            Name = name;
            TargetAllocations = new Dictionary<string, double>(allocations);
        }

        /// <summary>
        /// Creates a copy of this rebalancing profile
        /// </summary>
        public RebalancingProfile Clone()
        {
            return new RebalancingProfile
            {
                Name = this.Name + " (Copy)",
                TargetAllocations = new Dictionary<string, double>(this.TargetAllocations),
                TolerancePercentage = this.TolerancePercentage,
                RiskLevel = this.RiskLevel,
                Schedule = this.Schedule,
                LastRebalanceDate = this.LastRebalanceDate,
                EnableMarketConditionAdjustments = this.EnableMarketConditionAdjustments,
                VolatilityThreshold = this.VolatilityThreshold,
                MarketTrendSensitivity = this.MarketTrendSensitivity,
                MaxDeviationInAdverseConditions = this.MaxDeviationInAdverseConditions
            };
        }

        /// <summary>
        /// Validates that allocations sum approximately to 1.0 (100%)
        /// </summary>
        public bool ValidateAllocations()
        {
            if (TargetAllocations == null || TargetAllocations.Count == 0)
                return false;
                
            double total = TargetAllocations.Values.Sum();
            return Math.Abs(total - 1.0) < 0.0001;
        }

        /// <summary>
        /// Adjusts target allocations based on current market conditions
        /// </summary>
        /// <param name="marketConditions">Current market metrics</param>
        /// <returns>Adjusted allocations</returns>
        public Dictionary<string, double> GetMarketAdjustedAllocations(MarketConditions marketConditions)
        {
            if (!EnableMarketConditionAdjustments || marketConditions == null)
                return new Dictionary<string, double>(TargetAllocations);
                
            // Create a copy of the target allocations
            var adjustedAllocations = new Dictionary<string, double>(TargetAllocations);
            
            // Determine if market conditions warrant adjustments
            bool highVolatility = marketConditions.VolatilityIndex > VolatilityThreshold;
            bool bearishTrend = marketConditions.MarketTrend < -0.1 * MarketTrendSensitivity;
            
            if (highVolatility || bearishTrend)
            {
                // Adjust based on risk level and asset types
                foreach (var symbol in adjustedAllocations.Keys.ToList())
                {
                    bool isDefensiveAsset = marketConditions.IsDefensiveAsset(symbol);
                    
                    // Adjust allocation based on asset type and market conditions
                    if (isDefensiveAsset)
                    {
                        // Increase allocation to defensive assets
                        adjustedAllocations[symbol] *= 1 + (MaxDeviationInAdverseConditions * 0.5);
                    }
                    else
                    {
                        // Decrease allocation to risk assets
                        adjustedAllocations[symbol] *= 1 - (MaxDeviationInAdverseConditions * 0.5);
                    }
                }
                
                // Normalize allocations to ensure they still sum to 1.0
                NormalizeAllocations(adjustedAllocations);
            }
            
            return adjustedAllocations;
        }
        
        /// <summary>
        /// Calculates the next scheduled rebalance date based on rebalancing schedule
        /// </summary>
        private void CalculateNextRebalanceDate()
        {
            if (!LastRebalanceDate.HasValue)
            {
                NextScheduledRebalance = DateTime.Now; // Rebalance immediately if never rebalanced
                return;
            }
            
            DateTime nextDate;
            DateTime baseDate = LastRebalanceDate.Value;
            
            switch (Schedule)
            {
                case RebalancingSchedule.Daily:
                    nextDate = baseDate.AddDays(1);
                    break;
                    
                case RebalancingSchedule.Weekly:
                    nextDate = baseDate.AddDays(7);
                    break;
                    
                case RebalancingSchedule.Monthly:
                    nextDate = baseDate.AddMonths(1);
                    break;
                    
                case RebalancingSchedule.Quarterly:
                    nextDate = baseDate.AddMonths(3);
                    break;
                    
                case RebalancingSchedule.Annually:
                    nextDate = baseDate.AddYears(1);
                    break;
                    
                default:
                    nextDate = baseDate.AddMonths(1); // Default to monthly
                    break;
            }
            
            NextScheduledRebalance = nextDate;
        }
        
        /// <summary>
        /// Normalizes allocations to ensure they sum to 1.0
        /// </summary>
        private void NormalizeAllocations(Dictionary<string, double> allocations)
        {
            double total = allocations.Values.Sum();
            
            if (Math.Abs(total - 1.0) > 0.0001)
            {
                foreach (var symbol in allocations.Keys.ToList())
                {
                    allocations[symbol] = allocations[symbol] / total;
                }
            }
        }
        
        /// <summary>
        /// Raises the PropertyChanged event
        /// </summary>
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
    
    /// <summary>
    /// Risk levels for portfolio rebalancing profiles
    /// </summary>
    public enum RebalancingRiskLevel
    {
        /// <summary>
        /// Conservative - prioritizes capital preservation
        /// </summary>
        Conservative = 0,
        
        /// <summary>
        /// Balanced - equal emphasis on growth and preservation
        /// </summary>
        Balanced = 1,
        
        /// <summary>
        /// Growth - emphasis on capital appreciation with moderate risk
        /// </summary>
        Growth = 2,
        
        /// <summary>
        /// Aggressive - maximizes growth potential with higher risk
        /// </summary>
        Aggressive = 3
    }
    
    /// <summary>
    /// Schedule frequency for portfolio rebalancing
    /// </summary>
    public enum RebalancingSchedule
    {
        /// <summary>
        /// Rebalance daily
        /// </summary>
        Daily = 0,
        
        /// <summary>
        /// Rebalance weekly
        /// </summary>
        Weekly = 1,
        
        /// <summary>
        /// Rebalance monthly
        /// </summary>
        Monthly = 2,
        
        /// <summary>
        /// Rebalance quarterly
        /// </summary>
        Quarterly = 3,
        
        /// <summary>
        /// Rebalance annually
        /// </summary>
        Annually = 4
    }
}