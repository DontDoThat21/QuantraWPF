using System.ComponentModel.DataAnnotations;
using Quantra.Configuration.Validation;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// Trading configuration
    /// </summary>
    public class TradingConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable paper trading
        /// </summary>
        public bool EnablePaperTrading
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Risk level (Low, Medium, High, Aggressive)
        /// </summary>
        [Required]
        public string RiskLevel
        {
            get => Get("Low");
            set => Set(value);
        }
        
        /// <summary>
        /// Account size
        /// </summary>
        [ConfigurationRange(100, 100000000)]
        public double AccountSize
        {
            get => Get(100000.0);
            set => Set(value);
        }
        
        /// <summary>
        /// Base risk percentage (0.01 = 1%)
        /// </summary>
        [ConfigurationRange(0.001, 0.1)]
        public double BaseRiskPercentage
        {
            get => Get(0.01);
            set => Set(value);
        }
        
        /// <summary>
        /// Position sizing method (FixedRisk, FixedSize, Volatility, Kelly)
        /// </summary>
        [Required]
        public string PositionSizingMethod
        {
            get => Get("FixedRisk");
            set => Set(value);
        }
        
        /// <summary>
        /// Maximum position size as percentage of account (0.1 = 10%)
        /// </summary>
        [ConfigurationRange(0.01, 1.0)]
        public double MaxPositionSizePercent
        {
            get => Get(0.1);
            set => Set(value);
        }
        
        /// <summary>
        /// Fixed trade amount
        /// </summary>
        [ConfigurationRange(100, 1000000)]
        public double FixedTradeAmount
        {
            get => Get(5000.0);
            set => Set(value);
        }
        
        /// <summary>
        /// Use volatility-based sizing
        /// </summary>
        public bool UseVolatilityBasedSizing
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// ATR multiple for volatility sizing
        /// </summary>
        [ConfigurationRange(0.5, 5.0)]
        public double ATRMultiple
        {
            get => Get(2.0);
            set => Set(value);
        }
        
        /// <summary>
        /// Use Kelly criterion for position sizing
        /// </summary>
        public bool UseKellyCriterion
        {
            get => Get(false);
            set => Set(value);
        }
        
        /// <summary>
        /// Historical win rate for Kelly criterion
        /// </summary>
        [ConfigurationRange(0.1, 0.95)]
        public double HistoricalWinRate
        {
            get => Get(0.55);
            set => Set(value);
        }
        
        /// <summary>
        /// Historical reward/risk ratio for Kelly criterion
        /// </summary>
        [ConfigurationRange(0.5, 10.0)]
        public double HistoricalRewardRiskRatio
        {
            get => Get(2.0);
            set => Set(value);
        }
        
        /// <summary>
        /// Kelly fraction multiplier (0.5 = half Kelly)
        /// </summary>
        [ConfigurationRange(0.1, 1.0)]
        public double KellyFractionMultiplier
        {
            get => Get(0.5);
            set => Set(value);
        }
    }
}