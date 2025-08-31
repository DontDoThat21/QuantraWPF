using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using Quantra.DAL.Services.Interfaces;
using Quantra.Utilities;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy based on comprehensive support and resistance level identification
    /// using price action analysis, pivot points, Fibonacci retracements, and volume-based methods
    /// </summary>
    public class SupportResistanceStrategy : TradingStrategyProfile
    {
        private int _lookbackPeriods = 100;
        private int _minTouchesToConfirm = 2;
        private double _levelTolerance = 0.5; // Percentage of price
        private int _breakoutConfirmation = 2; // Candles to confirm breakout
        private bool _useVolumeConfirmation = true;
        private double _volumeThreshold = 1.5; // Multiplier of average volume
        
        // Detection method settings
        private bool _usePriceAction = true;
        private bool _usePivotPoints = true;
        private bool _useFibonacciLevels = true;
        private bool _useVolumeProfile = true;

        // The analyzer that handles level detection
        private PriceLevelAnalyzer _analyzer;
        
        // Cached levels for UI visualization
        private List<PriceLevelAnalyzer.PriceLevel> _cachedLevels;
        
        // Visual representations of detected levels
        private List<PriceLevelVisualization.VisualPriceLevel> _visualLevels;

        public SupportResistanceStrategy()
        {
            Name = "Support/Resistance Levels";
            Description = "Identifies key support and resistance levels using multiple techniques: " +
                          "1) Price action analysis (swing highs/lows) " +
                          "2) Pivot points (standard, Woodie's, Camarilla) " +
                          "3) Fibonacci retracements " +
                          "4) Volume-based levels. " +
                          "Generates signals on level breakouts, bounces, and rejections with volume confirmation.";
            
            RiskLevel = 0.6;
            MinConfidence = 0.7;
            
            // Initialize the price level analyzer
            _analyzer = new PriceLevelAnalyzer(_lookbackPeriods, _minTouchesToConfirm, _levelTolerance);
            
            // Initialize collections
            _cachedLevels = new List<PriceLevelAnalyzer.PriceLevel>();
            _visualLevels = new List<PriceLevelVisualization.VisualPriceLevel>();
        }

        #region Properties

        /// <summary>
        /// Number of historical bars to analyze for level detection
        /// </summary>
        public int LookbackPeriods
        {
            get => _lookbackPeriods;
            set
            {
                if (value > 10 && _lookbackPeriods != value)
                {
                    _lookbackPeriods = value;
                    _analyzer = new PriceLevelAnalyzer(_lookbackPeriods, _minTouchesToConfirm, _levelTolerance);
                    OnPropertyChanged(nameof(LookbackPeriods));
                }
            }
        }

        /// <summary>
        /// Minimum number of touches required to confirm a support/resistance level
        /// </summary>
        public int MinTouchesToConfirm
        {
            get => _minTouchesToConfirm;
            set
            {
                if (value > 0 && _minTouchesToConfirm != value)
                {
                    _minTouchesToConfirm = value;
                    _analyzer = new PriceLevelAnalyzer(_lookbackPeriods, _minTouchesToConfirm, _levelTolerance);
                    OnPropertyChanged(nameof(MinTouchesToConfirm));
                }
            }
        }

        /// <summary>
        /// Percentage tolerance for grouping nearby price levels
        /// </summary>
        public double LevelTolerance
        {
            get => _levelTolerance;
            set
            {
                if (value > 0 && value < 5 && _levelTolerance != value)
                {
                    _levelTolerance = value;
                    _analyzer = new PriceLevelAnalyzer(_lookbackPeriods, _minTouchesToConfirm, _levelTolerance);
                    OnPropertyChanged(nameof(LevelTolerance));
                }
            }
        }

        /// <summary>
        /// Number of candles required to confirm a breakout
        /// </summary>
        public int BreakoutConfirmation
        {
            get => _breakoutConfirmation;
            set
            {
                if (value >= 0 && _breakoutConfirmation != value)
                {
                    _breakoutConfirmation = value;
                    OnPropertyChanged(nameof(BreakoutConfirmation));
                }
            }
        }

        /// <summary>
        /// Whether to use volume for confirmation of breakouts
        /// </summary>
        public bool UseVolumeConfirmation
        {
            get => _useVolumeConfirmation;
            set
            {
                if (_useVolumeConfirmation != value)
                {
                    _useVolumeConfirmation = value;
                    OnPropertyChanged(nameof(UseVolumeConfirmation));
                }
            }
        }

        /// <summary>
        /// Minimum volume multiplier (compared to average) required for confirmation
        /// </summary>
        public double VolumeThreshold
        {
            get => _volumeThreshold;
            set
            {
                if (value > 0 && _volumeThreshold != value)
                {
                    _volumeThreshold = value;
                    OnPropertyChanged(nameof(VolumeThreshold));
                }
            }
        }
        
        /// <summary>
        /// Whether to use price action (swing highs/lows) for level detection
        /// </summary>
        public bool UsePriceAction
        {
            get => _usePriceAction;
            set
            {
                if (_usePriceAction != value)
                {
                    _usePriceAction = value;
                    OnPropertyChanged(nameof(UsePriceAction));
                }
            }
        }
        
        /// <summary>
        /// Whether to use pivot points for level detection
        /// </summary>
        public bool UsePivotPoints
        {
            get => _usePivotPoints;
            set
            {
                if (_usePivotPoints != value)
                {
                    _usePivotPoints = value;
                    OnPropertyChanged(nameof(UsePivotPoints));
                }
            }
        }
        
        /// <summary>
        /// Whether to use Fibonacci retracements for level detection
        /// </summary>
        public bool UseFibonacciLevels
        {
            get => _useFibonacciLevels;
            set
            {
                if (_useFibonacciLevels != value)
                {
                    _useFibonacciLevels = value;
                    OnPropertyChanged(nameof(UseFibonacciLevels));
                }
            }
        }
        
        /// <summary>
        /// Whether to use volume profile for level detection
        /// </summary>
        public bool UseVolumeProfile
        {
            get => _useVolumeProfile;
            set
            {
                if (_useVolumeProfile != value)
                {
                    _useVolumeProfile = value;
                    OnPropertyChanged(nameof(UseVolumeProfile));
                }
            }
        }

        #endregion

        public override IEnumerable<string> RequiredIndicators => new[] { "Price", "Volume" };

        #region Public Methods
        
        /// <summary>
        /// Get the most recent detected support/resistance levels
        /// </summary>
        /// <returns>List of detected price levels</returns>
        public List<PriceLevelAnalyzer.PriceLevel> GetDetectedLevels()
        {
            return _cachedLevels.ToList();
        }
        
        /// <summary>
        /// Get visual representations of support/resistance levels for chart rendering
        /// </summary>
        /// <returns>List of visual price levels</returns>
        public List<PriceLevelVisualization.VisualPriceLevel> GetVisualLevels()
        {
            return _visualLevels.ToList();
        }

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < LookbackPeriods + 5)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < LookbackPeriods || currentIndex >= prices.Count)
                return null;

            // Detect support and resistance levels using all methods
            DetectLevels(prices, currentIndex);
            if (_cachedLevels.Count == 0)
                return null;

            // Generate trading signals based on detected levels
            return AnalyzePriceLevelInteractions(prices, _cachedLevels, currentIndex);
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            // Simple validation - we need price data
            return indicators != null && indicators.ContainsKey("Price");
        }

        #endregion

        #region Private Methods
        
        /// <summary>
        /// Detect support and resistance levels from historical price data using all enabled methods
        /// </summary>
        private void DetectLevels(List<HistoricalPrice> prices, int currentIndex)
        {
            List<PriceLevelAnalyzer.PriceLevel> levels = new List<PriceLevelAnalyzer.PriceLevel>();
            
            // Apply each enabled detection method
            if (_usePriceAction)
            {
                levels.AddRange(_analyzer.DetectPriceActionLevels(prices, currentIndex));
            }
            
            if (_usePivotPoints)
            {
                levels.AddRange(_analyzer.CalculatePivotPoints(prices, currentIndex));
            }
            
            if (_useFibonacciLevels)
            {
                levels.AddRange(_analyzer.CalculateFibonacciLevels(prices, currentIndex));
            }
            
            if (_useVolumeProfile)
            {
                levels.AddRange(_analyzer.DetectVolumeLevels(prices, currentIndex));
            }
            
            // Update cached levels
            _cachedLevels = levels
                .OrderByDescending(l => l.Strength)
                .Take(10) // Limit to most significant levels
                .ToList();
                
            // Generate visual representations
            _visualLevels = PriceLevelVisualization.CreateVisuals(_cachedLevels);
        }
        
        /// <summary>
        /// Analyze price interactions with levels to generate signals
        /// </summary>
        private string AnalyzePriceLevelInteractions(List<HistoricalPrice> prices, List<PriceLevelAnalyzer.PriceLevel> levels, int currentIndex)
        {
            // Current and recent price action
            double currentPrice = prices[currentIndex].Close;
            double previousClose = prices[currentIndex - 1].Close;
            double highPrice = prices[currentIndex].High;
            double lowPrice = prices[currentIndex].Low;
            
            // Calculate average volume for comparison
            double avgVolume = prices
                .Skip(Math.Max(0, currentIndex - 20))
                .Take(20)
                .Average(p => p.Volume);
            
            // Volume confirmation check
            bool hasVolumeConfirmation = !UseVolumeConfirmation || 
                                       (prices[currentIndex].Volume > avgVolume * VolumeThreshold);
                                       
            // Find nearest support and resistance levels
            var nearestSupport = levels
                .Where(l => l.IsSupport && l.Price < currentPrice)
                .OrderByDescending(l => l.Price)
                .FirstOrDefault();
                
            var nearestResistance = levels
                .Where(l => l.IsResistance && l.Price > currentPrice)
                .OrderBy(l => l.Price)
                .FirstOrDefault();
                
            // Check for breakouts and bounces
            if (nearestResistance != null)
            {
                // Resistance breakout
                if (previousClose < nearestResistance.Price && 
                    currentPrice > nearestResistance.Price)
                {
                    // Confirm breakout if needed
                    if (ConfirmBreakout(prices, currentIndex, nearestResistance.Price, true) && hasVolumeConfirmation)
                    {
                        return "BUY";
                    }
                }
                
                // Resistance rejection
                if (highPrice >= nearestResistance.Price && 
                    currentPrice < nearestResistance.Price && 
                    hasVolumeConfirmation)
                {
                    return "SELL";
                }
            }
            
            if (nearestSupport != null)
            {
                // Support breakdown
                if (previousClose > nearestSupport.Price && 
                    currentPrice < nearestSupport.Price)
                {
                    // Confirm breakdown if needed
                    if (ConfirmBreakout(prices, currentIndex, nearestSupport.Price, false) && hasVolumeConfirmation)
                    {
                        return "SELL";
                    }
                }
                
                // Support bounce
                if (lowPrice <= nearestSupport.Price && 
                    currentPrice > nearestSupport.Price && 
                    hasVolumeConfirmation)
                {
                    return "BUY";
                }
            }
            
            return null;
        }
        
        /// <summary>
        /// Confirm a breakout of a price level
        /// </summary>
        private bool ConfirmBreakout(List<HistoricalPrice> prices, int currentIndex, double level, bool isBreakingUp)
        {
            // If no confirmation required, return true
            if (BreakoutConfirmation <= 0)
                return true;
                
            // Not enough bars to confirm
            if (currentIndex + BreakoutConfirmation >= prices.Count)
                return false;
                
            // Check future candles (if available) to see if breakout holds
            for (int i = 1; i <= BreakoutConfirmation; i++)
            {
                if (currentIndex + i >= prices.Count)
                    break;
                    
                // For upward breakout, confirm price stays above level
                if (isBreakingUp && prices[currentIndex + i].Close <= level)
                    return false;
                    
                // For downward breakout, confirm price stays below level
                if (!isBreakingUp && prices[currentIndex + i].Close >= level)
                    return false;
            }
            
            return true;
        }
        
        #endregion
    }
}