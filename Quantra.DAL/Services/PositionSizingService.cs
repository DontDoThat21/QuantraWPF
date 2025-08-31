using System;
using System.Linq;
using Quantra.DAL.Services.Interfaces;
using Quantra.Enums;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class PositionSizingService : IPositionSizingService
    {
        /// <summary>
        /// Calculates position size based on risk parameters
        /// </summary>
        public int CalculatePositionSizeByRisk(string symbol, double price, double stopLossPrice, double riskPercentage, double accountSize)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(symbol) || price <= 0 || riskPercentage <= 0 || accountSize <= 0)
                {
                    DatabaseMonolith.Log("Warning", "Invalid parameters for position sizing calculation");
                    return 0;
                }

                double riskAmount = accountSize * (riskPercentage / 100);
                double riskPerShare = Math.Abs(price - stopLossPrice);

                if (riskPerShare == 0)
                {
                    DatabaseMonolith.Log("Warning", $"Risk per share is zero for {symbol}, using 1% of price as default");
                    riskPerShare = price * 0.01;
                }

                int shares = (int)(riskAmount / riskPerShare);
                DatabaseMonolith.Log("Info", $"Position sizing for {symbol}: {shares} shares with {riskPercentage}% risk on ${accountSize:F2} account");
                return shares;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate position size for {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Calculates position size using adaptive risk adjustment
        /// </summary>
        public int CalculatePositionSizeByAdaptiveRisk(string symbol, double price, double stopLossPrice, 
            double riskPercentage, double accountSize, double volatility = 0, double winRate = 0.5)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(symbol) || price <= 0 || riskPercentage <= 0 || accountSize <= 0)
                {
                    DatabaseMonolith.Log("Warning", "Invalid parameters for adaptive position sizing calculation");
                    return 0;
                }

                // Base position size calculation
                double baseRiskAmount = accountSize * (riskPercentage / 100);
                double riskPerShare = Math.Abs(price - stopLossPrice);

                if (riskPerShare == 0)
                {
                    riskPerShare = price * 0.01; // Default 1% stop
                }

                // Adaptive adjustments based on volatility and win rate
                double volatilityAdjustment = 1.0;
                if (volatility > 0)
                {
                    // Reduce position size for high volatility (above 30%)
                    if (volatility > 0.30)
                        volatilityAdjustment = 0.7;
                    else if (volatility > 0.20)
                        volatilityAdjustment = 0.85;
                    else if (volatility < 0.10)
                        volatilityAdjustment = 1.15; // Slightly increase for low volatility
                }

                double winRateAdjustment = 1.0;
                if (winRate > 0.6)
                    winRateAdjustment = 1.2; // Increase position for high win rate
                else if (winRate < 0.4)
                    winRateAdjustment = 0.8; // Decrease position for low win rate

                double adjustedRiskAmount = baseRiskAmount * volatilityAdjustment * winRateAdjustment;
                int shares = (int)(adjustedRiskAmount / riskPerShare);

                DatabaseMonolith.Log("Info", $"Adaptive position sizing for {symbol}: {shares} shares (Vol adj: {volatilityAdjustment:F2}, WinRate adj: {winRateAdjustment:F2})");
                return shares;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate adaptive position size for {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Calculates position size using specified parameters and sizing method
        /// </summary>
        public int CalculatePositionSize(PositionSizingParameters parameters)
        {
            try
            {
                // Validate common parameters
                if (parameters == null || string.IsNullOrWhiteSpace(parameters.Symbol) || 
                    parameters.Price <= 0 || parameters.AccountSize <= 0)
                {
                    DatabaseMonolith.Log("Warning", "Invalid parameters for position sizing calculation");
                    return 0;
                }
                
                // Apply risk mode adjustments
                AdjustParametersForRiskMode(parameters);
                
                // Calculate position size based on the specified method
                int shares;
                switch (parameters.Method)
                {
                    case PositionSizingMethod.PercentageOfEquity:
                        shares = CalculatePositionSizeByEquityPercentage(parameters);
                        break;
                    case PositionSizingMethod.VolatilityBased:
                        shares = CalculatePositionSizeByVolatility(parameters);
                        break;
                    case PositionSizingMethod.KellyFormula:
                        shares = CalculatePositionSizeByKellyFormula(parameters);
                        break;
                    case PositionSizingMethod.FixedAmount:
                        shares = CalculatePositionSizeByFixedAmount(parameters);
                        break;
                    case PositionSizingMethod.TierBased:
                        shares = CalculatePositionSizeByTiers(parameters);
                        break;
                    case PositionSizingMethod.AdaptiveRisk:
                        shares = CalculatePositionSizeByAdaptiveRisk(parameters);
                        break;
                    case PositionSizingMethod.FixedRisk:
                    default:
                        shares = CalculatePositionSizeByFixedRisk(parameters);
                        break;
                }
                
                // Apply maximum position size constraint
                int maxShares = CalculateMaxPositionSize(parameters);
                shares = Math.Min(shares, maxShares);
                
                // Log the calculation
                string methodName = parameters.Method.ToString();
                DatabaseMonolith.Log("Info", $"Position sizing for {parameters.Symbol} using {methodName}: " +
                    $"{shares} shares at {parameters.Price:C2} with {parameters.RiskMode} risk mode");
                
                return shares;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate position size for {parameters.Symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Adjusts position sizing parameters based on risk mode
        /// </summary>
        private void AdjustParametersForRiskMode(PositionSizingParameters parameters)
        {
            switch (parameters.RiskMode)
            {
                case RiskMode.Conservative:
                    // Conservative mode - lower risk percentage and lower maximums
                    parameters.RiskPercentage *= 0.5; // Half the risk percentage
                    parameters.MaxPositionSizePercent *= 0.7; // 70% of normal maximum size
                    parameters.KellyFractionMultiplier = 0.3; // Lower Kelly fraction
                    parameters.ATRMultiple *= 0.7; // Tighter ATR multiple
                    break;
                
                case RiskMode.Moderate:
                    // Moderate mode - slightly reduced risk factors
                    parameters.RiskPercentage *= 0.8; // 80% of normal risk
                    parameters.KellyFractionMultiplier = 0.4; // Moderate Kelly fraction
                    break;
                
                case RiskMode.Aggressive:
                    // Aggressive mode - increased risk factors
                    parameters.RiskPercentage *= 1.5; // 150% of normal risk
                    parameters.MaxPositionSizePercent *= 1.3; // 130% of normal maximum size
                    parameters.KellyFractionMultiplier = 0.7; // Higher Kelly fraction
                    parameters.ATRMultiple *= 1.2; // Wider ATR multiple
                    break;
                
                case RiskMode.Normal:
                default:
                    // Normal mode - no adjustments needed
                    break;
            }
        }

        /// <summary>
        /// Calculates position size by equity percentage
        /// </summary>
        public int CalculatePositionSizeByEquityPercentage(PositionSizingParameters parameters)
        {
            double positionValue = parameters.AccountSize * (parameters.EquityPercentage / 100);
            return (int)(positionValue / parameters.Price);
        }

        /// <summary>
        /// Calculates position size based on volatility
        /// </summary>
        public int CalculatePositionSizeByVolatility(PositionSizingParameters parameters)
        {
            // Use ATR (Average True Range) for volatility-based sizing
            double atr = (double)(parameters.ATR ?? 0.0);
            double atrBasedRisk = atr * parameters.ATRMultiple;
            double riskAmount = parameters.AccountSize * (parameters.RiskPercentage / 100);
            return (int)(riskAmount / atrBasedRisk);
        }

        /// <summary>
        /// Calculates position size using Kelly Formula
        /// </summary>
        public int CalculatePositionSizeByKellyFormula(PositionSizingParameters parameters)
        {
            // Kelly formula: f = (bp - q) / b
            // where b = odds received on the wager, p = probability of winning, q = probability of losing
            double kellyFraction = (parameters.WinProbability * parameters.AvgWin - parameters.LossProbability) / parameters.AvgWin;
            
            // Apply multiplier to reduce risk (typically 0.25-0.5 of full Kelly)
            kellyFraction *= parameters.KellyFractionMultiplier;
            
            // Ensure fraction is within reasonable bounds
            kellyFraction = Math.Max(0, Math.Min(kellyFraction, 0.5)); // Cap at 50% of account
            
            double positionValue = parameters.AccountSize * kellyFraction;
            return (int)(positionValue / parameters.Price);
        }

        /// <summary>
        /// Calculates position size by fixed amount
        /// </summary>
        public int CalculatePositionSizeByFixedAmount(PositionSizingParameters parameters)
        {
            return (int)(parameters.FixedAmount / parameters.Price);
        }

        /// <summary>
        /// Calculates position size by tier system
        /// </summary>
        public int CalculatePositionSizeByTiers(PositionSizingParameters parameters)
        {
            // Implement tier-based position sizing
            // This could be based on account size tiers, volatility tiers, etc.
            
            double tierMultiplier = 1.0;
            
            // Account size tiers
            if (parameters.AccountSize < 10000)
                tierMultiplier = 0.5; // Small account - reduce size
            else if (parameters.AccountSize > 100000)
                tierMultiplier = 1.5; // Large account - can take bigger positions
            
            // Price tiers (adjust for stock price ranges)
            if (parameters.Price < 10)
                tierMultiplier *= 1.2; // Cheaper stocks - can buy more shares
            else if (parameters.Price > 500)
                tierMultiplier *= 0.8; // Expensive stocks - fewer shares
            
            double basePositionValue = parameters.AccountSize * (parameters.RiskPercentage / 100) * tierMultiplier;
            return (int)(basePositionValue / parameters.Price);
        }

        /// <summary>
        /// Calculates position size using adaptive risk management
        /// </summary>
        public int CalculatePositionSizeByAdaptiveRisk(PositionSizingParameters parameters)
        {
            // Base calculation using fixed risk
            int baseShares = CalculatePositionSizeByFixedRisk(parameters);
            
            // Apply adaptive adjustments
            double adaptiveMultiplier = 1.0;
            
            // Volatility adjustment
            if (parameters.CurrentVolatility > 0)
            {
                double volRatio = parameters.CurrentVolatility / parameters.BaselineVolatility;
                adaptiveMultiplier /= Math.Sqrt(volRatio); // Reduce size for higher volatility
            }
            
            // Market conditions adjustment
            if (parameters.MarketTrend == "Bearish")
                adaptiveMultiplier *= 0.8;
            else if (parameters.MarketTrend == "Bullish")
                adaptiveMultiplier *= 1.1;
            
            // Recent performance adjustment
            if (parameters.RecentWinRate > 0.6)
                adaptiveMultiplier *= 1.1;
            else if (parameters.RecentWinRate < 0.4)
                adaptiveMultiplier *= 0.9;
            
            return (int)(baseShares * adaptiveMultiplier);
        }

        /// <summary>
        /// Calculates position size by fixed risk
        /// </summary>
        public int CalculatePositionSizeByFixedRisk(PositionSizingParameters parameters)
        {
            double riskAmount = parameters.AccountSize * (parameters.RiskPercentage / 100);
            double riskPerShare = Math.Abs(parameters.Price - parameters.StopLossPrice);
            
            if (riskPerShare == 0)
            {
                // Default to 1% of price if no stop loss provided
                riskPerShare = parameters.Price * 0.01;
            }
            
            return (int)(riskAmount / riskPerShare);
        }

        /// <summary>
        /// Calculates maximum allowed position size
        /// </summary>
        public int CalculateMaxPositionSize(PositionSizingParameters parameters)
        {
            double maxPositionValue = parameters.AccountSize * (parameters.MaxPositionSizePercent / 100);
            return (int)(maxPositionValue / parameters.Price);
        }
    }
}