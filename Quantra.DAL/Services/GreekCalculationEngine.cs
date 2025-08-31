using System;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for calculating Greek letter metrics for options positions
    /// </summary>
    public class GreekCalculationEngine
    {
        private const double SQRT_2PI = 2.506628274631000502415765284811;
        
        /// <summary>
        /// Calculates all Greek metrics for a given position
        /// </summary>
        /// <param name="position">The options position</param>
        /// <param name="market">Current market conditions</param>
        /// <returns>Complete set of Greek metrics</returns>
        public GreekMetrics CalculateGreeks(Position position, MarketConditions market)
        {
            if (position == null)
                throw new ArgumentNullException(nameof(position));
            
            if (market == null)
                throw new ArgumentNullException(nameof(market));
            
            // Use market interest rate if available, otherwise use position's rate
            double riskFreeRate = market.InterestRate > 0 ? market.InterestRate : position.RiskFreeRate;
            
            return new GreekMetrics
            {
                Alpha = CalculateAlpha(position, market),
                Beta = CalculateBeta(position, market),
                Sigma = CalculateVolatility(position, market),
                Omega = CalculateOmega(position, market),
                Delta = CalculateDelta(position, market),
                Gamma = CalculateGamma(position, market),
                Theta = CalculateTheta(position, market),
                Vega = CalculateVega(position, market),
                Rho = CalculateRho(position, market)
            };
        }
        
        /// <summary>
        /// Calculates Theta - the rate of change of option value with respect to time
        /// </summary>
        /// <param name="position">The options position</param>
        /// <param name="market">Current market conditions</param>
        /// <returns>Theta value</returns>
        public double CalculateTheta(Position position, MarketConditions market)
        {
            if (position == null || position.TimeToExpiration <= 0)
                return 0.0;
            
            double S = position.UnderlyingPrice;
            double K = position.StrikePrice;
            double T = position.TimeToExpiration;
            double r = market?.InterestRate ?? position.RiskFreeRate;
            double sigma = position.Volatility;
            
            if (S <= 0 || K <= 0 || sigma <= 0)
                return 0.0;
            
            // Calculate d1 and d2 for Black-Scholes
            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double d2 = d1 - sigma * Math.Sqrt(T);
            
            // Standard normal density function
            double phi_d1 = 1.0 / Math.Sqrt(2 * Math.PI) * Math.Exp(-0.5 * d1 * d1);
            
            // Calculate Theta components
            double term1 = -(S * phi_d1 * sigma) / (2 * Math.Sqrt(T));
            double term2, term3;
            
            if (position.IsCall)
            {
                // Call option Theta
                term2 = -r * K * Math.Exp(-r * T) * CumulativeNormalDistribution(d2);
                term3 = 0; // No additional term for calls
            }
            else
            {
                // Put option Theta
                term2 = r * K * Math.Exp(-r * T) * CumulativeNormalDistribution(-d2);
                term3 = 0; // No additional term for puts
            }
            
            // Theta is typically expressed as the change per day, so divide by 365
            double theta = (term1 + term2) / 365.0;
            
            // Apply position quantity (negative for short positions)
            return theta * position.Quantity;
        }
        
        /// <summary>
        /// Placeholder for Alpha calculation (excess return generation)
        /// </summary>
        private double CalculateAlpha(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement sophisticated alpha calculation
            return 0.0;
        }
        
        /// <summary>
        /// Placeholder for Beta calculation (market exposure sensitivity)
        /// </summary>
        private double CalculateBeta(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement market beta calculation
            return 1.0;
        }
        
        /// <summary>
        /// Placeholder for Volatility calculation
        /// </summary>
        private double CalculateVolatility(Position position, MarketConditions market)
        {
            return position.Volatility;
        }
        
        /// <summary>
        /// Placeholder for Omega calculation (advanced risk-return optimization)
        /// </summary>
        private double CalculateOmega(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement Omega ratio calculation
            return 1.0;
        }
        
        /// <summary>
        /// Placeholder for Delta calculation (price sensitivity)
        /// </summary>
        private double CalculateDelta(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement full Delta calculation
            return 0.5;
        }
        
        /// <summary>
        /// Placeholder for Gamma calculation (rate of change of Delta)
        /// </summary>
        private double CalculateGamma(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement full Gamma calculation
            return 0.1;
        }
        
        /// <summary>
        /// Placeholder for Vega calculation (volatility sensitivity)
        /// </summary>
        private double CalculateVega(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement full Vega calculation
            return 0.2;
        }
        
        /// <summary>
        /// Placeholder for Rho calculation (interest rate sensitivity)
        /// </summary>
        private double CalculateRho(Position position, MarketConditions market)
        {
            // Simplified placeholder - would implement full Rho calculation
            return 0.05;
        }
        
        /// <summary>
        /// Cumulative normal distribution function
        /// </summary>
        /// <param name="x">Input value</param>
        /// <returns>Cumulative probability</returns>
        private double CumulativeNormalDistribution(double x)
        {
            if (x < 0)
                return 1.0 - CumulativeNormalDistribution(-x);
            
            double k = 1.0 / (1.0 + 0.2316419 * x);
            double result = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
            
            result = result * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
            return 1.0 - result;
        }
    }
}