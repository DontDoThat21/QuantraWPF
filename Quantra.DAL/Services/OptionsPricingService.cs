using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for options pricing calculations using Black-Scholes model
    /// </summary>
    public class OptionsPricingService
    {
        private readonly LoggingService _loggingService;
        private const double SQRT_2PI = 2.506628274631000502415765284811;

        public OptionsPricingService(LoggingService loggingService)
        {
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Calculates theoretical option price using Black-Scholes model
        /// </summary>
        /// <param name="spotPrice">Current underlying price</param>
        /// <param name="strikePrice">Option strike price</param>
        /// <param name="timeToExpiration">Time to expiration in years</param>
        /// <param name="riskFreeRate">Risk-free interest rate (annual)</param>
        /// <param name="volatility">Implied volatility (annual)</param>
        /// <param name="isCall">True for call option, false for put</param>
        /// <param name="dividendYield">Dividend yield (annual, default: 0)</param>
        /// <returns>Theoretical option price</returns>
        public double CalculateBlackScholesPrice(
            double spotPrice,
            double strikePrice,
            double timeToExpiration,
            double riskFreeRate,
            double volatility,
            bool isCall,
            double dividendYield = 0)
        {
            if (spotPrice <= 0 || strikePrice <= 0 || timeToExpiration <= 0 || volatility <= 0)
                return 0;

            try
            {
                // Calculate d1 and d2
                double d1 = (Math.Log(spotPrice / strikePrice) + 
                            (riskFreeRate - dividendYield + 0.5 * volatility * volatility) * timeToExpiration) /
                            (volatility * Math.Sqrt(timeToExpiration));
                
                double d2 = d1 - volatility * Math.Sqrt(timeToExpiration);

                // Calculate option price based on type
                double price;
                if (isCall)
                {
                    price = spotPrice * Math.Exp(-dividendYield * timeToExpiration) * CumulativeNormalDistribution(d1) -
                           strikePrice * Math.Exp(-riskFreeRate * timeToExpiration) * CumulativeNormalDistribution(d2);
                }
                else
                {
                    price = strikePrice * Math.Exp(-riskFreeRate * timeToExpiration) * CumulativeNormalDistribution(-d2) -
                           spotPrice * Math.Exp(-dividendYield * timeToExpiration) * CumulativeNormalDistribution(-d1);
                }

                return Math.Max(0, price);
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error calculating Black-Scholes price");
                return 0;
            }
        }

        /// <summary>
        /// Compares theoretical price vs market price to identify potential mispricing
        /// </summary>
        /// <param name="option">Option data with market price</param>
        /// <param name="spotPrice">Current underlying price</param>
        /// <param name="riskFreeRate">Risk-free interest rate</param>
        /// <param name="dividendYield">Dividend yield</param>
        /// <returns>Pricing analysis result</returns>
        public async Task<OptionPricingAnalysis> AnalyzePricingAsync(
            OptionData option,
            double spotPrice,
            double riskFreeRate,
            double dividendYield = 0)
        {
            return await Task.Run(() =>
            {
                var timeToExpiration = (option.ExpirationDate - DateTime.Now).TotalDays / 365.0;
                
                var theoreticalPrice = CalculateBlackScholesPrice(
                    spotPrice,
                    option.StrikePrice,
                    timeToExpiration,
                    riskFreeRate,
                    option.ImpliedVolatility,
                    option.OptionType == "CALL",
                    dividendYield);

                var marketPrice = option.MidPrice;
                var difference = marketPrice - theoreticalPrice;
                var percentDifference = theoreticalPrice > 0 ? (difference / theoreticalPrice) * 100 : 0;

                return new OptionPricingAnalysis
                {
                    Symbol = option.UnderlyingSymbol,
                    Strike = option.StrikePrice,
                    Expiration = option.ExpirationDate,
                    OptionType = option.OptionType,
                    TheoreticalPrice = theoreticalPrice,
                    MarketPrice = marketPrice,
                    PriceDifference = difference,
                    PercentDifference = percentDifference,
                    IsOverpriced = percentDifference > 5,
                    IsUnderpriced = percentDifference < -5
                };
            });
        }

        /// <summary>
        /// Calculates probability of profit for an option position
        /// </summary>
        /// <param name="option">Option data</param>
        /// <param name="spotPrice">Current underlying price</param>
        /// <param name="entryPrice">Price at which option was bought</param>
        /// <returns>Probability of profit (0-100%)</returns>
        public double CalculateProbabilityOfProfit(
            OptionData option,
            double spotPrice,
            double entryPrice)
        {
            try
            {
                var timeToExpiration = (option.ExpirationDate - DateTime.Now).TotalDays / 365.0;
                
                if (timeToExpiration <= 0)
                    return 0;

                // Calculate breakeven price
                double breakeven;
                if (option.OptionType == "CALL")
                {
                    breakeven = option.StrikePrice + entryPrice;
                }
                else // PUT
                {
                    breakeven = option.StrikePrice - entryPrice;
                }

                // Calculate probability using log-normal distribution
                var drift = 0.0; // Assume no drift for simplicity
                var stdDev = option.ImpliedVolatility * Math.Sqrt(timeToExpiration);
                
                double z;
                if (option.OptionType == "CALL")
                {
                    z = (Math.Log(breakeven / spotPrice) - drift) / stdDev;
                    return (1 - CumulativeNormalDistribution(z)) * 100;
                }
                else // PUT
                {
                    z = (Math.Log(breakeven / spotPrice) - drift) / stdDev;
                    return CumulativeNormalDistribution(z) * 100;
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error calculating probability of profit");
                return 0;
            }
        }

        /// <summary>
        /// Estimates P&L scenarios for an option position
        /// </summary>
        /// <param name="option">Option data</param>
        /// <param name="spotPrice">Current underlying price</param>
        /// <param name="quantity">Number of contracts (negative for short positions)</param>
        /// <param name="entryPrice">Entry price per contract</param>
        /// <param name="priceScenarios">List of underlying price scenarios to evaluate</param>
        /// <returns>P&L estimates for each scenario</returns>
        public async Task<List<PLScenario>> EstimatePLScenariosAsync(
            OptionData option,
            double spotPrice,
            int quantity,
            double entryPrice,
            List<double> priceScenarios)
        {
            return await Task.Run(() =>
            {
                var scenarios = new List<PLScenario>();
                var timeToExpiration = (option.ExpirationDate - DateTime.Now).TotalDays / 365.0;

                if (timeToExpiration <= 0)
                    return scenarios;

                foreach (var scenarioPrice in priceScenarios)
                {
                    // Calculate option value at scenario price
                    double optionValue;
                    if (option.OptionType == "CALL")
                    {
                        optionValue = Math.Max(0, scenarioPrice - option.StrikePrice);
                    }
                    else // PUT
                    {
                        optionValue = Math.Max(0, option.StrikePrice - scenarioPrice);
                    }

                    // Calculate P&L
                    var costBasis = entryPrice * quantity * 100; // Options are priced per share, 100 shares per contract
                    var currentValue = optionValue * quantity * 100;
                    var pl = currentValue - Math.Abs(costBasis);

                    scenarios.Add(new PLScenario
                    {
                        UnderlyingPrice = scenarioPrice,
                        OptionValue = optionValue,
                        ProfitLoss = pl,
                        ROI = Math.Abs(costBasis) > 0 ? (pl / Math.Abs(costBasis)) * 100 : 0
                    });
                }

                return scenarios;
            });
        }

        /// <summary>
        /// Calculates implied volatility from market price using Newton-Raphson method
        /// </summary>
        /// <param name="marketPrice">Market option price</param>
        /// <param name="spotPrice">Current underlying price</param>
        /// <param name="strikePrice">Option strike price</param>
        /// <param name="timeToExpiration">Time to expiration in years</param>
        /// <param name="riskFreeRate">Risk-free interest rate</param>
        /// <param name="isCall">True for call, false for put</param>
        /// <param name="dividendYield">Dividend yield</param>
        /// <returns>Implied volatility</returns>
        public double CalculateImpliedVolatility(
            double marketPrice,
            double spotPrice,
            double strikePrice,
            double timeToExpiration,
            double riskFreeRate,
            bool isCall,
            double dividendYield = 0)
        {
            const int maxIterations = 100;
            const double tolerance = 0.0001;
            
            double volatility = 0.3; // Initial guess: 30%

            try
            {
                for (int i = 0; i < maxIterations; i++)
                {
                    var theoreticalPrice = CalculateBlackScholesPrice(
                        spotPrice, strikePrice, timeToExpiration, 
                        riskFreeRate, volatility, isCall, dividendYield);

                    var priceDiff = theoreticalPrice - marketPrice;

                    if (Math.Abs(priceDiff) < tolerance)
                        return volatility;

                    // Calculate vega for Newton-Raphson
                    var vega = CalculateVega(spotPrice, strikePrice, timeToExpiration, 
                                           riskFreeRate, volatility, dividendYield);

                    if (vega < 0.0001)
                        break;

                    // Newton-Raphson update
                    volatility = volatility - priceDiff / vega;

                    // Keep volatility in reasonable range
                    volatility = Math.Max(0.01, Math.Min(5.0, volatility));
                }

                return volatility;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error calculating implied volatility");
                return 0.3; // Return initial guess on error
            }
        }

        #region Private Helper Methods

        /// <summary>
        /// Calculates Vega (sensitivity to volatility change)
        /// </summary>
        private double CalculateVega(
            double spotPrice,
            double strikePrice,
            double timeToExpiration,
            double riskFreeRate,
            double volatility,
            double dividendYield)
        {
            double d1 = (Math.Log(spotPrice / strikePrice) + 
                        (riskFreeRate - dividendYield + 0.5 * volatility * volatility) * timeToExpiration) /
                        (volatility * Math.Sqrt(timeToExpiration));

            double phi_d1 = 1.0 / Math.Sqrt(2 * Math.PI) * Math.Exp(-0.5 * d1 * d1);
            
            return spotPrice * Math.Exp(-dividendYield * timeToExpiration) * 
                   phi_d1 * Math.Sqrt(timeToExpiration) / 100.0; // Divide by 100 for percentage point
        }

        /// <summary>
        /// Cumulative normal distribution function
        /// </summary>
        private double CumulativeNormalDistribution(double x)
        {
            if (x < 0)
                return 1.0 - CumulativeNormalDistribution(-x);

            double k = 1.0 / (1.0 + 0.2316419 * x);
            double result = k * (0.319381530 + k * (-0.356563782 + 
                                k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));

            result = result * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
            return 1.0 - result;
        }

        #endregion
    }

    /// <summary>
    /// Result of pricing analysis
    /// </summary>
    public class OptionPricingAnalysis
    {
        public string Symbol { get; set; }
        public double Strike { get; set; }
        public DateTime Expiration { get; set; }
        public string OptionType { get; set; }
        public double TheoreticalPrice { get; set; }
        public double MarketPrice { get; set; }
        public double PriceDifference { get; set; }
        public double PercentDifference { get; set; }
        public bool IsOverpriced { get; set; }
        public bool IsUnderpriced { get; set; }
    }

    /// <summary>
    /// P&L scenario result
    /// </summary>
    public class PLScenario
    {
        public double UnderlyingPrice { get; set; }
        public double OptionValue { get; set; }
        public double ProfitLoss { get; set; }
        public double ROI { get; set; }
    }
}
