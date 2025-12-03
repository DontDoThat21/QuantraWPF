using System;

namespace Quantra.DAL.TradingEngine.Options
{
    /// <summary>
    /// Black-Scholes options pricing calculator
    /// </summary>
    public static class OptionsCalculator
    {
        /// <summary>
        /// Calculates the theoretical option price using Black-Scholes model
        /// </summary>
        /// <param name="underlyingPrice">Current price of the underlying</param>
        /// <param name="strikePrice">Strike price</param>
        /// <param name="timeToExpiry">Time to expiration in years</param>
        /// <param name="riskFreeRate">Risk-free interest rate (e.g., 0.03 for 3%)</param>
        /// <param name="volatility">Implied volatility (e.g., 0.25 for 25%)</param>
        /// <param name="isCall">True for call, false for put</param>
        /// <returns>Theoretical option price</returns>
        public static decimal CalculatePrice(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility,
            bool isCall)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
            {
                return 0;
            }

            double S = (double)underlyingPrice;
            double K = (double)strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double d2 = d1 - sigma * Math.Sqrt(T);

            double price;
            if (isCall)
            {
                price = S * CumulativeNormal(d1) - K * Math.Exp(-r * T) * CumulativeNormal(d2);
            }
            else
            {
                price = K * Math.Exp(-r * T) * CumulativeNormal(-d2) - S * CumulativeNormal(-d1);
            }

            return (decimal)Math.Max(0, price);
        }

        /// <summary>
        /// Calculates Delta - rate of change of option price with respect to underlying price
        /// </summary>
        public static decimal CalculateDelta(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility,
            bool isCall)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
            {
                return 0;
            }

            double S = (double)underlyingPrice;
            double K = (double)strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));

            if (isCall)
            {
                return (decimal)CumulativeNormal(d1);
            }
            else
            {
                return (decimal)(CumulativeNormal(d1) - 1);
            }
        }

        /// <summary>
        /// Calculates Gamma - rate of change of delta
        /// </summary>
        public static decimal CalculateGamma(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
            {
                return 0;
            }

            double S = (double)underlyingPrice;
            double K = (double)strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double gamma = NormalDensity(d1) / (S * sigma * Math.Sqrt(T));

            return (decimal)gamma;
        }

        /// <summary>
        /// Calculates Theta - rate of time decay (per day)
        /// </summary>
        public static decimal CalculateTheta(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility,
            bool isCall)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
            {
                return 0;
            }

            double S = (double)underlyingPrice;
            double K = (double)strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double d2 = d1 - sigma * Math.Sqrt(T);

            double term1 = -(S * NormalDensity(d1) * sigma) / (2 * Math.Sqrt(T));
            double theta;

            if (isCall)
            {
                theta = term1 - r * K * Math.Exp(-r * T) * CumulativeNormal(d2);
            }
            else
            {
                theta = term1 + r * K * Math.Exp(-r * T) * CumulativeNormal(-d2);
            }

            // Convert to daily theta
            return (decimal)(theta / 365);
        }

        /// <summary>
        /// Calculates Vega - sensitivity to volatility (per 1% change)
        /// </summary>
        public static decimal CalculateVega(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
            {
                return 0;
            }

            double S = (double)underlyingPrice;
            double K = (double)strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double vega = S * NormalDensity(d1) * Math.Sqrt(T);

            // Convert to per 1% volatility change
            return (decimal)(vega / 100);
        }

        /// <summary>
        /// Calculates Rho - sensitivity to interest rates
        /// </summary>
        public static decimal CalculateRho(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility,
            bool isCall)
        {
            if (underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0 || volatility <= 0)
            {
                return 0;
            }

            double S = (double)underlyingPrice;
            double K = (double)strikePrice;
            double T = timeToExpiry;
            double r = riskFreeRate;
            double sigma = volatility;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double d2 = d1 - sigma * Math.Sqrt(T);

            double rho;
            if (isCall)
            {
                rho = K * T * Math.Exp(-r * T) * CumulativeNormal(d2);
            }
            else
            {
                rho = -K * T * Math.Exp(-r * T) * CumulativeNormal(-d2);
            }

            // Convert to per 1% rate change
            return (decimal)(rho / 100);
        }

        /// <summary>
        /// Calculates all Greeks at once
        /// </summary>
        public static OptionGreeks CalculateAllGreeks(
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            double volatility,
            bool isCall)
        {
            return new OptionGreeks
            {
                Price = CalculatePrice(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, volatility, isCall),
                Delta = CalculateDelta(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, volatility, isCall),
                Gamma = CalculateGamma(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, volatility),
                Theta = CalculateTheta(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, volatility, isCall),
                Vega = CalculateVega(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, volatility),
                Rho = CalculateRho(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, volatility, isCall)
            };
        }

        /// <summary>
        /// Calculates implied volatility from market price using Newton-Raphson method
        /// </summary>
        public static double CalculateImpliedVolatility(
            decimal marketPrice,
            decimal underlyingPrice,
            decimal strikePrice,
            double timeToExpiry,
            double riskFreeRate,
            bool isCall,
            double initialGuess = 0.25,
            int maxIterations = 100,
            double tolerance = 0.0001)
        {
            if (marketPrice <= 0 || underlyingPrice <= 0 || strikePrice <= 0 || timeToExpiry <= 0)
            {
                return 0;
            }

            double sigma = initialGuess;
            double price = (double)marketPrice;

            for (int i = 0; i < maxIterations; i++)
            {
                double theoreticalPrice = (double)CalculatePrice(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, sigma, isCall);
                double vega = (double)CalculateVega(underlyingPrice, strikePrice, timeToExpiry, riskFreeRate, sigma) * 100; // Convert back

                if (Math.Abs(vega) < 0.0001)
                {
                    break;
                }

                double diff = theoreticalPrice - price;
                if (Math.Abs(diff) < tolerance)
                {
                    return sigma;
                }

                sigma -= diff / vega;

                // Ensure sigma stays positive and reasonable
                sigma = Math.Max(0.01, Math.Min(5.0, sigma));
            }

            return sigma;
        }

        /// <summary>
        /// Standard normal cumulative distribution function
        /// </summary>
        private static double CumulativeNormal(double x)
        {
            if (x < 0)
            {
                return 1.0 - CumulativeNormal(-x);
            }

            double k = 1.0 / (1.0 + 0.2316419 * x);
            double result = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));

            result = result * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
            return 1.0 - result;
        }

        /// <summary>
        /// Standard normal probability density function
        /// </summary>
        private static double NormalDensity(double x)
        {
            return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
        }
    }

    /// <summary>
    /// Container for all option Greeks
    /// </summary>
    public class OptionGreeks
    {
        /// <summary>
        /// Theoretical option price
        /// </summary>
        public decimal Price { get; set; }

        /// <summary>
        /// Delta - sensitivity to underlying price
        /// </summary>
        public decimal Delta { get; set; }

        /// <summary>
        /// Gamma - rate of change of delta
        /// </summary>
        public decimal Gamma { get; set; }

        /// <summary>
        /// Theta - daily time decay
        /// </summary>
        public decimal Theta { get; set; }

        /// <summary>
        /// Vega - sensitivity to volatility
        /// </summary>
        public decimal Vega { get; set; }

        /// <summary>
        /// Rho - sensitivity to interest rates
        /// </summary>
        public decimal Rho { get; set; }
    }
}
