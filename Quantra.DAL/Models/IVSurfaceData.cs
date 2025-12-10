using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Represents a 3D implied volatility surface for options analysis
    /// </summary>
    public class IVSurfaceData
    {
        /// <summary>
        /// Underlying symbol
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Collection of IV data points across strikes and expirations
        /// </summary>
        public List<IVPoint> DataPoints { get; set; } = new List<IVPoint>();

        /// <summary>
        /// Average implied volatility across all data points
        /// </summary>
        public double AverageIV { get; set; }

        /// <summary>
        /// IV skew (Put IV - Call IV differential at ATM)
        /// Positive value indicates puts are more expensive than calls
        /// </summary>
        public double IVSkew { get; set; }

        /// <summary>
        /// ATM (at-the-money) volatility
        /// </summary>
        public double ATMVolatility { get; set; }

        /// <summary>
        /// Term structure slope (change in IV vs time)
        /// </summary>
        public double TermStructureSlope { get; set; }

        /// <summary>
        /// Skew direction description
        /// </summary>
        public string SkewDirection
        {
            get
            {
                if (IVSkew > 0.02) return "Put Skew";
                if (IVSkew < -0.02) return "Call Skew";
                return "Flat";
            }
        }

        /// <summary>
        /// When the surface was calculated
        /// </summary>
        public DateTime CalculatedAt { get; set; }

        /// <summary>
        /// Current underlying price used for moneyness calculations
        /// </summary>
        public double UnderlyingPrice { get; set; }

        /// <summary>
        /// Gets IV smile data for a specific expiration
        /// </summary>
        public List<IVPoint> GetSmileForExpiration(DateTime expiration)
        {
            return DataPoints
                .Where(p => p.Expiration.Date == expiration.Date)
                .OrderBy(p => p.Moneyness)
                .ToList();
        }

        /// <summary>
        /// Gets term structure data for a specific strike
        /// </summary>
        public List<IVPoint> GetTermStructureForStrike(double strike)
        {
            return DataPoints
                .Where(p => Math.Abs(p.Strike - strike) < 0.01)
                .OrderBy(p => p.DaysToExpiration)
                .ToList();
        }

        /// <summary>
        /// Gets IV at a specific strike and expiration (interpolated if needed)
        /// </summary>
        public double GetIV(double strike, DateTime expiration)
        {
            // Find exact match first
            var exact = DataPoints.FirstOrDefault(p => 
                Math.Abs(p.Strike - strike) < 0.01 && 
                p.Expiration.Date == expiration.Date);

            if (exact != null)
                return exact.ImpliedVolatility;

            // Simple linear interpolation for now
            // TODO: Implement more sophisticated interpolation (cubic spline)
            var nearbyPoints = DataPoints
                .Where(p => p.Expiration.Date == expiration.Date)
                .OrderBy(p => Math.Abs(p.Strike - strike))
                .Take(2)
                .ToList();

            if (nearbyPoints.Count == 0)
                return AverageIV;

            if (nearbyPoints.Count == 1)
                return nearbyPoints[0].ImpliedVolatility;

            // Linear interpolation
            var p1 = nearbyPoints[0];
            var p2 = nearbyPoints[1];
            var weight = (strike - p1.Strike) / (p2.Strike - p1.Strike);
            
            return p1.ImpliedVolatility + weight * (p2.ImpliedVolatility - p1.ImpliedVolatility);
        }

        /// <summary>
        /// Analyzes the IV surface for anomalies or opportunities
        /// </summary>
        public IVSurfaceAnalysis Analyze()
        {
            var analysis = new IVSurfaceAnalysis
            {
                Symbol = Symbol,
                AnalyzedAt = DateTime.Now
            };

            if (DataPoints.Count == 0)
                return analysis;

            // Find highest/lowest IV points
            var maxIVPoint = DataPoints.OrderByDescending(p => p.ImpliedVolatility).First();
            var minIVPoint = DataPoints.OrderBy(p => p.ImpliedVolatility).First();

            analysis.MaxIV = maxIVPoint.ImpliedVolatility;
            analysis.MinIV = minIVPoint.ImpliedVolatility;
            analysis.MaxIVStrike = maxIVPoint.Strike;
            analysis.MinIVStrike = minIVPoint.Strike;

            // Calculate IV range
            analysis.IVRange = analysis.MaxIV - analysis.MinIV;

            // Analyze skew
            analysis.HasPutSkew = IVSkew > 0.02;
            analysis.HasCallSkew = IVSkew < -0.02;

            // Check for contango/backwardation in term structure
            var shortTerm = DataPoints.Where(p => p.DaysToExpiration <= 30).Average(p => p.ImpliedVolatility);
            var longTerm = DataPoints.Where(p => p.DaysToExpiration >= 90).Average(p => p.ImpliedVolatility);
            
            analysis.IsContango = longTerm > shortTerm;
            analysis.IsBackwardation = shortTerm > longTerm;

            return analysis;
        }
    }

    /// <summary>
    /// Represents a single data point in the IV surface
    /// </summary>
    public class IVPoint
    {
        /// <summary>
        /// Strike price
        /// </summary>
        public double Strike { get; set; }

        /// <summary>
        /// Option expiration date
        /// </summary>
        public DateTime Expiration { get; set; }

        /// <summary>
        /// Implied volatility at this strike/expiration
        /// </summary>
        public double ImpliedVolatility { get; set; }

        /// <summary>
        /// Moneyness ratio (Strike / Spot)
        /// </summary>
        public double Moneyness { get; set; }

        /// <summary>
        /// Days to expiration
        /// </summary>
        public double DaysToExpiration { get; set; }

        /// <summary>
        /// Option type (CALL or PUT)
        /// </summary>
        public string OptionType { get; set; }

        /// <summary>
        /// Bid-ask spread as percentage of mid price
        /// </summary>
        public double BidAskSpreadPercent { get; set; }

        /// <summary>
        /// Open interest at this point
        /// </summary>
        public long OpenInterest { get; set; }

        /// <summary>
        /// Volume at this point
        /// </summary>
        public long Volume { get; set; }

        /// <summary>
        /// Calculated vega at this point
        /// </summary>
        public double Vega { get; set; }

        /// <summary>
        /// Is this point liquid enough for trading?
        /// </summary>
        public bool IsLiquid => Volume > 50 || OpenInterest > 100;

        /// <summary>
        /// Moneyness category
        /// </summary>
        public string MoneynessCategory
        {
            get
            {
                if (Moneyness < 0.95) return "Deep OTM";
                if (Moneyness < 0.98) return "OTM";
                if (Moneyness < 1.02) return "ATM";
                if (Moneyness < 1.05) return "ITM";
                return "Deep ITM";
            }
        }
    }

    /// <summary>
    /// Analysis results from IV surface
    /// </summary>
    public class IVSurfaceAnalysis
    {
        public string Symbol { get; set; }
        public DateTime AnalyzedAt { get; set; }

        public double MaxIV { get; set; }
        public double MinIV { get; set; }
        public double MaxIVStrike { get; set; }
        public double MinIVStrike { get; set; }
        public double IVRange { get; set; }

        public bool HasPutSkew { get; set; }
        public bool HasCallSkew { get; set; }
        public bool IsContango { get; set; }
        public bool IsBackwardation { get; set; }

        public List<string> Observations { get; set; } = new List<string>();
        public List<string> TradingOpportunities { get; set; } = new List<string>();
    }
}
