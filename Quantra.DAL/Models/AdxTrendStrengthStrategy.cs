using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Models
{
    public class AdxTrendStrengthStrategy : StrategyProfile
    {
        private const int AdxPeriod = 14;
        private const double AdxThreshold = 25;
        
        public AdxTrendStrengthStrategy()
        {
            Name = "ADX Trend Strength";
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "ADX", "PLUS_DI", "MINUS_DI" };

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (!indicators.TryGetValue("ADX", out double adx) ||
                !indicators.TryGetValue("PLUS_DI", out double plusDi) ||
                !indicators.TryGetValue("MINUS_DI", out double minusDi))
                return false;

            return adx > AdxThreshold; // Only validate when ADX shows strong trend
        }

        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            var actualIndex = index ?? historical.Count - 1;
            if (actualIndex < AdxPeriod + 1)
                return null;

            var prices = historical.Take(actualIndex + 1).ToList();
            var (adx, plusDI, minusDI) = CalculateADX(prices);

            if (adx.Count == 0 || plusDI.Count == 0 || minusDI.Count == 0)
                return null;

            // Get current values
            double currentAdx = adx[adx.Count - 1];
            double currentPlusDI = plusDI[plusDI.Count - 1];
            double currentMinusDI = minusDI[minusDI.Count - 1];

            // Only generate signals when ADX indicates a strong trend
            if (currentAdx > AdxThreshold)
            {
                if (currentPlusDI > currentMinusDI)
                    return "BUY";
                else if (currentMinusDI > currentPlusDI)
                    return "SELL";
            }

            return null;
        }

        private (List<double> adx, List<double> plusDI, List<double> minusDI) CalculateADX(List<HistoricalPrice> prices)
        {
            if (prices.Count <= AdxPeriod)
                return (new List<double>(), new List<double>(), new List<double>());

            var plusDM = new List<double>();
            var minusDM = new List<double>();
            var trueRange = new List<double>();

            // Calculate Plus DM, Minus DM and True Range for each period
            for (int i = 1; i < prices.Count; i++)
            {
                var high = prices[i].High;
                var low = prices[i].Low;
                var prevHigh = prices[i - 1].High;
                var prevLow = prices[i - 1].Low;
                var prevClose = prices[i - 1].Close;

                // Plus DM
                var highDiff = high - prevHigh;
                var lowDiff = prevLow - low;
                
                if (highDiff > lowDiff && highDiff > 0)
                    plusDM.Add(highDiff);
                else
                    plusDM.Add(0);

                // Minus DM
                if (lowDiff > highDiff && lowDiff > 0)
                    minusDM.Add(lowDiff);
                else
                    minusDM.Add(0);

                // True Range
                var tr1 = Math.Abs(high - low);
                var tr2 = Math.Abs(high - prevClose);
                var tr3 = Math.Abs(low - prevClose);
                trueRange.Add(Math.Max(tr1, Math.Max(tr2, tr3)));
            }

            // Calculate smoothed values
            var smoothedTR = new List<double>();
            var smoothedPlusDM = new List<double>();
            var smoothedMinusDM = new List<double>();

            // Initial smoothed values
            smoothedTR.Add(trueRange.Take(AdxPeriod).Sum());
            smoothedPlusDM.Add(plusDM.Take(AdxPeriod).Sum());
            smoothedMinusDM.Add(minusDM.Take(AdxPeriod).Sum());

            // Calculate subsequent smoothed values
            for (int i = AdxPeriod; i < trueRange.Count; i++)
            {
                var prevTR = smoothedTR[smoothedTR.Count - 1];
                var prevPlusDM = smoothedPlusDM[smoothedPlusDM.Count - 1];
                var prevMinusDM = smoothedMinusDM[smoothedMinusDM.Count - 1];

                smoothedTR.Add(prevTR - (prevTR / AdxPeriod) + trueRange[i]);
                smoothedPlusDM.Add(prevPlusDM - (prevPlusDM / AdxPeriod) + plusDM[i]);
                smoothedMinusDM.Add(prevMinusDM - (prevMinusDM / AdxPeriod) + minusDM[i]);
            }

            // Calculate +DI and -DI
            var plusDI = smoothedPlusDM.Select((dm, i) => 100 * dm / smoothedTR[i]).ToList();
            var minusDI = smoothedMinusDM.Select((dm, i) => 100 * dm / smoothedTR[i]).ToList();

            // Calculate DX
            var dx = new List<double>();
            for (int i = 0; i < plusDI.Count; i++)
            {
                var diff = Math.Abs(plusDI[i] - minusDI[i]);
                var sum = plusDI[i] + minusDI[i];
                dx.Add(100 * diff / sum);
            }

            // Calculate ADX (14-period smoothed average of DX)
            var adx = new List<double>();
            double sumDx = dx.Take(AdxPeriod).Sum();
            adx.Add(sumDx / AdxPeriod);

            for (int i = AdxPeriod; i < dx.Count; i++)
            {
                var prev = adx[adx.Count - 1];
                var current = ((prev * (AdxPeriod - 1)) + dx[i]) / AdxPeriod;
                adx.Add(current);
            }

            return (adx, plusDI, minusDI);
        }
    }
}