using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Models
{
    public class StochRsiSwingStrategy : StrategyProfile
    {
        private const int RsiPeriod = 14;
        private const int StochPeriod = 14;
        private const int SignalPeriod = 3;
        private const double OverboughtLevel = 80;
        private const double OversoldLevel = 20;
        
        public StochRsiSwingStrategy()
        {
            Name = "Stochastic RSI Swing";
            Description = "Combines RSI and Stochastic oscillator to identify potential swing trading opportunities";
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "RSI", "StochasticRSI", "SignalLine" };

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            return indicators.ContainsKey("RSI") && 
                   indicators.ContainsKey("StochasticRSI") && 
                   indicators.ContainsKey("SignalLine");
        }

        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            if (historical == null || !historical.Any())
                return null;

            int targetIndex = index ?? historical.Count - 1;
            if (targetIndex < Math.Max(RsiPeriod + StochPeriod, SignalPeriod))
                return null;

            var prices = historical.Take(targetIndex + 1).ToList();
            var rsiValues = CalculateRSI(prices);
            var stochRsi = CalculateStochasticRSI(rsiValues);
            var signalLine = CalculateSignalLine(stochRsi);

            if (stochRsi.Count < 2 || signalLine.Count < 2)
                return null;

            var currentStochRsi = stochRsi[stochRsi.Count - 1];
            var previousStochRsi = stochRsi[stochRsi.Count - 2];
            var currentSignal = signalLine[signalLine.Count - 1];

            // Generate buy signal when StochRSI crosses above signal line from oversold
            if (previousStochRsi <= currentSignal && currentStochRsi > currentSignal && currentStochRsi < OversoldLevel)
            {
                return "BUY";
            }
            // Generate sell signal when StochRSI crosses below signal line from overbought
            else if (previousStochRsi >= currentSignal && currentStochRsi < currentSignal && currentStochRsi > OverboughtLevel)
            {
                return "SELL";
            }
            // Exit signals when StochRSI crosses the midpoint (50) in the opposite direction
            else if ((currentStochRsi > 50 && previousStochRsi <= 50) || (currentStochRsi < 50 && previousStochRsi >= 50))
            {
                return "EXIT";
            }

            return null;
        }

        public override double GetStopLossPercentage()
        {
            // Conservative stop loss at 2x the average true range
            return 0.02; // 2%
        }

        public override double GetTakeProfitPercentage()
        {
            // Take profit at 2:1 risk-reward ratio
            return 0.04; // 4%
        }

        private List<double> CalculateRSI(List<HistoricalPrice> prices)
        {
            var rsiValues = new List<double>();
            if (prices.Count <= RsiPeriod)
                return rsiValues;

            var gains = new List<double>();
            var losses = new List<double>();

            // Calculate first gains and losses
            for (int i = 1; i < RsiPeriod + 1; i++)
            {
                var difference = prices[i].Close - prices[i - 1].Close;
                gains.Add(Math.Max(0, difference));
                losses.Add(Math.Max(0, -difference));
            }

            // Calculate first RSI
            var avgGain = gains.Average();
            var avgLoss = losses.Average();
            var rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            var rsi = 100 - (100 / (1 + rs));
            rsiValues.Add(rsi);

            // Calculate subsequent RSIs
            for (int i = RsiPeriod + 1; i < prices.Count; i++)
            {
                var difference = prices[i].Close - prices[i - 1].Close;
                var gain = Math.Max(0, difference);
                var loss = Math.Max(0, -difference);

                avgGain = ((avgGain * (RsiPeriod - 1)) + gain) / RsiPeriod;
                avgLoss = ((avgLoss * (RsiPeriod - 1)) + loss) / RsiPeriod;

                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiValues.Add(rsi);
            }

            return rsiValues;
        }

        private List<double> CalculateStochasticRSI(List<double> rsiValues)
        {
            var stochRsi = new List<double>();
            if (rsiValues.Count < StochPeriod)
                return stochRsi;

            for (int i = StochPeriod - 1; i < rsiValues.Count; i++)
            {
                var periodRsi = rsiValues.Skip(i - StochPeriod + 1).Take(StochPeriod).ToList();
                var highestRsi = periodRsi.Max();
                var lowestRsi = periodRsi.Min();
                var currentRsi = periodRsi.Last();

                var stochValue = highestRsi - lowestRsi != 0 
                    ? ((currentRsi - lowestRsi) / (highestRsi - lowestRsi)) * 100
                    : 100;

                stochRsi.Add(stochValue);
            }

            return stochRsi;
        }

        private List<double> CalculateSignalLine(List<double> stochRsi)
        {
            var signalLine = new List<double>();
            if (stochRsi.Count < SignalPeriod)
                return signalLine;

            // Simple Moving Average of StochRSI
            for (int i = SignalPeriod - 1; i < stochRsi.Count; i++)
            {
                var average = stochRsi.Skip(i - SignalPeriod + 1).Take(SignalPeriod).Average();
                signalLine.Add(average);
            }

            return signalLine;
        }
    }
}