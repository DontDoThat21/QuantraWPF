using System;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Examples
{
    /// <summary>
    /// Example demonstrating Theta calculations for options trading
    /// </summary>
    public class ThetaCalculationExample
    {
        /// <summary>
        /// Demonstrates various Theta calculation scenarios
        /// </summary>
        public static void RunThetaExamples()
        {
            var greekEngine = new GreekCalculationEngine();
            var marketConditions = new MarketConditions(20.0, 0.0, 0.05, 0.0);

            Console.WriteLine("=== THETA CALCULATION EXAMPLES ===\n");

            // Example 1: Near-term call option (high time decay)
            var nearTermCall = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0, // ATM
                TimeToExpiration = 7.0 / 365.0, // 1 week
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = true,
                Quantity = 1
            };

            var nearTermGreeks = greekEngine.CalculateGreeks(nearTermCall, marketConditions);
            Console.WriteLine("1. NEAR-TERM ATM CALL (1 week to expiration):");
            Console.WriteLine($"   Theta: {nearTermGreeks.Theta:F4} (daily time decay)");
            Console.WriteLine($"   Weekly decay: ${Math.Abs(nearTermGreeks.Theta * 7):F2}");
            Console.WriteLine($"   Strategy: High time decay - avoid buying, consider selling\n");

            // Example 2: Long-term call option (low time decay)
            var longTermCall = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0, // ATM
                TimeToExpiration = 90.0 / 365.0, // 3 months
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = true,
                Quantity = 1
            };

            var longTermGreeks = greekEngine.CalculateGreeks(longTermCall, marketConditions);
            Console.WriteLine("2. LONG-TERM ATM CALL (3 months to expiration):");
            Console.WriteLine($"   Theta: {longTermGreeks.Theta:F4} (daily time decay)");
            Console.WriteLine($"   Weekly decay: ${Math.Abs(longTermGreeks.Theta * 7):F2}");
            Console.WriteLine($"   Strategy: Low time decay - suitable for directional plays\n");

            // Example 3: Short put option (theta positive)
            var shortPut = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 145.0, // OTM Put
                TimeToExpiration = 30.0 / 365.0, // 1 month
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = false,
                Quantity = -1 // Short position
            };

            var shortPutGreeks = greekEngine.CalculateGreeks(shortPut, marketConditions);
            Console.WriteLine("3. SHORT OTM PUT (1 month to expiration):");
            Console.WriteLine($"   Theta: {shortPutGreeks.Theta:F4} (daily time benefit)");
            Console.WriteLine($"   Weekly profit from time decay: ${shortPutGreeks.Theta * 7:F2}");
            Console.WriteLine($"   Strategy: Theta harvesting - collect time decay premium\n");

            // Example 4: Comparison of different strikes
            Console.WriteLine("4. THETA COMPARISON ACROSS STRIKES (30 days to expiration):");

            double[] strikes = { 140.0, 145.0, 150.0, 155.0, 160.0 };
            foreach (double strike in strikes)
            {
                var position = new Position
                {
                    Symbol = "AAPL",
                    UnderlyingPrice = 150.0,
                    StrikePrice = strike,
                    TimeToExpiration = 30.0 / 365.0,
                    RiskFreeRate = 0.05,
                    Volatility = 0.25,
                    IsCall = true,
                    Quantity = 1
                };

                var greeks = greekEngine.CalculateGreeks(position, marketConditions);
                string moneyness = strike < 150 ? "ITM" : strike > 150 ? "OTM" : "ATM";

                Console.WriteLine($"   ${strike} Strike ({moneyness}): Theta = {greeks.Theta:F4}");
            }

            Console.WriteLine("\n=== SUMMARY ===");
            Console.WriteLine("• Theta represents daily time decay of option value");
            Console.WriteLine("• Negative Theta = losing money to time decay (typical for long positions)");
            Console.WriteLine("• Positive Theta = gaining money from time decay (typical for short positions)");
            Console.WriteLine("• ATM options typically have highest absolute Theta values");
            Console.WriteLine("• Time decay accelerates as expiration approaches");
            Console.WriteLine("• Use Theta for income strategies and risk management");
        }
    }
}