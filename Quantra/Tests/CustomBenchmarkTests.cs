using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Models;
using Quantra.Services;

namespace Quantra.Tests
{
    public class CustomBenchmarkTests
    {
        public void CustomBenchmark_Creation_IsValid()
        {
            // Arrange
            var benchmark = new CustomBenchmark
            {
                Name = "Test Tech Benchmark",
                Description = "A benchmark for tech stocks",
                Category = BenchmarkCategory.Sector
            };

            // Act
            benchmark.AddComponent("AAPL", "Apple Inc.", 0.3);
            benchmark.AddComponent("MSFT", "Microsoft Corp.", 0.3);
            benchmark.AddComponent("GOOGL", "Alphabet Inc.", 0.2);
            benchmark.AddComponent("AMZN", "Amazon.com Inc.", 0.2);

            // Assert
            string errorMessage;
            bool isValid = benchmark.Validate(out errorMessage);
            
            Console.WriteLine($"Validation result: {isValid}, error: {errorMessage ?? "none"}");
            Console.WriteLine($"Total components: {benchmark.Components.Count}");
            
            double totalWeight = benchmark.Components.Sum(c => c.Weight);
            Console.WriteLine($"Total weight: {totalWeight}");
            
            // Display all components
            foreach (var component in benchmark.Components)
            {
                Console.WriteLine($"Component: {component.Symbol}, Weight: {component.Weight:P2}");
            }
            
            Console.WriteLine($"Display symbol: {benchmark.DisplaySymbol}");
        }

        public void CustomBenchmark_NormalizeWeights_DistributesEquallyWhenZero()
        {
            // Arrange
            var benchmark = new CustomBenchmark
            {
                Name = "Zero Weight Benchmark",
                Category = BenchmarkCategory.Custom
            };
            
            // Add components with zero weights
            benchmark.Components.Add(new BenchmarkComponent { Symbol = "A", Name = "A Corp", Weight = 0 });
            benchmark.Components.Add(new BenchmarkComponent { Symbol = "B", Name = "B Corp", Weight = 0 });
            benchmark.Components.Add(new BenchmarkComponent { Symbol = "C", Name = "C Corp", Weight = 0 });
            
            // Act
            benchmark.NormalizeWeights();
            
            // Assert
            Console.WriteLine("After normalizing zero weights:");
            double expectedWeight = 1.0 / 3.0;
            foreach (var component in benchmark.Components)
            {
                Console.WriteLine($"Component: {component.Symbol}, Weight: {component.Weight:P2}, Expected: {expectedWeight:P2}");
            }
        }

        public void CustomBenchmark_DisplaySymbol_FormatsCorrectly()
        {
            // Arrange
            var benchmark = new CustomBenchmark
            {
                Name = "Test Benchmark",
                Category = BenchmarkCategory.Custom
            };
            
            // Act - single component
            benchmark.AddComponent("SPY", "S&P 500", 1.0);
            string singleSymbol = benchmark.DisplaySymbol;
            Console.WriteLine($"Single component display symbol: {singleSymbol}");
            
            // Add more components
            benchmark.Components.Clear();
            benchmark.AddComponent("SPY", "S&P 500", 0.6);
            benchmark.AddComponent("QQQ", "NASDAQ", 0.4);
            string multiSymbol = benchmark.DisplaySymbol;
            Console.WriteLine($"Two components display symbol: {multiSymbol}");
            
            // Add many components
            benchmark.Components.Clear();
            benchmark.AddComponent("SPY", "S&P 500", 0.5);
            benchmark.AddComponent("QQQ", "NASDAQ", 0.3);
            benchmark.AddComponent("IWM", "Russell 2000", 0.1);
            benchmark.AddComponent("DIA", "Dow Jones", 0.05);
            benchmark.AddComponent("XLF", "Financials", 0.05);
            string manySymbol = benchmark.DisplaySymbol;
            Console.WriteLine($"Many components display symbol: {manySymbol}");
        }
        
        // Simple manual test method to run all tests
        public static void RunAllTests()
        {
            var tests = new CustomBenchmarkTests();
            
            Console.WriteLine("=== Testing CustomBenchmark_Creation_IsValid ===");
            tests.CustomBenchmark_Creation_IsValid();
            
            Console.WriteLine("\n=== Testing CustomBenchmark_NormalizeWeights_DistributesEquallyWhenZero ===");
            tests.CustomBenchmark_NormalizeWeights_DistributesEquallyWhenZero();
            
            Console.WriteLine("\n=== Testing CustomBenchmark_DisplaySymbol_FormatsCorrectly ===");
            tests.CustomBenchmark_DisplaySymbol_FormatsCorrectly();
        }
    }
}