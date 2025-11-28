using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Models;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Xunit;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for MultiSymbolAnalyzer service (MarketChat story 7).
    /// Tests multi-symbol comparative analysis, scoring algorithms, and formatting.
    /// </summary>
    public class MultiSymbolAnalyzerTests
    {
        private readonly MultiSymbolAnalyzer _analyzer;

        public MultiSymbolAnalyzerTests()
        {
            // Create analyzer without dependencies for unit testing
            _analyzer = new MultiSymbolAnalyzer(null, null, null, null);
        }

        #region CalculateCompositeScores Tests

        [Fact]
        public void CalculateCompositeScores_WithValidData_ReturnsScoresForAllSymbols()
        {
            // Arrange
            var result = CreateMockComparisonResult();

            // Act
            var scores = _analyzer.CalculateCompositeScores(result);

            // Assert
            Assert.NotNull(scores);
            Assert.Equal(3, scores.Count);
            Assert.Contains("AAPL", scores.Keys);
            Assert.Contains("MSFT", scores.Keys);
            Assert.Contains("GOOGL", scores.Keys);
        }

        [Fact]
        public void CalculateCompositeScores_WithEmptyResult_ReturnsEmptyDictionary()
        {
            // Arrange
            var result = new MultiSymbolComparisonResult();

            // Act
            var scores = _analyzer.CalculateCompositeScores(result);

            // Assert
            Assert.NotNull(scores);
            Assert.Empty(scores);
        }

        [Fact]
        public void CalculateCompositeScores_WithNullResult_ReturnsEmptyDictionary()
        {
            // Act
            var scores = _analyzer.CalculateCompositeScores(null);

            // Assert
            Assert.NotNull(scores);
            Assert.Empty(scores);
        }

        [Fact]
        public void CalculateCompositeScores_ScoresAreBetween0And100()
        {
            // Arrange
            var result = CreateMockComparisonResult();

            // Act
            var scores = _analyzer.CalculateCompositeScores(result);

            // Assert
            foreach (var score in scores.Values)
            {
                Assert.InRange(score, 0, 100);
            }
        }

        [Fact]
        public void CalculateCompositeScores_BuySignalGetsBonus()
        {
            // Arrange
            var result = new MultiSymbolComparisonResult();
            result.Symbols.AddRange(new[] { "BUY_STOCK", "HOLD_STOCK" });
            
            result.SymbolData["BUY_STOCK"] = new SymbolAnalysisData
            {
                Symbol = "BUY_STOCK",
                PredictedAction = "BUY",
                Confidence = 0.8,
                RiskMetrics = new SymbolRiskMetrics { RiskScore = 50 },
                HistoricalContext = new HistoricalContextSummary { MomentumScore = 50 }
            };
            
            result.SymbolData["HOLD_STOCK"] = new SymbolAnalysisData
            {
                Symbol = "HOLD_STOCK",
                PredictedAction = "HOLD",
                Confidence = 0.8, // Same confidence
                RiskMetrics = new SymbolRiskMetrics { RiskScore = 50 }, // Same risk
                HistoricalContext = new HistoricalContextSummary { MomentumScore = 50 } // Same momentum
            };

            // Act
            var scores = _analyzer.CalculateCompositeScores(result);

            // Assert - BUY signal should have higher score due to bonus
            Assert.True(scores["BUY_STOCK"] > scores["HOLD_STOCK"]);
        }

        #endregion

        #region IdentifySignalHighlights Tests

        [Fact]
        public void IdentifySignalHighlights_IdentifiesStrongestBullish()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);

            // Act
            var highlights = _analyzer.IdentifySignalHighlights(result);

            // Assert
            Assert.NotNull(highlights);
            Assert.NotNull(highlights.StrongestBullish);
            Assert.NotEmpty(highlights.StrongestBullishReason);
        }

        [Fact]
        public void IdentifySignalHighlights_IdentifiesHighestConfidence()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);

            // Act
            var highlights = _analyzer.IdentifySignalHighlights(result);

            // Assert
            Assert.NotNull(highlights.HighestConfidence);
            Assert.True(highlights.HighestConfidenceValue > 0);
        }

        [Fact]
        public void IdentifySignalHighlights_IdentifiesRecommendedPick()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);

            // Act
            var highlights = _analyzer.IdentifySignalHighlights(result);

            // Assert
            Assert.NotNull(highlights.RecommendedPick);
            Assert.NotEmpty(highlights.RecommendedPickReason);
        }

        [Fact]
        public void IdentifySignalHighlights_WithNoData_ReturnsEmptyHighlights()
        {
            // Arrange
            var result = new MultiSymbolComparisonResult();

            // Act
            var highlights = _analyzer.IdentifySignalHighlights(result);

            // Assert
            Assert.NotNull(highlights);
            Assert.Null(highlights.StrongestBullish);
            Assert.Null(highlights.HighestConfidence);
        }

        #endregion

        #region FormatComparisonAsMarkdown Tests

        [Fact]
        public void FormatComparisonAsMarkdown_ContainsComparisonTable()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);
            result.Highlights = _analyzer.IdentifySignalHighlights(result);

            // Act
            var markdown = _analyzer.FormatComparisonAsMarkdown(result);

            // Assert
            Assert.NotNull(markdown);
            Assert.Contains("| Symbol |", markdown);
            Assert.Contains("| AAPL |", markdown);
            Assert.Contains("| MSFT |", markdown);
        }

        [Fact]
        public void FormatComparisonAsMarkdown_ContainsTechnicalIndicatorsTable()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);
            result.Highlights = _analyzer.IdentifySignalHighlights(result);

            // Act
            var markdown = _analyzer.FormatComparisonAsMarkdown(result);

            // Assert
            Assert.Contains("### Technical Indicators", markdown);
            Assert.Contains("| RSI |", markdown);
        }

        [Fact]
        public void FormatComparisonAsMarkdown_ContainsSignalHighlights()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);
            result.Highlights = _analyzer.IdentifySignalHighlights(result);

            // Act
            var markdown = _analyzer.FormatComparisonAsMarkdown(result);

            // Assert
            Assert.Contains("### Signal Highlights", markdown);
            Assert.Contains("Recommended Pick", markdown);
        }

        [Fact]
        public void FormatComparisonAsMarkdown_WithNullResult_ReturnsNoDataMessage()
        {
            // Act
            var markdown = _analyzer.FormatComparisonAsMarkdown(null);

            // Assert
            Assert.Contains("No comparison data available", markdown);
        }

        #endregion

        #region GenerateAllocationRecommendations Tests

        [Fact]
        public void GenerateAllocationRecommendations_ContainsAllocationTable()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);

            // Act
            var recommendations = _analyzer.GenerateAllocationRecommendations(result, "moderate");

            // Assert
            Assert.NotNull(recommendations);
            Assert.Contains("Portfolio Allocation", recommendations);
            Assert.Contains("| Symbol |", recommendations);
        }

        [Fact]
        public void GenerateAllocationRecommendations_ConservativeHasLowerMaxAllocation()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);

            // Act
            var conservative = _analyzer.GenerateAllocationRecommendations(result, "conservative");
            var aggressive = _analyzer.GenerateAllocationRecommendations(result, "aggressive");

            // Assert - Conservative should have more even distribution (25% max vs 50% max)
            Assert.NotNull(conservative);
            Assert.NotNull(aggressive);
        }

        [Fact]
        public void GenerateAllocationRecommendations_ContainsRiskConsiderations()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);
            result.Highlights = _analyzer.IdentifySignalHighlights(result);

            // Act
            var recommendations = _analyzer.GenerateAllocationRecommendations(result, "moderate");

            // Assert
            Assert.Contains("Risk Considerations", recommendations);
        }

        [Fact]
        public void GenerateAllocationRecommendations_WithNullResult_ReturnsInsufficientDataMessage()
        {
            // Act
            var recommendations = _analyzer.GenerateAllocationRecommendations(null, "moderate");

            // Assert
            Assert.Contains("Insufficient data", recommendations);
        }

        #endregion

        #region BuildComparisonContext Tests

        [Fact]
        public void BuildComparisonContext_ContainsAllSymbolData()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);
            result.Highlights = _analyzer.IdentifySignalHighlights(result);

            // Act
            var context = _analyzer.BuildComparisonContext(result);

            // Assert
            Assert.Contains("AAPL:", context);
            Assert.Contains("MSFT:", context);
            Assert.Contains("GOOGL:", context);
        }

        [Fact]
        public void BuildComparisonContext_ContainsPredictionDetails()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);

            // Act
            var context = _analyzer.BuildComparisonContext(result);

            // Assert
            Assert.Contains("Predicted Action:", context);
            Assert.Contains("Confidence:", context);
            Assert.Contains("Target Price:", context);
        }

        [Fact]
        public void BuildComparisonContext_ContainsHighlights()
        {
            // Arrange
            var result = CreateMockComparisonResult();
            result.CompositeScores = _analyzer.CalculateCompositeScores(result);
            result.Highlights = _analyzer.IdentifySignalHighlights(result);

            // Act
            var context = _analyzer.BuildComparisonContext(result);

            // Assert
            Assert.Contains("Signal Highlights", context);
            Assert.Contains("Recommended Pick", context);
        }

        #endregion

        #region CompareSymbolsAsync Tests

        [Fact]
        public async Task CompareSymbolsAsync_WithEmptySymbols_ReturnsErrorResult()
        {
            // Act
            var result = await _analyzer.CompareSymbolsAsync(new List<string>());

            // Assert
            Assert.NotNull(result);
            Assert.False(result.IsSuccessful);
            Assert.Contains("No symbols provided", result.Errors.FirstOrDefault());
        }

        [Fact]
        public async Task CompareSymbolsAsync_WithNullSymbols_ReturnsErrorResult()
        {
            // Act
            var result = await _analyzer.CompareSymbolsAsync(null);

            // Assert
            Assert.NotNull(result);
            Assert.False(result.IsSuccessful);
        }

        [Fact]
        public async Task CompareSymbolsAsync_NormalizesSymbolsToUppercase()
        {
            // Act
            var result = await _analyzer.CompareSymbolsAsync(new[] { "aapl", "msft" });

            // Assert
            Assert.Contains("AAPL", result.Symbols);
            Assert.Contains("MSFT", result.Symbols);
        }

        [Fact]
        public async Task CompareSymbolsAsync_RemovesDuplicateSymbols()
        {
            // Act
            var result = await _analyzer.CompareSymbolsAsync(new[] { "AAPL", "aapl", "AAPL" });

            // Assert
            Assert.Single(result.Symbols);
        }

        [Fact]
        public async Task CompareSymbolsAsync_LimitsToMaxSymbols()
        {
            // Arrange
            var manySymbols = Enumerable.Range(1, 20).Select(i => $"SYM{i}").ToList();

            // Act
            var result = await _analyzer.CompareSymbolsAsync(manySymbols);

            // Assert - Should be limited to 10 symbols
            Assert.True(result.Symbols.Count <= 10);
        }

        #endregion

        #region Helper Methods

        private MultiSymbolComparisonResult CreateMockComparisonResult()
        {
            var result = new MultiSymbolComparisonResult
            {
                GeneratedAt = DateTime.UtcNow
            };

            result.Symbols.AddRange(new[] { "AAPL", "MSFT", "GOOGL" });

            result.SymbolData["AAPL"] = new SymbolAnalysisData
            {
                Symbol = "AAPL",
                CurrentPrice = 180.0,
                PredictedAction = "BUY",
                TargetPrice = 195.0,
                Confidence = 0.85,
                PotentialReturn = 0.083,
                Indicators = new Dictionary<string, double>
                {
                    { "RSI", 45.0 },
                    { "MACD", 0.25 },
                    { "ADX", 28.0 }
                },
                RiskMetrics = new SymbolRiskMetrics
                {
                    RiskScore = 45,
                    Volatility = 2.5,
                    ATR = 3.2
                },
                HistoricalContext = new HistoricalContextSummary
                {
                    FiveDayChange = 0.02,
                    TwentyDayChange = 0.05,
                    TrendDirection = "bullish",
                    MomentumScore = 65
                }
            };

            result.SymbolData["MSFT"] = new SymbolAnalysisData
            {
                Symbol = "MSFT",
                CurrentPrice = 380.0,
                PredictedAction = "BUY",
                TargetPrice = 400.0,
                Confidence = 0.78,
                PotentialReturn = 0.053,
                Indicators = new Dictionary<string, double>
                {
                    { "RSI", 55.0 },
                    { "MACD", 0.15 },
                    { "ADX", 22.0 }
                },
                RiskMetrics = new SymbolRiskMetrics
                {
                    RiskScore = 40,
                    Volatility = 2.0,
                    ATR = 5.5
                },
                HistoricalContext = new HistoricalContextSummary
                {
                    FiveDayChange = 0.01,
                    TwentyDayChange = 0.03,
                    TrendDirection = "bullish",
                    MomentumScore = 50
                }
            };

            result.SymbolData["GOOGL"] = new SymbolAnalysisData
            {
                Symbol = "GOOGL",
                CurrentPrice = 140.0,
                PredictedAction = "HOLD",
                TargetPrice = 145.0,
                Confidence = 0.65,
                PotentialReturn = 0.036,
                Indicators = new Dictionary<string, double>
                {
                    { "RSI", 48.0 },
                    { "MACD", -0.05 },
                    { "ADX", 18.0 }
                },
                RiskMetrics = new SymbolRiskMetrics
                {
                    RiskScore = 55,
                    Volatility = 2.8,
                    ATR = 2.8
                },
                HistoricalContext = new HistoricalContextSummary
                {
                    FiveDayChange = -0.01,
                    TwentyDayChange = 0.02,
                    TrendDirection = "neutral",
                    MomentumScore = 30
                }
            };

            return result;
        }

        #endregion
    }
}
