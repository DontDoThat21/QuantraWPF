using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Models;

namespace Quantra.Tests.Models
{
    /// <summary>
    /// Unit tests for the NaturalLanguageQueryResult model (MarketChat story 5).
    /// Validates that query results are properly formatted for display.
    /// </summary>
    [TestClass]
    public class NaturalLanguageQueryResultTests
    {
        #region ToMarkdownTable Tests

        [TestMethod]
        public void ToMarkdownTable_SuccessfulResult_ReturnsFormattedTable()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "Show high confidence predictions",
                "SELECT * FROM StockPredictions WHERE Confidence > 0.8",
                new List<string> { "Symbol", "Confidence", "PredictedAction" },
                new List<List<object>>
                {
                    new List<object> { "AAPL", 0.85, "BUY" },
                    new List<object> { "MSFT", 0.92, "BUY" }
                },
                new List<string> { "StockPredictions" },
                50);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("| Symbol | Confidence | PredictedAction |"));
            Assert.IsTrue(markdown.Contains("| --- | --- | --- |"));
            Assert.IsTrue(markdown.Contains("AAPL"));
            Assert.IsTrue(markdown.Contains("MSFT"));
        }

        [TestMethod]
        public void ToMarkdownTable_BlockedResult_ReturnsBlockedMessage()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateBlocked(
                "DROP TABLE Users",
                "Destructive operations are not allowed");

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("Query blocked"));
            Assert.IsTrue(markdown.Contains("Destructive operations are not allowed"));
        }

        [TestMethod]
        public void ToMarkdownTable_FailedResult_ReturnsErrorMessage()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateFailure(
                "SELECT * FROM BadQuery",
                "Invalid table name");

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("Query failed"));
            Assert.IsTrue(markdown.Contains("Invalid table name"));
        }

        [TestMethod]
        public void ToMarkdownTable_EmptyResults_ReturnsNoResultsMessage()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "SELECT * FROM StockPredictions WHERE Symbol = 'INVALID'",
                "SELECT * FROM StockPredictions WHERE Symbol = 'INVALID'",
                new List<string> { "Symbol", "Confidence" },
                new List<List<object>>(),
                new List<string> { "StockPredictions" },
                10);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.AreEqual("No results found.", markdown);
        }

        [TestMethod]
        public void ToMarkdownTable_LargeResult_TruncatesTo50Rows()
        {
            // Arrange
            var rows = new List<List<object>>();
            for (int i = 0; i < 100; i++)
            {
                rows.Add(new List<object> { $"STOCK{i}", 0.5 + i * 0.005, "HOLD" });
            }

            var result = NaturalLanguageQueryResult.CreateSuccess(
                "SELECT * FROM StockPredictions",
                "SELECT * FROM StockPredictions",
                new List<string> { "Symbol", "Confidence", "Action" },
                rows,
                new List<string> { "StockPredictions" },
                100);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("Showing 50 of 100 results"));
            // Count actual data rows (excluding header and separator)
            var lineCount = markdown.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
            // Should have header, separator, 50 data rows, empty line, and truncation message
            Assert.IsTrue(lineCount <= 55);
        }

        #endregion

        #region CreateSuccess Tests

        [TestMethod]
        public void CreateSuccess_SetsAllProperties()
        {
            // Act
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "original",
                "translated",
                new List<string> { "col1" },
                new List<List<object>> { new List<object> { "val1" } },
                new List<string> { "Table1" },
                42);

            // Assert
            Assert.IsTrue(result.Success);
            Assert.AreEqual("original", result.OriginalQuery);
            Assert.AreEqual("translated", result.TranslatedSql);
            Assert.AreEqual(1, result.Columns.Count);
            Assert.AreEqual(1, result.Rows.Count);
            Assert.AreEqual(1, result.RowCount);
            Assert.AreEqual(1, result.QueriedTables.Count);
            Assert.AreEqual(42, result.ExecutionTimeMs);
            Assert.IsFalse(result.WasBlocked);
        }

        #endregion

        #region CreateFailure Tests

        [TestMethod]
        public void CreateFailure_SetsErrorProperties()
        {
            // Act
            var result = NaturalLanguageQueryResult.CreateFailure("query", "error message");

            // Assert
            Assert.IsFalse(result.Success);
            Assert.AreEqual("query", result.OriginalQuery);
            Assert.AreEqual("error message", result.ErrorMessage);
            Assert.IsFalse(result.WasBlocked);
        }

        #endregion

        #region CreateBlocked Tests

        [TestMethod]
        public void CreateBlocked_SetsBlockedProperties()
        {
            // Act
            var result = NaturalLanguageQueryResult.CreateBlocked("dangerous query", "blocked reason");

            // Assert
            Assert.IsFalse(result.Success);
            Assert.IsTrue(result.WasBlocked);
            Assert.AreEqual("dangerous query", result.OriginalQuery);
            Assert.AreEqual("blocked reason", result.BlockedReason);
        }

        #endregion

        #region Value Formatting Tests

        [TestMethod]
        public void ToMarkdownTable_FormatsPercentages()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "query",
                "sql",
                new List<string> { "Confidence" },
                new List<List<object>> { new List<object> { 0.85 } },
                new List<string> { "StockPredictions" },
                10);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("85.0%"));
        }

        [TestMethod]
        public void ToMarkdownTable_FormatsNullValues()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "query",
                "sql",
                new List<string> { "Value" },
                new List<List<object>> { new List<object> { null } },
                new List<string> { "StockPredictions" },
                10);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("-"));
        }

        [TestMethod]
        public void ToMarkdownTable_FormatsDates()
        {
            // Arrange
            var testDate = new DateTime(2024, 1, 15, 14, 30, 0);
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "query",
                "sql",
                new List<string> { "Date" },
                new List<List<object>> { new List<object> { testDate } },
                new List<string> { "StockPredictions" },
                10);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("2024-01-15 14:30"));
        }

        [TestMethod]
        public void ToMarkdownTable_FormatsLargeNumbers()
        {
            // Arrange
            var result = NaturalLanguageQueryResult.CreateSuccess(
                "query",
                "sql",
                new List<string> { "Volume" },
                new List<List<object>> { new List<object> { 1234567890L } },
                new List<string> { "StockPredictions" },
                10);

            // Act
            var markdown = result.ToMarkdownTable();

            // Assert
            Assert.IsTrue(markdown.Contains("1,234,567,890"));
        }

        #endregion
    }
}
