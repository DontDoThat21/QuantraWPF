using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Models;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for the SafeQueryExecutor service (MarketChat story 5).
    /// Validates that SQL queries are properly validated and blocked for security.
    /// </summary>
    [TestClass]
    public class SafeQueryExecutorTests
    {
        private SafeQueryExecutor _executor;

        [TestInitialize]
        public void Setup()
        {
            // Note: This uses the parameterless constructor for testing without database
            _executor = new SafeQueryExecutor();
        }

        #region ValidateQuery Tests

        [TestMethod]
        public void ValidateQuery_EmptyQuery_ReturnsFalse()
        {
            // Act
            var (isValid, reason) = _executor.ValidateQuery("");

            // Assert
            Assert.IsFalse(isValid);
            Assert.IsNotNull(reason);
            Assert.IsTrue(reason.Contains("empty"));
        }

        [TestMethod]
        public void ValidateQuery_NullQuery_ReturnsFalse()
        {
            // Act
            var (isValid, reason) = _executor.ValidateQuery(null);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_ValidSelectQuery_ReturnsTrue()
        {
            // Arrange
            var query = "SELECT * FROM StockPredictions WHERE Confidence > 0.8";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsTrue(isValid);
            Assert.IsNull(reason);
        }

        [TestMethod]
        public void ValidateQuery_InsertQuery_ReturnsFalse()
        {
            // Arrange
            var query = "INSERT INTO StockPredictions (Symbol) VALUES ('AAPL')";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
            Assert.IsTrue(reason.Contains("forbidden") || reason.Contains("SELECT"));
        }

        [TestMethod]
        public void ValidateQuery_UpdateQuery_ReturnsFalse()
        {
            // Arrange
            var query = "UPDATE StockPredictions SET Confidence = 0 WHERE Symbol = 'AAPL'";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_DeleteQuery_ReturnsFalse()
        {
            // Arrange
            var query = "DELETE FROM StockPredictions WHERE Symbol = 'AAPL'";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_DropTableQuery_ReturnsFalse()
        {
            // Arrange
            var query = "DROP TABLE StockPredictions";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_TruncateQuery_ReturnsFalse()
        {
            // Arrange
            var query = "TRUNCATE TABLE StockPredictions";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_ExecQuery_ReturnsFalse()
        {
            // Arrange
            var query = "EXEC sp_executesql @sql";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_UnauthorizedTable_ReturnsFalse()
        {
            // Arrange
            var query = "SELECT * FROM Users WHERE Username = 'admin'";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
            Assert.IsTrue(reason.Contains("not allowed") || reason.Contains("Users"));
        }

        [TestMethod]
        public void ValidateQuery_AllowedTable_ReturnsTrue()
        {
            // Arrange
            var query = "SELECT * FROM PredictionCache WHERE Symbol = 'AAPL'";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsTrue(isValid);
        }

        [TestMethod]
        public void ValidateQuery_SqlInjectionWithComment_ReturnsFalse()
        {
            // Arrange
            var query = "SELECT * FROM StockPredictions -- WHERE Confidence > 0.5";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
            Assert.IsTrue(reason.Contains("suspicious") || reason.Contains("injection"));
        }

        [TestMethod]
        public void ValidateQuery_SqlInjectionWithOr1Equals1_ReturnsFalse()
        {
            // Arrange
            var query = "SELECT * FROM StockPredictions WHERE Symbol = '' OR 1=1";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        [TestMethod]
        public void ValidateQuery_JoinWithAllowedTables_ReturnsTrue()
        {
            // Arrange
            var query = "SELECT p.*, i.IndicatorName FROM StockPredictions p JOIN PredictionIndicators i ON p.Id = i.PredictionId";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsTrue(isValid);
        }

        [TestMethod]
        public void ValidateQuery_JoinWithUnauthorizedTable_ReturnsFalse()
        {
            // Arrange
            var query = "SELECT * FROM StockPredictions p JOIN Credentials c ON p.UserId = c.UserId";

            // Act
            var (isValid, reason) = _executor.ValidateQuery(query);

            // Assert
            Assert.IsFalse(isValid);
        }

        #endregion

        #region ExtractTableNames Tests

        [TestMethod]
        public void ExtractTableNames_SingleTable_ReturnsCorrectTable()
        {
            // Arrange
            var query = "SELECT * FROM StockPredictions WHERE Confidence > 0.8";

            // Act
            var tables = _executor.ExtractTableNames(query);

            // Assert
            Assert.AreEqual(1, tables.Count);
            Assert.IsTrue(tables.Contains("StockPredictions"));
        }

        [TestMethod]
        public void ExtractTableNames_MultipleTablesWithJoin_ReturnsAllTables()
        {
            // Arrange
            var query = "SELECT * FROM StockPredictions p JOIN PredictionIndicators i ON p.Id = i.PredictionId";

            // Act
            var tables = _executor.ExtractTableNames(query);

            // Assert
            Assert.AreEqual(2, tables.Count);
            Assert.IsTrue(tables.Contains("StockPredictions"));
            Assert.IsTrue(tables.Contains("PredictionIndicators"));
        }

        [TestMethod]
        public void ExtractTableNames_TableWithDboPrefix_ReturnsTableName()
        {
            // Arrange
            var query = "SELECT * FROM dbo.StockPredictions";

            // Act
            var tables = _executor.ExtractTableNames(query);

            // Assert
            Assert.IsTrue(tables.Contains("StockPredictions"));
        }

        [TestMethod]
        public void ExtractTableNames_TableWithBrackets_ReturnsTableName()
        {
            // Arrange
            var query = "SELECT * FROM [StockPredictions]";

            // Act
            var tables = _executor.ExtractTableNames(query);

            // Assert
            Assert.IsTrue(tables.Contains("StockPredictions"));
        }

        #endregion

        #region AllowedTables Tests

        [TestMethod]
        public void AllowedTables_ContainsStockPredictions()
        {
            // Assert
            Assert.IsTrue(_executor.AllowedTables.Contains("StockPredictions"));
        }

        [TestMethod]
        public void AllowedTables_ContainsPredictionCache()
        {
            // Assert
            Assert.IsTrue(_executor.AllowedTables.Contains("PredictionCache"));
        }

        [TestMethod]
        public void AllowedTables_ContainsStockSymbols()
        {
            // Assert
            Assert.IsTrue(_executor.AllowedTables.Contains("StockSymbols"));
        }

        [TestMethod]
        public void AllowedTables_DoesNotContainUserCredentials()
        {
            // Assert - sensitive tables should NOT be in the allowed list
            Assert.IsFalse(_executor.AllowedTables.Contains("UserCredentials"));
            Assert.IsFalse(_executor.AllowedTables.Contains("Users"));
            Assert.IsFalse(_executor.AllowedTables.Contains("Passwords"));
        }

        #endregion

        #region AllowedOperations Tests

        [TestMethod]
        public void AllowedOperations_OnlyContainsSelect()
        {
            // Assert
            Assert.AreEqual(1, _executor.AllowedOperations.Count);
            Assert.IsTrue(_executor.AllowedOperations.Contains("SELECT"));
        }

        [TestMethod]
        public void AllowedOperations_DoesNotContainInsert()
        {
            // Assert
            Assert.IsFalse(_executor.AllowedOperations.Contains("INSERT"));
        }

        [TestMethod]
        public void AllowedOperations_DoesNotContainUpdate()
        {
            // Assert
            Assert.IsFalse(_executor.AllowedOperations.Contains("UPDATE"));
        }

        [TestMethod]
        public void AllowedOperations_DoesNotContainDelete()
        {
            // Assert
            Assert.IsFalse(_executor.AllowedOperations.Contains("DELETE"));
        }

        #endregion
    }
}
