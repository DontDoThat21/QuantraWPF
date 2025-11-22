using System;
using System.Data.SQLite;

//using System.Data.SQLite;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra;
using Quantra.DAL.Data;

namespace Quantra.Tests
{
    [TestClass]
    public class DatabaseMonolithTests
    {
        private const string TestSymbol = "TEST_DELETE_SYMBOL";

        [TestInitialize]
        public void Setup()
        {
            // Ensure database is initialized
            DatabaseMonolith.Initialize();
            
            // Clean up any existing test data
            CleanupTestData();
        }

        [TestCleanup]
        public void Cleanup()
        {
            // Clean up test data
            CleanupTestData();
        }

        private void CleanupTestData()
        {
            try
            {
                using (var connection = ConnectionHelper.GetConnection())
                {
                    connection.Open();
                    
                    // Delete from both tables
                    var deleteCacheQuery = "DELETE FROM StockDataCache WHERE Symbol = @Symbol";
                    using (var command = new SQLiteCommand(deleteCacheQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", TestSymbol);
                        command.ExecuteNonQuery();
                    }
                    
                    var deleteSymbolQuery = "DELETE FROM StockSymbols WHERE Symbol = @Symbol";
                    using (var command = new SQLiteCommand(deleteSymbolQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", TestSymbol);
                        command.ExecuteNonQuery();
                    }
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }

        [TestMethod]
        public void DeleteStockSymbol_ValidSymbol_ReturnsTrue()
        {
            // Arrange - Add a test symbol to the database first
            using (var connection = ConnectionHelper.GetConnection())
            {
                connection.Open();
                
                // Insert into StockSymbols table
                var insertSymbolQuery = @"
                    INSERT INTO StockSymbols (Symbol, Name, Sector, Industry, LastUpdated)
                    VALUES (@Symbol, @Name, @Sector, @Industry, @LastUpdated)";
                using (var command = new SQLiteCommand(insertSymbolQuery, connection))
                {
                    command.Parameters.AddWithValue("@Symbol", TestSymbol);
                    command.Parameters.AddWithValue("@Name", "Test Company");
                    command.Parameters.AddWithValue("@Sector", "Technology");
                    command.Parameters.AddWithValue("@Industry", "Software");
                    command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                    command.ExecuteNonQuery();
                }
                
                // Insert into StockDataCache table
                var insertCacheQuery = @"
                    INSERT INTO StockDataCache (Symbol, TimeRange, Interval, Data, CacheTime)
                    VALUES (@Symbol, @TimeRange, @Interval, @Data, @CacheTime)";
                using (var command = new SQLiteCommand(insertCacheQuery, connection))
                {
                    command.Parameters.AddWithValue("@Symbol", TestSymbol);
                    command.Parameters.AddWithValue("@TimeRange", "1d");
                    command.Parameters.AddWithValue("@Interval", "1m");
                    command.Parameters.AddWithValue("@Data", "test_data");
                    command.Parameters.AddWithValue("@CacheTime", DateTime.Now);
                    command.ExecuteNonQuery();
                }
            }

            // Act
            bool result = DatabaseMonolith.DeleteStockSymbol(TestSymbol);

            // Assert
            Assert.IsTrue(result, "DeleteStockSymbol should return true for valid symbol");
            
            // Verify the symbol was deleted from both tables
            using (var connection = ConnectionHelper.GetConnection())
            {
                connection.Open();
                
                // Check StockSymbols table
                var checkSymbolQuery = "SELECT COUNT(*) FROM StockSymbols WHERE Symbol = @Symbol";
                using (var command = new SQLiteCommand(checkSymbolQuery, connection))
                {
                    command.Parameters.AddWithValue("@Symbol", TestSymbol);
                    var symbolCount = Convert.ToInt32(command.ExecuteScalar());
                    Assert.AreEqual(0, symbolCount, "Symbol should be deleted from StockSymbols table");
                }
                
                // Check StockDataCache table
                var checkCacheQuery = "SELECT COUNT(*) FROM StockDataCache WHERE Symbol = @Symbol";
                using (var command = new SQLiteCommand(checkCacheQuery, connection))
                {
                    command.Parameters.AddWithValue("@Symbol", TestSymbol);
                    var cacheCount = Convert.ToInt32(command.ExecuteScalar());
                    Assert.AreEqual(0, cacheCount, "Symbol should be deleted from StockDataCache table");
                }
            }
        }

        [TestMethod]
        public void DeleteStockSymbol_NonExistentSymbol_ReturnsFalse()
        {
            // Arrange
            string nonExistentSymbol = "NONEXISTENT_SYMBOL_12345";

            // Act
            bool result = DatabaseMonolith.DeleteStockSymbol(nonExistentSymbol);

            // Assert
            Assert.IsFalse(result, "DeleteStockSymbol should return false for non-existent symbol");
        }

        [TestMethod]
        public void DeleteStockSymbol_NullOrEmptySymbol_ReturnsFalse()
        {
            // Act & Assert
            Assert.IsFalse(DatabaseMonolith.DeleteStockSymbol(null), "DeleteStockSymbol should return false for null symbol");
            Assert.IsFalse(DatabaseMonolith.DeleteStockSymbol(""), "DeleteStockSymbol should return false for empty symbol");
            Assert.IsFalse(DatabaseMonolith.DeleteStockSymbol("   "), "DeleteStockSymbol should return false for whitespace symbol");
        }

        [TestMethod]
        public void DeleteStockSymbol_OnlyInStockSymbols_ReturnsTrue()
        {
            // Arrange - Add symbol only to StockSymbols table (not cache)
            using (var connection = ConnectionHelper.GetConnection())
            {
                connection.Open();
                
                var insertSymbolQuery = @"
                    INSERT INTO StockSymbols (Symbol, Name, Sector, Industry, LastUpdated)
                    VALUES (@Symbol, @Name, @Sector, @Industry, @LastUpdated)";
                using (var command = new SQLiteCommand(insertSymbolQuery, connection))
                {
                    command.Parameters.AddWithValue("@Symbol", TestSymbol);
                    command.Parameters.AddWithValue("@Name", "Test Company");
                    command.Parameters.AddWithValue("@Sector", "Technology");
                    command.Parameters.AddWithValue("@Industry", "Software");
                    command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                    command.ExecuteNonQuery();
                }
            }

            // Act
            bool result = DatabaseMonolith.DeleteStockSymbol(TestSymbol);

            // Assert
            Assert.IsTrue(result, "DeleteStockSymbol should return true even if symbol is only in StockSymbols table");
            
            // Verify deletion
            using (var connection = ConnectionHelper.GetConnection())
            {
                connection.Open();
                var checkQuery = "SELECT COUNT(*) FROM StockSymbols WHERE Symbol = @Symbol";
                using (var command = new SQLiteCommand(checkQuery, connection))
                {
                    command.Parameters.AddWithValue("@Symbol", TestSymbol);
                    var count = Convert.ToInt32(command.ExecuteScalar());
                    Assert.AreEqual(0, count, "Symbol should be deleted from StockSymbols table");
                }
            }
        }
    }
}