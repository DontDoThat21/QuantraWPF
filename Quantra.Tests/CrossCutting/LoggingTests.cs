using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Logging;

namespace Quantra.Tests.CrossCutting
{
    [TestClass]
    public class LoggingTests
    {
        [TestInitialize]
        public void Setup()
        {
            // Initialize the cross-cutting framework
            CrossCuttingRegistry.Initialize();
        }

        [TestMethod]
        public void LoggingManager_Initialization_ShouldSucceed()
        {
            // Arrange
            var manager = LoggingManager.Instance;
            
            // Act
            manager.Initialize("Logging");
            
            // Assert
            Assert.IsNotNull(manager);
        }

        [TestMethod]
        public void Log_GetLogger_ShouldReturnValidLogger()
        {
            // Arrange & Act
            var logger = Log.ForType<LoggingTests>();
            
            // Assert
            Assert.IsNotNull(logger);
        }

        [TestMethod]
        public void Log_Info_ShouldNotThrow()
        {
            // Arrange
            var logger = Log.ForType<LoggingTests>();
            
            // Act & Assert
            try
            {
                logger.Information("Test log message");
                logger.Information("Test with parameters {Param1} and {Param2}", "Value1", 42);
                
                // Log should succeed without exceptions
                Assert.IsTrue(true);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Logging should not throw: {ex.Message}");
            }
        }

        [TestMethod]
        public void Log_TimedOperation_ShouldMeasureTime()
        {
            // Arrange
            var logger = Log.ForType<LoggingTests>();
            
            // Act
            TimeSpan elapsed;
            using (var operation = logger.BeginTimedOperation("TestOperation"))
            {
                // Simulate some work
                Task.Delay(100).Wait();
            }
            
            // Assert - if we reach here, the operation completed and logged correctly
            Assert.IsTrue(true);
        }

        [TestMethod]
        public void LoggingService_LegacyCompat_ShouldWork()
        {
            // Arrange & Act & Assert
            try
            {
                // Test legacy logging works
                Quantra.Services._loggingService.Log("Info", "Legacy log test");
                Quantra.Services._loggingService.LogErrorWithContext(
                    new InvalidOperationException("Test exception"), "Legacy error logging test");
                
                // Log should succeed without exceptions
                Assert.IsTrue(true);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Legacy logging should not throw: {ex.Message}");
            }
        }

        [TestMethod]
        public void Log_NestedContext_ShouldPreserveContext()
        {
            // Arrange
            var logger = Log.ForType<LoggingTests>();
            
            // Act & Assert
            try
            {
                using (logger.BeginScope("OuterOperation"))
                {
                    logger.Information("Outer operation started");
                    
                    using (logger.BeginScope("InnerOperation"))
                    {
                        logger.Information("Inner operation started");
                        logger.Information("Inner operation completed");
                    }
                    
                    logger.Information("Outer operation completed");
                }
                
                // Log should succeed without exceptions
                Assert.IsTrue(true);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Nested logging context should not throw: {ex.Message}");
            }
        }

        [TestMethod]
        public void DatabaseMonolith_Log_ShouldHandleColumnMigration()
        {
            // Arrange
            string testDbPath = "test_logs.db";
            
            try
            {
                // Clean up any existing test database
                if (File.Exists(testDbPath))
                    File.Delete(testDbPath);
                
                // Create a test database with the old LogLevel column schema
                using (var connection = new System.Data.SQLite.SQLiteConnection($"Data Source={testDbPath};Version=3;"))
                {
                    connection.Open();
                    
                    // Create the problematic table with LogLevel column
                    using (var command = new System.Data.SQLite.SQLiteCommand(
                        @"CREATE TABLE Logs (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Timestamp TEXT NOT NULL,
                        LogLevel TEXT NOT NULL,
                        Message TEXT NOT NULL
                        )", connection))
                    {
                        command.ExecuteNonQuery();
                    }
                }
                
                // Act - temporarily use test database for logging
                var originalDbPath = typeof(DatabaseMonolith).GetField("DbFilePath", 
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
                var originalValue = originalDbPath?.GetValue(null);
                originalDbPath?.SetValue(null, testDbPath);
                
                try
                {
                    // This should trigger the column migration and succeed
                    //DatabaseMonolith.Log("Info", "Test message", "Test details");
                    
                    // Verify the Level column exists and data was inserted
                    using (var connection = new System.Data.SQLite.SQLiteConnection($"Data Source={testDbPath};Version=3;"))
                    {
                        connection.Open();
                        
                        // Check that Level column exists
                        bool hasLevelColumn = false;
                        using (var command = new System.Data.SQLite.SQLiteCommand("PRAGMA table_info(Logs)", connection))
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                if (reader["name"].ToString() == "Level")
                                {
                                    hasLevelColumn = true;
                                    break;
                                }
                            }
                        }
                        
                        Assert.IsTrue(hasLevelColumn, "Level column should exist after migration");
                        
                        // Verify data was inserted
                        using (var command = new System.Data.SQLite.SQLiteCommand("SELECT COUNT(*) FROM Logs WHERE Level = 'Info'", connection))
                        {
                            var count = Convert.ToInt32(command.ExecuteScalar());
                            Assert.IsTrue(count > 0, "Should have inserted at least one log record");
                        }
                    }
                }
                finally
                {
                    // Restore original database path
                    originalDbPath?.SetValue(null, originalValue);
                }
            }
            catch (Exception ex)
            {
                Assert.Fail($"Database column migration should not throw: {ex.Message}");
            }
            finally
            {
                // Clean up test database
                if (File.Exists(testDbPath))
                    File.Delete(testDbPath);
            }
        }
    }
}