using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SQLite;
using Dapper; // Add Dapper namespace import
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    public class TransactionService : ITransactionService
    {
        public List<TransactionModel> GetTransactions()
        {
            return ResilienceHelper.Retry(() =>
            {
                try
                {
                    // Use DatabaseMonolith to get orders from the OrderHistory table
                    return GetOrdersFromDatabase();
                }
                catch (Exception ex)
                {
                    // Log the error
                    //DatabaseMonolith.Log("Error", "Failed to retrieve transactions from database", ex.ToString());
                
                // Return empty list in case of error
                return new List<TransactionModel>();
            }
            }, RetryOptions.ForUserFacingOperation());
        }

        private List<TransactionModel> GetOrdersFromDatabase()
        {
            var transactions = new List<TransactionModel>();
            
            using (var connection = DatabaseMonolith.GetConnection())
            {
                connection.Open();
                
                // Check if the OrderHistory table exists
                var tableExists = connection.ExecuteScalar<int>(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='OrderHistory'");
                
                if (tableExists == 0)
                {
                    // Create the OrderHistory table if it doesn't exist
                    CreateOrderHistoryTable(connection);
                    return transactions; // Return empty list as the table was just created
                }
                
                // Query to get all orders from the OrderHistory table
                string query = @"
                    SELECT 
                        Symbol, 
                        OrderType as TransactionType, 
                        Quantity, 
                        Price as ExecutionPrice,
                        (Price * Quantity) as TotalValue,
                        Timestamp as ExecutionTime,
                        IsPaperTrade,
                        StopLoss,
                        TakeProfit,
                        PredictionSource as Notes,
                        0.0 as Fees, -- Default for now, could be updated later
                        0.0 as RealizedPnL, -- Default for now, could be calculated
                        0.0 as RealizedPnLPercentage, -- Default for now
                        Status
                    FROM OrderHistory
                    ORDER BY Timestamp DESC";
                
                using (var command = new SQLiteCommand(query, connection))
                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        var transaction = new TransactionModel
                        {
                            Symbol = reader["Symbol"].ToString(),
                            TransactionType = reader["TransactionType"].ToString(),
                            Quantity = Convert.ToInt32(reader["Quantity"]),
                            ExecutionPrice = Convert.ToDouble(reader["ExecutionPrice"]),
                            TotalValue = Convert.ToDouble(reader["TotalValue"]),
                            ExecutionTime = Convert.ToDateTime(reader["ExecutionTime"]),
                            IsPaperTrade = Convert.ToBoolean(reader["IsPaperTrade"]),
                            Fees = Convert.ToDouble(reader["Fees"]),
                            RealizedPnL = Convert.ToDouble(reader["RealizedPnL"]),
                            RealizedPnLPercentage = Convert.ToDouble(reader["RealizedPnLPercentage"]),
                            Notes = reader["Notes"].ToString(),
                            OrderSource = string.IsNullOrEmpty(reader["Notes"].ToString()) ? "Manual" : "Automated"
                        };
                        transactions.Add(transaction);
                    }
                }
            }
            
            //DatabaseMonolith.Log("Info", $"Retrieved {transactions.Count} transactions from database");
            return transactions;
        }

        private void CreateOrderHistoryTable(SQLiteConnection connection)
        {
            // Create the OrderHistory table if it doesn't exist
            var createTableCommand = new SQLiteCommand(@"
                CREATE TABLE OrderHistory (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Symbol TEXT NOT NULL,
                    OrderType TEXT NOT NULL,
                    Quantity INTEGER NOT NULL,
                    Price REAL NOT NULL,
                    StopLoss REAL,
                    TakeProfit REAL,
                    IsPaperTrade INTEGER NOT NULL,
                    Status TEXT NOT NULL,
                    PredictionSource TEXT,
                    Timestamp DATETIME NOT NULL
                )", connection);
            
            createTableCommand.ExecuteNonQuery();
            //DatabaseMonolith.Log("Info", "Created OrderHistory table");
        }

        // Sample data method preserved for reference or testing
        private List<TransactionModel> GetSampleTransactions()
        {
            var transactions = new List<TransactionModel>();
            //
            //// Generate sample data
            //transactions.Add(new TransactionModel
            //{
            //    Symbol = "AAPL",
            //    TransactionType = "BUY",
            //    Quantity = 100,
            //    ExecutionPrice = 182.50,
            //    TotalValue = 18250.00,
            //    ExecutionTime = DateTime.Now.AddDays(-30),
            //    IsPaperTrade = false,
            //    Fees = 4.95,
            //    RealizedPnL = 650.50,
            //    RealizedPnLPercentage = 0.0356,
            //    Notes = "Quarterly earnings expectation"
            //});
            // ... other sample transactions
            return transactions;
        }

        // In a real app, methods below would be implemented to interact with the database

        public TransactionModel GetTransaction(int id)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    string query = @"
                        SELECT 
                            Symbol, 
                            OrderType as TransactionType, 
                            Quantity, 
                            Price as ExecutionPrice,
                            (Price * Quantity) as TotalValue,
                            Timestamp as ExecutionTime,
                            IsPaperTrade,
                            StopLoss,
                            TakeProfit,
                            PredictionSource as Notes,
                            0.0 as Fees,
                            0.0 as RealizedPnL,
                            0.0 as RealizedPnLPercentage,
                            Status
                        FROM OrderHistory
                        WHERE Id = @Id";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Id", id);
                        using (var reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                return new TransactionModel
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    TransactionType = reader["TransactionType"].ToString(),
                                    Quantity = Convert.ToInt32(reader["Quantity"]),
                                    ExecutionPrice = Convert.ToDouble(reader["ExecutionPrice"]),
                                    TotalValue = Convert.ToDouble(reader["TotalValue"]),
                                    ExecutionTime = Convert.ToDateTime(reader["ExecutionTime"]),
                                    IsPaperTrade = Convert.ToBoolean(reader["IsPaperTrade"]),
                                    Fees = Convert.ToDouble(reader["Fees"]),
                                    RealizedPnL = Convert.ToDouble(reader["RealizedPnL"]),
                                    RealizedPnLPercentage = Convert.ToDouble(reader["RealizedPnLPercentage"]),
                                    Notes = reader["Notes"].ToString()
                                };
                            }
                        }
                    }
                }
                return null;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get transaction with ID {id}", ex.ToString());
                throw;
            }
        }

        public List<TransactionModel> GetTransactionsByDateRange(DateTime startDate, DateTime endDate)
        {
            try
            {
                var transactions = new List<TransactionModel>();
                
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    string query = @"
                        SELECT 
                            Symbol, 
                            OrderType as TransactionType, 
                            Quantity, 
                            Price as ExecutionPrice,
                            (Price * Quantity) as TotalValue,
                            Timestamp as ExecutionTime,
                            IsPaperTrade,
                            StopLoss,
                            TakeProfit,
                            PredictionSource as Notes,
                            0.0 as Fees,
                            0.0 as RealizedPnL,
                            0.0 as RealizedPnLPercentage,
                            Status
                        FROM OrderHistory
                        WHERE Timestamp BETWEEN @StartDate AND @EndDate
                        ORDER BY Timestamp DESC";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@StartDate", startDate.ToString("yyyy-MM-dd HH:mm:ss"));
                        command.Parameters.AddWithValue("@EndDate", endDate.ToString("yyyy-MM-dd HH:mm:ss"));
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                transactions.Add(new TransactionModel
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    TransactionType = reader["TransactionType"].ToString(),
                                    Quantity = Convert.ToInt32(reader["Quantity"]),
                                    ExecutionPrice = Convert.ToDouble(reader["ExecutionPrice"]),
                                    TotalValue = Convert.ToDouble(reader["TotalValue"]),
                                    ExecutionTime = Convert.ToDateTime(reader["ExecutionTime"]),
                                    IsPaperTrade = Convert.ToBoolean(reader["IsPaperTrade"]),
                                    Fees = Convert.ToDouble(reader["Fees"]),
                                    RealizedPnL = Convert.ToDouble(reader["RealizedPnL"]),
                                    RealizedPnLPercentage = Convert.ToDouble(reader["RealizedPnLPercentage"]),
                                    Notes = reader["Notes"].ToString()
                                });
                            }
                        }
                    }
                }
                
                return transactions;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to get transactions by date range", ex.ToString());
                throw;
            }
        }

        public List<TransactionModel> GetTransactionsBySymbol(string symbol)
        {
            try
            {
                var transactions = new List<TransactionModel>();
                
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    string query = @"
                        SELECT 
                            Symbol, 
                            OrderType as TransactionType, 
                            Quantity, 
                            Price as ExecutionPrice,
                            (Price * Quantity) as TotalValue,
                            Timestamp as ExecutionTime,
                            IsPaperTrade,
                            StopLoss,
                            TakeProfit,
                            PredictionSource as Notes,
                            0.0 as Fees,
                            0.0 as RealizedPnL,
                            0.0 as RealizedPnLPercentage,
                            Status
                        FROM OrderHistory
                        WHERE Symbol = @Symbol
                        ORDER BY Timestamp DESC";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol);
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                transactions.Add(new TransactionModel
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    TransactionType = reader["TransactionType"].ToString(),
                                    Quantity = Convert.ToInt32(reader["Quantity"]),
                                    ExecutionPrice = Convert.ToDouble(reader["ExecutionPrice"]),
                                    TotalValue = Convert.ToDouble(reader["TotalValue"]),
                                    ExecutionTime = Convert.ToDateTime(reader["ExecutionTime"]),
                                    IsPaperTrade = Convert.ToBoolean(reader["IsPaperTrade"]),
                                    Fees = Convert.ToDouble(reader["Fees"]),
                                    RealizedPnL = Convert.ToDouble(reader["RealizedPnL"]),
                                    RealizedPnLPercentage = Convert.ToDouble(reader["RealizedPnLPercentage"]),
                                    Notes = reader["Notes"].ToString()
                                });
                            }
                        }
                    }
                }
                
                return transactions;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get transactions for symbol {symbol}", ex.ToString());
                throw;
            }
        }

        public void SaveTransaction(TransactionModel transaction)
        {
            ResilienceHelper.Retry(() =>
            {
                try
                {
                    using (var connection = DatabaseMonolith.GetConnection())
                    {
                        connection.Open();
                        
                        string query = @"
                            INSERT INTO OrderHistory (
                                Symbol, OrderType, Quantity, Price, 
                                StopLoss, TakeProfit, IsPaperTrade, 
                                Status, PredictionSource, Timestamp
                            )
                        VALUES (
                            @Symbol, @OrderType, @Quantity, @Price,
                            @StopLoss, @TakeProfit, @IsPaperTrade,
                            @Status, @PredictionSource, @Timestamp
                        )";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", transaction.Symbol);
                        command.Parameters.AddWithValue("@OrderType", transaction.TransactionType);
                        command.Parameters.AddWithValue("@Quantity", transaction.Quantity);
                        command.Parameters.AddWithValue("@Price", transaction.ExecutionPrice);
                        command.Parameters.AddWithValue("@StopLoss", 0.0); // Default value, can be updated
                        command.Parameters.AddWithValue("@TakeProfit", 0.0); // Default value, can be updated
                        command.Parameters.AddWithValue("@IsPaperTrade", transaction.IsPaperTrade ? 1 : 0);
                        command.Parameters.AddWithValue("@Status", "Executed");
                        command.Parameters.AddWithValue("@PredictionSource", transaction.Notes ?? string.Empty);
                        command.Parameters.AddWithValue("@Timestamp", transaction.ExecutionTime.ToString("yyyy-MM-dd HH:mm:ss.fff"));
                        
                        command.ExecuteNonQuery();
                    }
                }
                
                //DatabaseMonolith.Log("Info", $"Saved transaction for {transaction.Symbol} ({transaction.TransactionType})");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to save transaction", ex.ToString());
                throw;
            }
            }, RetryOptions.ForCriticalOperation());
        }

        public void DeleteTransaction(int id)
        {
            ResilienceHelper.Retry(() =>
            {
                try
                {
                    using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    string query = "DELETE FROM OrderHistory WHERE Id = @Id";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Id", id);
                        command.ExecuteNonQuery();
                    }
                }
                
                //DatabaseMonolith.Log("Info", $"Deleted transaction with ID {id}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to delete transaction with ID {id}", ex.ToString());
                throw;
            }
            }, RetryOptions.ForCriticalOperation());
        }
    }
}
