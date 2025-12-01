using System;
using Microsoft.Data.SqlClient;
using Quantra.DAL.Services;

namespace Quantra.DAL.Data
{
    /// <summary>
    /// Helper class for managing SQL Server LocalDB connections for the QuantraRelational database
    /// </summary>
    public class ConnectionHelper
    {
        private const string DefaultConnectionString =
 "Data Source=(localdb)\\MSSQLLocalDB;" +
     "Initial Catalog=QuantraRelational;" +
            "Integrated Security=True;" +
      "Persist Security Info=False;" +
         "Pooling=True;" +
            "MultipleActiveResultSets=True;" +
       "Encrypt=True;" +
            "TrustServerCertificate=False;" +
      "Application Name=\"Quantra Trading Platform\";" +
        "Command Timeout=0";

        private static string _connectionString;

        /// <summary>
        /// Gets or sets the connection string. If not set, uses the default LocalDB connection string.
        /// </summary>
        public static string ConnectionString
        {
            get => _connectionString ?? DefaultConnectionString;
            set => _connectionString = value;
        }

        /// <summary>
        /// Gets a new SQL Server connection to the QuantraRelational database
        /// </summary>
        /// <returns>A SqlConnection instance (not opened)</returns>
        /// <exception cref="InvalidOperationException">Thrown when connection cannot be created</exception>
        public static SqlConnection GetConnection()
        {
            try
            {
                var connection = new SqlConnection(ConnectionString);
                return connection;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to create database connection. See inner exception for details.", ex);
            }
        }

        /// <summary>
        /// Gets a new SQL Server connection and opens it
        /// </summary>
        /// <returns>An open SqlConnection instance</returns>
        /// <exception cref="InvalidOperationException">Thrown when connection cannot be created or opened</exception>
        public static SqlConnection GetOpenConnection()
        {
            SqlConnection connection = null;
            try
            {
                connection = GetConnection();
                connection.Open();
                return connection;
            }
            catch (Exception ex)
            {
                connection?.Dispose();
                throw new InvalidOperationException("Failed to open database connection. Ensure LocalDB is installed and running.", ex);
            }
        }

        /// <summary>
        /// Tests the connection to ensure it can be established
        /// </summary>
        /// <returns>True if connection is successful, false otherwise</returns>
        public static bool TestConnection()
        {
            try
            {
                using (var connection = GetOpenConnection())
                {
                    return true;
                }
            }
            catch (Exception ex)
            {
                return false;
            }
        }

        /// <summary>
        /// Ensures the QuantraRelational database exists. Creates it if it doesn't exist.
        /// </summary>
        /// <returns>True if database exists or was created successfully, false otherwise</returns>
        public static bool EnsureDatabaseExists()
        {
            try
            {
                // First try to connect to the database
                using (var connection = GetOpenConnection())
                {
                    return true;
                }
            }
            catch
            {
                // Database doesn't exist, try to create it
                try
                {
                    // Connect to master database to create the database
                    var masterConnectionString =
               "Data Source=(localdb)\\MSSQLLocalDB;" +
                          "Initial Catalog=QuantraRelational;" +
                "Integrated Security=True;" +
                            "Encrypt=True;" +
                                "TrustServerCertificate=False";

                    using (var connection = new SqlConnection(masterConnectionString))
                    {
                        connection.Open();
                        using (var command = connection.CreateCommand())
                        {
                            command.CommandText = @"
   IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'QuantraRelational')
            BEGIN
           CREATE DATABASE [QuantraRelational]
           END";
                            command.ExecuteNonQuery();
                        }
                    }
                    return true;
                }
                catch (Exception ex)
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets the database name from the connection string
        /// </summary>
        /// <returns>The database name</returns>
        public static string GetDatabaseName()
        {
            try
            {
                using (var connection = GetConnection())
                {
                    return connection.Database;
                }
            }
            catch
            {
                return "QuantraRelational";
            }
        }

        /// <summary>
        /// Gets the server name from the connection string
        /// </summary>
        /// <returns>The server name</returns>
        public static string GetServerName()
        {
            try
            {
                using (var connection = GetConnection())
                {
                    return connection.DataSource;
                }
            }
            catch
            {
                return "(localdb)\\MSSQLLocalDB";
            }
        }
    }
}
