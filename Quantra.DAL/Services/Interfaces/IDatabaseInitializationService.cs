namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for database initialization and table creation using Entity Framework Core
    /// </summary>
    public interface IDatabaseInitializationService
    {
        /// <summary>
        /// Ensures all required database tables exist
        /// </summary>
        void EnsureAllTablesExist();

        /// <summary>
        /// Ensures UserAppSettings table exists
        /// </summary>
        void EnsureUserAppSettingsTable();

        /// <summary>
        /// Ensures UserCredentials table exists
        /// </summary>
        void EnsureUserCredentialsTable();

        /// <summary>
        /// Ensures Logs table exists
        /// </summary>
        void EnsureLogsTable();

        /// <summary>
        /// Ensures TabConfigs table exists
        /// </summary>
        void EnsureTabConfigsTable();

        /// <summary>
        /// Ensures Settings table exists with default values
        /// </summary>
        void EnsureSettingsTable();

        /// <summary>
        /// Applies any pending database migrations
        /// </summary>
        void ApplyMigrations();

        /// <summary>
        /// Tests if the database connection is working
        /// </summary>
        bool TestConnection();

        /// <summary>
        /// Gets information about the database
        /// </summary>
        string GetDatabaseInfo();
    }
}
