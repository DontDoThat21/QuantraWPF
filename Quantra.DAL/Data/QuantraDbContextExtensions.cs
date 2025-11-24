using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Quantra.DAL.Data.Repositories;
using System;

namespace Quantra.DAL.Data
{
    /// <summary>
    /// Extension methods for configuring Quantra database services with dependency injection
    /// </summary>
    public static class QuantraDbContextExtensions
    {
        /// <summary>
        /// Adds Quantra database services to the service collection
        /// </summary>
        /// <param name="services">The service collection</param>
        /// <param name="connectionString">Optional connection string (defaults to SQL Server LocalDB)</param>
        /// <returns>The service collection for chaining</returns>
        public static IServiceCollection AddQuantraDatabase(
         this IServiceCollection services,
       string connectionString = null)
        {
            connectionString ??= ConnectionHelper.ConnectionString;

            // Register DbContext
            services.AddDbContext<QuantraDbContext>(options =>
            {
                options.UseSqlServer(connectionString, sqlServerOptions =>
                         {
                             sqlServerOptions.CommandTimeout(30);
                         });

                // Enable sensitive data logging in development
#if DEBUG
                options.EnableSensitiveDataLogging();
                options.EnableDetailedErrors();
#endif
            });

            // Register repositories
            services.AddScoped<ILogRepository, LogRepository>();
            services.AddScoped<IStockSymbolRepository, StockSymbolRepository>();
            services.AddScoped<ITradingRuleRepository, TradingRuleRepository>();

            // Register generic repository
            services.AddScoped(typeof(IRepository<>), typeof(Repository<>));

            return services;
        }

        /// <summary>
        /// Initializes the database (creates tables, applies migrations)
        /// Call this during application startup
        /// </summary>
        public static IServiceProvider InitializeQuantraDatabase(this IServiceProvider serviceProvider)
        {
            using (var scope = serviceProvider.CreateScope())
            {
                var context = scope.ServiceProvider.GetRequiredService<QuantraDbContext>();
                context.Initialize();
            }

            return serviceProvider;
        }
    }
}
