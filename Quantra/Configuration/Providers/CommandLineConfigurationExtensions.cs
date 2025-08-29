using System;
using System.Collections.Generic;
using Microsoft.Extensions.Configuration;

namespace Quantra.Configuration.Providers
{
    /// <summary>
    /// Extension methods for adding command-line configuration
    /// </summary>
    public static class CommandLineConfigurationExtensions
    {
        /// <summary>
        /// Adds the command line configuration provider to the builder
        /// </summary>
        /// <param name="builder">The configuration builder to add to</param>
        /// <param name="args">The command line arguments</param>
        /// <returns>The configuration builder</returns>
        /// todo: seems like this is unused. Remove?
        public static IConfigurationBuilder AddCommandLine(this IConfigurationBuilder builder, string[] args)
        {
            if (builder == null)
                throw new ArgumentNullException(nameof(builder));
                
            if (args == null || args.Length == 0)
                return builder;
                
            return builder.AddCommandLine(args, new Dictionary<string, string>
            {
                // Map command line parameters to configuration keys
                { "--environment", "Application:Environment" },
                { "--api-key", "Api:AlphaVantage:ApiKey" },
                { "--news-api-key", "Api:News:ApiKey" },
                { "--risk", "Trading:RiskLevel" },
                { "--paper-trading", "Trading:EnablePaperTrading" },
                { "--dark-mode", "UI:EnableDarkMode" },
                { "--cache-minutes", "Cache:CacheDurationMinutes" },
                { "--no-cache", "Cache:EnableHistoricalDataCache" },
                { "--account-size", "Trading:AccountSize" }
            });
        }
    }
}