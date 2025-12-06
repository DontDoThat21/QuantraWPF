using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for checking connectivity to external APIs used by Quantra.
    /// Provides health monitoring for Alpha Vantage and other critical data sources.
    /// </summary>
    public interface IApiConnectivityService
    {
        /// <summary>
        /// Checks if all monitored APIs are reachable and responding correctly.
        /// Returns status information about the first failing endpoint, or success if all are operational.
        /// </summary>
        /// <returns>Status information about API connectivity</returns>
        Task<ApiConnectivityStatus> CheckConnectivityAsync();

        /// <summary>
        /// Gets the last successful connection time for a specific API endpoint.
        /// </summary>
        /// <param name="apiName">Name of the API endpoint</param>
        /// <returns>DateTime of last successful connection, or null if never connected</returns>
        DateTime? GetLastSuccessfulConnection(string apiName);

        /// <summary>
        /// Gets the names of all monitored API endpoints.
        /// </summary>
        /// <returns>Read-only list of monitored endpoint names</returns>
        IReadOnlyList<string> GetMonitoredEndpoints();
    }

    /// <summary>
    /// Represents the connectivity status of an external API endpoint
    /// </summary>
    public class ApiConnectivityStatus
    {
        /// <summary>
        /// Whether the API is connected and responding correctly
        /// </summary>
        public bool IsConnected { get; set; }
        
        /// <summary>
        /// The name of the API being checked
        /// </summary>
        public string ApiName { get; set; }
        
        /// <summary>
        /// Status message with details about the connection or error
        /// </summary>
        public string StatusMessage { get; set; }
        
        /// <summary>
        /// When the API was last successfully connected
        /// </summary>
        public DateTime? LastSuccessfulConnection { get; set; }
        
        /// <summary>
        /// Response time in milliseconds (null if request failed before completion)
        /// </summary>
        public double? ResponseTimeMs { get; set; }

        /// <summary>
        /// Description of what this API endpoint provides
        /// </summary>
        public string Description { get; set; }
    }
}