using System;
using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for checking connectivity to external APIs
    /// </summary>
    public interface IApiConnectivityService
    {
        /// <summary>
        /// Checks if the API is reachable and responding correctly
        /// </summary>
        Task<ApiConnectivityStatus> CheckConnectivityAsync();
    }

    /// <summary>
    /// Represents the connectivity status of an external API
    /// </summary>
    public class ApiConnectivityStatus
    {
        /// <summary>
        /// Whether the API is connected and responding
        /// </summary>
        public bool IsConnected { get; set; }
        
        /// <summary>
        /// The name of the API being checked
        /// </summary>
        public string ApiName { get; set; }
        
        /// <summary>
        /// Status message with details about the connection
        /// </summary>
        public string StatusMessage { get; set; }
        
        /// <summary>
        /// When the API was last successfully connected
        /// </summary>
        public DateTime? LastSuccessfulConnection { get; set; }
        
        /// <summary>
        /// Response time in milliseconds (if available)
        /// </summary>
        public double? ResponseTimeMs { get; set; }
    }
}