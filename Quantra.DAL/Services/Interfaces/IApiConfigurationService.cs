namespace Quantra.DAL.Services
{
    /// <summary>
    /// Interface for API configuration service that manages API keys and configuration settings.
    /// </summary>
    public interface IApiConfigurationService
    {
        /// <summary>
        /// Gets the Alpha Vantage API key
        /// </summary>
        string AlphaVantageApiKey { get; }

        /// <summary>
        /// Gets the News API key
        /// </summary>
        string NewsApiKey { get; }

        /// <summary>
        /// Gets the OpenAI API key
        /// </summary>
        string OpenAiApiKey { get; }

        /// <summary>
        /// Validates that required API keys are configured
        /// </summary>
        /// <returns>True if all required API keys are present, false otherwise</returns>
        bool ValidateApiKeys();

        /// <summary>
        /// Refreshes API keys from all configuration sources
        /// </summary>
        void RefreshApiKeys();
    }
}
