using System;

namespace Quantra.CrossCutting
{
    /// <summary>
    /// Base interface for all cross-cutting concern modules.
    /// </summary>
    public interface ICrossCuttingModule
    {
        /// <summary>
        /// Initializes the module with optional configuration.
        /// </summary>
        /// <param name="configurationSection">Optional configuration section name.</param>
        void Initialize(string configurationSection = null);

        /// <summary>
        /// Gets the name of the module.
        /// </summary>
        string ModuleName { get; }
    }
}