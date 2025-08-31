using System;
using System.Collections.Generic;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.CrossCutting.Monitoring;
using Quantra.CrossCutting.Security;

namespace Quantra.CrossCutting
{
    /// <summary>
    /// Central registry for all cross-cutting concerns.
    /// </summary>
    public static class CrossCuttingRegistry
    {
        private static bool _isInitialized;
        private static readonly object _lock = new object();
        
        /// <summary>
        /// Initializes all cross-cutting modules.
        /// </summary>
        public static void Initialize()
        {
            if (_isInitialized)
            {
                return;
            }
            
            lock (_lock)
            {
                if (_isInitialized)
                {
                    return;
                }
                
                // Initialize in correct order
                // 1. Logging (needed by everything else)
                LoggingManager.Instance.Initialize();
                
                // 2. Security (needed for secure logging)
                SecurityManager.Instance.Initialize();
                
                // 3. Error handling
                ErrorHandlingManager.Instance.Initialize();
                
                // 4. Monitoring
                MonitoringManager.Instance.Initialize();
                
                _isInitialized = true;
                
                // Now log that we've initialized
                var logger = Log.ForType(typeof(CrossCuttingRegistry));
                logger.Information("Cross-cutting concerns framework initialized successfully");
            }
        }
        
        /// <summary>
        /// Gets all registered cross-cutting modules.
        /// </summary>
        public static IEnumerable<ICrossCuttingModule> GetAllModules()
        {
            yield return LoggingManager.Instance;
            yield return SecurityManager.Instance;
            yield return ErrorHandlingManager.Instance;
            yield return MonitoringManager.Instance;
        }
    }
}