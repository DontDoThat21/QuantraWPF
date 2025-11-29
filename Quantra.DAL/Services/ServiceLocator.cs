using System;
using System.Collections.Generic;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Simple service locator pattern for retrieving application services
    /// DEPRECATED: New code should use proper dependency injection through constructors.
    /// This class is maintained only for backward compatibility with legacy code.
    /// 
    /// For new services:
    /// 1. Register services in ServiceCollectionExtensions.AddQuantraServices()
    /// 2. Inject services through constructor parameters
    /// 3. Use Microsoft.Extensions.DependencyInjection framework
    /// 
    /// Example:
    /// public class MyService
    /// {
    ///     private readonly IAnalystRatingService _analystRatingService;
    ///     
    ///     public MyService(IAnalystRatingService analystRatingService)
    ///     {
    ///         _analystRatingService = analystRatingService;
    ///     }
    /// }
    /// </summary>
    [Obsolete("Use proper dependency injection through constructors instead. See ServiceCollectionExtensions.AddQuantraServices()")]
    public static class ServiceLocator
    {
        private static readonly Dictionary<Type, object> _services = new Dictionary<Type, object>();
        private static readonly Dictionary<string, object> _namedServices = new Dictionary<string, object>();

        /// <summary>
        /// Register a service instance for a given type
        /// </summary>
        /// <typeparam name="T">The service interface type</typeparam>
        /// <param name="implementation">The service implementation</param>
        public static void RegisterService<T>(T implementation)
        {
            var serviceType = typeof(T);
            if (_services.ContainsKey(serviceType))
            {
                _services[serviceType] = implementation;
            }
            else
            {
                _services.Add(serviceType, implementation);
            }
        }

        /// <summary>
        /// Register a named service instance
        /// </summary>
        /// <param name="serviceName">The unique name for the service</param>
        /// <param name="implementation">The service implementation</param>
        public static void RegisterService(string serviceName, object implementation)
        {
            if (_namedServices.ContainsKey(serviceName))
            {
                _namedServices[serviceName] = implementation;
            }
            else
            {
                _namedServices.Add(serviceName, implementation);
            }
        }

        /// <summary>
        /// Get a service of the specified type
        /// </summary>
        /// <typeparam name="T">The type of service to retrieve</typeparam>
        /// <returns>The service instance</returns>
        public static T GetService<T>() where T : class
        {
            var serviceType = typeof(T);
            if (_services.TryGetValue(serviceType, out var service))
            {
                return (T)service;
            }

            throw new InvalidOperationException($"Service of type {serviceType.Name} is not registered");
        }

        /// <summary>
        /// Check if a service of the specified type is registered
        /// </summary>
        /// <typeparam name="T">The type of service to check</typeparam>
        /// <returns>True if the service is registered, false otherwise</returns>
        public static bool IsServiceRegistered<T>()
        {
            return _services.ContainsKey(typeof(T));
        }

        /// <summary>
        /// Resolve a service of the specified type (alias for GetService)
        /// </summary>
        /// <typeparam name="T">The type of service to retrieve</typeparam>
        /// <returns>The service instance</returns>
        public static T Resolve<T>() where T : class
        {
            // Try to use the registered service first
            var serviceType = typeof(T);
            if (_services.TryGetValue(serviceType, out var service))
            {
                return (T)service;
            }

            // For the technical indicator service, create a mock implementation if needed
            if (typeof(T).Name == "ITechnicalIndicatorService")
            {
                // Create a dynamic mock for testing purposes
                return new MockTechnicalIndicatorService() as T;
            }

            throw new InvalidOperationException($"Service of type {serviceType.Name} is not registered");
        }

        /// <summary>
        /// Resolve a named service of the specified type
        /// </summary>
        /// <typeparam name="T">The type of service to retrieve</typeparam>
        /// <param name="serviceName">The name of the service to retrieve</param>
        /// <returns>The service instance</returns>
        public static T Resolve<T>(string serviceName) where T : class
        {
            // Try to find the named service first
            if (_namedServices.TryGetValue(serviceName, out var namedService))
            {
                if (namedService is T service)
                {
                    return service;
                }
                throw new InvalidOperationException($"Named service '{serviceName}' is not of type {typeof(T).Name}");
            }

            throw new InvalidOperationException($"Named service '{serviceName}' of type {typeof(T).Name} is not registered");
        }
    }
}