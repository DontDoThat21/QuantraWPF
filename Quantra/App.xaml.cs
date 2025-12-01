using System;
using System.Configuration;
using System.Data;
using System.IO;
using System.Windows;
using Quantra.DAL.Services.Interfaces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.Json;
using Quantra.Extensions;
using Quantra.Configuration;
using Quantra.Configuration.Providers;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.CrossCutting.Monitoring;
using Quantra.Utilities;
using Quantra.DAL.Services;
using System.Reflection;
using Quantra.DAL.Data;
using Microsoft.EntityFrameworkCore;
using Quantra.ViewModels;

namespace Quantra
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public static ServiceProvider ServiceProvider { get; private set; }
        public static IConfiguration Configuration { get; private set; }
        private ILogger _logger;

        protected override void OnStartup(StartupEventArgs e)
        {
            // Initialize cross-cutting concerns first
            CrossCuttingRegistry.Initialize();
            _logger = Log.ForType<App>();

            _logger.Information("Application starting");

            // Determine the environment
            string environment = Environment.GetEnvironmentVariable("QUANTRA_ENVIRONMENT")
                ?? Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT")
                ?? "Development";

            _logger.ForContext("Environment", environment)
                  .Information("Running in {Environment} environment", environment);

            try
            {
                // Build hierarchical configuration from multiple sources
                var builder = new ConfigurationBuilder()
                    .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
                    .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                    .AddJsonFile($"appsettings.{environment}.json", optional: true, reloadOnChange: true)
                    .AddJsonFile("usersettings.json", optional: true, reloadOnChange: true)
                    .AddEnvironmentVariables();
                // Disambiguate AddCommandLine by calling the static method
                Microsoft.Extensions.Configuration.CommandLineConfigurationExtensions.AddCommandLine(builder, e.Args);

                // Add custom Quantra environment variables provider
                builder.AddQuantraEnvironmentVariables();

                // Build the final configuration
                Configuration = builder.Build();
                _logger.Information("Configuration loaded successfully");
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to initialize application configuration");
                ResilienceHelper.HandleException(ex, "ApplicationStartup");
                MessageBox.Show($"Failed to initialize application: {ex.Message}", "Startup Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                Environment.Exit(1);
            }

            base.OnStartup(e);

            try
            {
                // Set up Dependency Injection
                using (_logger.BeginTimedOperation("ConfigureServices"))
                {
                    var serviceCollection = new ServiceCollection();
                    serviceCollection.AddSingleton(Configuration);

                    ConfigureServices(serviceCollection);
                    ServiceProvider = serviceCollection.BuildServiceProvider();
                    _logger.Information("Services configured successfully");
                }

                // Initialize DatabaseMonolith with configuration
                var settingsService = ServiceProvider.GetRequiredService<SettingsService>();
                var loggingService = new LoggingService();
                var databaseMonolith = new DatabaseMonolith(settingsService, loggingService);
                databaseMonolith.SetConfiguration(Configuration);

                // Initialize the system health monitoring service
                var healthMonitor = ServiceProvider.GetService<SystemHealthMonitorService>();
                if (healthMonitor != null)
                {
                    // Register the service in ServiceLocator
                    ServiceLocator.RegisterService(healthMonitor);

                    // Run an initial health check
                    _logger.Information("Running initial system health check");
                    _ = healthMonitor.CheckSystemHealthAsync();
                }

                // Initialize database config bridge for config-database synchronization
                var configBridge = ServiceProvider.GetService<DatabaseConfigBridge>();

                // Migrate legacy configuration if needed
                var configManager = ServiceProvider.GetService<Quantra.Configuration.IConfigurationManager>();
                if (configManager != null)
                {
                    _ = ConfigurationMigration.MigrateFromLegacySources(configManager);
                }

                // Register to hook each window as it's created to add resize functionality
                this.Startup += App_Startup;

                // Create and show LoginWindow with dependency injection
                var userSettingsService = ServiceProvider.GetRequiredService<UserSettingsService>();
                var historicalDataService = ServiceProvider.GetRequiredService<HistoricalDataService>();
                var alphaVantageService = ServiceProvider.GetRequiredService<AlphaVantageService>();
                var technicalIndicatorService = ServiceProvider.GetRequiredService<TechnicalIndicatorService>();
                var authenticationService = ServiceProvider.GetRequiredService<AuthenticationService>();

                var loginViewModel = new ViewModels.LoginWindowViewModel(
                    userSettingsService,
                    historicalDataService,
                    alphaVantageService,
                    technicalIndicatorService,
                    authenticationService);

                var loginWindow = new LoginWindow(loginViewModel);
                loginWindow.Show();

                _logger.Information("Application startup completed successfully");

                // Record initial memory usage
                Performance.RecordMemoryUsage("ApplicationStart");
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to complete application startup");
                ResilienceHelper.HandleException(ex, "ApplicationStartup");
                MessageBox.Show($"Failed to complete application startup: {ex.Message}",
                    "Startup Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ConfigureServices(IServiceCollection services)
        {
            try
            {
                // Register LoggingService first (needed by other services)
                services.AddSingleton<LoggingService>();

                // Register application services using the extension method
                // This will register DbContext, Settings services, and all other services
                Quantra.Extensions.ServiceCollectionExtensions.AddQuantraServices(services);

                // Register AlertPublisher so non-UI services can emit alerts via DI
                services.AddSingleton<IAlertPublisher, AlertPublisher>();

                // Initialize SystemHealthMonitorService (will be registered via AddQuantraServices)

                // Register services in ServiceLocator for components that don't use DI
                // Note: We'll register these after building the ServiceProvider

                _logger.Debug("Service registration completed");
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to configure services");
                throw; // Rethrow to be handled by caller
            }
        }

        private void App_Startup(object sender, StartupEventArgs e)
        {
            // Instead of trying to hook into Windows.SourceInitialized (which doesn't exist),
            // we'll monitor window creation through the Application.Current.Activated event
            // and attach the WindowOpened event to catch new windows
            Application.Current.Activated += Application_Activated;
            _logger.Debug("Application startup event handler registered");

            // Register services in ServiceLocator after ServiceProvider is built
            try
            {
                var healthMonitor = ServiceProvider.GetService<SystemHealthMonitorService>();
                if (healthMonitor != null)
                {
                    ServiceLocator.RegisterService(healthMonitor);
                }

                var technicalIndicatorService = ServiceProvider.GetService<ITechnicalIndicatorService>();
                if (technicalIndicatorService != null)
                {
                    ServiceLocator.RegisterService<ITechnicalIndicatorService>(technicalIndicatorService);
                }

                var stockDataCacheService = ServiceProvider.GetService<IStockDataCacheService>();
                if (stockDataCacheService != null)
                {
                    ServiceLocator.RegisterService<IStockDataCacheService>(stockDataCacheService);
                }

                // Register sentiment analysis services for legacy ServiceLocator compatibility
                var financialNewsSentimentService = ServiceProvider.GetService<FinancialNewsSentimentService>();
                if (financialNewsSentimentService != null)
                {
                    ServiceLocator.RegisterService(financialNewsSentimentService);
                }

                var socialMediaSentimentService = ServiceProvider.GetService<ISocialMediaSentimentService>();
                if (socialMediaSentimentService != null)
                {
                    ServiceLocator.RegisterService(socialMediaSentimentService);
                }

                var analystRatingService = ServiceProvider.GetService<IAnalystRatingService>();
                if (analystRatingService != null)
                {
                    ServiceLocator.RegisterService(analystRatingService);
                }

                var insiderTradingService = ServiceProvider.GetService<IInsiderTradingService>();
                if (insiderTradingService != null)
                {
                    ServiceLocator.RegisterService(insiderTradingService);
                }

                var sectorSentimentService = ServiceProvider.GetService<SectorSentimentAnalysisService>();
                if (sectorSentimentService != null)
                {
                    ServiceLocator.RegisterService(sectorSentimentService);
                }

                var sectorMomentumService = ServiceProvider.GetService<SectorMomentumService>();
                if (sectorMomentumService != null)
                {
                    ServiceLocator.RegisterService(sectorMomentumService);
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to register services in ServiceLocator");
            }
        }

        private void Application_Activated(object sender, System.EventArgs e)
        {
            try
            {
                // Check all current windows and apply resize behavior if needed
                foreach (Window window in Application.Current.Windows)
                {
                    ApplyResizableBehaviorIfNeeded(window);
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error in Application_Activated");
                ResilienceHelper.HandleException(ex, "WindowActivation");
            }
        }

        private void ApplyResizableBehaviorIfNeeded(Window window)
        {
            // Only apply to borderless windows that don't have the behavior yet
            if (window != null &&
                window.AllowsTransparency &&
                window.WindowStyle == WindowStyle.None &&
                !window.Tag?.ToString().Contains("ResizableApplied") == true)
            {
                // Set minimum dimensions if not already set
                if (window.MinWidth <= 0)
                    window.MinWidth = 300;
                if (window.MinHeight <= 0)
                    window.MinHeight = 200;

                // Attach resize behavior via reflection to avoid hard project dependency here
                ResizableBehaviorHelper.Attach(window);

                // Mark the window to avoid applying behavior multiple times
                window.Tag = "ResizableApplied";

                _logger.Debug("Applied resize behavior to window {WindowType}", window.GetType().Name);
            }
        }

        protected override void OnExit(ExitEventArgs e)
        {
            try
            {
                // Flush all logs to ensure everything is written
                Log.Flush();
                _logger.Information("Application exiting with code {ExitCode}", e.ApplicationExitCode);
            }
            catch (Exception ex)
            {
                // Direct console logging as a last resort
                Console.Error.WriteLine($"Error during application shutdown: {ex.Message}");
            }

            base.OnExit(e);
        }

        private static class ResizableBehaviorHelper
        {
            public static void Attach(Window window)
            {
                try
                {
                    if (window == null) return;

                    var type = Type.GetType("Quantra.WindowResizeBehavior");
                    if (type == null) return;

                    var method = type.GetMethod(
                        name: "AttachResizeBehavior",
                        BindingFlags.Public | BindingFlags.Static);

                    if (method == null) return;

                    var parameters = method.GetParameters();
                    if (parameters == null || parameters.Length == 0) return;

                    // Build arguments array with default values for optional parameters
                    object[] args = new object[parameters.Length];
                    args[0] = window;

                    // Fill in default values for optional parameters
                    for (int i = 1; i < parameters.Length; i++)
                    {
                        args[i] = parameters[i].DefaultValue ?? Type.Missing;
                    }

                    method.Invoke(null, args);
                }
                catch
                {
                    // Intentionally ignore any reflection errors to avoid crashing on startup
                }
            }
        }
    }
}
