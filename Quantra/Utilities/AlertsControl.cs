using System;
using System.Collections.Generic;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Utilities
{
    public static class AlertManager
    {
        private static readonly List<Action<AlertModel>> _alertHandlers = new();
        private static IAudioService _audioService;
        private static INotificationService _notificationService;
        private static ISettingsService _settingsService;
        private static readonly ILogger _logger = Log.ForType(typeof(AlertManager));
        
        static AlertManager()
        {
            // Ensure cross-cutting concerns are initialized
            CrossCuttingRegistry.Initialize();
        }

        public static void Initialize(IAudioService audioService, INotificationService notificationService)
        {
            _audioService = audioService ?? throw new ArgumentNullException(nameof(audioService));
            _notificationService = notificationService ?? throw new ArgumentNullException(nameof(notificationService));
            
            // Try to resolve settings service from DI container or create fallback
            try
            {
                _settingsService = ServiceLocator.Resolve<ISettingsService>();
            }
            catch (Exception ex)
            {
                _logger.Warning(ex.Message, audioService);
                _settingsService = new SettingsService();
            }
            
            _logger.Information("AlertManager initialized with audio and notification services");
        }

        public static void RegisterAlertHandler(Action<AlertModel> handler)
        {
            if (!_alertHandlers.Contains(handler))
            {
                _alertHandlers.Add(handler);
                _logger.Debug("Alert handler registered, total handlers: {HandlerCount}", _alertHandlers.Count);
            }
        }

        public static void UnregisterAlertHandler(Action<AlertModel> handler)
        {
            _alertHandlers.Remove(handler);
            _logger.Debug("Alert handler unregistered, total handlers: {HandlerCount}", _alertHandlers.Count);
        }

        public static void EmitGlobalAlert(AlertModel alert)
        {
            if (alert == null)
            {
                _logger.Warning("Attempted to emit null alert");
                return;
            }

            try
            {
                // Use the resilience patterns for the alert emission
                ResilienceHelper.Retry(() =>
                {
                    using var perfTimer = _logger.BeginTimedOperation("EmitGlobalAlert");
                    
                    _logger.ForContext("AlertName", alert.Name)
                           .ForContext("AlertCategory", alert.Category)
                           .ForContext("AlertPriority", alert.Priority)
                           .Information("Emitting alert: {AlertCondition}", alert.Condition);
                    
                    // Log the alert to the database for backward compatibility
                    DatabaseMonolith.Log(alert.Category.ToString(), alert.Name, alert.Notes);

                    // Play sound if enabled
                    if (_audioService != null && alert.EnableSound)
                    {
                        _audioService.PlayAlertSound(alert);
                    }

                    // Show visual notification if notification service is available
                    if (_notificationService != null)
                    {
                        _notificationService.ShowAlertNotification(alert);
                    }

                    // Get the current database settings profile
                    var settings = _settingsService.GetDefaultSettingsProfile();
                    
                    // Send email notification if enabled
                    if (settings != null)
                    {
                        // Send SMS notification if enabled
                        ResilienceHelper.Retry(() => _smsAlertService.SendAlertSms(alert, settings));
                        
                        // Send email notification if enabled
                        ResilienceHelper.Retry(() => _emailAlertService.SendAlertEmail(alert, settings));

                        // Send push notification if enabled
                        ResilienceHelper.Retry(() => _pushNotificationAlertService.SendAlertPushNotification(alert, settings));
                    }

                    // Notify all registered handlers
                    foreach (var handler in _alertHandlers)
                    {
                        try
                        {
                            handler?.Invoke(alert);
                        }
                        catch (Exception handlerEx)
                        {
                            // Log but continue with other handlers
                            _logger.Warning(handlerEx.Message, "Alert handler failed for alert {AlertName}", alert.Name);
                        }
                    }
                }, RetryOptions.ForUserFacingOperation());
            }
            catch (Exception ex)
            {
                // Log using our new framework
                _logger.Error(ex, "Failed to emit global alert: {AlertName}", alert.Name);
                
                // Also log to database for backward compatibility
                DatabaseMonolith.Log("Error", "Failed to emit global alert", ex.ToString());
            }
        }

        /// <summary>
        /// Emits a standardized error alert that is UI-agnostic (Helpers/DAL layer responsibility).
        /// </summary>
        public static void EmitGlobalError(string message, Exception ex = null)
        {
            try
            {
                var alert = new AlertModel
                {
                    Name = message,
                    Condition = "Error",
                    AlertType = "Error",
                    IsActive = true,
                    Priority = 1,
                    CreatedDate = DateTime.Now,
                    Category = AlertCategory.Global,
                    Notes = ex?.ToString() ?? string.Empty
                };

                EmitGlobalAlert(alert);
            }
            catch (Exception emitEx)
            {
                _logger.Error(emitEx, "Failed to emit global error: {ErrorMessage}", message);
                DatabaseMonolith.Log("Error", message, emitEx.ToString());
            }
        }
    }
}