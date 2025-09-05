using System;
using System.Reflection;
using Quantra.Models;

namespace Quantra.Utilities
{
    /// <summary>
    /// DAL-level alerting helper to emit error alerts without referencing UI components.
    /// </summary>
    public static class Alerting
    {
        /// <summary>
        /// Emits a global alert in a UI-agnostic way. Logs to the database and, if available at runtime,
        /// forwards to the UI AlertManager via reflection without taking a compile-time dependency.
        /// </summary>
        public static void EmitGlobalAlert(AlertModel alert)
        {
            if (alert == null)
            {
                EmitGlobalError("Attempted to emit null alert");
                return;
            }

            try
            {
                // Log to DB first so alerts are persisted even without UI present
                Quantra.DatabaseMonolith.Log(alert.Category.ToString(), alert.Name, alert.Notes);
            }
            catch
            {
                // Last resort: write to console to avoid throwing from alert reporter
                Console.WriteLine($"[Alert] {alert?.Name} :: {alert?.Condition} :: {alert?.Notes}");
            }

            // Best-effort forward to UI layer if it's loaded (no compile-time dependency)
            try
            {
                var alertManagerType = Type.GetType("Quantra.Utilities.AlertManager, Quantra", throwOnError: false);
                var emitMethod = alertManagerType?.GetMethod("EmitGlobalAlert", BindingFlags.Public | BindingFlags.Static);
                emitMethod?.Invoke(null, new object[] { alert });
            }
            catch
            {
                // Silently ignore; DAL should not fail if UI is unavailable
            }
        }

        /// <summary>
        /// Emits a global error alert by logging to the database. The database logger will handle notifying listeners.
        /// </summary>
        public static void EmitGlobalError(string message, Exception ex = null)
        {
            try
            {
                // Log as Error; DatabaseMonolith will emit an alert to any registered listeners/UI.
                Quantra.DatabaseMonolith.Log("Error", message, ex?.ToString());
            }
            catch
            {
                // Last resort: write to console to avoid throwing from error reporter
                Console.WriteLine($"[Error] {message} :: {ex}");
            }
        }
    }
}
