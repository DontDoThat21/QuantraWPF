using System;
using System.Reflection;
using System.Threading.Tasks;

namespace Quantra.Utilities
{
    // Utility to abstract UI thread invocation without compile-time WPF dependency
    public static class UiThreadHelper
    {
        private static readonly Lazy<bool> _hasDispatcher = new Lazy<bool>(() => TryGetDispatcher(out _));

        public static bool HasDispatcher => _hasDispatcher.Value;

        public static Task InvokeAsync(Action action)
        {
            if (action == null) return Task.CompletedTask;

            if (TryGetDispatcher(out var dispatcher))
            {
                try
                {
                    dynamic d = dispatcher;
                    dynamic op = d.InvokeAsync(action); // Uses default priority
                    try
                    {
                        Task task = (Task)op.Task;
                        return task ?? Task.CompletedTask;
                    }
                    catch
                    {
                        return Task.CompletedTask;
                    }
                }
                catch
                {
                    // Fall back to direct invoke if reflection/dynamic fails
                    action();
                    return Task.CompletedTask;
                }
            }

            // No dispatcher available
            action();
            return Task.CompletedTask;
        }

        public static async Task InvokeAsync(Func<Task> func)
        {
            if (func == null) return;

            if (TryGetDispatcher(out _))
            {
                Task inner = null;
                await InvokeAsync(() => { inner = func(); });
                if (inner != null)
                {
                    await inner.ConfigureAwait(false);
                }
                return;
            }

            await func().ConfigureAwait(false);
        }

        private static bool TryGetDispatcher(out object dispatcher)
        {
            dispatcher = null;
            try
            {
                var appType = Type.GetType("System.Windows.Application, PresentationFramework");
                if (appType == null) return false;

                var currentProp = appType.GetProperty("Current", BindingFlags.Public | BindingFlags.Static);
                var app = currentProp?.GetValue(null);
                if (app == null) return false;

                // Use dynamic to access Dispatcher to avoid compile-time reference
                dynamic dApp = app;
                dispatcher = dApp.Dispatcher;
                return dispatcher != null;
            }
            catch
            {
                dispatcher = null;
                return false;
            }
        }
    }
}
