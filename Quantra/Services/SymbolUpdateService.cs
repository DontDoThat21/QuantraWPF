using System;
using System.Windows;

namespace Quantra.Services
{
    /// <summary>
    /// Service to coordinate symbol updates across different components
    /// </summary>
    public static class SymbolUpdateService
    {
        public static event EventHandler<SymbolUpdatedEventArgs> SymbolUpdated;
        public static event EventHandler<SymbolUpdatedEventArgs> CachedSymbolUpdated;

        /// <summary>
        /// Notify that symbol data has been retrieved from API
        /// </summary>
        /// <param name="symbol">The symbol that was updated</param>
        /// <param name="source">The source component that triggered the update</param>
        public static void NotifySymbolDataRetrieved(string symbol, string source = "Unknown")
        {
            try
            {
                // Ensure the event is raised on the UI thread to avoid STA thread issues
                if (Application.Current?.Dispatcher?.CheckAccess() == true)
                {
                    // Already on UI thread, invoke directly
                    SymbolUpdated?.Invoke(null, new SymbolUpdatedEventArgs(symbol, source, false));
                }
                else if (Application.Current?.Dispatcher != null)
                {
                    // Update dispatcher monitoring before making the call
                    SharedTitleBar.UpdateDispatcherMonitoring("SymbolUpdateService_NotifyDataRetrieved");
                    
                    // Marshal to UI thread
                    Application.Current.Dispatcher.InvokeAsync(() =>
                    {
                        try
                        {
                            SymbolUpdated?.Invoke(null, new SymbolUpdatedEventArgs(symbol, source, false));
                        }
                        catch (Exception ex)
                        {
                            DatabaseMonolith.Log("Error", $"Failed to invoke symbol data retrieved event for {symbol} on UI thread", ex.ToString());
                        }
                    });
                }
                
                else
                {
                    // Fallback: invoke directly (application may not be fully initialized)
                    SymbolUpdated?.Invoke(null, new SymbolUpdatedEventArgs(symbol, source, false));
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to notify symbol data retrieved for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Notify that cached symbol data has been accessed
        /// </summary>
        /// <param name="symbol">The symbol that was accessed from cache</param>
        /// <param name="source">The source component that triggered the update</param>
        public static void NotifyCachedSymbolDataAccessed(string symbol, string source = "Unknown")
        {
            try
            {
                // Ensure the event is raised on the UI thread to avoid STA thread issues
                if (Application.Current?.Dispatcher?.CheckAccess() == true)
                {
                    // Already on UI thread, invoke directly
                    CachedSymbolUpdated?.Invoke(null, new SymbolUpdatedEventArgs(symbol, source, true));
                }
                else if (Application.Current?.Dispatcher != null)
                {
                    // Update dispatcher monitoring before making the call
                    SharedTitleBar.UpdateDispatcherMonitoring("SymbolUpdateService_NotifyCachedAccess");
                    
                    // Marshal to UI thread
                    Application.Current.Dispatcher.InvokeAsync(() =>
                    {
                        try
                        {
                            CachedSymbolUpdated?.Invoke(null, new SymbolUpdatedEventArgs(symbol, source, true));
                        }
                        catch (Exception ex)
                        {
                            DatabaseMonolith.Log("Error", $"Failed to invoke cached symbol data accessed event for {symbol} on UI thread", ex.ToString());
                        }
                    });
                }
                else
                {
                    // Fallback: invoke directly (application may not be fully initialized)
                    CachedSymbolUpdated?.Invoke(null, new SymbolUpdatedEventArgs(symbol, source, true));
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to notify cached symbol data accessed for {symbol}", ex.ToString());
            }
        }
    }

    public class SymbolUpdatedEventArgs : EventArgs
    {
        public string Symbol { get; }
        public string Source { get; }
        public bool FromCache { get; }
        public DateTime UpdateTime { get; }

        public SymbolUpdatedEventArgs(string symbol, string source, bool fromCache)
        {
            Symbol = symbol;
            Source = source;
            FromCache = fromCache;
            UpdateTime = DateTime.Now;
        }
    }
}