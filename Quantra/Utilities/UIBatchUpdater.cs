using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace Quantra.Utilities
{
    /// <summary>
    /// Utility class for batching UI updates to improve performance by reducing frequent dispatcher calls
    /// </summary>
    public class UIBatchUpdater
    {
        private class UpdateItem
        {
            public Action Action { get; set; }
            public DispatcherPriority Priority { get; set; }
        }

        private readonly Dispatcher _dispatcher;
        private readonly Dictionary<string, UpdateItem> _pendingUpdates;
        private readonly DispatcherTimer _batchTimer;
        private readonly object _lock = new object();

        /// <summary>
        /// Gets or sets the batch interval in milliseconds. Default is 100ms.
        /// </summary>
        public int BatchIntervalMs { get; set; } = 100;

        public UIBatchUpdater(Dispatcher dispatcher)
        {
            _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
            _pendingUpdates = new Dictionary<string, UpdateItem>();
            
            _batchTimer = new DispatcherTimer(DispatcherPriority.Background, _dispatcher)
            {
                Interval = TimeSpan.FromMilliseconds(BatchIntervalMs)
            };
            _batchTimer.Tick += BatchTimer_Tick;
        }

        /// <summary>
        /// Queues a UI update action to be executed in the next batch
        /// </summary>
        /// <param name="key">Unique key for the update (allows overwriting previous pending updates with same key)</param>
        /// <param name="updateAction">Action to execute on UI thread</param>
        /// <param name="priority">Dispatcher priority for the update (default: Background)</param>
        public void QueueUpdate(string key, Action updateAction, DispatcherPriority priority = DispatcherPriority.Background)
        {
            lock (_lock)
            {
                _pendingUpdates[key] = new UpdateItem { Action = updateAction, Priority = priority };
                
                if (!_batchTimer.IsEnabled)
                {
                    _batchTimer.Start();
                }
            }
        }

        /// <summary>
        /// Queues a UI update action to be executed in the next batch (backward compatibility)
        /// </summary>
        /// <param name="key">Unique key for the update (allows overwriting previous pending updates with same key)</param>
        /// <param name="updateAction">Action to execute on UI thread</param>
        public void QueueUpdate(string key, Action updateAction)
        {
            QueueUpdate(key, updateAction, DispatcherPriority.Background);
        }

        /// <summary>
        /// Immediately executes all pending updates and clears the queue
        /// </summary>
        public async Task FlushUpdates()
        {
            Dictionary<string, UpdateItem> updates;
            
            lock (_lock)
            {
                if (_pendingUpdates.Count == 0)
                    return;
                    
                updates = new Dictionary<string, UpdateItem>(_pendingUpdates);
                _pendingUpdates.Clear();
                _batchTimer.Stop();
            }

            // Group updates by priority for more efficient dispatching
            var groupedUpdates = updates.Values
                .GroupBy(update => update.Priority)
                .OrderByDescending(g => g.Key); // Higher priority first

            foreach (var priorityGroup in groupedUpdates)
            {
                await _dispatcher.InvokeAsync(() =>
                {
                    foreach (var update in priorityGroup)
                    {
                        try
                        {
                            update.Action?.Invoke();
                        }
                        catch (Exception ex)
                        {
                            // Log but don't let one bad update break the batch
                            //DatabaseMonolith.Log("Warning", $"Error executing batched UI update: {ex.Message}");
                        }
                    }
                }, priorityGroup.Key);
            }
        }

        private async void BatchTimer_Tick(object sender, EventArgs e)
        {
            await FlushUpdates();
        }

        /// <summary>
        /// Disposes the batch updater and flushes any pending updates
        /// </summary>
        public void Dispose()
        {
            _batchTimer?.Stop();
            FlushUpdates().Wait(1000); // Wait up to 1 second for final flush
        }
    }
}