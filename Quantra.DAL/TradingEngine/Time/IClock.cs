using System;

namespace Quantra.DAL.TradingEngine.Time
{
    /// <summary>
    /// Interface for clock abstraction supporting both backtesting and real-time trading
    /// </summary>
    public interface IClock
    {
        /// <summary>
        /// Gets the current time according to this clock
        /// </summary>
        DateTime CurrentTime { get; }

        /// <summary>
        /// Event raised when time changes (for backtesting progression or real-time updates)
        /// </summary>
        event EventHandler<DateTime>? TimeChanged;

        /// <summary>
        /// Starts the clock
        /// </summary>
        void Start();

        /// <summary>
        /// Stops the clock
        /// </summary>
        void Stop();

        /// <summary>
        /// Advances the clock to a specific time (primarily for backtesting)
        /// </summary>
        /// <param name="time">Target time to advance to</param>
        void AdvanceTo(DateTime time);

        /// <summary>
        /// Indicates if the clock is currently running
        /// </summary>
        bool IsRunning { get; }

        /// <summary>
        /// Gets whether this is a simulated clock (backtesting) or real clock
        /// </summary>
        bool IsSimulated { get; }
    }
}
