using System;

namespace Quantra.DAL.TradingEngine.Time
{
    /// <summary>
    /// Backtest clock for batch time progression through historical data
    /// </summary>
    public class BacktestClock : IClock
    {
        private DateTime _currentTime;
        private bool _isRunning;

        public event EventHandler<DateTime>? TimeChanged;

        /// <summary>
        /// Creates a new backtest clock starting at the specified time
        /// </summary>
        /// <param name="startTime">Start time for the backtest</param>
        public BacktestClock(DateTime startTime)
        {
            _currentTime = startTime;
            _isRunning = false;
        }

        /// <summary>
        /// Creates a new backtest clock starting at current UTC time
        /// </summary>
        public BacktestClock() : this(DateTime.UtcNow)
        {
        }

        /// <summary>
        /// Gets the current simulated time
        /// </summary>
        public DateTime CurrentTime => _currentTime;

        /// <summary>
        /// Gets whether the clock is running
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Returns true as this is a simulated clock
        /// </summary>
        public bool IsSimulated => true;

        /// <summary>
        /// Starts the backtest clock
        /// </summary>
        public void Start()
        {
            _isRunning = true;
        }

        /// <summary>
        /// Stops the backtest clock
        /// </summary>
        public void Stop()
        {
            _isRunning = false;
        }

        /// <summary>
        /// Advances the clock to a specific time
        /// </summary>
        /// <param name="time">Target time</param>
        public void AdvanceTo(DateTime time)
        {
            if (time <= _currentTime) return;

            _currentTime = time;
            TimeChanged?.Invoke(this, _currentTime);
        }

        /// <summary>
        /// Advances the clock by a specified time span
        /// </summary>
        /// <param name="duration">Duration to advance</param>
        public void AdvanceBy(TimeSpan duration)
        {
            AdvanceTo(_currentTime + duration);
        }

        /// <summary>
        /// Advances to the next day (at market open 9:30 AM ET)
        /// </summary>
        public void AdvanceToNextDay()
        {
            DateTime nextDay = _currentTime.Date.AddDays(1);
            // Set to market open time (9:30 AM Eastern)
            DateTime marketOpen = nextDay.AddHours(9).AddMinutes(30);
            AdvanceTo(marketOpen);
        }

        /// <summary>
        /// Resets the clock to a new start time
        /// </summary>
        /// <param name="startTime">New start time</param>
        public void Reset(DateTime startTime)
        {
            _currentTime = startTime;
            _isRunning = false;
        }
    }
}
