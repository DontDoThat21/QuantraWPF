using System;
using System.Timers;

namespace Quantra.DAL.TradingEngine.Time
{
    /// <summary>
    /// Real-time clock for live/paper trading
    /// </summary>
    public class RealTimeClock : IClock, IDisposable
    {
        private readonly Timer _timer;
        private bool _isRunning;
        private bool _disposed;

        public event EventHandler<DateTime>? TimeChanged;

        /// <summary>
        /// Creates a new real-time clock with optional tick interval
        /// </summary>
        /// <param name="tickIntervalMs">Interval in milliseconds for time updates (default: 1000ms)</param>
        public RealTimeClock(int tickIntervalMs = 1000)
        {
            _timer = new Timer(tickIntervalMs);
            _timer.Elapsed += OnTimerElapsed;
            _timer.AutoReset = true;
        }

        /// <summary>
        /// Gets the current system time in UTC
        /// </summary>
        public DateTime CurrentTime => DateTime.UtcNow;

        /// <summary>
        /// Gets whether the clock is running
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Returns false as this is not a simulated clock
        /// </summary>
        public bool IsSimulated => false;

        /// <summary>
        /// Starts the real-time clock
        /// </summary>
        public void Start()
        {
            if (!_isRunning)
            {
                _timer.Start();
                _isRunning = true;
            }
        }

        /// <summary>
        /// Stops the real-time clock
        /// </summary>
        public void Stop()
        {
            if (_isRunning)
            {
                _timer.Stop();
                _isRunning = false;
            }
        }

        /// <summary>
        /// AdvanceTo is not supported for real-time clocks
        /// </summary>
        public void AdvanceTo(DateTime time)
        {
            // Real-time clocks cannot be advanced manually
            throw new NotSupportedException("Cannot manually advance a real-time clock");
        }

        private void OnTimerElapsed(object? sender, ElapsedEventArgs e)
        {
            TimeChanged?.Invoke(this, DateTime.UtcNow);
        }

        /// <summary>
        /// Disposes of the timer resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _timer.Stop();
                    _timer.Dispose();
                }
                _disposed = true;
            }
        }

        ~RealTimeClock()
        {
            Dispose(false);
        }
    }
}
