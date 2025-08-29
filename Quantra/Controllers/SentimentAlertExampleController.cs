using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Threading;
using Quantra.Services.Interfaces;

namespace Quantra.Controllers
{
    /// <summary>
    /// Example controller showing how to use the SentimentShiftAlertService
    /// </summary>
    public class SentimentAlertExampleController
    {
        private readonly ISentimentShiftAlertService _sentimentShiftAlertService;
        private readonly DispatcherTimer _monitorTimer;
        private readonly List<string> _watchlist;
        
        public SentimentAlertExampleController(ISentimentShiftAlertService sentimentShiftAlertService)
        {
            _sentimentShiftAlertService = sentimentShiftAlertService;
            _watchlist = new List<string> { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA" };
            
            // Set up timer to periodically check for sentiment shifts
            _monitorTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMinutes(15) // Check every 15 minutes
            };
            _monitorTimer.Tick += MonitorTimer_Tick;
        }
        
        /// <summary>
        /// Start monitoring sentiment for watchlist symbols
        /// </summary>
        public void StartMonitoring()
        {
            _monitorTimer.Start();
            // Run immediately on start
            _ = MonitorWatchlistSentiment();
        }
        
        /// <summary>
        /// Stop monitoring sentiment
        /// </summary>
        public void StopMonitoring()
        {
            _monitorTimer.Stop();
        }
        
        /// <summary>
        /// Add a symbol to the watchlist for sentiment monitoring
        /// </summary>
        public void AddSymbolToWatchlist(string symbol)
        {
            if (!string.IsNullOrWhiteSpace(symbol) && !_watchlist.Contains(symbol))
            {
                _watchlist.Add(symbol);
                // Monitor it immediately
                _ = _sentimentShiftAlertService.MonitorSentimentShiftsAsync(symbol);
            }
        }
        
        /// <summary>
        /// Remove a symbol from the watchlist
        /// </summary>
        public void RemoveSymbolFromWatchlist(string symbol)
        {
            _watchlist.Remove(symbol);
        }
        
        /// <summary>
        /// Timer tick handler to monitor watchlist sentiment
        /// </summary>
        private void MonitorTimer_Tick(object sender, EventArgs e)
        {
            _ = MonitorWatchlistSentiment();
        }
        
        /// <summary>
        /// Monitor sentiment for all symbols in the watchlist
        /// </summary>
        private async Task MonitorWatchlistSentiment()
        {
            try
            {
                await _sentimentShiftAlertService.MonitorWatchlistAsync(_watchlist);
            }
            catch (Exception ex)
            {
                // Log error
                DatabaseMonolith.Log("Error", "Failed to monitor sentiment", ex.ToString());
            }
        }
    }
}