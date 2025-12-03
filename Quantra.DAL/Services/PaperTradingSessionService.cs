using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing paper trading sessions in the database
    /// </summary>
    public class PaperTradingSessionService
    {
        private readonly string _connectionString;
        private readonly LoggingService _loggingService;

        public PaperTradingSessionService(string connectionString, LoggingService loggingService)
        {
            _connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Creates a new DbContext instance for each operation to avoid threading issues
        /// </summary>
        private QuantraDbContext CreateDbContext()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(_connectionString);
            return new QuantraDbContext(optionsBuilder.Options);
        }

        /// <summary>
        /// Starts a new paper trading session
        /// </summary>
        /// <param name="initialBalance">Starting cash balance</param>
        /// <param name="name">Optional session name</param>
        /// <returns>The session ID as string</returns>
        public async Task<string> StartSessionAsync(decimal initialBalance, string name = null)
        {
            try
            {
                using var context = CreateDbContext();
                
                var sessionId = Guid.NewGuid().ToString();
                var now = DateTime.UtcNow;

                var session = new PaperTradingSessionEntity
                {
                    SessionId = sessionId,
                    Name = name,
                    InitialCash = initialBalance,
                    CashBalance = initialBalance,
                    RealizedPnL = 0,
                    IsActive = true,
                    StartedAt = now,
                    LastUpdatedAt = now
                };

                context.PaperTradingSessions.Add(session);
                await context.SaveChangesAsync();

                _loggingService.Log("Info", $"Paper trading session started: {sessionId}");

                return sessionId;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to start paper trading session", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Stops an active paper trading session
        /// </summary>
        /// <param name="sessionId">Session to stop</param>
        /// <param name="finalBalance">Final cash balance</param>
        /// <param name="realizedPnL">Realized profit/loss</param>
        /// <returns>True if successful</returns>
        public async Task<bool> StopSessionAsync(string sessionId, decimal finalBalance, decimal realizedPnL)
        {
            try
            {
                using var context = CreateDbContext();
                
                var session = await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.IsActive);

                if (session == null)
                {
                    _loggingService.Log("Warning", $"Active session not found: {sessionId}");
                    return false;
                }

                var now = DateTime.UtcNow;
                session.EndedAt = now;
                session.CashBalance = finalBalance;
                session.RealizedPnL = realizedPnL;
                session.IsActive = false;
                session.LastUpdatedAt = now;

                await context.SaveChangesAsync();

                _loggingService.Log("Info", $"Paper trading session stopped: {sessionId}, Realized PnL: {realizedPnL:C}");

                return true;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to stop paper trading session: {sessionId}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Updates session statistics during trading
        /// </summary>
        /// <param name="sessionId">Session to update</param>
        /// <param name="currentBalance">Current cash balance</param>
        /// <param name="realizedPnL">Current realized profit/loss</param>
        /// <returns>True if successful</returns>
        public async Task<bool> UpdateSessionAsync(string sessionId, decimal currentBalance, decimal realizedPnL)
        {
            try
            {
                using var context = CreateDbContext();
                
                var session = await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.IsActive);

                if (session == null)
                {
                    return false;
                }

                session.CashBalance = currentBalance;
                session.RealizedPnL = realizedPnL;
                session.LastUpdatedAt = DateTime.UtcNow;

                await context.SaveChangesAsync();

                return true;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to update paper trading session: {sessionId}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Gets the active session ID
        /// </summary>
        /// <returns>Active session ID or null</returns>
        public async Task<string> GetActiveSessionIdAsync()
        {
            try
            {
                using var context = CreateDbContext();
                
                var session = await context.PaperTradingSessions
                    .Where(s => s.IsActive)
                    .OrderByDescending(s => s.StartedAt)
                    .FirstOrDefaultAsync();

                return session?.SessionId;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to get active session ID", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Gets a session by ID
        /// </summary>
        /// <param name="sessionId">Session ID to retrieve</param>
        /// <returns>Session entity or null</returns>
        public async Task<PaperTradingSessionEntity> GetSessionAsync(string sessionId)
        {
            try
            {
                using var context = CreateDbContext();
                
                return await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId);
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get session: {sessionId}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Resets a session (marks it as inactive and creates a new one)
        /// </summary>
        /// <param name="sessionId">Session to reset</param>
        /// <param name="newInitialBalance">New starting balance</param>
        /// <returns>New session ID</returns>
        public async Task<string> ResetSessionAsync(string sessionId, decimal newInitialBalance)
        {
            try
            {
                using var context = CreateDbContext();
                
                var oldSession = await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId);

                if (oldSession != null && oldSession.IsActive)
                {
                    // Mark old session as inactive
                    oldSession.IsActive = false;
                    oldSession.EndedAt = DateTime.UtcNow;
                    oldSession.LastUpdatedAt = DateTime.UtcNow;
                    await context.SaveChangesAsync();
                }

                // Create new session (will use its own context)
                var newSessionId = await StartSessionAsync(newInitialBalance, oldSession?.Name);

                _loggingService.Log("Info", $"Paper trading session reset: {sessionId} -> {newSessionId}");

                return newSessionId;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to reset session: {sessionId}", ex.ToString());
                throw;
            }
        }
    }
}
