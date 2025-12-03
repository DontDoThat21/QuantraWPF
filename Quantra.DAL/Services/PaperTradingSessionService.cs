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
        /// <param name="userId">Optional user ID</param>
        /// <returns>The session ID</returns>
        public async Task<Guid> StartSessionAsync(decimal initialBalance, int? userId = null)
        {
            try
            {
                using var context = CreateDbContext();
                
                var sessionId = Guid.NewGuid();
                var now = DateTime.UtcNow;

                var session = new PaperTradingSessionEntity
                {
                    SessionId = sessionId,
                    UserId = userId,
                    StartTime = now,
                    InitialBalance = initialBalance,
                    Status = "Active",
                    TradeCount = 0,
                    WinningTrades = 0,
                    LosingTrades = 0,
                    CreatedAt = now,
                    UpdatedAt = now
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
        /// <param name="finalPortfolioValue">Final total portfolio value</param>
        /// <param name="realizedPnL">Realized profit/loss</param>
        /// <param name="unrealizedPnL">Unrealized profit/loss</param>
        /// <returns>True if successful</returns>
        public async Task<bool> StopSessionAsync(Guid sessionId, decimal finalBalance, decimal finalPortfolioValue, 
            decimal realizedPnL, decimal unrealizedPnL)
        {
            try
            {
                using var context = CreateDbContext();
                
                var session = await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.Status == "Active");

                if (session == null)
                {
                    _loggingService.Log("Warning", $"Active session not found: {sessionId}");
                    return false;
                }

                var now = DateTime.UtcNow;
                session.EndTime = now;
                session.FinalBalance = finalBalance;
                session.FinalPortfolioValue = finalPortfolioValue;
                session.RealizedPnL = realizedPnL;
                session.UnrealizedPnL = unrealizedPnL;
                session.TotalPnL = realizedPnL + unrealizedPnL;
                session.Status = "Completed";
                session.UpdatedAt = now;

                // Calculate win rate if there are trades
                if (session.TradeCount > 0)
                {
                    session.WinRate = (decimal)session.WinningTrades / session.TradeCount * 100;
                }

                await context.SaveChangesAsync();

                _loggingService.Log("Info", $"Paper trading session stopped: {sessionId}, Total PnL: {session.TotalPnL:C}");

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
        /// <param name="currentPortfolioValue">Current total portfolio value</param>
        /// <param name="realizedPnL">Current realized profit/loss</param>
        /// <param name="unrealizedPnL">Current unrealized profit/loss</param>
        /// <param name="tradeCount">Total number of trades</param>
        /// <param name="winningTrades">Number of winning trades</param>
        /// <param name="losingTrades">Number of losing trades</param>
        /// <returns>True if successful</returns>
        public async Task<bool> UpdateSessionAsync(Guid sessionId, decimal currentBalance, decimal currentPortfolioValue,
            decimal realizedPnL, decimal unrealizedPnL, int tradeCount, int winningTrades, int losingTrades)
        {
            try
            {
                using var context = CreateDbContext();
                
                var session = await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId && s.Status == "Active");

                if (session == null)
                {
                    return false;
                }

                session.FinalBalance = currentBalance;
                session.FinalPortfolioValue = currentPortfolioValue;
                session.RealizedPnL = realizedPnL;
                session.UnrealizedPnL = unrealizedPnL;
                session.TotalPnL = realizedPnL + unrealizedPnL;
                session.TradeCount = tradeCount;
                session.WinningTrades = winningTrades;
                session.LosingTrades = losingTrades;
                session.UpdatedAt = DateTime.UtcNow;

                // Calculate win rate
                if (tradeCount > 0)
                {
                    session.WinRate = (decimal)winningTrades / tradeCount * 100;
                }

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
        /// Gets the active session ID for the current user
        /// </summary>
        /// <param name="userId">Optional user ID</param>
        /// <returns>Active session ID or null</returns>
        public async Task<Guid?> GetActiveSessionIdAsync(int? userId = null)
        {
            try
            {
                using var context = CreateDbContext();
                
                var query = context.PaperTradingSessions
                    .Where(s => s.Status == "Active");

                if (userId.HasValue)
                {
                    query = query.Where(s => s.UserId == userId);
                }

                var session = await query
                    .OrderByDescending(s => s.StartTime)
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
        public async Task<PaperTradingSessionEntity> GetSessionAsync(Guid sessionId)
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
        /// Resets a session (marks it as reset and creates a new one)
        /// </summary>
        /// <param name="sessionId">Session to reset</param>
        /// <param name="newInitialBalance">New starting balance</param>
        /// <returns>New session ID</returns>
        public async Task<Guid> ResetSessionAsync(Guid sessionId, decimal newInitialBalance)
        {
            try
            {
                using var context = CreateDbContext();
                
                var oldSession = await context.PaperTradingSessions
                    .FirstOrDefaultAsync(s => s.SessionId == sessionId);

                if (oldSession != null && oldSession.Status == "Active")
                {
                    // Mark old session as reset
                    oldSession.Status = "Reset";
                    oldSession.EndTime = DateTime.UtcNow;
                    oldSession.UpdatedAt = DateTime.UtcNow;
                    await context.SaveChangesAsync();
                }

                // Create new session (will use its own context)
                var newSessionId = await StartSessionAsync(newInitialBalance, oldSession?.UserId);

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
