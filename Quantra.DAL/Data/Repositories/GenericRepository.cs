using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.DAL.Data.Repositories
{
    /// <summary>
    /// Generic repository pattern implementation for common database operations
    /// </summary>
    public interface IRepository<T> where T : class
    {
        Task<T> GetByIdAsync(object id);
        Task<IEnumerable<T>> GetAllAsync();
        Task<T> AddAsync(T entity);
        Task UpdateAsync(T entity);
        Task DeleteAsync(T entity);
        Task<IEnumerable<T>> FindAsync(System.Linq.Expressions.Expression<Func<T, bool>> predicate);
    }

    public class Repository<T> : IRepository<T> where T : class
    {
        protected readonly QuantraDbContext _context;
        protected readonly DbSet<T> _dbSet;

        public Repository(QuantraDbContext context)
        {
            _context = context;
            _dbSet = context.Set<T>();
        }

        public virtual async Task<T> GetByIdAsync(object id)
        {
            return await _dbSet.FindAsync(id);
        }

        public virtual async Task<IEnumerable<T>> GetAllAsync()
        {
            return await _dbSet.ToListAsync();
        }

        public virtual async Task<T> AddAsync(T entity)
        {
            await _dbSet.AddAsync(entity);
            await _context.SaveChangesAsync();
            return entity;
        }

        public virtual async Task UpdateAsync(T entity)
        {
            _dbSet.Update(entity);
            await _context.SaveChangesAsync();
        }

        public virtual async Task DeleteAsync(T entity)
        {
            _dbSet.Remove(entity);
            await _context.SaveChangesAsync();
        }

        public virtual async Task<IEnumerable<T>> FindAsync(System.Linq.Expressions.Expression<Func<T, bool>> predicate)
        {
            return await _dbSet.Where(predicate).ToListAsync();
        }
    }

    /// <summary>
    /// Specialized repository for logging operations
    /// </summary>
    public interface ILogRepository : IRepository<LogEntry>
    {
        Task<IEnumerable<LogEntry>> GetLogsByLevelAsync(string level, int count = 100);
        Task DeleteOldLogsAsync(DateTime olderThan);
    }

    public class LogRepository : Repository<LogEntry>, ILogRepository
    {
        public LogRepository(QuantraDbContext context) : base(context) { }

        public async Task<IEnumerable<LogEntry>> GetLogsByLevelAsync(string level, int count = 100)
        {
            return await _dbSet
         .Where(l => l.Level == level)
          .OrderByDescending(l => l.Timestamp)
          .Take(count)
          .ToListAsync();
        }

        public async Task DeleteOldLogsAsync(DateTime olderThan)
        {
            var oldLogs = await _dbSet.Where(l => l.Timestamp < olderThan).ToListAsync();
            _dbSet.RemoveRange(oldLogs);
            await _context.SaveChangesAsync();
        }
    }

    /// <summary>
    /// Specialized repository for stock symbols
    /// </summary>
    public interface IStockSymbolRepository : IRepository<StockSymbolEntity>
    {
        Task<StockSymbolEntity> GetBySymbolAsync(string symbol);
        Task<IEnumerable<StockSymbolEntity>> SearchSymbolsAsync(string searchTerm, int limit = 100);
        Task<bool> IsSymbolCacheValidAsync(int maxAgeDays = 7);
    }

    public class StockSymbolRepository : Repository<StockSymbolEntity>, IStockSymbolRepository
    {
        public StockSymbolRepository(QuantraDbContext context) : base(context) { }

        public async Task<StockSymbolEntity> GetBySymbolAsync(string symbol)
        {
            return await _dbSet.FirstOrDefaultAsync(s => s.Symbol == symbol.ToUpper());
        }

        public async Task<IEnumerable<StockSymbolEntity>> SearchSymbolsAsync(string searchTerm, int limit = 100)
        {
            return await _dbSet
    .Where(s => s.Symbol.Contains(searchTerm) || s.Name.Contains(searchTerm))
       .OrderBy(s => s.Symbol)
            .Take(limit)
                 .ToListAsync();
        }

        public async Task<bool> IsSymbolCacheValidAsync(int maxAgeDays = 7)
        {
            var count = await _dbSet.CountAsync();
            if (count == 0) return false;

            var oldestUpdate = await _dbSet.MinAsync(s => s.LastUpdated);
            if (!oldestUpdate.HasValue) return false;

            var cacheAge = (DateTime.Now - oldestUpdate.Value).TotalDays;
            return cacheAge <= maxAgeDays;
        }
    }

    /// <summary>
    /// Specialized repository for trading rules
    /// </summary>
    public interface ITradingRuleRepository : IRepository<TradingRuleEntity>
    {
        Task<IEnumerable<TradingRuleEntity>> GetActiveRulesAsync(string symbol = null);
    }

    public class TradingRuleRepository : Repository<TradingRuleEntity>, ITradingRuleRepository
    {
        public TradingRuleRepository(QuantraDbContext context) : base(context) { }

        public async Task<IEnumerable<TradingRuleEntity>> GetActiveRulesAsync(string symbol = null)
        {
            var query = _dbSet.Where(r => r.IsActive);

            if (!string.IsNullOrEmpty(symbol))
            {
                query = query.Where(r => r.Symbol == symbol);
            }

            return await query.OrderBy(r => r.Name).ToListAsync();
        }
    }
}
