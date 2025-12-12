using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing saved filter configurations in StockExplorer
    /// </summary>
    public class SavedFilterService
    {
        private readonly IDbContextFactory<QuantraDbContext> _contextFactory;
        private readonly LoggingService _loggingService;

        public SavedFilterService(IDbContextFactory<QuantraDbContext> contextFactory, LoggingService loggingService)
        {
            _contextFactory = contextFactory ?? throw new ArgumentNullException(nameof(contextFactory));
            _loggingService = loggingService;
        }

        /// <summary>
        /// Get all saved filters for the current user and system filters
        /// </summary>
        public async Task<List<SavedFilter>> GetAllFiltersAsync(int? userId = null)
        {
            try
            {
                using var context = await _contextFactory.CreateDbContextAsync();

                var query = context.SavedFilters.AsQueryable();

                // Get system filters plus user's personal filters
                if (userId.HasValue)
                {
                    query = query.Where(f => f.IsSystemFilter || f.UserId == userId.Value);
                }
                else
                {
                    // If no user ID, only return system filters
                    query = query.Where(f => f.IsSystemFilter);
                }

                return await query
                    .OrderBy(f => f.IsSystemFilter ? 0 : 1) // System filters first
                    .ThenBy(f => f.Name)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to get saved filters", ex.ToString());
                return new List<SavedFilter>();
            }
        }

        /// <summary>
        /// Get a specific saved filter by ID
        /// </summary>
        public async Task<SavedFilter?> GetFilterByIdAsync(int id)
        {
            try
            {
                using var context = await _contextFactory.CreateDbContextAsync();
                return await context.SavedFilters.FindAsync(id);
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to get saved filter with ID {id}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Save a new filter
        /// </summary>
        public async Task<SavedFilter?> SaveFilterAsync(SavedFilter filter)
        {
            try
            {
                using var context = await _contextFactory.CreateDbContextAsync();

                filter.CreatedAt = DateTime.UtcNow;
                filter.LastModified = DateTime.UtcNow;

                context.SavedFilters.Add(filter);
                await context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Saved filter '{filter.Name}'");
                return filter;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to save filter '{filter?.Name}'", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Update an existing filter
        /// </summary>
        public async Task<bool> UpdateFilterAsync(SavedFilter filter)
        {
            try
            {
                using var context = await _contextFactory.CreateDbContextAsync();

                var existing = await context.SavedFilters.FindAsync(filter.Id);
                if (existing == null)
                {
                    _loggingService?.Log("Warning", $"Filter with ID {filter.Id} not found");
                    return false;
                }

                // Don't allow updating system filters
                if (existing.IsSystemFilter)
                {
                    _loggingService?.Log("Warning", $"Cannot update system filter '{existing.Name}'");
                    return false;
                }

                // Update properties
                existing.Name = filter.Name;
                existing.Description = filter.Description;
                existing.SymbolFilter = filter.SymbolFilter;
                existing.PriceFilter = filter.PriceFilter;
                existing.PeRatioFilter = filter.PeRatioFilter;
                existing.VwapFilter = filter.VwapFilter;
                existing.RsiFilter = filter.RsiFilter;
                existing.ChangePercentFilter = filter.ChangePercentFilter;
                existing.MarketCapFilter = filter.MarketCapFilter;
                existing.LastModified = DateTime.UtcNow;

                await context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Updated filter '{filter.Name}'");
                return true;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to update filter '{filter?.Name}'", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Delete a saved filter
        /// </summary>
        public async Task<bool> DeleteFilterAsync(int id)
        {
            try
            {
                using var context = await _contextFactory.CreateDbContextAsync();

                var filter = await context.SavedFilters.FindAsync(id);
                if (filter == null)
                {
                    _loggingService?.Log("Warning", $"Filter with ID {id} not found");
                    return false;
                }

                // Don't allow deleting system filters
                if (filter.IsSystemFilter)
                {
                    _loggingService?.Log("Warning", $"Cannot delete system filter '{filter.Name}'");
                    return false;
                }

                context.SavedFilters.Remove(filter);
                await context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Deleted filter '{filter.Name}'");
                return true;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to delete filter with ID {id}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Check if a filter name already exists for the user
        /// </summary>
        public async Task<bool> FilterNameExistsAsync(string name, int? userId = null, int? excludeId = null)
        {
            try
            {
                using var context = await _contextFactory.CreateDbContextAsync();

                var query = context.SavedFilters.AsQueryable();

                if (userId.HasValue)
                {
                    // Check within user's filters and system filters
                    query = query.Where(f => (f.UserId == userId.Value || f.IsSystemFilter) &&
                                            f.Name.ToLower() == name.ToLower());
                }
                else
                {
                    // Check only system filters
                    query = query.Where(f => f.IsSystemFilter && f.Name.ToLower() == name.ToLower());
                }

                if (excludeId.HasValue)
                {
                    query = query.Where(f => f.Id != excludeId.Value);
                }

                return await query.AnyAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to check if filter name exists: '{name}'", ex.ToString());
                return false;
            }
        }
    }
}
