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
    /// Service for managing saved stock filters
    /// </summary>
    public class SavedFilterService
    {
        private readonly QuantraDbContext _dbContext;
        private readonly LoggingService _loggingService;
        private readonly AuthenticationService _authenticationService;

        public SavedFilterService(
            QuantraDbContext dbContext,
            LoggingService loggingService,
            AuthenticationService authenticationService)
        {
            _dbContext = dbContext;
            _loggingService = loggingService;
            _authenticationService = authenticationService;
        }

        /// <summary>
        /// Get all filters for the current user (or all system filters if no user is logged in)
        /// </summary>
        public async Task<List<SavedFilter>> GetAllFiltersAsync()
        {
            try
            {
                var currentUserId = AuthenticationService.CurrentUserId;

                var filters = await _dbContext.SavedFilters
                    .Where(f => f.UserId == currentUserId || f.IsSystemFilter || f.UserId == null)
                    .OrderBy(f => f.IsSystemFilter ? 0 : 1) // System filters first
                    .ThenBy(f => f.Name)
                    .AsNoTracking()
                    .ToListAsync();

                _loggingService?.Log("Info", $"Retrieved {filters.Count} saved filters for user {currentUserId}");
                return filters;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to retrieve saved filters", ex.ToString());
                return new List<SavedFilter>();
            }
        }

        /// <summary>
        /// Get a specific filter by ID
        /// </summary>
        public async Task<SavedFilter?> GetFilterByIdAsync(int id)
        {
            try
            {
                var filter = await _dbContext.SavedFilters
                    .AsNoTracking()
                    .FirstOrDefaultAsync(f => f.Id == id);

                return filter;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to retrieve filter with ID {id}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Check if a filter name already exists for the current user
        /// </summary>
        public async Task<bool> FilterNameExistsAsync(string name)
        {
            try
            {
                var currentUserId = AuthenticationService.CurrentUserId;

                var exists = await _dbContext.SavedFilters
                    .AnyAsync(f => f.Name == name && (f.UserId == currentUserId || f.UserId == null));

                return exists;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to check if filter name '{name}' exists", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Save a new filter or update an existing one
        /// </summary>
        public async Task<SavedFilter?> SaveFilterAsync(SavedFilter filter)
        {
            try
            {
                var currentUserId = AuthenticationService.CurrentUserId;

                // Check if updating existing filter
                var existing = await _dbContext.SavedFilters
                    .FirstOrDefaultAsync(f => f.Name == filter.Name && (f.UserId == currentUserId || f.UserId == null));

                if (existing != null)
                {
                    // Update existing filter
                    existing.Description = filter.Description;
                    existing.SymbolFilter = filter.SymbolFilter;
                    existing.PriceFilter = filter.PriceFilter;
                    existing.PeRatioFilter = filter.PeRatioFilter;
                    existing.VwapFilter = filter.VwapFilter;
                    existing.RsiFilter = filter.RsiFilter;
                    existing.ChangePercentFilter = filter.ChangePercentFilter;
                    existing.MarketCapFilter = filter.MarketCapFilter;
                    existing.LastModified = DateTime.Now;

                    await _dbContext.SaveChangesAsync();
                    _loggingService?.Log("Info", $"Updated existing filter '{filter.Name}'");
                    return existing;
                }
                else
                {
                    // Create new filter
                    filter.UserId = currentUserId;
                    filter.CreatedAt = DateTime.Now;
                    filter.LastModified = DateTime.Now;

                    _dbContext.SavedFilters.Add(filter);
                    await _dbContext.SaveChangesAsync();

                    _loggingService?.Log("Info", $"Saved new filter '{filter.Name}' with ID {filter.Id}");
                    return filter;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to save filter '{filter?.Name}'", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Delete a filter by ID
        /// </summary>
        public async Task<bool> DeleteFilterAsync(int id)
        {
            try
            {
                var filter = await _dbContext.SavedFilters.FindAsync(id);

                if (filter == null)
                {
                    _loggingService?.Log("Warning", $"Filter with ID {id} not found for deletion");
                    return false;
                }

                if (filter.IsSystemFilter)
                {
                    _loggingService?.Log("Warning", $"Cannot delete system filter '{filter.Name}'");
                    return false;
                }

                _dbContext.SavedFilters.Remove(filter);
                await _dbContext.SaveChangesAsync();

                _loggingService?.Log("Info", $"Deleted filter '{filter.Name}' (ID: {id})");
                return true;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to delete filter with ID {id}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Create default system filters if they don't exist
        /// </summary>
        public async Task EnsureDefaultFiltersExistAsync()
        {
            try
            {
                // Check if any system filters exist
                var hasSystemFilters = await _dbContext.SavedFilters
                    .AnyAsync(f => f.IsSystemFilter);

                if (!hasSystemFilters)
                {
                    var defaultFilters = new List<SavedFilter>
                    {
                        new SavedFilter
                        {
                            Name = "High RSI (>70)",
                            Description = "Stocks with RSI above 70 (potentially overbought)",
                            IsSystemFilter = true,
                            RsiFilter = ">70",
                            CreatedAt = DateTime.Now,
                            LastModified = DateTime.Now
                        },
                        new SavedFilter
                        {
                            Name = "Low RSI (<30)",
                            Description = "Stocks with RSI below 30 (potentially oversold)",
                            IsSystemFilter = true,
                            RsiFilter = "<30",
                            CreatedAt = DateTime.Now,
                            LastModified = DateTime.Now
                        },
                        new SavedFilter
                        {
                            Name = "Low P/E (<15)",
                            Description = "Stocks with P/E ratio below 15",
                            IsSystemFilter = true,
                            PeRatioFilter = "<15",
                            CreatedAt = DateTime.Now,
                            LastModified = DateTime.Now
                        },
                        new SavedFilter
                        {
                            Name = "Large Cap (>10B)",
                            Description = "Stocks with market cap above 10 billion",
                            IsSystemFilter = true,
                            MarketCapFilter = ">10000000000",
                            CreatedAt = DateTime.Now,
                            LastModified = DateTime.Now
                        }
                    };

                    _dbContext.SavedFilters.AddRange(defaultFilters);
                    await _dbContext.SaveChangesAsync();

                    _loggingService?.Log("Info", $"Created {defaultFilters.Count} default system filters");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to create default filters", ex.ToString());
            }
        }
    }
}
