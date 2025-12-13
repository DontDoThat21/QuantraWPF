using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.Repositories
{
    /// <summary>
    /// Repository for managing tab configurations using Entity Framework Core
    /// </summary>
    public class TabRepository
    {
        private readonly QuantraDbContext _dbContext;

        public TabRepository(QuantraDbContext dbContext)
        {
            _dbContext = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
        }

        /// <summary>
        /// Gets all tabs for the current user ordered by TabOrder
        /// </summary>
        public List<(string TabName, int TabOrder)> GetTabs()
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabs = _dbContext.UserAppSettings
                    .AsNoTracking()
                    .Where(t => t.UserId == currentUserId)
                    .OrderBy(t => t.TabOrder)
                    .Select(t => new { t.TabName, t.TabOrder })
                    .ToList()
                    .Select(t => (t.TabName, t.TabOrder))
                    .ToList();

                return tabs;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting tabs: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Gets all tabs for the current user ordered by TabOrder (async version)
        /// </summary>
        public async Task<List<(string TabName, int TabOrder)>> GetTabsAsync()
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabs = await _dbContext.UserAppSettings
                    .AsNoTracking()
                    .Where(t => t.UserId == currentUserId)
                    .OrderBy(t => t.TabOrder)
                    .Select(t => new { t.TabName, t.TabOrder })
                    .ToListAsync();

                return tabs.Select(t => (t.TabName, t.TabOrder)).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting tabs: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Inserts a new tab with specified grid dimensions for the current user
        /// </summary>
        public void InsertTab(string tabName, int tabOrder, int rows, int columns)
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabSetting = new UserAppSetting
                {
                    UserId = currentUserId,
                    TabName = tabName,
                    TabOrder = tabOrder,
                    GridRows = rows,
                    GridColumns = columns
                };

                _dbContext.UserAppSettings.Add(tabSetting);
                _dbContext.SaveChanges();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error inserting tab '{tabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Inserts a new tab with specified grid dimensions for the current user (async version)
        /// </summary>
        public async Task InsertTabAsync(string tabName, int tabOrder, int rows, int columns)
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabSetting = new UserAppSetting
                {
                    UserId = currentUserId,
                    TabName = tabName,
                    TabOrder = tabOrder,
                    GridRows = rows,
                    GridColumns = columns
                };

                await _dbContext.UserAppSettings.AddAsync(tabSetting);
                await _dbContext.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error inserting tab '{tabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Updates the name of a tab for the current user
        /// </summary>
        public void UpdateTabName(string oldTabName, string newTabName)
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabSetting = _dbContext.UserAppSettings
                    .FirstOrDefault(t => t.TabName == oldTabName && t.UserId == currentUserId);

                if (tabSetting != null)
                {
                    tabSetting.TabName = newTabName;
                    _dbContext.SaveChanges();
                }
                else
                {
                    Console.WriteLine($"Warning: Tab '{oldTabName}' not found for rename");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error updating tab name from '{oldTabName}' to '{newTabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Updates the name of a tab (async version)
        /// </summary>
        public async Task UpdateTabNameAsync(string oldTabName, string newTabName)
        {
            try
            {
                var tabSetting = await _dbContext.UserAppSettings
                    .FirstOrDefaultAsync(t => t.TabName == oldTabName);

                if (tabSetting != null)
                {
                    tabSetting.TabName = newTabName;
                    await _dbContext.SaveChangesAsync();
                }
                else
                {
                    Console.WriteLine($"Warning: Tab '{oldTabName}' not found for rename");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error updating tab name from '{oldTabName}' to '{newTabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Deletes a tab by name for the current user
        /// </summary>
        public void DeleteTab(string tabName)
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabSetting = _dbContext.UserAppSettings
                    .FirstOrDefault(t => t.TabName == tabName && t.UserId == currentUserId);

                if (tabSetting != null)
                {
                    _dbContext.UserAppSettings.Remove(tabSetting);
                    _dbContext.SaveChanges();
                }
                else
                {
                    Console.WriteLine($"Warning: Tab '{tabName}' not found for deletion");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error deleting tab '{tabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Deletes a tab by name (async version)
        /// </summary>
        public async Task DeleteTabAsync(string tabName)
        {
            try
            {
                var tabSetting = await _dbContext.UserAppSettings
                    .FirstOrDefaultAsync(t => t.TabName == tabName);

                if (tabSetting != null)
                {
                    _dbContext.UserAppSettings.Remove(tabSetting);
                    await _dbContext.SaveChangesAsync();
                }
                else
                {
                    Console.WriteLine($"Warning: Tab '{tabName}' not found for deletion");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error deleting tab '{tabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Updates the order of a tab for the current user
        /// </summary>
        public void UpdateTabOrder(string tabName, int tabOrder)
        {
            try
            {
                var currentUserId = Quantra.DAL.Services.AuthenticationService.CurrentUserId;

                var tabSetting = _dbContext.UserAppSettings
                    .FirstOrDefault(t => t.TabName == tabName && t.UserId == currentUserId);

                if (tabSetting != null)
                {
                    tabSetting.TabOrder = tabOrder;
                    _dbContext.SaveChanges();
                }
                else
                {
                    Console.WriteLine($"Warning: Tab '{tabName}' not found for order update");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error updating tab order for '{tabName}': {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Updates the order of a tab (async version)
        /// </summary>
        public async Task UpdateTabOrderAsync(string tabName, int tabOrder)
        {
            try
            {
                var tabSetting = await _dbContext.UserAppSettings
                    .FirstOrDefaultAsync(t => t.TabName == tabName);

                if (tabSetting != null)
                {
                    tabSetting.TabOrder = tabOrder;
                    await _dbContext.SaveChangesAsync();
                }
                else
                {
                    Console.WriteLine($"Warning: Tab '{tabName}' not found for order update");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error updating tab order for '{tabName}': {ex.Message}");
                throw;
            }
        }
    }
}
