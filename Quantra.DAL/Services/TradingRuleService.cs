using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing trading rules using Entity Framework Core
    /// </summary>
    public interface ITradingRuleService
    {
        Task<List<TradingRule>> GetTradingRulesAsync(string symbol = null);
        Task SaveTradingRuleAsync(TradingRule rule);
        Task<bool> DeleteTradingRuleAsync(int ruleId);
        Task<TradingRule> GetTradingRuleByIdAsync(int ruleId);
        Task<List<TradingRule>> GetActiveTradingRulesAsync(string symbol = null);
    }

    public class TradingRuleService : ITradingRuleService
    {
        private readonly QuantraDbContext _context;

        public TradingRuleService(QuantraDbContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
        }

        /// <summary>
        /// Gets trading rules, optionally filtered by symbol
        /// </summary>
        /// <param name="symbol">Optional stock symbol to filter rules (null for all rules)</param>
        /// <returns>List of trading rules matching the criteria</returns>
        public async Task<List<TradingRule>> GetTradingRulesAsync(string symbol = null)
        {
            try
            {
                var query = _context.TradingRules.AsQueryable();

                if (!string.IsNullOrWhiteSpace(symbol))
                {
                    query = query.Where(r => r.Symbol == symbol.ToUpper());
                }

                var entities = await query
                    .OrderBy(r => r.Name)
                    .ToListAsync();

                return entities.Select(MapToModel).ToList();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to retrieve trading rules for symbol '{symbol}'", ex);
            }
        }

        /// <summary>
        /// Gets only active trading rules, optionally filtered by symbol
        /// </summary>
        /// <param name="symbol">Optional stock symbol to filter rules (null for all active rules)</param>
        /// <returns>List of active trading rules matching the criteria</returns>
        public async Task<List<TradingRule>> GetActiveTradingRulesAsync(string symbol = null)
        {
            try
            {
                var query = _context.TradingRules.Where(r => r.IsActive);

                if (!string.IsNullOrWhiteSpace(symbol))
                {
                    query = query.Where(r => r.Symbol == symbol.ToUpper());
                }

                var entities = await query
                    .OrderBy(r => r.Name)
                    .ToListAsync();

                return entities.Select(MapToModel).ToList();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to retrieve active trading rules for symbol '{symbol}'", ex);
            }
        }

        /// <summary>
        /// Gets a specific trading rule by ID
        /// </summary>
        /// <param name="ruleId">The ID of the rule to retrieve</param>
        /// <returns>The trading rule or null if not found</returns>
        public async Task<TradingRule> GetTradingRuleByIdAsync(int ruleId)
        {
            try
            {
                var entity = await _context.TradingRules.FindAsync(ruleId);
                return entity != null ? MapToModel(entity) : null;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to retrieve trading rule with ID {ruleId}", ex);
            }
        }

        /// <summary>
        /// Saves a trading rule (creates new or updates existing)
        /// </summary>
        /// <param name="rule">The trading rule to save</param>
        public async Task SaveTradingRuleAsync(TradingRule rule)
        {
            if (rule == null)
                throw new ArgumentNullException(nameof(rule));

            try
            {
                if (rule.Id == 0)
                {
                    // Create new rule
                    var entity = MapToEntity(rule);
                    entity.CreatedDate = DateTime.Now;
                    entity.LastModified = DateTime.Now;

                    await _context.TradingRules.AddAsync(entity);
                    await _context.SaveChangesAsync();

                    // Update the model with the generated ID
                    rule.Id = entity.Id;
                }
                else
                {
                    // Update existing rule
                    var existingEntity = await _context.TradingRules.FindAsync(rule.Id);
                    if (existingEntity == null)
                    {
                        throw new InvalidOperationException($"Trading rule with ID {rule.Id} not found");
                    }

                    // Update properties
                    existingEntity.Name = rule.Name;
                    existingEntity.Symbol = rule.Symbol?.ToUpper();
                    existingEntity.OrderType = rule.OrderType;
                    existingEntity.IsActive = rule.IsActive;
                    existingEntity.Conditions = SerializeConditions(rule.Conditions);
                    existingEntity.LastModified = DateTime.Now;
                    existingEntity.MinConfidence = rule.MinConfidence;
                    existingEntity.EntryPrice = rule.EntryPrice;
                    existingEntity.ExitPrice = rule.ExitPrice;
                    existingEntity.StopLoss = rule.StopLoss;
                    existingEntity.Quantity = rule.Quantity;

                    _context.TradingRules.Update(existingEntity);
                    await _context.SaveChangesAsync();
                }
            }
            catch (DbUpdateException ex)
            {
                throw new InvalidOperationException("Failed to save trading rule to database", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to save trading rule '{rule.Name}'", ex);
            }
        }

        /// <summary>
        /// Deletes a trading rule by ID
        /// </summary>
        /// <param name="ruleId">The ID of the rule to delete</param>
        /// <returns>True if deleted successfully, false if rule not found</returns>
        public async Task<bool> DeleteTradingRuleAsync(int ruleId)
        {
            try
            {
                var entity = await _context.TradingRules.FindAsync(ruleId);
                if (entity == null)
                    return false;

                _context.TradingRules.Remove(entity);
                await _context.SaveChangesAsync();
                return true;
            }
            catch (DbUpdateException ex)
            {
                throw new InvalidOperationException($"Failed to delete trading rule with ID {ruleId} from database", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to delete trading rule with ID {ruleId}", ex);
            }
        }

        #region Mapping Methods

        /// <summary>
        /// Maps TradingRuleEntity to TradingRule model
        /// </summary>
        private TradingRule MapToModel(TradingRuleEntity entity)
        {
            if (entity == null)
                return null;

            return new TradingRule
            {
                Id = entity.Id,
                Name = entity.Name,
                Symbol = entity.Symbol,
                OrderType = entity.OrderType,
                IsActive = entity.IsActive,
                Conditions = DeserializeConditions(entity.Conditions),
                CreatedDate = entity.CreatedDate,
                LastModified = entity.LastModified,
                MinConfidence = entity.MinConfidence,
                EntryPrice = entity.EntryPrice,
                ExitPrice = entity.ExitPrice,
                StopLoss = entity.StopLoss,
                Quantity = entity.Quantity
            };
        }

        /// <summary>
        /// Maps TradingRule model to TradingRuleEntity
        /// </summary>
        private TradingRuleEntity MapToEntity(TradingRule model)
        {
            if (model == null)
                return null;

            return new TradingRuleEntity
            {
                Id = model.Id,
                Name = model.Name,
                Symbol = model.Symbol?.ToUpper(),
                OrderType = model.OrderType,
                IsActive = model.IsActive,
                Conditions = SerializeConditions(model.Conditions),
                CreatedDate = model.CreatedDate,
                LastModified = model.LastModified,
                MinConfidence = model.MinConfidence,
                EntryPrice = model.EntryPrice,
                ExitPrice = model.ExitPrice,
                StopLoss = model.StopLoss,
                Quantity = model.Quantity
            };
        }

        /// <summary>
        /// Serializes conditions list to JSON string
        /// </summary>
        private string SerializeConditions(List<string> conditions)
        {
            if (conditions == null || conditions.Count == 0)
                return "[]";

            try
            {
                return JsonConvert.SerializeObject(conditions);
            }
            catch
            {
                return "[]";
            }
        }

        /// <summary>
        /// Deserializes conditions from JSON string to list
        /// </summary>
        private List<string> DeserializeConditions(string conditionsJson)
        {
            if (string.IsNullOrWhiteSpace(conditionsJson))
                return new List<string>();

            try
            {
                var conditions = JsonConvert.DeserializeObject<List<string>>(conditionsJson);
                return conditions ?? new List<string>();
            }
            catch
            {
                // Fallback: try to split by comma if JSON deserialization fails
                return conditionsJson
                    .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(s => s.Trim())
                    .ToList();
            }
        }

        #endregion
    }
}
