using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.Models;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    public class SettingsService : ISettingsService
    {
        private readonly QuantraDbContext _context;

        public SettingsService(QuantraDbContext context)
        {
            _context = context;
        }

        // Parameterless constructor for backward compatibility
        public SettingsService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        // Ensure the settings profiles table exists
        public void EnsureSettingsProfilesTable()
        {
            ResilienceHelper.Retry(() =>
   {
       // With EF Core, tables are created via migrations or EnsureCreated
       // We'll use EnsureCreated for backward compatibility
       _context.Database.EnsureCreated();

       // Check if any profiles exist, if not create a default one
       if (!_context.SettingsProfiles.Any())
       {
           var defaultProfile = new DatabaseSettingsProfile
           {
               Name = "Default",
               Description = "Default system settings",
               IsDefault = true,
               CreatedDate = DateTime.Now,
               ModifiedDate = DateTime.Now,
               EnableApiModalChecks = true,
               ApiTimeoutSeconds = 30,
               CacheDurationMinutes = 15,
               EnableHistoricalDataCache = true,
               EnableDarkMode = true,
               ChartUpdateIntervalSeconds = 2,
               DefaultGridRows = 4,
               DefaultGridColumns = 4,
               GridBorderColor = "#FF00FFFF",
               EnablePriceAlerts = true,
               EnableTradeNotifications = true,
               EnablePaperTrading = true,
               RiskLevel = "Low",
               AlertEmail = "test@gmail.com",
               EnableEmailAlerts = false,
               EnableStandardAlertEmails = false,
               EnableOpportunityAlertEmails = false,
               EnablePredictionAlertEmails = false,
               EnableGlobalAlertEmails = false,
               EnableSystemHealthAlertEmails = false
           };
           CreateSettingsProfile(defaultProfile);
       }
   }, RetryOptions.ForCriticalOperation());
        }

        // Create a new settings profile
        public int CreateSettingsProfile(DatabaseSettingsProfile profile)
        {
            return ResilienceHelper.Retry(() =>
               {
                   // If this is set as default, clear other defaults first
                   if (profile.IsDefault)
                   {
                       var existingDefaults = _context.SettingsProfiles.Where(p => p.IsDefault);
                       foreach (var p in existingDefaults)
                       {
                           p.IsDefault = false;
                       }
                   }

                   // Map DatabaseSettingsProfile to SettingsProfile entity
                   var entity = MapToEntity(profile);

                   _context.SettingsProfiles.Add(entity);
                   _context.SaveChanges();

                   return entity.Id;
               }, RetryOptions.ForCriticalOperation());
        }

        // Update an existing settings profile
        public bool UpdateSettingsProfile(DatabaseSettingsProfile profile)
        {
            return ResilienceHelper.Retry(() =>
         {
             var entity = _context.SettingsProfiles.Find(profile.Id);
             if (entity == null)
                 return false;

             // If this is set as default, clear other defaults first
             if (profile.IsDefault)
             {
                 var existingDefaults = _context.SettingsProfiles
                     .Where(p => p.IsDefault && p.Id != profile.Id);
                 foreach (var p in existingDefaults)
                 {
                     p.IsDefault = false;
                 }
             }

             // Update entity properties
             UpdateEntityFromProfile(entity, profile);
             entity.LastModified = DateTime.Now;

             _context.SaveChanges();
             return true;
         }, RetryOptions.ForCriticalOperation());
        }

        // Delete a settings profile
        public bool DeleteSettingsProfile(int profileId)
        {
            return ResilienceHelper.Retry(() =>
       {
           var entity = _context.SettingsProfiles.Find(profileId);
           if (entity == null)
               return false;

           // Don't allow deleting the default profile if it's the only one
           if (entity.IsDefault)
           {
               var count = _context.SettingsProfiles.Count();
               if (count <= 1)
                   return false; // Can't delete the only profile
           }

           bool wasDefault = entity.IsDefault;
           _context.SettingsProfiles.Remove(entity);
           _context.SaveChanges();

           // If we deleted the default profile, set another one as default
           if (wasDefault)
           {
               var newDefault = _context.SettingsProfiles.FirstOrDefault();
               if (newDefault != null)
               {
                   newDefault.IsDefault = true;
                   _context.SaveChanges();
               }
           }

           return true;
       }, RetryOptions.ForCriticalOperation());
        }

        // Get a specific settings profile
        public DatabaseSettingsProfile GetSettingsProfile(int profileId)
        {
            return ResilienceHelper.Retry(() =>
        {
            var entity = _context.SettingsProfiles
                         .AsNoTracking()
          .FirstOrDefault(p => p.Id == profileId);

            return entity != null ? MapFromEntity(entity) : null;
        }, RetryOptions.ForUserFacingOperation());
        }

        // Get the default settings profile
        public DatabaseSettingsProfile GetDefaultSettingsProfile()
        {
            return ResilienceHelper.Retry(() =>
        {
            // Ensure database is created
            _context.Database.EnsureCreated();

            // Try to get default profile
            var entity = _context.SettingsProfiles
               .AsNoTracking()
             .FirstOrDefault(p => p.IsDefault);

            if (entity != null)
                return MapFromEntity(entity);

            // If no default, get first profile
            entity = _context.SettingsProfiles
              .AsNoTracking()
            .FirstOrDefault();

            if (entity != null)
                return MapFromEntity(entity);

            // If still no profile, ensure profiles exist
            EnsureSettingsProfiles();

            // Try one more time
            entity = _context.SettingsProfiles
                  .AsNoTracking()
              .FirstOrDefault(p => p.IsDefault);

            return entity != null ? MapFromEntity(entity) : null;
        }, RetryOptions.ForUserFacingOperation());
        }

        // Get the default settings profile asynchronously
        public async Task<DatabaseSettingsProfile> GetDefaultSettingsProfileAsync()
        {
            await _context.Database.EnsureCreatedAsync();

            // Try to get default profile
            var entity = await _context.SettingsProfiles
            .AsNoTracking()
             .FirstOrDefaultAsync(p => p.IsDefault);

            if (entity != null)
                return MapFromEntity(entity);

            // If no default, get first profile
            entity = await _context.SettingsProfiles
               .AsNoTracking()
                   .FirstOrDefaultAsync();

            if (entity != null)
                return MapFromEntity(entity);

            // If still no profile, ensure profiles exist
            EnsureSettingsProfiles();

            // Try one more time
            entity = await _context.SettingsProfiles
                 .AsNoTracking()
            .FirstOrDefaultAsync(p => p.IsDefault);

            return entity != null ? MapFromEntity(entity) : null;
        }

        // Get all settings profiles
        public List<DatabaseSettingsProfile> GetAllSettingsProfiles()
        {
            return ResilienceHelper.Retry(() =>
                  {
                      // Ensure database is created
                      _context.Database.EnsureCreated();

                      var entities = _context.SettingsProfiles
     .AsNoTracking()
           .OrderByDescending(p => p.IsDefault)
        .ThenBy(p => p.Name)
               .ToList();

                      return entities.Select(MapFromEntity).ToList();
                  }, RetryOptions.ForUserFacingOperation());
        }

        // Ensure at least one settings profile exists
        public void EnsureSettingsProfiles()
        {
            ResilienceHelper.Retry(() =>
          {
              // First ensure the table exists
              EnsureSettingsProfilesTable();

              // Check if any profiles exist
              if (!_context.SettingsProfiles.Any())
              {
                  var defaultProfile = new DatabaseSettingsProfile
                  {
                      Name = "Default",
                      Description = "Default system settings",
                      IsDefault = true,
                      CreatedDate = DateTime.Now,
                      ModifiedDate = DateTime.Now,
                      EnableApiModalChecks = true,
                      ApiTimeoutSeconds = 30,
                      CacheDurationMinutes = 15,
                      EnableHistoricalDataCache = true,
                      EnableDarkMode = true,
                      ChartUpdateIntervalSeconds = 2,
                      DefaultGridRows = 4,
                      DefaultGridColumns = 4,
                      GridBorderColor = "#FF00FFFF",
                      EnablePriceAlerts = true,
                      EnableTradeNotifications = true,
                      EnablePaperTrading = true,
                      RiskLevel = "Low",
                      AlertEmail = "test@gmail.com",
                      EnableEmailAlerts = true,
                      EnableStandardAlertEmails = true,
                      EnableOpportunityAlertEmails = true,
                      EnablePredictionAlertEmails = true,
                      EnableGlobalAlertEmails = true,
                      EnableSystemHealthAlertEmails = true,
                      EnableVixMonitoring = true
                  };
                  CreateSettingsProfile(defaultProfile);
              }
          }, RetryOptions.ForCriticalOperation());
        }

        // Set a profile as the default
        public bool SetProfileAsDefault(int profileId)
        {
            return ResilienceHelper.Retry(() =>
            {
                // Clear existing defaults
                var existingDefaults = _context.SettingsProfiles.Where(p => p.IsDefault);
                foreach (var p in existingDefaults)
                {
                    p.IsDefault = false;
                }

                // Set new default
                var entity = _context.SettingsProfiles.Find(profileId);
                if (entity == null)
                    return false;

                entity.IsDefault = true;
                _context.SaveChanges();
                return true;
            }, RetryOptions.ForCriticalOperation());
        }

        #region Mapping Methods

        /// <summary>
        /// Maps DatabaseSettingsProfile to SettingsProfile entity
        /// </summary>
        private SettingsProfile MapToEntity(DatabaseSettingsProfile profile)
        {
            return new SettingsProfile
            {
                Id = profile.Id,
                Name = profile.Name,
                Description = profile.Description,
                IsDefault = profile.IsDefault,
                EnableApiModalChecks = profile.EnableApiModalChecks,
                ApiTimeoutSeconds = profile.ApiTimeoutSeconds,
                CacheDurationMinutes = profile.CacheDurationMinutes,
                EnableHistoricalDataCache = profile.EnableHistoricalDataCache,
                EnableDarkMode = profile.EnableDarkMode,
                ChartUpdateIntervalSeconds = profile.ChartUpdateIntervalSeconds,
                EnablePriceAlerts = profile.EnablePriceAlerts,
                EnableTradeNotifications = profile.EnableTradeNotifications,
                EnablePaperTrading = profile.EnablePaperTrading,
                RiskLevel = profile.RiskLevel,
                DefaultGridRows = profile.DefaultGridRows,
                DefaultGridColumns = profile.DefaultGridColumns,
                GridBorderColor = profile.GridBorderColor,
                
                // Email alerts
                AlertEmail = profile.AlertEmail,
                EnableEmailAlerts = profile.EnableEmailAlerts,
                EnableStandardAlertEmails = profile.EnableStandardAlertEmails,
                EnableOpportunityAlertEmails = profile.EnableOpportunityAlertEmails,
                EnablePredictionAlertEmails = profile.EnablePredictionAlertEmails,
                EnableGlobalAlertEmails = profile.EnableGlobalAlertEmails,
                EnableSystemHealthAlertEmails = profile.EnableSystemHealthAlertEmails,
                
                // SMS alerts
                AlertPhoneNumber = profile.AlertPhoneNumber,
                EnableSmsAlerts = profile.EnableSmsAlerts,
                EnableStandardAlertSms = profile.EnableStandardAlertSms,
                EnableOpportunityAlertSms = profile.EnableOpportunityAlertSms,
                EnablePredictionAlertSms = profile.EnablePredictionAlertSms,
                EnableGlobalAlertSms = profile.EnableGlobalAlertSms,
                
                // Push notifications
                PushNotificationUserId = profile.PushNotificationUserId,
                EnablePushNotifications = profile.EnablePushNotifications,
                EnableStandardAlertPushNotifications = profile.EnableStandardAlertPushNotifications,
                EnableOpportunityAlertPushNotifications = profile.EnableOpportunityAlertPushNotifications,
                EnablePredictionAlertPushNotifications = profile.EnablePredictionAlertPushNotifications,
                EnableGlobalAlertPushNotifications = profile.EnableGlobalAlertPushNotifications,
                EnableTechnicalIndicatorAlertPushNotifications = profile.EnableTechnicalIndicatorAlertPushNotifications,
                EnableSentimentShiftAlertPushNotifications = profile.EnableSentimentShiftAlertPushNotifications,
                EnableSystemHealthAlertPushNotifications = profile.EnableSystemHealthAlertPushNotifications,
                EnableTradeExecutionPushNotifications = profile.EnableTradeExecutionPushNotifications,
                
                // VIX monitoring
                EnableVixMonitoring = profile.EnableVixMonitoring,
                
                // API Keys
                AlphaVantageApiKey = profile.AlphaVantageApiKey,
                
                CreatedAt = profile.CreatedDate,
                LastModified = profile.ModifiedDate
            };
        }

        /// <summary>
        /// Updates entity from DatabaseSettingsProfile
        /// </summary>
        private void UpdateEntityFromProfile(SettingsProfile entity, DatabaseSettingsProfile profile)
        {
            entity.Name = profile.Name;
            entity.Description = profile.Description;
            entity.IsDefault = profile.IsDefault;
            entity.EnableApiModalChecks = profile.EnableApiModalChecks;
            entity.ApiTimeoutSeconds = profile.ApiTimeoutSeconds;
            entity.CacheDurationMinutes = profile.CacheDurationMinutes;
            entity.EnableHistoricalDataCache = profile.EnableHistoricalDataCache;
            entity.EnableDarkMode = profile.EnableDarkMode;
            entity.ChartUpdateIntervalSeconds = profile.ChartUpdateIntervalSeconds;
            entity.EnablePriceAlerts = profile.EnablePriceAlerts;
            entity.EnableTradeNotifications = profile.EnableTradeNotifications;
            entity.EnablePaperTrading = profile.EnablePaperTrading;
            entity.RiskLevel = profile.RiskLevel;
            entity.DefaultGridRows = profile.DefaultGridRows;
            entity.DefaultGridColumns = profile.DefaultGridColumns;
            entity.GridBorderColor = profile.GridBorderColor;
            
            // Email alerts
            entity.AlertEmail = profile.AlertEmail;
            entity.EnableEmailAlerts = profile.EnableEmailAlerts;
            entity.EnableStandardAlertEmails = profile.EnableStandardAlertEmails;
            entity.EnableOpportunityAlertEmails = profile.EnableOpportunityAlertEmails;
            entity.EnablePredictionAlertEmails = profile.EnablePredictionAlertEmails;
            entity.EnableGlobalAlertEmails = profile.EnableGlobalAlertEmails;
            entity.EnableSystemHealthAlertEmails = profile.EnableSystemHealthAlertEmails;
            
            // SMS alerts
            entity.AlertPhoneNumber = profile.AlertPhoneNumber;
            entity.EnableSmsAlerts = profile.EnableSmsAlerts;
            entity.EnableStandardAlertSms = profile.EnableStandardAlertSms;
            entity.EnableOpportunityAlertSms = profile.EnableOpportunityAlertSms;
            entity.EnablePredictionAlertSms = profile.EnablePredictionAlertSms;
            entity.EnableGlobalAlertSms = profile.EnableGlobalAlertSms;
            
            // Push notifications
            entity.PushNotificationUserId = profile.PushNotificationUserId;
            entity.EnablePushNotifications = profile.EnablePushNotifications;
            entity.EnableStandardAlertPushNotifications = profile.EnableStandardAlertPushNotifications;
            entity.EnableOpportunityAlertPushNotifications = profile.EnableOpportunityAlertPushNotifications;
            entity.EnablePredictionAlertPushNotifications = profile.EnablePredictionAlertPushNotifications;
            entity.EnableGlobalAlertPushNotifications = profile.EnableGlobalAlertPushNotifications;
            entity.EnableTechnicalIndicatorAlertPushNotifications = profile.EnableTechnicalIndicatorAlertPushNotifications;
            entity.EnableSentimentShiftAlertPushNotifications = profile.EnableSentimentShiftAlertPushNotifications;
            entity.EnableSystemHealthAlertPushNotifications = profile.EnableSystemHealthAlertPushNotifications;
            entity.EnableTradeExecutionPushNotifications = profile.EnableTradeExecutionPushNotifications;
            
            // VIX monitoring
            entity.EnableVixMonitoring = profile.EnableVixMonitoring;
            
            // API Keys
            entity.AlphaVantageApiKey = profile.AlphaVantageApiKey;
        }

        /// <summary>
        /// Maps SettingsProfile entity to DatabaseSettingsProfile
        /// </summary>
        private DatabaseSettingsProfile MapFromEntity(SettingsProfile entity)
        {
            return new DatabaseSettingsProfile
            {
                Id = entity.Id,
                Name = entity.Name,
                Description = entity.Description ?? "",
                IsDefault = entity.IsDefault,
                CreatedDate = entity.CreatedAt,
                ModifiedDate = entity.LastModified,
                EnableApiModalChecks = entity.EnableApiModalChecks,
                ApiTimeoutSeconds = entity.ApiTimeoutSeconds,
                CacheDurationMinutes = entity.CacheDurationMinutes,
                EnableHistoricalDataCache = entity.EnableHistoricalDataCache,
                EnableDarkMode = entity.EnableDarkMode,
                ChartUpdateIntervalSeconds = entity.ChartUpdateIntervalSeconds,
                DefaultGridRows = entity.DefaultGridRows,
                DefaultGridColumns = entity.DefaultGridColumns,
                GridBorderColor = entity.GridBorderColor ?? "#FF00FFFF",
                EnablePriceAlerts = entity.EnablePriceAlerts,
                EnableTradeNotifications = entity.EnableTradeNotifications,
                EnablePaperTrading = entity.EnablePaperTrading,
                RiskLevel = entity.RiskLevel ?? "Low",
                
                // Email alerts
                AlertEmail = entity.AlertEmail ?? "test@gmail.com",
                EnableEmailAlerts = entity.EnableEmailAlerts,
                EnableStandardAlertEmails = entity.EnableStandardAlertEmails,
                EnableOpportunityAlertEmails = entity.EnableOpportunityAlertEmails,
                EnablePredictionAlertEmails = entity.EnablePredictionAlertEmails,
                EnableGlobalAlertEmails = entity.EnableGlobalAlertEmails,
                EnableSystemHealthAlertEmails = entity.EnableSystemHealthAlertEmails,
                
                // SMS alerts
                AlertPhoneNumber = entity.AlertPhoneNumber ?? "",
                EnableSmsAlerts = entity.EnableSmsAlerts,
                EnableStandardAlertSms = entity.EnableStandardAlertSms,
                EnableOpportunityAlertSms = entity.EnableOpportunityAlertSms,
                EnablePredictionAlertSms = entity.EnablePredictionAlertSms,
                EnableGlobalAlertSms = entity.EnableGlobalAlertSms,
                
                // Push notifications
                PushNotificationUserId = entity.PushNotificationUserId ?? "",
                EnablePushNotifications = entity.EnablePushNotifications,
                EnableStandardAlertPushNotifications = entity.EnableStandardAlertPushNotifications,
                EnableOpportunityAlertPushNotifications = entity.EnableOpportunityAlertPushNotifications,
                EnablePredictionAlertPushNotifications = entity.EnablePredictionAlertPushNotifications,
                EnableGlobalAlertPushNotifications = entity.EnableGlobalAlertPushNotifications,
                EnableTechnicalIndicatorAlertPushNotifications = entity.EnableTechnicalIndicatorAlertPushNotifications,
                EnableSentimentShiftAlertPushNotifications = entity.EnableSentimentShiftAlertPushNotifications,
                EnableSystemHealthAlertPushNotifications = entity.EnableSystemHealthAlertPushNotifications,
                EnableTradeExecutionPushNotifications = entity.EnableTradeExecutionPushNotifications,
                
                // VIX monitoring
                EnableVixMonitoring = entity.EnableVixMonitoring,
                
                // API Keys
                AlphaVantageApiKey = entity.AlphaVantageApiKey ?? ""
            };
        }

        #endregion
    }
}
