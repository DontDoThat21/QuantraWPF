using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using System;
using System.Linq;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing tab configuration and control placement
    /// </summary>
    public class TabConfigurationService
    {
        private readonly QuantraDbContext _dbContext;
        private readonly LoggingService _loggingService;

        public TabConfigurationService(QuantraDbContext dbContext, LoggingService loggingService)
        {
            _dbContext = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Loads controls configuration for a specific tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <returns>The controls configuration string, or empty string if not found</returns>
        public string LoadControlsConfig(string tabName)
        {
            try
            {
                // Query the UserAppSettings table for the tab
                var tabConfig = _dbContext.UserAppSettings
                    .AsNoTracking()
                    .FirstOrDefault(t => t.TabName == tabName);

                if (tabConfig != null && !string.IsNullOrWhiteSpace(tabConfig.ControlsConfig))
                {
                    return tabConfig.ControlsConfig;
                }

                // Return empty string if no config found
                return string.Empty;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to load controls configuration for tab '{tabName}'", ex.ToString());
                return string.Empty;
            }
        }

        /// <summary>
        /// Saves controls configuration for a specific tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <param name="controlsConfig">The controls configuration string</param>
        public void SaveControlsConfig(string tabName, string controlsConfig)
        {
            try
            {
                // Find or create tab configuration
                var tabConfig = _dbContext.UserAppSettings
                    .FirstOrDefault(t => t.TabName == tabName);

                if (tabConfig == null)
                {
                    // Create new tab configuration
                    tabConfig = new Data.Entities.UserAppSetting
                    {
                        TabName = tabName,
                        ControlsConfig = controlsConfig,
                        GridRows = 4, // Default
                        GridColumns = 4, // Default
                        TabOrder = _dbContext.UserAppSettings.Count()
                    };
                    _dbContext.UserAppSettings.Add(tabConfig);
                }
                else
                {
                    // Update existing configuration
                    tabConfig.ControlsConfig = controlsConfig;
                }

                _dbContext.SaveChanges();
                _loggingService.Log("Info", $"Saved controls configuration for tab '{tabName}'");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to save controls configuration for tab '{tabName}'", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Adds a custom control with span configuration to a tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <param name="controlType">The type of control to add</param>
        /// <param name="row">The row position</param>
        /// <param name="column">The column position</param>
        /// <param name="rowSpan">The row span</param>
        /// <param name="columnSpan">The column span</param>
        public void AddCustomControlWithSpans(string tabName, string controlType, int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                var controlDefinition = $"{controlType},{row},{column},{rowSpan},{columnSpan}";
                var currentControls = LoadControlsConfig(tabName) ?? string.Empty;

                // Fix: Ensure we're using semicolons as separators instead of newlines
                // Trim any trailing semicolons to avoid empty entries
                currentControls = currentControls.Trim().TrimEnd(';');

                // Append the new control with proper semicolon separator
                var updatedControls = string.IsNullOrEmpty(currentControls)
                    ? controlDefinition
                    : currentControls + ";" + controlDefinition;

                SaveControlsConfig(tabName, updatedControls);

                _loggingService.Log("Info", $"Added control to tab '{tabName}' with spans: {controlDefinition}");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to add control with spans to tab '{tabName}'", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Updates the position of a control in a tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <param name="controlIndex">The index of the control (0-based)</param>
        /// <param name="row">The new row position</param>
        /// <param name="column">The new column position</param>
        /// <param name="rowSpan">The new row span</param>
        /// <param name="columnSpan">The new column span</param>
        public void UpdateControlPosition(string tabName, int controlIndex, int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                var controlsConfig = LoadControlsConfig(tabName);
                if (string.IsNullOrWhiteSpace(controlsConfig))
                {
                    _loggingService.Log("Warning", $"No controls config found for tab '{tabName}' when trying to update control position");
                    return;
                }

                // Split the controls config
                var controls = controlsConfig
                    .Replace("\r\n", ";")
                    .Replace("\n", ";")
                    .Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries)
                    .ToList();

                // Validate control index
                if (controlIndex < 0 || controlIndex >= controls.Count)
                {
                    _loggingService.Log("Warning", $"Invalid control index {controlIndex} for tab '{tabName}' with {controls.Count} controls");
                    return;
                }

                // Parse the control at the specified index
                var controlParts = controls[controlIndex].Split(',');
                if (controlParts.Length < 3)
                {
                    _loggingService.Log("Warning", $"Invalid control definition at index {controlIndex} in tab '{tabName}'");
                    return;
                }

                // Update the control position with the new values
                string controlType = controlParts[0].Trim();
                controls[controlIndex] = $"{controlType},{row},{column},{rowSpan},{columnSpan}";

                // Rebuild the configuration string
                var updatedConfig = string.Join(";", controls);

                // Save back to database
                SaveControlsConfig(tabName, updatedConfig);

                _loggingService.Log("Info", $"Updated control position in tab '{tabName}' at index {controlIndex} to ({row},{column}) with spans ({rowSpan},{columnSpan})");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to update control position in tab '{tabName}'", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Removes a control from a tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <param name="controlIndex">The index of the control to remove (0-based)</param>
        public void RemoveControl(string tabName, int controlIndex)
        {
            try
            {
                var controlsConfig = LoadControlsConfig(tabName);
                if (string.IsNullOrWhiteSpace(controlsConfig))
                {
                    _loggingService.Log("Warning", $"No controls config found for tab '{tabName}' when trying to remove control");
                    return;
                }

                // Split the controls config
                var controls = controlsConfig
                    .Replace("\r\n", ";")
                    .Replace("\n", ";")
                    .Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries)
                    .ToList();

                // Validate control index
                if (controlIndex < 0 || controlIndex >= controls.Count)
                {
                    _loggingService.Log("Warning", $"Invalid control index {controlIndex} for tab '{tabName}' with {controls.Count} controls");
                    return;
                }

                // Remove the control at the specified index
                controls.RemoveAt(controlIndex);

                // Rebuild the configuration string
                var updatedConfig = string.Join(";", controls);

                // Save back to database
                SaveControlsConfig(tabName, updatedConfig);

                _loggingService.Log("Info", $"Removed control at index {controlIndex} from tab '{tabName}'");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to remove control from tab '{tabName}'", ex.ToString());
                throw;
            }
        }
    }
}
