using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    public class IndicatorSettingsModel
    {
        public int Id { get; set; }
        // New property linking to the controls table primary key
        public int ControlId { get; set; }
        public string IndicatorName { get; set; }
        public bool IsEnabled { get; set; }
        public DateTime LastUpdated { get; set; }

        // Example indicator settings; adjust names/properties as needed
        public bool UseVwap { get; set; }
        public bool UseMacd { get; set; }
        public bool UseRsi { get; set; }
        public bool UseBollinger { get; set; }
        public bool UseMa { get; set; }
        public bool UseVolume { get; set; }
        // Add Breadth Thrust indicator setting
        public bool UseBreadthThrust { get; set; }

        // Constructor for creating indicator settings
        public IndicatorSettingsModel(int controlId, string indicatorName, bool isEnabled)
        {
            ControlId = controlId;
            IndicatorName = indicatorName;
            IsEnabled = isEnabled;
            LastUpdated = DateTime.Now;
        }

        // Default constructor for deserialization
        public IndicatorSettingsModel()
        {
        }
    }
}
