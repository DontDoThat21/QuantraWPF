using System;
using System.Collections.Generic;
using System.Windows.Media;
using Quantra.Models;
using System.Windows.Controls;


namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl : UserControl
    {
        // UI configuration and display settings
        public class DisplaySettings
        {
            // Color settings for text
            public static Brush DefaultTextColor => Brushes.GhostWhite;
            public static Brush HeaderTextColor => Brushes.LightBlue;
            public static Brush WarningTextColor => Brushes.Orange;
            public static Brush ErrorTextColor => Brushes.Red;
            public static Brush PositiveSignalColor => Brushes.LimeGreen;
            public static Brush NegativeSignalColor => Brushes.Crimson;
            public static Brush NeutralSignalColor => Brushes.Gray;

            // Colors for confidence ratings
            public static Brush HighConfidenceColor => Brushes.LimeGreen;
            public static Brush MediumConfidenceColor => Brushes.Gold;
            public static Brush LowConfidenceColor => Brushes.Orange;

            // Font sizes for different elements
            public const double HeaderFontSize = 16;
            public const double SubheaderFontSize = 14;
            public const double NormalFontSize = 12;
            public const double SmallFontSize = 10;

            // Default formatting for values
            public const string PercentageFormat = "P2"; // Percentage with 2 decimal places
            public const string PriceFormat = "C2"; // Currency with 2 decimal places
            public const string DecimalFormat = "F2"; // Fixed decimal with 2 decimal places
        }

        // PredictionModel has been removed to eliminate ambiguity.
        // Please use Quantra.Models.PredictionModel instead.
    }
}

// Ensure no mock/sample/placeholder data is used in model instantiation or test logic
// If you have any static or test data, remove it and ensure all models are populated from real AlphaVantageService data

public class PatternPoint
{
    public DateTime Timestamp { get; set; }
    public double Price { get; set; }
    public double Volume { get; set; }
    public string SignalStrength { get; set; } // e.g., "Strong", "Weak", "Neutral"
    public string ConfirmationStatus { get; set; } // e.g., "Confirmed", "Unconfirmed"
}