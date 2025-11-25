using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Quantra.Models
{
    /// <summary>
    /// Serializable definition of a custom indicator
    /// </summary>
    public class CustomIndicatorDefinition
    {
        /// <summary>
        /// Unique identifier for the indicator
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// Display name for the indicator
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Description of the indicator
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Category of the indicator (e.g., Momentum, Volume, Trend)
        /// </summary>
        public string Category { get; set; }

        /// <summary>
        /// The type of indicator (Simple, Composite, etc.)
        /// </summary>
        public string IndicatorType { get; set; }

        /// <summary>
        /// For composite indicators, this contains the expression or formula
        /// </summary>
        public string Formula { get; set; }

        /// <summary>
        /// List of other indicators this one depends on
        /// </summary>
        public List<string> Dependencies { get; set; }

        /// <summary>
        /// Parameters and their configurations
        /// </summary>
        public Dictionary<string, IndicatorParameterDefinition> Parameters { get; set; }

        /// <summary>
        /// Visual styling options for chart display
        /// </summary>
        public IndicatorVisualization Visualization { get; set; }

        /// <summary>
        /// Date the indicator was created
        /// </summary>
        public DateTime CreatedDate { get; set; }

        /// <summary>
        /// Date the indicator was last modified
        /// </summary>
        public DateTime ModifiedDate { get; set; }

        /// <summary>
        /// The user who created this indicator
        /// </summary>
        public string CreatedBy { get; set; }

        /// <summary>
        /// Whether this indicator is visible in lists
        /// </summary>
        public bool IsVisible { get; set; }

        /// <summary>
        /// Whether this indicator is a built-in system one
        /// </summary>
        [JsonIgnore]
        public bool IsBuiltIn { get; set; }

        /// <summary>
        /// The output keys this indicator produces
        /// </summary>
        public List<string> OutputKeys { get; set; }

        public CustomIndicatorDefinition()
        {
            Id = Guid.NewGuid().ToString();
            Dependencies = new List<string>();
            Parameters = new Dictionary<string, IndicatorParameterDefinition>();
            OutputKeys = new List<string> { "Value" };
            CreatedDate = DateTime.Now;
            ModifiedDate = DateTime.Now;
            IsVisible = true;
            IsBuiltIn = false;
            Visualization = new IndicatorVisualization();
        }
    }

    /// <summary>
    /// Serializable definition of an indicator parameter
    /// </summary>
    public class IndicatorParameterDefinition
    {
        /// <summary>
        /// Name of the parameter
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Description of the parameter
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Default value for the parameter
        /// </summary>
        public object DefaultValue { get; set; }

        /// <summary>
        /// Current value for the parameter
        /// </summary>
        public object Value { get; set; }

        /// <summary>
        /// Type of the parameter (int, double, etc.)
        /// </summary>
        public string ParameterType { get; set; }

        /// <summary>
        /// Minimum allowed value (for numeric parameters)
        /// </summary>
        public object MinValue { get; set; }

        /// <summary>
        /// Maximum allowed value (for numeric parameters)
        /// </summary>
        public object MaxValue { get; set; }

        /// <summary>
        /// Whether the parameter is optional
        /// </summary>
        public bool IsOptional { get; set; }

        /// <summary>
        /// For selection parameters, the available options
        /// </summary>
        public string[] Options { get; set; }

        public IndicatorParameterDefinition()
        {
            IsOptional = false;
        }
    }

    /// <summary>
    /// Visual styling options for indicator display
    /// </summary>
    public class IndicatorVisualization
    {
        /// <summary>
        /// The color to use when rendering the indicator
        /// </summary>
        public string Color { get; set; }

        /// <summary>
        /// The thickness of the line when rendering
        /// </summary>
        public double LineThickness { get; set; }

        /// <summary>
        /// The style of the line (Solid, Dashed, etc.)
        /// </summary>
        public string LineStyle { get; set; }

        /// <summary>
        /// Whether the indicator should be displayed in its own panel
        /// </summary>
        public bool ShowInSeparatePanel { get; set; }

        /// <summary>
        /// Whether the indicator should be displayed on the price chart
        /// </summary>
        public bool ShowOnPriceChart { get; set; }

        /// <summary>
        /// Height of the panel when ShowInSeparatePanel is true (as a proportion)
        /// </summary>
        public double PanelHeight { get; set; }

        public IndicatorVisualization()
        {
            Color = "#FF1E90FF";  // Default to DodgerBlue
            LineThickness = 1.5;
            LineStyle = "Solid";
            ShowInSeparatePanel = false;
            ShowOnPriceChart = true;
            PanelHeight = 0.2;
        }
    }
}