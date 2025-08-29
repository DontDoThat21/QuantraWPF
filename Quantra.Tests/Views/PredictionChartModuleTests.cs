using System;
using System.Collections.Generic;
using Quantra.Views.PredictionAnalysis.Components;
using Xunit;

namespace Quantra.Tests.Views
{
    public class PredictionChartModuleTests
    {
        [Fact]
        public void Constructor_ShouldInitializePropertiesBeforeDataContextBinding()
        {
            // Arrange & Act
            var chartModule = new PredictionChartModule();

            // Assert
            Assert.NotNull(chartModule.Labels);
            Assert.NotNull(chartModule.YFormatter);
            Assert.NotNull(chartModule.ChartSeries);
            
            // Test that YFormatter actually works
            var formattedValue = chartModule.YFormatter(123.456);
            Assert.Equal("123.46", formattedValue);
        }

        [Fact]
        public void Labels_WhenSetToNewValue_ShouldTriggerPropertyChanged()
        {
            // Arrange
            var chartModule = new PredictionChartModule();
            bool propertyChangedTriggered = false;
            chartModule.PropertyChanged += (sender, args) =>
            {
                if (args.PropertyName == nameof(chartModule.Labels))
                    propertyChangedTriggered = true;
            };

            // Act
            chartModule.Labels = new List<string> { "Label1", "Label2" };

            // Assert
            Assert.True(propertyChangedTriggered);
            Assert.Equal(2, chartModule.Labels.Count);
        }

        [Fact]
        public void YFormatter_WhenSetToNewValue_ShouldTriggerPropertyChanged()
        {
            // Arrange
            var chartModule = new PredictionChartModule();
            bool propertyChangedTriggered = false;
            chartModule.PropertyChanged += (sender, args) =>
            {
                if (args.PropertyName == nameof(chartModule.YFormatter))
                    propertyChangedTriggered = true;
            };

            // Act
            chartModule.YFormatter = value => $"${value:F1}";

            // Assert
            Assert.True(propertyChangedTriggered);
            var result = chartModule.YFormatter(100.5);
            Assert.Equal("$100.5", result);
        }
    }
}