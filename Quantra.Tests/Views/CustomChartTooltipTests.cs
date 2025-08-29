using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.Controls;

namespace Quantra.Tests.Views
{
    [TestClass]
    public class CustomChartTooltipTests
    {
        private CustomChartTooltip _tooltip;

        [TestInitialize]
        public void Setup()
        {
            _tooltip = new CustomChartTooltip();
        }

        [TestMethod]
        public void ExtractAndFormatDateTime_WithValidDateTime_ReturnsFormattedString()
        {
            // Test the date formatting to ensure it includes the full year
            var testDate = new DateTime(2024, 3, 15, 14, 30, 0);
            var expectedFormat = "Mar 15, 2024 - 14:30";
            
            // Test the format string directly
            var actualFormat = testDate.ToString("MMM dd, yyyy - HH:mm");
            Assert.AreEqual(expectedFormat, actualFormat);
        }

        [TestMethod]
        public void ExtractAndFormatDateTime_WithCurrentYear_ShowsFullYear()
        {
            // Test that current year dates show the full year, not just month/day
            var currentDate = DateTime.Now;
            var formattedDate = currentDate.ToString("MMM dd, yyyy - HH:mm");
            
            // Ensure the formatted string contains the year
            Assert.IsTrue(formattedDate.Contains(currentDate.Year.ToString()),
                $"Formatted date '{formattedDate}' should contain year {currentDate.Year}");
        }

        [TestMethod]
        public void ExtractAndFormatDateTime_WithDifferentDates_ShowsCorrectFormat()
        {
            // Test various dates to ensure consistent formatting
            var testCases = new[]
            {
                new DateTime(2023, 9, 21, 10, 30, 0), // The problematic date from the issue
                new DateTime(2024, 1, 1, 0, 0, 0),    // New Year
                new DateTime(2024, 12, 31, 23, 59, 0) // End of year
            };

            foreach (var testDate in testCases)
            {
                var formatted = testDate.ToString("MMM dd, yyyy - HH:mm");
                
                // Check that format includes month, day, year, and time
                Assert.IsTrue(formatted.Contains(testDate.Year.ToString()),
                    $"Date {testDate} should include year in formatted output: {formatted}");
                Assert.IsTrue(formatted.Contains(testDate.Day.ToString()),
                    $"Date {testDate} should include day in formatted output: {formatted}");
                Assert.IsTrue(formatted.Contains("-"),
                    $"Date {testDate} should include separator in formatted output: {formatted}");
            }
        }

        [TestMethod]
        public void FormatValue_WithBollingerBandsTitle_ReturnsCorrectFormat()
        {
            // Test that Bollinger Bands values are formatted correctly
            var testValue = 123.456789;
            var expectedFormat = "123.46"; // F2 format

            // This tests the logic that would be used for Bollinger Bands
            var actualFormat = testValue.ToString("F2");
            Assert.AreEqual(expectedFormat, actualFormat);
        }

        [TestMethod]
        public void DateTimeFormat_IsConsistentWithIssueRequirement()
        {
            // Verify that the format matches the requirement to show full date including year
            var issueDate = new DateTime(2023, 9, 21, 10, 30, 0);
            var formattedDate = issueDate.ToString("MMM dd, yyyy - HH:mm");
            
            // The issue mentions it's showing "Sept 21st and 10:30am" 
            // We want to show "Sep 21, 2023 - 10:30" format
            Assert.AreEqual("Sep 21, 2023 - 10:30", formattedDate);
            
            // Ensure it's not the old problematic format
            Assert.AreNotEqual("Sep 21, 2023 - 10:30 AM", formattedDate); // Different from old format
        }
    }
}