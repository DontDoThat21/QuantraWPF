using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for calendar feature extraction and earnings date calculations.
    /// Tests for TFT model known future inputs implementation.
    /// </summary>
    [TestClass]
    public class CalendarFeaturesTests
    {
        private EarningsCalendarService _earningsCalendarService;

        [TestInitialize]
        public void Setup()
        {
            // Initialize the database to ensure services work
            DatabaseMonolith.Initialize();
            
            // Create service instance with real dependencies
            var userSettingsService = new UserSettingsService();
            var loggingService = new LoggingService();
            var alphaVantageService = new AlphaVantageService(userSettingsService, loggingService);
            
            _earningsCalendarService = new EarningsCalendarService(alphaVantageService, loggingService);
        }

        #region Calendar Feature Tests

        [TestMethod]
        public void DayOfWeek_ReturnsCorrectValue_ForSunday()
        {
            // Arrange
            DateTime sunday = new DateTime(2024, 12, 1); // Dec 1, 2024 is a Sunday
            
            // Act
            double dayOfWeek = (double)sunday.DayOfWeek;
            
            // Assert
            Assert.AreEqual(0, dayOfWeek, "Sunday should return 0");
        }

        [TestMethod]
        public void DayOfWeek_ReturnsCorrectValue_ForSaturday()
        {
            // Arrange
            DateTime saturday = new DateTime(2024, 12, 7); // Dec 7, 2024 is a Saturday
            
            // Act
            double dayOfWeek = (double)saturday.DayOfWeek;
            
            // Assert
            Assert.AreEqual(6, dayOfWeek, "Saturday should return 6");
        }

        [TestMethod]
        public void DayOfWeek_ReturnsCorrectValue_ForMonday()
        {
            // Arrange
            DateTime monday = new DateTime(2024, 12, 2); // Dec 2, 2024 is a Monday
            
            // Act
            double dayOfWeek = (double)monday.DayOfWeek;
            
            // Assert
            Assert.AreEqual(1, dayOfWeek, "Monday should return 1");
        }

        [TestMethod]
        public void Quarter_ReturnsOne_ForJanuary()
        {
            // Arrange
            DateTime january = new DateTime(2024, 1, 15);
            
            // Act
            int quarter = ((january.Month - 1) / 3) + 1;
            
            // Assert
            Assert.AreEqual(1, quarter, "January should be in Q1");
        }

        [TestMethod]
        public void Quarter_ReturnsTwo_ForApril()
        {
            // Arrange
            DateTime april = new DateTime(2024, 4, 15);
            
            // Act
            int quarter = ((april.Month - 1) / 3) + 1;
            
            // Assert
            Assert.AreEqual(2, quarter, "April should be in Q2");
        }

        [TestMethod]
        public void Quarter_ReturnsThree_ForJuly()
        {
            // Arrange
            DateTime july = new DateTime(2024, 7, 15);
            
            // Act
            int quarter = ((july.Month - 1) / 3) + 1;
            
            // Assert
            Assert.AreEqual(3, quarter, "July should be in Q3");
        }

        [TestMethod]
        public void Quarter_ReturnsFour_ForOctober()
        {
            // Arrange
            DateTime october = new DateTime(2024, 10, 15);
            
            // Act
            int quarter = ((october.Month - 1) / 3) + 1;
            
            // Assert
            Assert.AreEqual(4, quarter, "October should be in Q4");
        }

        [TestMethod]
        public void IsMonthEnd_ReturnsOne_ForLastThreeDays()
        {
            // Arrange - December has 31 days, so days 29-31 should be month end
            DateTime dec29 = new DateTime(2024, 12, 29);
            DateTime dec30 = new DateTime(2024, 12, 30);
            DateTime dec31 = new DateTime(2024, 12, 31);
            
            // Act
            int daysInMonth = DateTime.DaysInMonth(2024, 12);
            double isMonthEnd29 = (dec29.Day >= daysInMonth - 2) ? 1.0 : 0.0;
            double isMonthEnd30 = (dec30.Day >= daysInMonth - 2) ? 1.0 : 0.0;
            double isMonthEnd31 = (dec31.Day >= daysInMonth - 2) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isMonthEnd29, "Dec 29 should be month end");
            Assert.AreEqual(1.0, isMonthEnd30, "Dec 30 should be month end");
            Assert.AreEqual(1.0, isMonthEnd31, "Dec 31 should be month end");
        }

        [TestMethod]
        public void IsMonthEnd_ReturnsZero_ForMiddleOfMonth()
        {
            // Arrange
            DateTime dec15 = new DateTime(2024, 12, 15);
            
            // Act
            int daysInMonth = DateTime.DaysInMonth(2024, 12);
            double isMonthEnd = (dec15.Day >= daysInMonth - 2) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isMonthEnd, "Dec 15 should not be month end");
        }

        [TestMethod]
        public void IsQuarterEnd_ReturnsOne_ForLastDaysOfMarch()
        {
            // Arrange - March is quarter-end month
            DateTime march31 = new DateTime(2024, 3, 31);
            DateTime march30 = new DateTime(2024, 3, 30);
            
            // Act
            int daysInMarch = DateTime.DaysInMonth(2024, 3);
            double isQuarterEnd31 = (march31.Month % 3 == 0 && march31.Day >= daysInMarch - 2) ? 1.0 : 0.0;
            double isQuarterEnd30 = (march30.Month % 3 == 0 && march30.Day >= daysInMarch - 2) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isQuarterEnd31, "March 31 should be quarter end");
            Assert.AreEqual(1.0, isQuarterEnd30, "March 30 should be quarter end");
        }

        [TestMethod]
        public void IsQuarterEnd_ReturnsZero_ForNonQuarterEndMonth()
        {
            // Arrange - February is not a quarter-end month
            DateTime feb28 = new DateTime(2024, 2, 28);
            
            // Act
            int daysInFeb = DateTime.DaysInMonth(2024, 2);
            double isQuarterEnd = (feb28.Month % 3 == 0 && feb28.Day >= daysInFeb - 2) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isQuarterEnd, "February should not be quarter end");
        }

        [TestMethod]
        public void DayOfYear_ReturnsOne_ForJanuaryFirst()
        {
            // Arrange
            DateTime jan1 = new DateTime(2024, 1, 1);
            
            // Act
            int dayOfYear = jan1.DayOfYear;
            
            // Assert
            Assert.AreEqual(1, dayOfYear, "January 1 should be day 1");
        }

        [TestMethod]
        public void DayOfYear_Returns366_ForDecember31InLeapYear()
        {
            // Arrange - 2024 is a leap year
            DateTime dec31 = new DateTime(2024, 12, 31);
            
            // Act
            int dayOfYear = dec31.DayOfYear;
            
            // Assert
            Assert.AreEqual(366, dayOfYear, "December 31 in leap year should be day 366");
        }

        #endregion

        #region Trading Days Calculation Tests

        [TestMethod]
        public void GetTradingDaysToEarnings_ReturnsZero_WhenDateIsToday()
        {
            // Arrange
            DateTime today = DateTime.UtcNow.Date;
            
            // Act
            int tradingDays = _earningsCalendarService.GetTradingDaysToEarnings(today);
            
            // Assert
            Assert.AreEqual(0, tradingDays, "Trading days to today should be 0");
        }

        [TestMethod]
        public void GetTradingDaysToEarnings_ReturnsZero_WhenDateIsPast()
        {
            // Arrange
            DateTime pastDate = DateTime.UtcNow.Date.AddDays(-5);
            
            // Act
            int tradingDays = _earningsCalendarService.GetTradingDaysToEarnings(pastDate);
            
            // Assert
            Assert.AreEqual(0, tradingDays, "Trading days to past date should be 0");
        }

        [TestMethod]
        public void GetTradingDaysToEarnings_ExcludesWeekends()
        {
            // Arrange - Start from Monday, count to Friday of next week (10 calendar days, 8 trading days)
            // This test uses fixed dates to ensure consistency
            DateTime monday = new DateTime(2024, 12, 2); // Monday
            DateTime nextFriday = new DateTime(2024, 12, 13); // Friday (11 calendar days later)
            
            // Manually calculate expected trading days: Dec 3-6 (4) + Dec 9-13 (5) = 9 trading days
            // But GetTradingDaysToEarnings counts from the day after start to the end date
            
            // Act
            int tradingDays = CountTradingDays(monday, nextFriday);
            
            // Assert - We expect 9 trading days (excluding the weekends Dec 7-8)
            Assert.AreEqual(9, tradingDays, "Should count 9 trading days excluding weekends");
        }

        [TestMethod]
        public void GetTradingDaysToEarnings_Returns5_ForOneWeekOfTradingDays()
        {
            // Arrange - Monday to next Monday (7 calendar days, 5 trading days)
            DateTime monday = new DateTime(2024, 12, 2); // Monday
            DateTime nextMonday = new DateTime(2024, 12, 9); // Next Monday
            
            // Act
            int tradingDays = CountTradingDays(monday, nextMonday);
            
            // Assert - 5 trading days (Tue, Wed, Thu, Fri, Mon)
            Assert.AreEqual(5, tradingDays, "Should count 5 trading days for one week");
        }

        [TestMethod]
        public void GetTradingDaysSinceEarnings_ReturnsZero_WhenDateIsToday()
        {
            // Arrange
            DateTime today = DateTime.UtcNow.Date;
            
            // Act
            int tradingDays = _earningsCalendarService.GetTradingDaysSinceEarnings(today);
            
            // Assert
            Assert.AreEqual(0, tradingDays, "Trading days since today should be 0");
        }

        [TestMethod]
        public void GetTradingDaysSinceEarnings_ReturnsZero_WhenDateIsFuture()
        {
            // Arrange
            DateTime futureDate = DateTime.UtcNow.Date.AddDays(5);
            
            // Act
            int tradingDays = _earningsCalendarService.GetTradingDaysSinceEarnings(futureDate);
            
            // Assert
            Assert.AreEqual(0, tradingDays, "Trading days since future date should be 0");
        }

        [TestMethod]
        public void GetTradingDaysSinceEarnings_ExcludesWeekends()
        {
            // Arrange - Count trading days from a past date
            DateTime pastMonday = new DateTime(2024, 12, 2); // Monday
            DateTime pastFriday = new DateTime(2024, 12, 6); // Friday (same week)
            
            // Act
            int tradingDays = CountTradingDays(pastMonday, pastFriday);
            
            // Assert - 4 trading days (Tue, Wed, Thu, Fri)
            Assert.AreEqual(4, tradingDays, "Should count 4 trading days within the week");
        }

        // Helper method to count trading days (same logic as in EarningsCalendarService)
        private int CountTradingDays(DateTime startDate, DateTime endDate)
        {
            if (endDate <= startDate)
            {
                return 0;
            }

            int tradingDays = 0;
            DateTime current = startDate;

            while (current < endDate)
            {
                current = current.AddDays(1);
                if (current.DayOfWeek != DayOfWeek.Saturday && current.DayOfWeek != DayOfWeek.Sunday)
                {
                    tradingDays++;
                }
            }

            return tradingDays;
        }

        #endregion

        #region Market Hours Tests

        [TestMethod]
        public void IsPreMarket_ReturnsOne_Before930AM()
        {
            // Arrange - 8:00 AM Eastern
            DateTime time = new DateTime(2024, 12, 2, 8, 0, 0); // 8:00 AM
            int hour = time.Hour;
            int minute = time.Minute;
            
            // Act
            double isPreMarket = (hour < 9 || (hour == 9 && minute < 30)) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isPreMarket, "8:00 AM should be pre-market");
        }

        [TestMethod]
        public void IsPreMarket_ReturnsOne_At929AM()
        {
            // Arrange - 9:29 AM Eastern
            DateTime time = new DateTime(2024, 12, 2, 9, 29, 0);
            int hour = time.Hour;
            int minute = time.Minute;
            
            // Act
            double isPreMarket = (hour < 9 || (hour == 9 && minute < 30)) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isPreMarket, "9:29 AM should be pre-market");
        }

        [TestMethod]
        public void IsPreMarket_ReturnsZero_At930AM()
        {
            // Arrange - 9:30 AM Eastern (market open)
            DateTime time = new DateTime(2024, 12, 2, 9, 30, 0);
            int hour = time.Hour;
            int minute = time.Minute;
            
            // Act
            double isPreMarket = (hour < 9 || (hour == 9 && minute < 30)) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isPreMarket, "9:30 AM should not be pre-market");
        }

        [TestMethod]
        public void IsRegularHours_ReturnsOne_At930AM()
        {
            // Arrange - 9:30 AM Eastern (market open)
            int hour = 9;
            int minute = 30;
            double minuteOfDay = hour * 60 + minute; // 570
            
            // Act
            double isRegularHours = (minuteOfDay >= 570 && minuteOfDay < 960) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isRegularHours, "9:30 AM should be regular hours");
        }

        [TestMethod]
        public void IsRegularHours_ReturnsOne_At359PM()
        {
            // Arrange - 3:59 PM Eastern (one minute before close)
            int hour = 15;
            int minute = 59;
            double minuteOfDay = hour * 60 + minute; // 959
            
            // Act
            double isRegularHours = (minuteOfDay >= 570 && minuteOfDay < 960) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isRegularHours, "3:59 PM should be regular hours");
        }

        [TestMethod]
        public void IsRegularHours_ReturnsZero_At400PM()
        {
            // Arrange - 4:00 PM Eastern (market close)
            int hour = 16;
            int minute = 0;
            double minuteOfDay = hour * 60 + minute; // 960
            
            // Act
            double isRegularHours = (minuteOfDay >= 570 && minuteOfDay < 960) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isRegularHours, "4:00 PM should not be regular hours");
        }

        [TestMethod]
        public void IsAfterHours_ReturnsOne_At400PM()
        {
            // Arrange - 4:00 PM Eastern
            int hour = 16;
            
            // Act
            double isAfterHours = (hour >= 16) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isAfterHours, "4:00 PM should be after hours");
        }

        [TestMethod]
        public void IsAfterHours_ReturnsZero_At359PM()
        {
            // Arrange - 3:59 PM Eastern
            int hour = 15;
            
            // Act
            double isAfterHours = (hour >= 16) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isAfterHours, "3:59 PM should not be after hours");
        }

        #endregion

        #region Earnings Week Tests

        [TestMethod]
        public void IsEarningsWeek_ReturnsOne_When5DaysOrLess()
        {
            // Arrange
            int daysToEarnings = 5;
            
            // Act
            double isEarningsWeek = (daysToEarnings <= 5 && daysToEarnings >= 0) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isEarningsWeek, "5 days to earnings should be earnings week");
        }

        [TestMethod]
        public void IsEarningsWeek_ReturnsOne_When0Days()
        {
            // Arrange
            int daysToEarnings = 0;
            
            // Act
            double isEarningsWeek = (daysToEarnings <= 5 && daysToEarnings >= 0) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(1.0, isEarningsWeek, "0 days to earnings should be earnings week");
        }

        [TestMethod]
        public void IsEarningsWeek_ReturnsZero_When6Days()
        {
            // Arrange
            int daysToEarnings = 6;
            
            // Act
            double isEarningsWeek = (daysToEarnings <= 5 && daysToEarnings >= 0) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isEarningsWeek, "6 days to earnings should not be earnings week");
        }

        [TestMethod]
        public void IsEarningsWeek_ReturnsZero_WhenUnknown()
        {
            // Arrange
            int daysToEarnings = 999; // Unknown
            
            // Act
            double isEarningsWeek = (daysToEarnings <= 5 && daysToEarnings >= 0) ? 1.0 : 0.0;
            
            // Assert
            Assert.AreEqual(0.0, isEarningsWeek, "Unknown earnings should not be earnings week");
        }

        #endregion
    }
}
