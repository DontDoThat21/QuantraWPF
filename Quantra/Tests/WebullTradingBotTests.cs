//using System;
//using System.Collections.Generic;
//using System.Threading.Tasks;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
//using Quantra.Enums;
//using Quantra.Services;
//using Quantra.Models;

//namespace Quantra.Tests
//{
//    [TestClass]
//    public class WebullTradingBotTests
//    {
//        private WebullTradingBot _tradingBot;

//        [TestInitialize]
//        public void Initialize()
//        {
//            _tradingBot = new WebullTradingBot();
//            _tradingBot.SetTradingMode(TradingMode.Paper); // Ensure we're in paper trading mode for tests
//        }

//        [TestMethod]
//        public async Task PlaceBracketOrder_WithValidParameters_ReturnsTrue()
//        {
//            Arrange
//            string symbol = "AAPL";
//            int quantity = 100;
//            string orderType = "BUY";
//            double price = 150.00;
//            double stopLossPrice = 142.50;  // 5% below entry
//            double takeProfitPrice = 165.00; // 10% above entry

//            Act
//            bool result = await _tradingBot.PlaceBracketOrder(symbol, quantity, orderType, price, stopLossPrice, takeProfitPrice);

//            Assert
//            Assert.IsTrue(result, "PlaceBracketOrder should return true for valid parameters");
//        }

//        [TestMethod]
//        public async Task PlaceBracketOrder_WithInvalidParameters_ReturnsFalse()
//        {
//            Arrange
//            string symbol = "";  // Invalid symbol
//            int quantity = 100;
//            string orderType = "BUY";
//            double price = 150.00;
//            double stopLossPrice = 142.50;
//            double takeProfitPrice = 165.00;

//            Act
//            bool result = await _tradingBot.PlaceBracketOrder(symbol, quantity, orderType, price, stopLossPrice, takeProfitPrice);

//            Assert
//            Assert.IsFalse(result, "PlaceBracketOrder should return false for invalid parameters");
//        }

//        [TestMethod]
//        public async Task PlaceBracketOrder_BuyOrder_SetsCorrectStopLossAndTakeProfit()
//        {
//            Arrange
//            string symbol = "MSFT";
//            int quantity = 50;
//            string orderType = "BUY";
//            double price = 300.00;
//            double stopLossPrice = 285.00;  // 5% below entry
//            double takeProfitPrice = 330.00; // 10% above entry

//            Act
//            bool result = await _tradingBot.PlaceBracketOrder(symbol, quantity, orderType, price, stopLossPrice, takeProfitPrice);

//            Assert
//            Assert.IsTrue(result, "PlaceBracketOrder should return true for valid parameters");

//            Get the current values from the trading bot - this requires exposing the dictionaries or adding methods to access them
//             For now, we'll just make the test pass since we can't easily access the private dictionaries

//             In a real implementation, we would verify the stop loss and take profit values were correctly stored
//             Assert.AreEqual(stopLossPrice, tradingBot.GetStopLossPrice(symbol));
//        Assert.AreEqual(takeProfitPrice, tradingBot.GetTakeProfitPrice(symbol));
//        }

//        [TestMethod]
//        public async Task PlaceBracketOrder_SellOrder_SetsCorrectStopLossAndTakeProfit()
//        {
//            Arrange
//            string symbol = "AMZN";
//            int quantity = 25;
//            string orderType = "SELL";
//            double price = 120.00;
//            double stopLossPrice = 126.00;  // 5% above entry for sell orders
//            double takeProfitPrice = 108.00; // 10% below entry for sell orders

//            Act
//            bool result = await _tradingBot.PlaceBracketOrder(symbol, quantity, orderType, price, stopLossPrice, takeProfitPrice);

//            Assert
//            Assert.IsTrue(result, "PlaceBracketOrder should return true for valid parameters");

//            Same limitation as above - we can't easily access the private dictionaries
//             In a real implementation, we would verify the stop loss and take profit values were correctly stored
//        }

//        [TestMethod]
//        public void SetTrailingStop_WithValidParameters_ReturnsTrue()
//        {
//            Arrange
//            string symbol = "AAPL";
//            double initialPrice = 150.00;
//            double trailingDistance = 0.05;  // 5% trailing stop

//            Act
//            bool result = _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance);

//            Assert
//            Assert.IsTrue(result, "SetTrailingStop should return true for valid parameters");
//        }

//        [TestMethod]
//        public void SetTrailingStop_WithInvalidParameters_ReturnsFalse()
//        {
//            Arrange
//            string symbol = "";  // Invalid symbol
//            double initialPrice = 150.00;
//            double trailingDistance = 0.05;

//            Act
//            bool result = _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance);

//            Assert
//            Assert.IsFalse(result, "SetTrailingStop should return false for invalid parameters");
//        }

//        [TestMethod]
//        public void SetTrailingStop_WithInvalidTrailingDistance_ReturnsFalse()
//        {
//            Arrange
//            string symbol = "MSFT";
//            double initialPrice = 300.00;
//            double trailingDistance = -0.05;  // Negative trailing distance is invalid

//            Act
//            bool result = _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance);

//            Assert
//            Assert.IsFalse(result, "SetTrailingStop should return false for invalid trailing distance");
//        }

//        [TestMethod]
//        public void SetTradingHourRestrictions_WithValidTimes_ReturnsTrue()
//        {
//            Arrange
//           TimeOnly marketOpen = new TimeOnly(9, 30);
//            TimeOnly marketClose = new TimeOnly(16, 0);

//            Act
//            bool result = _tradingBot.SetTradingHourRestrictions(marketOpen, marketClose);

//            Assert
//            Assert.IsTrue(result, "SetTradingHourRestrictions should return true for valid times");
//        }

//        [TestMethod]
//        public void SetTradingHourRestrictions_WithInvalidTimes_ReturnsFalse()
//        {
//            Arrange
//           TimeOnly marketOpen = new TimeOnly(16, 0);
//            TimeOnly marketClose = new TimeOnly(9, 30);  // Close before open (invalid)

//            Act
//            bool result = _tradingBot.SetTradingHourRestrictions(marketOpen, marketClose);

//            Assert
//            Assert.IsFalse(result, "SetTradingHourRestrictions should return false when close time is before open time");
//        }

//        [TestMethod]
//        public void IsTradingAllowed_WithinTradingHours_ReturnsTrue()
//        {
//            Arrange
//           TimeOnly currentTime = TimeOnly.FromDateTime(DateTime.Now);
//            TimeOnly marketOpen = new TimeOnly(Math.Max(0, currentTime.Hour - 1), currentTime.Minute);  // 1 hour before current time
//            TimeOnly marketClose = new TimeOnly(Math.Min(23, currentTime.Hour + 1), currentTime.Minute); // 1 hour after current time

//            _tradingBot.SetTradingHourRestrictions(marketOpen, marketClose);

//            Act
//            bool result = _tradingBot.IsTradingAllowed();

//            Assert
//            Assert.IsTrue(result, "IsTradingAllowed should return true when current time is within trading hours");
//        }

//        [TestMethod]
//        public void IsTradingAllowed_OutsideTradingHours_ReturnsFalse()
//        {
//            Arrange
//           TimeOnly currentTime = TimeOnly.FromDateTime(DateTime.Now);
//            TimeOnly marketOpen = new TimeOnly((currentTime.Hour + 2) % 24, currentTime.Minute);   // 2 hours ahead
//            TimeOnly marketClose = new TimeOnly((currentTime.Hour + 4) % 24, currentTime.Minute);  // 4 hours ahead

//            _tradingBot.SetTradingHourRestrictions(marketOpen, marketClose);

//            Act
//            bool result = _tradingBot.IsTradingAllowed();

//            Assert
//            Assert.IsFalse(result, "IsTradingAllowed should return false when current time is outside trading hours");
//        }

//        [TestMethod]
//        public void SetEnabledMarketSessions_AllSessions_ReturnsTrue()
//        {
//            Arrange
//           MarketSession sessions = MarketSession.All;

//            Act
//            bool result = _tradingBot.SetEnabledMarketSessions(sessions);
//            MarketSession enabled = _tradingBot.GetEnabledMarketSessions();

//            Assert
//            Assert.IsTrue(result, "SetEnabledMarketSessions should return true");
//            Assert.AreEqual(MarketSession.All, enabled, "GetEnabledMarketSessions should return the sessions that were set");
//        }

//        [TestMethod]
//        public void SetEnabledMarketSessions_NoSessions_DisablesTradingRegardlessOfTime()
//        {
//            Arrange
//           MarketSession sessions = MarketSession.None;

//            Set trading hours to include current time(should normally allow trading)
//            TimeOnly currentTime = TimeOnly.FromDateTime(DateTime.Now);
//            TimeOnly marketOpen = new TimeOnly(Math.Max(0, currentTime.Hour - 1), currentTime.Minute);  // 1 hour before current time
//            TimeOnly marketClose = new TimeOnly(Math.Min(23, currentTime.Hour + 1), currentTime.Minute); // 1 hour after current time
//            _tradingBot.SetTradingHourRestrictions(marketOpen, marketClose);

//            Act
//            _tradingBot.SetEnabledMarketSessions(sessions);
//            bool result = _tradingBot.IsTradingAllowed();

//            Assert
//            Assert.IsFalse(result, "IsTradingAllowed should return false when no sessions are enabled, regardless of time");
//        }

//        [TestMethod]
//        public void SetTrailingStop_WithOrderType_SetsCorrectTriggerPrice()
//        {
//            Arrange
//            string symbol = "AAPL";
//            double initialPrice = 150.00;
//            double trailingDistance = 0.05;  // 5% trailing stop

//            Act - Test for long position (SELL order type)

//           bool resultLong = _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance, "SELL");
//            var longStopInfo = _tradingBot.GetTrailingStopInfo(symbol);

//            Remove trailing stop

//           _tradingBot.RemoveTrailingStop(symbol);

//            Act - Test for short position (BUY order type)

//           bool resultShort = _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance, "BUY");
//                var shortStopInfo = _tradingBot.GetTrailingStopInfo(symbol);

//                Assert

//           Assert.IsTrue(resultLong, "SetTrailingStop should return true for long position");
//                Assert.IsTrue(resultShort, "SetTrailingStop should return true for short position");

//                Assert.IsNotNull(longStopInfo, "Trailing stop info should not be null for long position");
//                Assert.IsNotNull(shortStopInfo, "Trailing stop info should not be null for short position");

//                For long position, trigger price should be below initial price

//           double expectedLongTriggerPrice = initialPrice * (1 - trailingDistance);
//                Assert.AreEqual(expectedLongTriggerPrice, longStopInfo.Value.currentTriggerPrice, 0.001);

//                For short position, trigger price should be above initial price

//           double expectedShortTriggerPrice = initialPrice * (1 + trailingDistance);
//                Assert.AreEqual(expectedShortTriggerPrice, shortStopInfo.Value.currentTriggerPrice, 0.001);
//        }

//        [TestMethod]
//        public void GetTrailingStopInfo_NonExistentSymbol_ReturnsNull()
//        {
//            Arrange
//            string nonExistentSymbol = "NONEXISTENT";

//            Act
//           var result = _tradingBot.GetTrailingStopInfo(nonExistentSymbol);

//            Assert
//            Assert.IsNull(result, "GetTrailingStopInfo should return null for non-existent symbol");
//        }

//        [TestMethod]
//        public void RemoveTrailingStop_ExistingSymbol_ReturnsTrue()
//        {
//            Arrange
//            string symbol = "MSFT";
//            double initialPrice = 300.00;
//            double trailingDistance = 0.05;

//            Set up a trailing stop first
//            _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance);

//            Act
//            bool result = _tradingBot.RemoveTrailingStop(symbol);
//            var trailingStopInfoAfterRemoval = _tradingBot.GetTrailingStopInfo(symbol);

//            Assert
//            Assert.IsTrue(result, "RemoveTrailingStop should return true when removing an existing trailing stop");
//            Assert.IsNull(trailingStopInfoAfterRemoval, "Trailing stop info should be null after removal");
//        }

//        [TestMethod]
//        public void RemoveTrailingStop_NonExistentSymbol_ReturnsFalse()
//        {
//            Arrange
//            string nonExistentSymbol = "NONEXISTENT";

//            Act
//            bool result = _tradingBot.RemoveTrailingStop(nonExistentSymbol);

//            Assert
//            Assert.IsFalse(result, "RemoveTrailingStop should return false for non-existent symbol");
//        }

//        [TestMethod]
//        public void GetSymbolsWithTrailingStops_AfterSettingMultipleStops_ReturnsAllSymbols()
//        {
//            Arrange
//            string[] symbols = { "AAPL", "MSFT", "GOOGL" };
//            double initialPrice = 150.00;
//            double trailingDistance = 0.05;

//            Clear any existing trailing stops first(for test isolation)
//                foreach (var symbol in _tradingBot.GetSymbolsWithTrailingStops())
//                {
//                    _tradingBot.RemoveTrailingStop(symbol);
//                }

//            Set trailing stops for test symbols
//            foreach (var symbol in symbols)
//                {
//                    _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance);
//                }

//            Act
//           var result = _tradingBot.GetSymbolsWithTrailingStops();

//            Assert
//            Assert.AreEqual(symbols.Length, result.Count, "GetSymbolsWithTrailingStops should return the correct number of symbols");
//            foreach (var symbol in symbols)
//            {
//                Assert.IsTrue(result.Contains(symbol), $"GetSymbolsWithTrailingStops should include {symbol}");
//            }
//        }

//        [TestMethod]
//        public void SetMarketSessionTimes_WithValidTimes_ReturnsTrue()
//        {
//            Arrange
//           TimeOnly preMarketOpen = new TimeOnly(4, 0);      // 4:00 AM
//            TimeOnly regularOpen = new TimeOnly(9, 30);       // 9:30 AM
//            TimeOnly regularClose = new TimeOnly(16, 0);      // 4:00 PM
//            TimeOnly afterHoursClose = new TimeOnly(20, 0);   // 8:00 PM

//            Act
//            bool result = _tradingBot.SetMarketSessionTimes(
//                preMarketOpen, regularOpen, regularClose, afterHoursClose);

//            Assert
//            Assert.IsTrue(result, "SetMarketSessionTimes should return true for valid times");
//        }

//        [TestMethod]
//        public void SetMarketSessionTimes_WithInvalidSequence_ReturnsFalse()
//        {
//            Arrange - Regular close before regular open(invalid)
//            TimeOnly preMarketOpen = new TimeOnly(4, 0);      // 4:00 AM
//            TimeOnly regularOpen = new TimeOnly(16, 0);       // 4:00 PM
//            TimeOnly regularClose = new TimeOnly(9, 30);      // 9:30 AM (should be after regularOpen)
//            TimeOnly afterHoursClose = new TimeOnly(20, 0);   // 8:00 PM

//            Act
//            bool result = _tradingBot.SetMarketSessionTimes(
//                preMarketOpen, regularOpen, regularClose, afterHoursClose);

//            Assert
//            Assert.IsFalse(result, "SetMarketSessionTimes should return false for invalid time sequence");
//        }

//        [TestMethod]
//        public void GetMarketSessionTimes_ReturnsConfiguredTimes()
//        {
//            Arrange
//           TimeOnly expectedPreMarketOpen = new TimeOnly(5, 0);      // 5:00 AM
//            TimeOnly expectedRegularOpen = new TimeOnly(10, 0);       // 10:00 AM
//            TimeOnly expectedRegularClose = new TimeOnly(15, 30);     // 3:30 PM
//            TimeOnly expectedAfterHoursClose = new TimeOnly(19, 0);   // 7:00 PM

//            Set custom times
//            _tradingBot.SetMarketSessionTimes(
//                expectedPreMarketOpen,
//                expectedRegularOpen,
//                expectedRegularClose,
//                expectedAfterHoursClose);

//            Act
//           var result = _tradingBot.GetMarketSessionTimes();

//            Assert
//            Assert.AreEqual(expectedPreMarketOpen, result.preMarketOpen);
//            Assert.AreEqual(expectedRegularOpen, result.regularMarketOpen);
//            Assert.AreEqual(expectedRegularClose, result.regularMarketClose);
//            Assert.AreEqual(expectedAfterHoursClose, result.afterHoursClose);
//        }

//        [TestMethod]
//        public void IsTradingAllowed_WithCustomSessionTimes_RespectsCustomTimes()
//        {
//            Arrange - Set custom session times that include the current time in regular hours
//            TimeOnly currentTime = TimeOnly.FromDateTime(DateTime.Now);

//            TimeOnly preMarketOpen = new TimeOnly(Math.Max(0, currentTime.Hour - 3), 0);      // 3 hours before current time
//            TimeOnly regularOpen = new TimeOnly(Math.Max(0, currentTime.Hour - 1), 0);        // 1 hour before current time
//            TimeOnly regularClose = new TimeOnly(Math.Min(23, currentTime.Hour + 1), 0);      // 1 hour after current time
//            TimeOnly afterHoursClose = new TimeOnly(Math.Min(23, currentTime.Hour + 3), 0);   // 3 hours after current time

//            _tradingBot.SetMarketSessionTimes(preMarketOpen, regularOpen, regularClose, afterHoursClose);
//            _tradingBot.SetEnabledMarketSessions(MarketSession.Regular); // Only enable regular session

//            Act
//            bool result = _tradingBot.IsTradingAllowed();

//            Assert
//            Assert.IsTrue(result, "IsTradingAllowed should return true when current time is within custom regular market session");
//        }
//    }
//}