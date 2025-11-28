using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using Quantra;
using Quantra.Models;

namespace Quantra.Tests.Models
{
    [TestClass]
    public class QuoteDataOptionTests
    {
        [TestMethod]
        public void QuoteData_OptionChain_InitializesEmpty()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00
            };

            // Assert
            Assert.IsNotNull(quoteData.OptionChain);
            Assert.AreEqual(0, quoteData.OptionChain.Count);
        }

        [TestMethod]
        public void QuoteData_OptionDataFetchTimestamp_CanBeSet()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var fetchTime = DateTime.Now;

            // Act
            quoteData.OptionDataFetchTimestamp = fetchTime;

            // Assert
            Assert.AreEqual(fetchTime, quoteData.OptionDataFetchTimestamp);
        }

        [TestMethod]
        public void QuoteData_OptionDataCacheWindow_CanBeSet()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var cacheWindow = TimeSpan.FromMinutes(15);

            // Act
            quoteData.OptionDataCacheWindow = cacheWindow;

            // Assert
            Assert.AreEqual(cacheWindow, quoteData.OptionDataCacheWindow);
        }

        [TestMethod]
        public void QuoteData_AddOptionData_WorksCorrectly()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var optionData = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 150.0,
                ExpirationDate = DateTime.Now.AddDays(30),
                OptionType = "CALL",
                ImpliedVolatility = 0.25,
                Delta = 0.6,
                Theta = -0.05,
                Bid = 5.0,
                Ask = 5.5,
                LastPrice = 5.25,
                FetchTimestamp = DateTime.Now
            };

            // Act
            quoteData.OptionChain.Add(optionData);

            // Assert
            Assert.AreEqual(1, quoteData.OptionChain.Count);
            Assert.AreEqual("AAPL", quoteData.OptionChain[0].UnderlyingSymbol);
            Assert.AreEqual(150.0, quoteData.OptionChain[0].StrikePrice);
            Assert.AreEqual(0.25, quoteData.OptionChain[0].ImpliedVolatility);
            Assert.AreEqual(0.6, quoteData.OptionChain[0].Delta);
            Assert.AreEqual(-0.05, quoteData.OptionChain[0].Theta);
            Assert.IsNotNull(quoteData.OptionChain[0].FetchTimestamp);
        }

        [TestMethod]
        public void QuoteData_GetOptionsByStrikeAndExpiration_ReturnsCorrectOptions()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var expirationDate = DateTime.Now.AddDays(30);

            var option1 = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 150.0,
                ExpirationDate = expirationDate,
                OptionType = "CALL",
                ImpliedVolatility = 0.25,
                Delta = 0.6,
                Theta = -0.05
            };

            var option2 = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 150.0,
                ExpirationDate = expirationDate,
                OptionType = "PUT",
                ImpliedVolatility = 0.26,
                Delta = -0.4,
                Theta = -0.04
            };

            var option3 = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 155.0,
                ExpirationDate = expirationDate,
                OptionType = "CALL",
                ImpliedVolatility = 0.24,
                Delta = 0.5,
                Theta = -0.06
            };

            quoteData.OptionChain.AddRange(new[] { option1, option2, option3 });

            // Act
            var result = quoteData.GetOptionsByStrikeAndExpiration(150.0, expirationDate);

            // Assert
            Assert.AreEqual(2, result.Count);
            Assert.IsTrue(result.All(o => o.StrikePrice == 150.0));
            Assert.IsTrue(result.All(o => o.ExpirationDate.Date == expirationDate.Date));
        }

        [TestMethod]
        public void QuoteData_GetAvailableStrikes_ReturnsCorrectStrikes()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var expirationDate = DateTime.Now.AddDays(30);

            var options = new[]
            {
                new OptionData { StrikePrice = 145.0, ExpirationDate = expirationDate, OptionType = "CALL" },
                new OptionData { StrikePrice = 150.0, ExpirationDate = expirationDate, OptionType = "CALL" },
                new OptionData { StrikePrice = 150.0, ExpirationDate = expirationDate, OptionType = "PUT" },
                new OptionData { StrikePrice = 155.0, ExpirationDate = expirationDate, OptionType = "CALL" },
                new OptionData { StrikePrice = 160.0, ExpirationDate = DateTime.Now.AddDays(60), OptionType = "CALL" }
            };

            quoteData.OptionChain.AddRange(options);

            // Act
            var result = quoteData.GetAvailableStrikes(expirationDate);

            // Assert
            Assert.AreEqual(3, result.Count);
            Assert.AreEqual(145.0, result[0]);
            Assert.AreEqual(150.0, result[1]);
            Assert.AreEqual(155.0, result[2]);
        }

        [TestMethod]
        public void QuoteData_GetAvailableExpirations_ReturnsCorrectExpirations()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var exp1 = DateTime.Now.AddDays(30);
            var exp2 = DateTime.Now.AddDays(60);
            var exp3 = DateTime.Now.AddDays(90);

            var options = new[]
            {
                new OptionData { StrikePrice = 150.0, ExpirationDate = exp1, OptionType = "CALL" },
                new OptionData { StrikePrice = 150.0, ExpirationDate = exp2, OptionType = "CALL" },
                new OptionData { StrikePrice = 150.0, ExpirationDate = exp1, OptionType = "PUT" },
                new OptionData { StrikePrice = 155.0, ExpirationDate = exp3, OptionType = "CALL" }
            };

            quoteData.OptionChain.AddRange(options);

            // Act
            var result = quoteData.GetAvailableExpirations();

            // Assert
            Assert.AreEqual(3, result.Count);
            Assert.AreEqual(exp1.Date, result[0].Date);
            Assert.AreEqual(exp2.Date, result[1].Date);
            Assert.AreEqual(exp3.Date, result[2].Date);
        }

        [TestMethod]
        public void QuoteData_ClearChartData_ClearsOptionChain()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var optionData = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 150.0,
                ImpliedVolatility = 0.25,
                Delta = 0.6,
                Theta = -0.05
            };

            quoteData.OptionChain.Add(optionData);
            Assert.AreEqual(1, quoteData.OptionChain.Count);

            // Act
            quoteData.ClearChartData();

            // Assert
            Assert.AreEqual(0, quoteData.OptionChain.Count);
        }

        [TestMethod]
        public void QuoteData_Dispose_ClearsOptionChain()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var optionData = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 150.0,
                ImpliedVolatility = 0.25,
                Delta = 0.6,
                Theta = -0.05
            };

            quoteData.OptionChain.Add(optionData);
            Assert.AreEqual(1, quoteData.OptionChain.Count);

            // Act
            quoteData.Dispose();

            // Assert
            Assert.AreEqual(0, quoteData.OptionChain.Count);
        }

        [TestMethod]
        public void QuoteData_GetOptionsByStrikeAndExpiration_EmptyChain_ReturnsEmptyList()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var expirationDate = DateTime.Now.AddDays(30);

            // Act
            var result = quoteData.GetOptionsByStrikeAndExpiration(150.0, expirationDate);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        [TestMethod]
        public void QuoteData_GetAvailableStrikes_EmptyChain_ReturnsEmptyList()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var expirationDate = DateTime.Now.AddDays(30);

            // Act
            var result = quoteData.GetAvailableStrikes(expirationDate);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        [TestMethod]
        public void QuoteData_GetAvailableExpirations_EmptyChain_ReturnsEmptyList()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };

            // Act
            var result = quoteData.GetAvailableExpirations();

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Count);
        }

        [TestMethod]
        public void OptionData_FetchTimestamp_CanBeSet()
        {
            // Arrange
            var optionData = new OptionData
            {
                UnderlyingSymbol = "AAPL",
                StrikePrice = 150.0,
                ExpirationDate = DateTime.Now.AddDays(30),
                OptionType = "CALL"
            };
            var fetchTime = DateTime.Now;

            // Act
            optionData.FetchTimestamp = fetchTime;

            // Assert
            Assert.AreEqual(fetchTime, optionData.FetchTimestamp);
        }

        [TestMethod]
        public void QuoteData_IsOptionDataFresh_WorksCorrectly()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var fetchTime = DateTime.Now.AddMinutes(-10);
            var cacheWindow = TimeSpan.FromMinutes(15);

            quoteData.OptionDataFetchTimestamp = fetchTime;
            quoteData.OptionDataCacheWindow = cacheWindow;

            // Act
            var isFreshNow = quoteData.IsOptionDataFresh(DateTime.Now);
            var isFreshFuture = quoteData.IsOptionDataFresh(DateTime.Now.AddMinutes(10)); // 20 minutes after fetch = expired

            // Assert
            Assert.IsTrue(isFreshNow, "Data should be fresh within cache window");
            Assert.IsFalse(isFreshFuture, "Data should be expired outside cache window");
        }

        [TestMethod]
        public void QuoteData_IsOptionDataFresh_NoTimestamp_ReturnsFalse()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            quoteData.OptionDataCacheWindow = TimeSpan.FromMinutes(15);
            // OptionDataFetchTimestamp not set

            // Act
            var isFresh = quoteData.IsOptionDataFresh(DateTime.Now);

            // Assert
            Assert.IsFalse(isFresh, "Should return false when no fetch timestamp is set");
        }

        [TestMethod]
        public void QuoteData_IsOptionDataFresh_NoCacheWindow_ReturnsFalse()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            quoteData.OptionDataFetchTimestamp = DateTime.Now;
            // OptionDataCacheWindow not set

            // Act
            var isFresh = quoteData.IsOptionDataFresh(DateTime.Now);

            // Assert
            Assert.IsFalse(isFresh, "Should return false when no cache window is set");
        }
    }
}