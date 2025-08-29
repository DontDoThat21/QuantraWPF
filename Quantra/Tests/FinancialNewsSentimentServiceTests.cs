using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.Models;
using Quantra.Services;

namespace Quantra.Tests
{
    [TestClass]
    public class FinancialNewsSentimentServiceTests
    {
        private UserSettings _testSettings;
        
        [TestInitialize]
        public void Initialize()
        {
            _testSettings = new UserSettings
            {
                MaxNewsArticlesPerSymbol = 5,
                NewsArticleRefreshIntervalMinutes = 5,
                EnableNewsSentimentAnalysis = true,
                EnabledNewsSources = new Dictionary<string, bool>
                {
                    { "bloomberg.com", true },
                    { "cnbc.com", true },
                    { "wsj.com", true },
                    { "reuters.com", true },
                    { "marketwatch.com", false },  // Disabled for testing
                    { "finance.yahoo.com", true },
                    { "ft.com", true }
                }
            };
        }
        
        [TestMethod]
        public void TestNewsSourceConfigInitialization()
        {
            // Arrange
            var service = new FinancialNewsSentimentService(_testSettings);
            
            // Act - this will invoke the private InitializeNewsSourceConfigs method
            var newsSourcesField = typeof(FinancialNewsSentimentService)
                .GetField("_newsSourceConfigs", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var newsSourceConfigs = newsSourcesField?.GetValue(service) as List<NewsSourceConfig>;
            
            // Assert
            Assert.IsNotNull(newsSourceConfigs);
            Assert.IsTrue(newsSourceConfigs.Count >= 6); // We should have at least 6 sources
            
            // MarketWatch should be disabled as per our test settings
            var marketWatchConfig = newsSourceConfigs.FirstOrDefault(c => c.Domain == "marketwatch.com");
            Assert.IsNotNull(marketWatchConfig);
            Assert.IsFalse(marketWatchConfig.IsEnabled);
            
            // Bloomberg should have a higher weight
            var bloombergConfig = newsSourceConfigs.FirstOrDefault(c => c.Domain == "bloomberg.com");
            Assert.IsNotNull(bloombergConfig);
            Assert.IsTrue(bloombergConfig.IsEnabled);
            Assert.IsTrue(bloombergConfig.Weight >= 1.0);
        }
        
        [TestMethod]
        [Ignore("Requires News API key to run")]
        public async Task TestFetchNewsArticles()
        {
            // Arrange
            var service = new FinancialNewsSentimentService(_testSettings);
            string testSymbol = "MSFT"; // Microsoft is a widely covered stock
            
            // Act
            var articles = await service.FetchNewsArticlesAsync(testSymbol, 5);
            
            // Assert
            Assert.IsNotNull(articles);
            Assert.IsTrue(articles.Count > 0, "Should find at least one news article for Microsoft");
            
            // Check that articles have the required fields populated
            foreach (var article in articles)
            {
                Assert.IsFalse(string.IsNullOrWhiteSpace(article.Title), "Article title should not be empty");
                Assert.IsFalse(string.IsNullOrWhiteSpace(article.GetCombinedContent()), "Article content should not be empty");
                Assert.IsFalse(string.IsNullOrWhiteSpace(article.SourceDomain), "Source domain should not be empty");
            }
        }
        
        [TestMethod]
        public void TestArticleSimilarityDetection()
        {
            // Arrange
            var article1 = new NewsArticle
            {
                Title = "AAPL Stock Price Surges After Earnings Report"
            };
            
            var article2 = new NewsArticle
            {
                Title = "Apple Stock Price Surges Following Earnings Report Release"
            };
            
            var article3 = new NewsArticle
            {
                Title = "Market Overview: Tech Sector Performance Analyzed"
            };
            
            // Act & Assert
            Assert.IsTrue(article1.IsSimilarTo(article2), "Articles with similar titles should be detected as similar");
            Assert.IsFalse(article1.IsSimilarTo(article3), "Articles with different topics should not be detected as similar");
        }
    }
}