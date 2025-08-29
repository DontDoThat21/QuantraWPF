using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.Services;
using Quantra.Controls.Components;

namespace Quantra.Controllers
{
    /// <summary>
    /// Controller to coordinate sector sentiment analysis and visualization
    /// </summary>
    public class SectorSentimentController
    {
        private readonly SectorSentimentAnalysisService _sectorSentimentService;
        private readonly SentimentPriceCorrelationAnalysis _sentimentCorrelationAnalysis;
        private SectorSentimentVisualizationView _visualizationView;
        
        public SectorSentimentController()
        {
            _sectorSentimentService = new SectorSentimentAnalysisService();
            _sentimentCorrelationAnalysis = new SentimentPriceCorrelationAnalysis();
        }
        
        /// <summary>
        /// Initialize the controller with a visualization view
        /// </summary>
        public void Initialize(SectorSentimentVisualizationView view)
        {
            _visualizationView = view;
            
            // Wire up events
            if (_visualizationView != null)
            {
                _visualizationView.SectorSelectionChanged += OnSectorSelected;
            }
            
            // Load initial data
            _ = LoadAllSectorsDataAsync();
        }
        
        /// <summary>
        /// Loads and displays data for all sectors
        /// </summary>
        public async Task LoadAllSectorsDataAsync()
        {
            try
            {
                // Get sentiment data for all sectors
                var sectorSentiments = await _sectorSentimentService.GetAllSectorsSentimentAsync();
                
                if (_visualizationView != null)
                {
                    // Update sector comparison chart
                    _visualizationView.UpdateSectorComparison(sectorSentiments);
                    
                    // Load details for the selected sector
                    string selectedSector = _visualizationView.SelectedSector;
                    if (!string.IsNullOrEmpty(selectedSector))
                    {
                        await LoadSectorDetailsAsync(selectedSector);
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error loading all sectors data", ex.ToString());
            }
        }
        
        /// <summary>
        /// Loads and displays detailed data for a specific sector
        /// </summary>
        public async Task LoadSectorDetailsAsync(string sector)
        {
            try
            {
                // Get detailed sentiment data for the sector
                var sentimentTrend = await _sectorSentimentService.GetSectorSentimentTrendAsync(sector, 30);
                
                // Get sector news analysis
                var (sentimentBySource, articles) = await _sectorSentimentService.GetSectorNewsAnalysisAsync(sector);
                
                // Generate trending topics from articles
                var trendingTopics = GenerateTrendingTopics(articles);
                
                if (_visualizationView != null)
                {
                    // Update all UI components
                    _visualizationView.UpdateSentimentTrend(sentimentTrend);
                    _visualizationView.UpdateSourceBreakdown(sentimentBySource);
                    _visualizationView.UpdateSectorNewsArticles(articles);
                    _visualizationView.UpdateTrendingTopics(trendingTopics);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error loading details for sector {sector}", ex.ToString());
            }
        }
        
        /// <summary>
        /// Analyzes and displays correlation between sector sentiment and performance
        /// </summary>
        public async Task LoadSectorCorrelationAsync(string sector)
        {
            try
            {
                var correlation = await _sentimentCorrelationAnalysis.AnalyzeSectorSentimentCorrelation(sector);
                
                // This data could be displayed in a separate view or dialog
                // For now, we'll just log some of the key findings
                DatabaseMonolith.Log("Info", 
                    $"Sector {sector} correlation analysis: " +
                    $"Overall Correlation: {correlation.OverallCorrelation:F2}, " +
                    $"Lead/Lag: {correlation.LeadLagRelationship:F1} days, " +
                    $"Sentiment Shifts: {correlation.SentimentShiftEvents.Count}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error analyzing sector correlation for {sector}", ex.ToString());
            }
        }
        
        /// <summary>
        /// Compares multiple sectors and displays their sentiment trends
        /// </summary>
        public async Task CompareSectorsAsync(List<string> sectors)
        {
            try
            {
                var trends = await _sectorSentimentService.CompareSectorSentimentTrendsAsync(sectors);
                
                // This would be displayed in a separate comparison view
                // For now, we'll just log the trends
                foreach (var sector in trends.Keys)
                {
                    double averageSentiment = trends[sector].Average(t => t.Sentiment);
                    DatabaseMonolith.Log("Info", $"Sector {sector} average sentiment: {averageSentiment:F2}");
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error comparing sectors", ex.ToString());
            }
        }
        
        #region Event Handlers
        
        private async void OnSectorSelected(object sender, string sector)
        {
            if (!string.IsNullOrEmpty(sector))
            {
                await LoadSectorDetailsAsync(sector);
            }
        }
        
        #endregion
        
        #region Helper Methods
        
        /// <summary>
        /// Generates trending topics from news articles
        /// </summary>
        private List<string> GenerateTrendingTopics(List<NewsArticle> articles)
        {
            var topics = new Dictionary<string, int>();
            
            // In a real implementation, this would use NLP techniques like topic modeling
            // For this example, we'll just use a simple keyword frequency approach
            
            // Define some potential topic keywords for each sector
            var topicKeywords = new Dictionary<string, List<string>>
            {
                ["Technology"] = new List<string> { "AI", "cloud", "software", "chip", "semiconductor", "innovation" },
                ["Financial"] = new List<string> { "rates", "banking", "fintech", "inflation", "economy", "debt" },
                ["Healthcare"] = new List<string> { "biotech", "pharma", "research", "clinical", "treatment", "medicare" },
                ["Energy"] = new List<string> { "oil", "renewable", "sustainability", "price", "green", "EV" }
            };
            
            // Get topics based on article content
            foreach (var article in articles)
            {
                if (string.IsNullOrEmpty(article.PrimarySector))
                    continue;
                
                // Check if we have keywords for this sector
                if (!topicKeywords.TryGetValue(article.PrimarySector, out var keywords))
                    continue;
                
                // Check for keyword mentions in the article
                string content = article.GetCombinedContent().ToLowerInvariant();
                foreach (var keyword in keywords)
                {
                    if (content.Contains(keyword.ToLowerInvariant()))
                    {
                        // Increment topic count
                        if (topics.ContainsKey(keyword))
                            topics[keyword]++;
                        else
                            topics[keyword] = 1;
                    }
                }
            }
            
            // Get top 10 trending topics
            return topics
                .OrderByDescending(t => t.Value)
                .Select(t => t.Key)
                .Take(10)
                .ToList();
        }
        
        #endregion
    }
}