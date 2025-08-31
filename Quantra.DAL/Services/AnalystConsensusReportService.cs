using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for generating analyst consensus reports and trend analyses
    /// </summary>
    public class AnalystConsensusReportService : IAnalystConsensusReportService
    {
        private readonly IAnalystRatingService _analystRatingService;
        
        public AnalystConsensusReportService(IAnalystRatingService analystRatingService)
        {
            _analystRatingService = analystRatingService ?? throw new ArgumentNullException(nameof(analystRatingService));
        }
        
        /// <summary>
        /// Generates a detailed consensus report for a symbol
        /// </summary>
        public async Task<ConsensusReport> GenerateConsensusReportAsync(string symbol, int historyDays = 90)
        {
            try
            {
                // Get current consensus
                var currentConsensus = await _analystRatingService.GetAggregatedRatingsAsync(symbol);
                
                // Get historical data
                var historyData = await _analystRatingService.GetConsensusHistoryAsync(symbol, historyDays);
                
                // Get recent ratings
                var recentRatings = await _analystRatingService.GetRecentRatingsAsync(symbol, 50);
                
                var report = new ConsensusReport
                {
                    Symbol = symbol,
                    GeneratedDate = DateTime.Now,
                    CurrentConsensus = currentConsensus,
                    ReportPeriodDays = historyDays
                };
                
                if (historyData.Any())
                {
                    // Calculate consensus change stats
                    var oldestData = historyData.OrderBy(h => h.LastUpdated).FirstOrDefault();
                    
                    if (oldestData != null)
                    {
                        report.ConsensusChangeStats = new ConsensusChangeStats
                        {
                            StartDate = oldestData.LastUpdated,
                            EndDate = currentConsensus.LastUpdated,
                            StartConsensusRating = oldestData.ConsensusRating,
                            EndConsensusRating = currentConsensus.ConsensusRating,
                            StartConsensusScore = oldestData.ConsensusScore,
                            EndConsensusScore = currentConsensus.ConsensusScore,
                            ScoreChange = currentConsensus.ConsensusScore - oldestData.ConsensusScore,
                            BuyCountChange = currentConsensus.BuyCount - oldestData.BuyCount,
                            HoldCountChange = currentConsensus.HoldCount - oldestData.HoldCount,
                            SellCountChange = currentConsensus.SellCount - oldestData.SellCount
                        };
                    }
                    
                    // Calculate rating distribution trends
                    report.RatingDistributionTrend = CalculateRatingDistributionTrend(historyData);
                }
                
                // Calculate analyst influence scores
                report.AnalystInfluenceScores = CalculateAnalystInfluence(recentRatings);
                
                // Identify significant changes
                report.SignificantChanges = IdentifySignificantChanges(recentRatings, historyDays);
                
                return report;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to generate consensus report for {symbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Calculates trends in rating distribution over time
        /// </summary>
        private RatingDistributionTrend CalculateRatingDistributionTrend(List<AnalystRatingAggregate> historyData)
        {
            if (historyData == null || !historyData.Any())
                return null;
                
            // Group data by week for trend analysis
            var weeklyData = historyData.GroupBy(h => 
                new { h.LastUpdated.Year, Week = GetWeekNumber(h.LastUpdated) })
                .Select(g => new
                {
                    YearWeek = $"{g.Key.Year}-W{g.Key.Week}",
                    MidDate = g.OrderBy(x => x.LastUpdated).Skip(g.Count() / 2).First().LastUpdated,
                    AvgBuyCount = g.Average(x => x.BuyCount),
                    AvgHoldCount = g.Average(x => x.HoldCount),
                    AvgSellCount = g.Average(x => x.SellCount),
                    AvgConsensusScore = g.Average(x => x.ConsensusScore)
                })
                .OrderBy(w => w.MidDate)
                .ToList();
                
            var trend = new RatingDistributionTrend
            {
                WeeklyLabels = weeklyData.Select(w => w.YearWeek).ToList(),
                BuyCountTrend = weeklyData.Select(w => w.AvgBuyCount).ToList(),
                HoldCountTrend = weeklyData.Select(w => w.AvgHoldCount).ToList(),
                SellCountTrend = weeklyData.Select(w => w.AvgSellCount).ToList(),
                ConsensusScoreTrend = weeklyData.Select(w => w.AvgConsensusScore).ToList()
            };
            
            return trend;
        }
        
        /// <summary>
        /// Gets the ISO week number for a date
        /// </summary>
        private int GetWeekNumber(DateTime date)
        {
            var day = (int)date.DayOfWeek;
            return (date.DayOfYear - day + 10) / 7;
        }
        
        /// <summary>
        /// Calculates influence scores for analysts based on their ratings
        /// </summary>
        private List<AnalystInfluenceScore> CalculateAnalystInfluence(List<AnalystRating> ratings)
        {
            if (ratings == null || ratings.Count == 0)
                return new List<AnalystInfluenceScore>();
                
            // Group ratings by analyst
            var analystGroups = ratings
                .GroupBy(r => r.AnalystName)
                .Select(g => new AnalystInfluenceScore
                {
                    AnalystName = g.Key,
                    RatingCount = g.Count(),
                    LastRating = g.OrderByDescending(r => r.RatingDate).First().Rating,
                    LastRatingDate = g.OrderByDescending(r => r.RatingDate).First().RatingDate,
                    AverageScore = g.Average(r => r.SentimentScore),
                    // A simple influence score based on recency and activity
                    InfluenceScore = g.Count() * (1.0 + g.Where(r => r.RatingDate >= DateTime.Now.AddDays(-30)).Count() * 0.5)
                })
                .OrderByDescending(a => a.InfluenceScore)
                .ToList();
                
            return analystGroups;
        }
        
        /// <summary>
        /// Identifies significant changes in ratings within the specified timeframe
        /// </summary>
        private List<SignificantChange> IdentifySignificantChanges(List<AnalystRating> ratings, int days)
        {
            if (ratings == null || ratings.Count == 0)
                return new List<SignificantChange>();
                
            var dateThreshold = DateTime.Now.AddDays(-days);
            var recentRatings = ratings.Where(r => r.RatingDate >= dateThreshold).ToList();
            
            var changes = new List<SignificantChange>();
            
            // Find significant upgrades
            var significantUpgrades = recentRatings
                .Where(r => r.ChangeType == RatingChangeType.Upgrade && r.SentimentScore > 0.5)
                .OrderByDescending(r => r.RatingDate)
                .Take(5)
                .Select(r => new SignificantChange
                {
                    Date = r.RatingDate,
                    AnalystName = r.AnalystName,
                    ChangeType = "Significant Upgrade",
                    FromRating = r.PreviousRating,
                    ToRating = r.Rating,
                    PriceTarget = r.PriceTarget,
                    PreviousPriceTarget = r.PreviousPriceTarget,
                    SentimentShift = r.SentimentScore - AnalystRating.GetSentimentScoreFromRating(r.PreviousRating)
                })
                .ToList();
                
            changes.AddRange(significantUpgrades);
            
            // Find significant downgrades
            var significantDowngrades = recentRatings
                .Where(r => r.ChangeType == RatingChangeType.Downgrade && r.SentimentScore < -0.3)
                .OrderByDescending(r => r.RatingDate)
                .Take(5)
                .Select(r => new SignificantChange
                {
                    Date = r.RatingDate,
                    AnalystName = r.AnalystName,
                    ChangeType = "Significant Downgrade",
                    FromRating = r.PreviousRating,
                    ToRating = r.Rating,
                    PriceTarget = r.PriceTarget,
                    PreviousPriceTarget = r.PreviousPriceTarget,
                    SentimentShift = r.SentimentScore - AnalystRating.GetSentimentScoreFromRating(r.PreviousRating)
                })
                .ToList();
                
            changes.AddRange(significantDowngrades);
            
            // Find significant price target changes
            var significantPriceTargetChanges = recentRatings
                .Where(r => r.PriceTarget > 0 && r.PreviousPriceTarget > 0 && 
                      Math.Abs(r.PriceTarget - r.PreviousPriceTarget) / r.PreviousPriceTarget > 0.15) // >15% change
                .OrderByDescending(r => r.RatingDate)
                .Take(5)
                .Select(r => new SignificantChange
                {
                    Date = r.RatingDate,
                    AnalystName = r.AnalystName,
                    ChangeType = r.PriceTarget > r.PreviousPriceTarget ? "Significant PT Increase" : "Significant PT Decrease",
                    FromRating = r.Rating,
                    ToRating = r.Rating,
                    PriceTarget = r.PriceTarget,
                    PreviousPriceTarget = r.PreviousPriceTarget,
                    SentimentShift = 0,
                    PriceTargetChangePercent = (r.PriceTarget - r.PreviousPriceTarget) / r.PreviousPriceTarget * 100
                })
                .ToList();
                
            changes.AddRange(significantPriceTargetChanges);
            
            return changes.OrderByDescending(c => c.Date).ToList();
        }
    }
    
    #region Report Models
    
    /// <summary>
    /// Comprehensive report on analyst consensus
    /// </summary>
    public class ConsensusReport
    {
        public string Symbol { get; set; }
        public DateTime GeneratedDate { get; set; }
        public int ReportPeriodDays { get; set; }
        public AnalystRatingAggregate CurrentConsensus { get; set; }
        public ConsensusChangeStats ConsensusChangeStats { get; set; }
        public RatingDistributionTrend RatingDistributionTrend { get; set; }
        public List<AnalystInfluenceScore> AnalystInfluenceScores { get; set; } = new List<AnalystInfluenceScore>();
        public List<SignificantChange> SignificantChanges { get; set; } = new List<SignificantChange>();
        
        /// <summary>
        /// Gets a summary of the consensus report
        /// </summary>
        public string GetSummary()
        {
            if (CurrentConsensus == null)
                return "No consensus data available";
                
            string changeDescription = "unchanged";
            if (ConsensusChangeStats != null)
            {
                if (ConsensusChangeStats.ScoreChange > 0.1)
                    changeDescription = "improved";
                else if (ConsensusChangeStats.ScoreChange < -0.1)
                    changeDescription = "deteriorated";
            }
            
            string upgradesToReport = SignificantChanges
                .Where(c => c.ChangeType == "Significant Upgrade")
                .Count().ToString();
                
            string downgradesToReport = SignificantChanges
                .Where(c => c.ChangeType == "Significant Downgrade")
                .Count().ToString();
            
            return $"Analyst consensus for {Symbol} is {CurrentConsensus.ConsensusRating} (Score: {CurrentConsensus.ConsensusScore:F2}). " +
                   $"Consensus has {changeDescription} over the last {ReportPeriodDays} days. " +
                   $"Current breakdown: {CurrentConsensus.BuyCount} Buy, {CurrentConsensus.HoldCount} Hold, {CurrentConsensus.SellCount} Sell. " +
                   $"Average price target: ${CurrentConsensus.AveragePriceTarget:F2}. " +
                   $"Significant changes: {upgradesToReport} upgrades, {downgradesToReport} downgrades.";
        }
    }
    
    /// <summary>
    /// Statistics about changes in consensus over time
    /// </summary>
    public class ConsensusChangeStats
    {
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string StartConsensusRating { get; set; }
        public string EndConsensusRating { get; set; }
        public double StartConsensusScore { get; set; }
        public double EndConsensusScore { get; set; }
        public double ScoreChange { get; set; }
        public int BuyCountChange { get; set; }
        public int HoldCountChange { get; set; }
        public int SellCountChange { get; set; }
    }
    
    /// <summary>
    /// Trends in rating distribution over time
    /// </summary>
    public class RatingDistributionTrend
    {
        public List<string> WeeklyLabels { get; set; } = new List<string>();
        public List<double> BuyCountTrend { get; set; } = new List<double>();
        public List<double> HoldCountTrend { get; set; } = new List<double>();
        public List<double> SellCountTrend { get; set; } = new List<double>();
        public List<double> ConsensusScoreTrend { get; set; } = new List<double>();
    }
    
    /// <summary>
    /// Measures an analyst's influence based on activity and recency
    /// </summary>
    public class AnalystInfluenceScore
    {
        public string AnalystName { get; set; }
        public int RatingCount { get; set; }
        public string LastRating { get; set; }
        public DateTime LastRatingDate { get; set; }
        public double AverageScore { get; set; }
        public double InfluenceScore { get; set; }
    }
    
    /// <summary>
    /// Details about a significant rating change
    /// </summary>
    public class SignificantChange
    {
        public DateTime Date { get; set; }
        public string AnalystName { get; set; }
        public string ChangeType { get; set; }
        public string FromRating { get; set; }
        public string ToRating { get; set; }
        public double PriceTarget { get; set; }
        public double PreviousPriceTarget { get; set; }
        public double SentimentShift { get; set; }
        public double PriceTargetChangePercent { get; set; }
    }
    
    #endregion
}