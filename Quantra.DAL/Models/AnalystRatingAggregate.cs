using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Represents aggregated analyst ratings information for a stock
    /// </summary>
    public class AnalystRatingAggregate : INotifyPropertyChanged
    {
        /// <summary>
        /// Stock symbol
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// Unique identifier for database storage
        /// </summary>
        public int Id { get; set; }

        private List<AnalystRating> ratings = new List<AnalystRating>();
        /// <summary>
        /// List of individual analyst ratings
        /// </summary>
        public List<AnalystRating> Ratings
        {
            get => ratings;
            set
            {
                ratings = value;
                RecalculateAggregates();
                OnPropertyChanged(nameof(Ratings));
            }
        }

        private double consensusScore;
        /// <summary>
        /// Consensus score from all ratings (-1.0 to 1.0)
        /// </summary>
        public double ConsensusScore
        {
            get => consensusScore;
            internal set
            {
                consensusScore = value;
                OnPropertyChanged(nameof(ConsensusScore));
            }
        }

        private string consensusRating;
        /// <summary>
        /// Text representation of the consensus (Buy/Hold/Sell)
        /// </summary>
        public string ConsensusRating
        {
            get => consensusRating;
            internal set
            {
                consensusRating = value;
                OnPropertyChanged(nameof(ConsensusRating));
            }
        }

        private double averagePriceTarget;
        /// <summary>
        /// Average of all analyst price targets
        /// </summary>
        public double AveragePriceTarget
        {
            get => averagePriceTarget;
            internal set
            {
                averagePriceTarget = value;
                OnPropertyChanged(nameof(AveragePriceTarget));
            }
        }

        private double highestPriceTarget;
        /// <summary>
        /// Highest price target among analysts
        /// </summary>
        public double HighestPriceTarget
        {
            get => highestPriceTarget;
            private set
            {
                highestPriceTarget = value;
                OnPropertyChanged(nameof(HighestPriceTarget));
            }
        }

        private double lowestPriceTarget;
        /// <summary>
        /// Lowest price target among analysts
        /// </summary>
        public double LowestPriceTarget
        {
            get => lowestPriceTarget;
            private set
            {
                lowestPriceTarget = value;
                OnPropertyChanged(nameof(LowestPriceTarget));
            }
        }

        private int buyCount;
        /// <summary>
        /// Number of Buy ratings
        /// </summary>
        public int BuyCount
        {
            get => buyCount;
            internal set
            {
                buyCount = value;
                OnPropertyChanged(nameof(BuyCount));
            }
        }

        private int holdCount;
        /// <summary>
        /// Number of Hold ratings
        /// </summary>
        public int HoldCount
        {
            get => holdCount;
            internal set
            {
                holdCount = value;
                OnPropertyChanged(nameof(HoldCount));
            }
        }

        private int sellCount;
        /// <summary>
        /// Number of Sell ratings
        /// </summary>
        public int SellCount
        {
            get => sellCount;
            internal set
            {
                sellCount = value;
                OnPropertyChanged(nameof(SellCount));
            }
        }

        private DateTime lastUpdated;
        /// <summary>
        /// Last time the aggregate was updated
        /// </summary>
        public DateTime LastUpdated
        {
            get => lastUpdated;
            set
            {
                lastUpdated = value;
                OnPropertyChanged(nameof(LastUpdated));
            }
        }

        private RatingChangeType? lastChange;
        /// <summary>
        /// Most recent change type in ratings
        /// </summary>
        public RatingChangeType? LastChange
        {
            get => lastChange;
            set
            {
                lastChange = value;
                OnPropertyChanged(nameof(LastChange));
            }
        }
        
        private int upgradeCount;
        /// <summary>
        /// Number of upgrades in the current dataset
        /// </summary>
        public int UpgradeCount
        {
            get => upgradeCount;
            internal set
            {
                upgradeCount = value;
                OnPropertyChanged(nameof(UpgradeCount));
            }
        }
        
        private int downgradeCount;
        /// <summary>
        /// Number of downgrades in the current dataset
        /// </summary>
        public int DowngradeCount
        {
            get => downgradeCount;
            internal set
            {
                downgradeCount = value;
                OnPropertyChanged(nameof(DowngradeCount));
            }
        }
        
        private double ratingsStrengthIndex;
        /// <summary>
        /// Index showing the strength of recent ratings (-1.0 to 1.0) where positive means more upgrades
        /// </summary>
        public double RatingsStrengthIndex
        {
            get => ratingsStrengthIndex;
            internal set
            {
                ratingsStrengthIndex = value;
                OnPropertyChanged(nameof(RatingsStrengthIndex));
            }
        }
        
        private string consensusTrend;
        /// <summary>
        /// Trend description of the consensus (Improving, Stable, Deteriorating)
        /// </summary>
        public string ConsensusTrend
        {
            get => consensusTrend;
            set
            {
                consensusTrend = value;
                OnPropertyChanged(nameof(ConsensusTrend));
            }
        }
        
        private double previousConsensusScore;
        /// <summary>
        /// Previous consensus score for tracking changes
        /// </summary>
        public double PreviousConsensusScore
        {
            get => previousConsensusScore;
            set
            {
                previousConsensusScore = value;
                OnPropertyChanged(nameof(PreviousConsensusScore));
            }
        }
        
        private string previousConsensusRating;
        /// <summary>
        /// Previous consensus rating for tracking changes
        /// </summary>
        public string PreviousConsensusRating
        {
            get => previousConsensusRating;
            set
            {
                previousConsensusRating = value;
                OnPropertyChanged(nameof(PreviousConsensusRating));
            }
        }

        public AnalystRatingAggregate()
        {
            LastUpdated = DateTime.Now;
        }

        /// <summary>
        /// Recalculates aggregate values based on current ratings
        /// </summary>
        public void RecalculateAggregates()
        {
            if (ratings == null || ratings.Count == 0)
            {
                ConsensusScore = 0;
                ConsensusRating = "No Ratings";
                AveragePriceTarget = 0;
                HighestPriceTarget = 0;
                LowestPriceTarget = 0;
                BuyCount = 0;
                HoldCount = 0;
                SellCount = 0;
                UpgradeCount = 0;
                DowngradeCount = 0;
                RatingsStrengthIndex = 0;
                ConsensusTrend = "Neutral";
                return;
            }

            // Calculate consensus score
            ConsensusScore = ratings.Average(r => r.SentimentScore);

            // Count ratings by category
            BuyCount = ratings.Count(r => r.SentimentScore > 0.3);
            HoldCount = ratings.Count(r => r.SentimentScore >= -0.3 && r.SentimentScore <= 0.3);
            SellCount = ratings.Count(r => r.SentimentScore < -0.3);

            // Calculate upgrades and downgrades
            UpgradeCount = ratings.Count(r => r.ChangeType == RatingChangeType.Upgrade);
            DowngradeCount = ratings.Count(r => r.ChangeType == RatingChangeType.Downgrade);

            // Calculate ratings strength index
            if (UpgradeCount + DowngradeCount > 0)
            {
                RatingsStrengthIndex = ((double)UpgradeCount - DowngradeCount) / (UpgradeCount + DowngradeCount);
            }
            else
            {
                RatingsStrengthIndex = 0;
            }

            // Set consensus trend based on upgrades vs downgrades and rating movement
            if (PreviousConsensusScore != 0)
            {
                if (ConsensusScore > PreviousConsensusScore + 0.1)
                    ConsensusTrend = "Improving";
                else if (ConsensusScore < PreviousConsensusScore - 0.1)
                    ConsensusTrend = "Deteriorating";
                else
                    ConsensusTrend = "Stable";
            }
            else if (UpgradeCount > DowngradeCount)
                ConsensusTrend = "Improving";
            else if (DowngradeCount > UpgradeCount)
                ConsensusTrend = "Deteriorating";
            else
                ConsensusTrend = "Stable";

            // Set consensus rating based on score
            if (ConsensusScore > 0.3)
                ConsensusRating = "Buy";
            else if (ConsensusScore < -0.3)
                ConsensusRating = "Sell";
            else
                ConsensusRating = "Hold";

            // Calculate price target metrics
            var validPriceTargets = ratings.Where(r => r.PriceTarget > 0).Select(r => r.PriceTarget).ToList();
            if (validPriceTargets.Any())
            {
                AveragePriceTarget = validPriceTargets.Average();
                HighestPriceTarget = validPriceTargets.Max();
                LowestPriceTarget = validPriceTargets.Min();
            }
            else
            {
                AveragePriceTarget = 0;
                HighestPriceTarget = 0;
                LowestPriceTarget = 0;
            }

            // Find most recent change
            if (ratings.Any())
            {
                var mostRecentRating = ratings.OrderByDescending(r => r.RatingDate).First();
                LastChange = mostRecentRating.ChangeType;
            }
            else
            {
                LastChange = null;
            }

            LastUpdated = DateTime.Now;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}