using System;
using System.ComponentModel;

namespace Quantra.Models
{
    /// <summary>
    /// Represents an analyst rating for a stock
    /// </summary>
    public class AnalystRating : INotifyPropertyChanged
    {
        /// <summary>
        /// Unique identifier for the rating
        /// </summary>
        public int Id { get; set; }

        private string symbol;
        /// <summary>
        /// Stock symbol
        /// </summary>
        public string Symbol
        {
            get => symbol;
            set
            {
                symbol = value;
                OnPropertyChanged(nameof(Symbol));
            }
        }

        private string analystName;
        /// <summary>
        /// Name of the analyst or firm
        /// </summary>
        public string AnalystName
        {
            get => analystName;
            set
            {
                analystName = value;
                OnPropertyChanged(nameof(AnalystName));
            }
        }

        private string rating;
        /// <summary>
        /// Current rating (e.g., Buy, Hold, Sell, Overweight, etc.)
        /// </summary>
        public string Rating
        {
            get => rating;
            set
            {
                rating = value;
                OnPropertyChanged(nameof(Rating));
            }
        }

        private string previousRating;
        /// <summary>
        /// Previous rating (if available)
        /// </summary>
        public string PreviousRating
        {
            get => previousRating;
            set
            {
                previousRating = value;
                OnPropertyChanged(nameof(PreviousRating));
            }
        }

        private double priceTarget;
        /// <summary>
        /// Price target set by the analyst
        /// </summary>
        public double PriceTarget
        {
            get => priceTarget;
            set
            {
                priceTarget = value;
                OnPropertyChanged(nameof(PriceTarget));
            }
        }

        private double previousPriceTarget;
        /// <summary>
        /// Previous price target (if available)
        /// </summary>
        public double PreviousPriceTarget
        {
            get => previousPriceTarget;
            set
            {
                previousPriceTarget = value;
                OnPropertyChanged(nameof(PreviousPriceTarget));
            }
        }

        private DateTime ratingDate;
        /// <summary>
        /// Date when the rating was issued
        /// </summary>
        public DateTime RatingDate
        {
            get => ratingDate;
            set
            {
                ratingDate = value;
                OnPropertyChanged(nameof(RatingDate));
            }
        }

        private RatingChangeType changeType;
        /// <summary>
        /// Type of rating change
        /// </summary>
        public RatingChangeType ChangeType
        {
            get => changeType;
            set
            {
                changeType = value;
                OnPropertyChanged(nameof(ChangeType));
            }
        }

        /// <summary>
        /// Calculated sentiment score for this rating (-1.0 to 1.0)
        /// Where -1.0 is most negative (strong sell), 0.0 is neutral, and 1.0 is most positive (strong buy)
        /// </summary>
        public double SentimentScore
        {
            get
            {
                return GetSentimentScoreFromRating(Rating);
            }
        }

        /// <summary>
        /// Calculates a sentiment score based on a standardized rating string
        /// </summary>
        public static double GetSentimentScoreFromRating(string rating)
        {
            if (string.IsNullOrEmpty(rating))
                return 0.0;

            rating = rating.ToLower().Trim();

            // Strong positive ratings
            if (rating == "strong buy" || rating == "buy" || rating == "overweight" || 
                rating == "outperform" || rating == "add")
                return 0.8;

            // Moderate positive ratings
            if (rating == "accumulate" || rating == "moderate buy")
                return 0.5;

            // Neutral/hold ratings
            if (rating == "hold" || rating == "neutral" || rating == "market perform" || 
                rating == "in-line" || rating == "sector perform")
                return 0.0;

            // Moderate negative ratings
            if (rating == "moderate sell" || rating == "underperform" || 
                rating == "underweight" || rating == "reduce")
                return -0.5;

            // Strong negative ratings
            if (rating == "strong sell" || rating == "sell")
                return -0.8;

            // Default to neutral if rating isn't recognized
            return 0.0;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    /// <summary>
    /// Represents the type of change in an analyst rating
    /// </summary>
    public enum RatingChangeType
    {
        /// <summary>
        /// Initial coverage with a rating
        /// </summary>
        Initiation,

        /// <summary>
        /// Upgrade from a lower rating
        /// </summary>
        Upgrade,

        /// <summary>
        /// Downgrade from a higher rating
        /// </summary>
        Downgrade,

        /// <summary>
        /// Same rating as before
        /// </summary>
        Reiteration,

        /// <summary>
        /// Price target change only
        /// </summary>
        PriceTargetChange,

        /// <summary>
        /// End of coverage
        /// </summary>
        Coverage_Drop
    }
}