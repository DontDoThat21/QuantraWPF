using System;
using System.ComponentModel;

namespace Quantra.Models
{
    /// <summary>
    /// Represents statistics about an analyst firm's historical accuracy
    /// </summary>
    public class AnalystFirmStatistics : INotifyPropertyChanged
    {
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

        private double accuracyRating;
        /// <summary>
        /// Overall accuracy rating (0.0 to 1.0)
        /// </summary>
        public double AccuracyRating
        {
            get => accuracyRating;
            set
            {
                accuracyRating = value;
                OnPropertyChanged(nameof(AccuracyRating));
            }
        }

        private double avgPriceTargetAccuracy;
        /// <summary>
        /// Average accuracy of price targets
        /// </summary>
        public double AvgPriceTargetAccuracy
        {
            get => avgPriceTargetAccuracy;
            set
            {
                avgPriceTargetAccuracy = value;
                OnPropertyChanged(nameof(AvgPriceTargetAccuracy));
            }
        }

        private int totalRatings;
        /// <summary>
        /// Total number of ratings issued
        /// </summary>
        public int TotalRatings
        {
            get => totalRatings;
            set
            {
                totalRatings = value;
                OnPropertyChanged(nameof(TotalRatings));
            }
        }

        private double upgradeSuccessRate;
        /// <summary>
        /// Success rate of upgrade recommendations
        /// </summary>
        public double UpgradeSuccessRate
        {
            get => upgradeSuccessRate;
            set
            {
                upgradeSuccessRate = value;
                OnPropertyChanged(nameof(UpgradeSuccessRate));
            }
        }

        private double downgradeSuccessRate;
        /// <summary>
        /// Success rate of downgrade recommendations
        /// </summary>
        public double DowngradeSuccessRate
        {
            get => downgradeSuccessRate;
            set
            {
                downgradeSuccessRate = value;
                OnPropertyChanged(nameof(DowngradeSuccessRate));
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}