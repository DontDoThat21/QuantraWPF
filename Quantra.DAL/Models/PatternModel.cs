using System;
using System.ComponentModel;

namespace Quantra.Models
{
    public class PatternModel : INotifyPropertyChanged
    {
        private string _symbol;
        public string Symbol
        {
            get => _symbol;
            set
            {
                _symbol = value;
                OnPropertyChanged(nameof(Symbol));
            }
        }

        private string _patternName;
        public string PatternName
        {
            get => _patternName;
            set
            {
                _patternName = value;
                OnPropertyChanged(nameof(PatternName));
            }
        }

        private double _reliability;
        public double Reliability
        {
            get => _reliability;
            set
            {
                _reliability = value;
                OnPropertyChanged(nameof(Reliability));
            }
        }

        private string _description;
        public string Description
        {
            get => _description;
            set
            {
                _description = value;
                OnPropertyChanged(nameof(Description));
            }
        }

        private string _predictedOutcome;
        public string PredictedOutcome
        {
            get => _predictedOutcome;
            set
            {
                _predictedOutcome = value;
                OnPropertyChanged(nameof(PredictedOutcome));
            }
        }

        private DateTime _detectionDate;
        public DateTime DetectionDate
        {
            get => _detectionDate;
            set
            {
                _detectionDate = value;
                OnPropertyChanged(nameof(DetectionDate));
            }
        }

        private double _historicalAccuracy;
        public double HistoricalAccuracy
        {
            get => _historicalAccuracy;
            set
            {
                _historicalAccuracy = value;
                OnPropertyChanged(nameof(HistoricalAccuracy));
            }
        }

        // INotifyPropertyChanged Implementation
        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}