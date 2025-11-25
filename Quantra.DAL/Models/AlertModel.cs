using System;
using System.ComponentModel;

namespace Quantra.Models
{
    public enum AlertCategory
    {
        Standard,
        Opportunity,
        Prediction,
        Global,
        TechnicalIndicator,
        Sentiment,
        Pattern,
        SystemHealth
    }

    public enum ComparisonOperator
    {
        Equal,
        NotEqual,
        GreaterThan,
        LessThan,
        GreaterThanOrEqual,
        LessThanOrEqual,
        CrossesAbove,
        CrossesBelow
    }

    public enum VisualIndicatorType
    {
        Toast,
        Banner,
        Popup,
        Flashcard
    }

    public class AlertModel : INotifyPropertyChanged
    {
        public int Id { get; set; }

        private string name;
        public string Name
        {
            get => name;
            set
            {
                name = value;
                OnPropertyChanged(nameof(Name));
            }
        }

        private string symbol;
        public string Symbol
        {
            get => symbol;
            set
            {
                symbol = value;
                OnPropertyChanged(nameof(Symbol));
            }
        }

        private string condition;
        public string Condition
        {
            get => condition;
            set
            {
                condition = value;
                OnPropertyChanged(nameof(Condition));
            }
        }

        private double triggerPrice;
        public double TriggerPrice
        {
            get => triggerPrice;
            set
            {
                triggerPrice = value;
                OnPropertyChanged(nameof(TriggerPrice));
            }
        }

        private string alertType;
        public string AlertType
        {
            get => alertType;
            set
            {
                alertType = value;
                OnPropertyChanged(nameof(AlertType));
            }
        }

        private bool isActive;
        public bool IsActive
        {
            get => isActive;
            set
            {
                isActive = value;
                OnPropertyChanged(nameof(IsActive));
            }
        }

        private bool isTriggered;
        public bool IsTriggered
        {
            get => isTriggered;
            set
            {
                isTriggered = value;
                OnPropertyChanged(nameof(IsTriggered));
            }
        }

        private DateTime createdDate;
        public DateTime CreatedDate
        {
            get => createdDate;
            set
            {
                createdDate = value;
                OnPropertyChanged(nameof(CreatedDate));
            }
        }

        private DateTime? triggeredDate;
        public DateTime? TriggeredDate
        {
            get => triggeredDate;
            set
            {
                triggeredDate = value;
                OnPropertyChanged(nameof(TriggeredDate));
            }
        }

        private string notes;
        public string Notes
        {
            get => notes;
            set
            {
                notes = value;
                OnPropertyChanged(nameof(Notes));
            }
        }

        private int priority;
        public int Priority
        {
            get => priority;
            set
            {
                priority = value;
                OnPropertyChanged(nameof(Priority));
            }
        }

        private AlertCategory category;
        public AlertCategory Category
        {
            get => category;
            set
            {
                category = value;
                OnPropertyChanged(nameof(Category));
            }
        }

        private string indicatorName;
        public string IndicatorName
        {
            get => indicatorName;
            set
            {
                indicatorName = value;
                OnPropertyChanged(nameof(IndicatorName));
            }
        }

        private ComparisonOperator comparisonOperator;
        public ComparisonOperator ComparisonOperator
        {
            get => comparisonOperator;
            set
            {
                comparisonOperator = value;
                OnPropertyChanged(nameof(ComparisonOperator));
            }
        }

        private double thresholdValue;
        public double ThresholdValue
        {
            get => thresholdValue;
            set
            {
                thresholdValue = value;
                OnPropertyChanged(nameof(ThresholdValue));
            }
        }

        private double currentIndicatorValue;
        public double CurrentIndicatorValue
        {
            get => currentIndicatorValue;
            set
            {
                currentIndicatorValue = value;
                OnPropertyChanged(nameof(CurrentIndicatorValue));
            }
        }

        // Sound and Visual Indicator properties
        private string soundFileName;
        public string SoundFileName
        {
            get => soundFileName;
            set
            {
                soundFileName = value;
                OnPropertyChanged(nameof(SoundFileName));
            }
        }

        private bool enableSound;
        public bool EnableSound
        {
            get => enableSound;
            set
            {
                enableSound = value;
                OnPropertyChanged(nameof(EnableSound));
            }
        }

        private VisualIndicatorType visualIndicatorType;
        public VisualIndicatorType VisualIndicatorType
        {
            get => visualIndicatorType;
            set
            {
                visualIndicatorType = value;
                OnPropertyChanged(nameof(VisualIndicatorType));
            }
        }

        private string visualIndicatorColor;
        public string VisualIndicatorColor
        {
            get => visualIndicatorColor;
            set
            {
                visualIndicatorColor = value;
                OnPropertyChanged(nameof(VisualIndicatorColor));
            }
        }

        public AlertModel()
        {
            CreatedDate = DateTime.Now;
            IsActive = true;
            IsTriggered = false;
            Priority = 2; // Medium priority by default
            Category = AlertCategory.Standard;
            ComparisonOperator = ComparisonOperator.GreaterThan;
            ThresholdValue = 0;

            // Default sound and visual indicator settings
            EnableSound = true;
            VisualIndicatorType = VisualIndicatorType.Toast;
            VisualIndicatorColor = "#FFFF00"; // Yellow by default
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
