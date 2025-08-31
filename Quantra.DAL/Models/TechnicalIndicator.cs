using System;
using System.ComponentModel;

namespace Quantra.Models
{
    public class TechnicalIndicator : INotifyPropertyChanged
    {
        private string _name;
        public string Name
        {
            get => _name;
            set
            {
                if (_name != value)
                {
                    _name = value;
                    OnPropertyChanged(nameof(Name));
                }
            }
        }

        private double _value;
        public double Value
        {
            get => _value;
            set
            {
                if (_value != value)
                {
                    _value = value;
                    OnPropertyChanged(nameof(Value));
                }
            }
        }

        private string _category;
        public string Category
        {
            get => _category;
            set
            {
                if (_category != value)
                {
                    _category = value;
                    OnPropertyChanged(nameof(Category));
                }
            }
        }

        private DateTime _timestamp;
        public DateTime Timestamp
        {
            get => _timestamp;
            set
            {
                if (_timestamp != value)
                {
                    _timestamp = value;
                    OnPropertyChanged(nameof(Timestamp));
                }
            }
        }

        private string _description;
        public string Description
        {
            get => _description;
            set
            {
                if (_description != value)
                {
                    _description = value;
                    OnPropertyChanged(nameof(Description));
                }
            }
        }

        public string FormattedValue
        {
            get
            {
                return Name switch
                {
                    "RSI" or "StochRSI" or "StochK" or "StochD" or "MFI" => $"{Value:F1}",
                    "MACD" or "MACDSignal" or "MACDHistogram" => $"{Value:F3}",
                    "ADX" => $"{Value:F1}",
                    "ATR" => $"${Value:F2}",
                    "VWAP" => $"${Value:F2}",
                    "ROC" => $"{Value:F1}%",
                    "BullPower" or "BearPower" => $"${Value:F2}",
                    "Price" => $"${Value:F2}",
                    "Volume" => $"{Value:N0}",
                    "MomentumScore" => $"{Value:F0}",
                    "P/E Ratio" or "PE Ratio" => $"{Value:F1}",
                    _ => $"{Value:F2}"
                };
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public TechnicalIndicator()
        {
            Timestamp = DateTime.Now;
        }
    }
}