using System;
using System.ComponentModel;

namespace Quantra.Models
{
    public class StockSymbol : INotifyPropertyChanged
    {
        private string symbol;
        private string name;
        private string sector;
        private string industry;
        private DateTime lastUpdated;

        public string Symbol
        {
            get => symbol;
            set
            {
                if (symbol != value)
                {
                    symbol = value;
                    OnPropertyChanged(nameof(Symbol));
                }
            }
        }

        public string Name
        {
            get => name;
            set
            {
                if (name != value)
                {
                    name = value;
                    OnPropertyChanged(nameof(Name));
                }
            }
        }

        public string Sector
        {
            get => sector;
            set
            {
                if (sector != value)
                {
                    sector = value;
                    OnPropertyChanged(nameof(Sector));
                }
            }
        }

        public string Industry
        {
            get => industry;
            set
            {
                if (industry != value)
                {
                    industry = value;
                    OnPropertyChanged(nameof(Industry));
                }
            }
        }

        public DateTime LastUpdated
        {
            get => lastUpdated;
            set
            {
                if (lastUpdated != value)
                {
                    lastUpdated = value;
                    OnPropertyChanged(nameof(LastUpdated));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
