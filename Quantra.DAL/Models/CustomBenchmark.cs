using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Class for defining custom benchmark compositions for backtesting
    /// </summary>
    public class CustomBenchmark : INotifyPropertyChanged
    {
        private string _id;
        /// <summary>
        /// Unique identifier for the custom benchmark
        /// </summary>
        public string Id
        {
            get => _id;
            set
            {
                if (_id != value)
                {
                    _id = value;
                    OnPropertyChanged(nameof(Id));
                }
            }
        }

        private string _name;
        /// <summary>
        /// Name of the custom benchmark
        /// </summary>
        public string Name
        {
            get => !string.IsNullOrWhiteSpace(_name) ? _name : "Unnamed Custom Benchmark";
            set
            {
                if (_name != value)
                {
                    _name = value;
                    OnPropertyChanged(nameof(Name));
                }
            }
        }

        private string _description;
        /// <summary>
        /// Description of the custom benchmark
        /// </summary>
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

        private BenchmarkCategory _category;
        /// <summary>
        /// Category of the custom benchmark (e.g., Sector, Strategy, Asset Class)
        /// </summary>
        public BenchmarkCategory Category
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

        private DateTime _createdDate;
        /// <summary>
        /// Date when the benchmark was created
        /// </summary>
        public DateTime CreatedDate
        {
            get => _createdDate;
            set
            {
                if (_createdDate != value)
                {
                    _createdDate = value;
                    OnPropertyChanged(nameof(CreatedDate));
                }
            }
        }

        private DateTime _modifiedDate;
        /// <summary>
        /// Date when the benchmark was last modified
        /// </summary>
        public DateTime ModifiedDate
        {
            get => _modifiedDate;
            set
            {
                if (_modifiedDate != value)
                {
                    _modifiedDate = value;
                    OnPropertyChanged(nameof(ModifiedDate));
                }
            }
        }

        private List<BenchmarkComponent> _components;
        /// <summary>
        /// Components that make up the custom benchmark
        /// </summary>
        public List<BenchmarkComponent> Components
        {
            get => _components;
            set
            {
                if (_components != value)
                {
                    _components = value;
                    OnPropertyChanged(nameof(Components));
                }
            }
        }

        /// <summary>
        /// Calculated display symbol for the benchmark (derived from components)
        /// </summary>
        public string DisplaySymbol => GenerateDisplaySymbol();

        /// <summary>
        /// Default constructor
        /// </summary>
        public CustomBenchmark()
        {
            Id = Guid.NewGuid().ToString();
            CreatedDate = DateTime.Now;
            ModifiedDate = DateTime.Now;
            Components = new List<BenchmarkComponent>();
            Category = BenchmarkCategory.Custom;
        }

        /// <summary>
        /// Constructor with name and description
        /// </summary>
        public CustomBenchmark(string name, string description = null) : this()
        {
            Name = name;
            Description = description;
        }

        /// <summary>
        /// Add a component to the benchmark
        /// </summary>
        /// <param name="symbol">Symbol of the component</param>
        /// <param name="name">Name of the component</param>
        /// <param name="weight">Weight of the component (0-1)</param>
        public void AddComponent(string symbol, string name, double weight)
        {
            Components.Add(new BenchmarkComponent
            {
                Symbol = symbol,
                Name = name,
                Weight = weight
            });

            // Normalize weights after adding
            NormalizeWeights();

            // Update modified date
            ModifiedDate = DateTime.Now;

            OnPropertyChanged(nameof(Components));
            OnPropertyChanged(nameof(DisplaySymbol));
        }

        /// <summary>
        /// Remove a component from the benchmark
        /// </summary>
        /// <param name="symbol">Symbol of the component to remove</param>
        public void RemoveComponent(string symbol)
        {
            int initialCount = Components.Count;
            Components.RemoveAll(c => c.Symbol == symbol);

            if (Components.Count != initialCount)
            {
                // Normalize weights after removing
                NormalizeWeights();

                // Update modified date
                ModifiedDate = DateTime.Now;

                OnPropertyChanged(nameof(Components));
                OnPropertyChanged(nameof(DisplaySymbol));
            }
        }

        /// <summary>
        /// Normalize weights to ensure they sum to 1.0
        /// </summary>
        public void NormalizeWeights()
        {
            if (Components.Count == 0) return;

            double totalWeight = Components.Sum(c => c.Weight);

            if (Math.Abs(totalWeight) < 0.000001) // Avoid division by zero
            {
                double equalWeight = 1.0 / Components.Count;
                foreach (var component in Components)
                {
                    component.Weight = equalWeight;
                }
            }
            else if (Math.Abs(totalWeight - 1.0) > 0.000001) // Only normalize if not already normalized
            {
                foreach (var component in Components)
                {
                    component.Weight = component.Weight / totalWeight;
                }
            }
        }

        /// <summary>
        /// Validate that the benchmark is properly defined
        /// </summary>
        /// <returns>True if valid, false otherwise</returns>
        public bool Validate(out string errorMessage)
        {
            errorMessage = null;

            if (string.IsNullOrWhiteSpace(Name))
            {
                errorMessage = "Benchmark name is required";
                return false;
            }

            if (Components.Count == 0)
            {
                errorMessage = "Benchmark must have at least one component";
                return false;
            }

            double totalWeight = Components.Sum(c => c.Weight);
            if (Math.Abs(totalWeight - 1.0) > 0.01)
            {
                errorMessage = $"Component weights must sum to 1.0 (current sum: {totalWeight})";
                return false;
            }

            return true;
        }

        /// <summary>
        /// Generate a display symbol for the benchmark based on components
        /// </summary>
        /// <returns>Display symbol string</returns>
        private string GenerateDisplaySymbol()
        {
            if (Components.Count == 0)
                return "CUSTOM";

            if (Components.Count == 1)
                return Components[0].Symbol;

            // For multiple components, create a composite symbol like "60% SPY + 40% QQQ"
            var topComponents = Components
                .OrderByDescending(c => c.Weight)
                .Take(3) // Take top 3 by weight
                .ToList();

            string symbol = string.Join(" + ", topComponents.Select(c => $"{c.Weight:P0} {c.Symbol}"));

            // If there are more components not shown, indicate with "..."
            if (Components.Count > 3)
                symbol += " + ...";

            return symbol;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Override ToString to return the Name for display purposes
        /// </summary>
        /// <returns>The name of the custom benchmark</returns>
        public override string ToString()
        {
            return !string.IsNullOrWhiteSpace(Name) ? Name : "Unnamed Custom Benchmark";
        }
    }

    /// <summary>
    /// Component of a custom benchmark
    /// </summary>
    public class BenchmarkComponent : INotifyPropertyChanged
    {
        private string _symbol;
        /// <summary>
        /// Symbol of the component (e.g., "SPY", "QQQ")
        /// </summary>
        public string Symbol
        {
            get => _symbol;
            set
            {
                if (_symbol != value)
                {
                    _symbol = value;
                    OnPropertyChanged(nameof(Symbol));
                }
            }
        }

        private string _name;
        /// <summary>
        /// Name of the component (e.g., "S&P 500", "NASDAQ")
        /// </summary>
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

        private double _weight;
        /// <summary>
        /// Weight of the component in the benchmark (0-1)
        /// </summary>
        public double Weight
        {
            get => _weight;
            set
            {
                if (_weight != value)
                {
                    _weight = value;
                    OnPropertyChanged(nameof(Weight));
                    OnPropertyChanged(nameof(WeightPercentage));
                }
            }
        }

        /// <summary>
        /// Weight formatted as a percentage
        /// </summary>
        public string WeightPercentage => Weight.ToString("P2");

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    /// <summary>
    /// Categories for custom benchmarks
    /// </summary>
    public enum BenchmarkCategory
    {
        Sector,
        Industry,
        AssetClass,
        Strategy,
        Geography,
        Custom
    }
}