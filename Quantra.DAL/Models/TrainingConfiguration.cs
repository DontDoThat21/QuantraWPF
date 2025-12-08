using System;
using System.Collections.Generic;
using System.Text.Json;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Comprehensive training configuration for ML models.
    /// Supports all model types (PyTorch, TensorFlow, Random Forest, TFT).
    /// </summary>
    public class TrainingConfiguration
    {
        // Model Selection
        public string ModelType { get; set; } = "auto";
        public string ArchitectureType { get; set; } = "lstm";

        // Data Configuration
        public int? MaxSymbols { get; set; }
        public List<string> SelectedSymbols { get; set; }
        public double TrainTestSplit { get; set; } = 0.8;
        public int? RandomSeed { get; set; } = 42;

        // Neural Network Hyperparameters
        public int Epochs { get; set; } = 50;
        public int BatchSize { get; set; } = 32;
        public double LearningRate { get; set; } = 0.001;
        public double Dropout { get; set; } = 0.1;
        public int HiddenDim { get; set; } = 128;
        public int NumLayers { get; set; } = 2;

        // TFT-Specific Parameters
        public int NumHeads { get; set; } = 4;
        public int NumAttentionLayers { get; set; } = 2;
        public List<int> ForecastHorizons { get; set; } = new List<int> { 5, 10, 20, 30 };

        // Random Forest Parameters
        public int NumberOfTrees { get; set; } = 100;
        public int MaxDepth { get; set; } = 10;
        public int MinSamplesSplit { get; set; } = 2;

        // Training Optimization
        public string Optimizer { get; set; } = "adam";
        public double WeightDecay { get; set; } = 0.0001;
        public bool UseEarlyStopping { get; set; } = true;
        public int EarlyStoppingPatience { get; set; } = 10;
        public bool UseLearningRateScheduler { get; set; } = true;
        public double LRSchedulerFactor { get; set; } = 0.5;
        public int LRSchedulerPatience { get; set; } = 5;

        // Feature Engineering
        public string FeatureType { get; set; } = "balanced"; // minimal, balanced, comprehensive
        public bool UseFeatureEngineering { get; set; } = true;
        public int LookbackPeriod { get; set; } = 60;

        // Advanced Options
        public bool UseGPU { get; set; } = true;
        public int NumWorkers { get; set; } = 4;
        public bool VerboseLogging { get; set; } = true;
        public bool SaveCheckpoints { get; set; } = true;
        public int CheckpointFrequency { get; set; } = 10; // Save every N epochs

        // Metadata
        public string ConfigurationName { get; set; } = "Default";
        public string Description { get; set; }
        public DateTime CreatedDate { get; set; } = DateTime.Now;
        public DateTime? LastModifiedDate { get; set; }

        /// <summary>
        /// Create a default configuration
        /// </summary>
        public static TrainingConfiguration CreateDefault()
        {
            return new TrainingConfiguration
            {
                ConfigurationName = "Default",
                Description = "Standard configuration for general stock prediction",
                ModelType = "auto",
                ArchitectureType = "lstm",
                Epochs = 50,
                BatchSize = 32,
                LearningRate = 0.001,
                Dropout = 0.1,
                HiddenDim = 128,
                NumLayers = 2
            };
        }

        /// <summary>
        /// Create a fast training configuration for testing
        /// </summary>
        public static TrainingConfiguration CreateFastTraining()
        {
            return new TrainingConfiguration
            {
                ConfigurationName = "Fast Training",
                Description = "Quick training for testing and development",
                ModelType = "pytorch",
                ArchitectureType = "lstm",
                Epochs = 10,
                BatchSize = 64,
                LearningRate = 0.002,
                Dropout = 0.1,
                HiddenDim = 96,
                NumLayers = 1,
                UseEarlyStopping = false
            };
        }

        /// <summary>
        /// Create a high-accuracy configuration for production
        /// Uses Transformer architecture for proven high accuracy
        /// For TFT with future features, use CreateTFTOptimized() instead
        /// </summary>
        public static TrainingConfiguration CreateHighAccuracy()
        {
            return new TrainingConfiguration
            {
                ConfigurationName = "High Accuracy",
                Description = "Maximum accuracy configuration for production deployment using Transformer",
                ModelType = "pytorch",
                ArchitectureType = "transformer",
                Epochs = 100,
                BatchSize = 64, // Increased from 32 for better convergence
                LearningRate = 0.001, // Increased from 0.0005 for faster initial learning
                Dropout = 0.2,
                HiddenDim = 256,
                NumLayers = 3,
                NumHeads = 8, // Still used by Transformer
                NumAttentionLayers = 3, // Still used by Transformer
                UseEarlyStopping = true,
                EarlyStoppingPatience = 15,
                FeatureType = "balanced", // Explicit feature engineering
                UseFeatureEngineering = true
            };
        }

        /// <summary>
        /// Create a TFT-optimized configuration
        /// TFT (Temporal Fusion Transformer) now fully implemented with future features support
        /// Trains with known-future covariates (calendar features) for multi-horizon forecasting
        /// </summary>
        public static TrainingConfiguration CreateTFTOptimized()
        {
            return new TrainingConfiguration
            {
                ConfigurationName = "TFT Optimized",
                Description = "Optimized for Temporal Fusion Transformer with future features",
                ModelType = "pytorch",
                ArchitectureType = "tft",
                Epochs = 50,
                BatchSize = 64,
                LearningRate = 0.001,
                Dropout = 0.15,
                HiddenDim = 160,
                NumLayers = 2,
                NumHeads = 4,
                NumAttentionLayers = 2,
                ForecastHorizons = new List<int> { 5, 10, 20, 30 },
                UseEarlyStopping = true,
                EarlyStoppingPatience = 10
            };
        }

        /// <summary>
        /// Create configuration specifically to fix low R� score issues
        /// Uses proven Transformer architecture with optimized hyperparameters
        /// Includes target scaling fix from train_from_database.py (RobustScaler)
        /// </summary>
        public static TrainingConfiguration CreateR2ScoreFix()
        {
            return new TrainingConfiguration
            {
                ConfigurationName = "R� Score Fix",
                Description = "Optimized configuration to fix R� score ~0 issues with target scaling and Transformer",
                ModelType = "pytorch",
                ArchitectureType = "transformer", // Known working architecture
                Epochs = 100, // Sufficient for convergence
                BatchSize = 64, // Good balance of speed and stability
                LearningRate = 0.001, // Standard rate for Adam optimizer
                Dropout = 0.2, // Prevent overfitting
                HiddenDim = 256, // High capacity for pattern learning
                NumLayers = 3, // Deep enough for complex patterns
                NumHeads = 8, // Multi-head attention
                NumAttentionLayers = 3, // Attention depth
                UseEarlyStopping = true,
                EarlyStoppingPatience = 15, // Allow time to converge
                UseLearningRateScheduler = true,
                LRSchedulerPatience = 5,
                LRSchedulerFactor = 0.5,
                FeatureType = "balanced", // Good feature set without overload
                UseFeatureEngineering = true, // Enhanced features
                LookbackPeriod = 60, // 60-day window
                TrainTestSplit = 0.8 // 80/20 split
            };
        }

        /// <summary>
        /// Validate the configuration
        /// </summary>
        public List<string> Validate()
        {
            var errors = new List<string>();

            if (Epochs <= 0)
                errors.Add("Epochs must be greater than 0");
            if (BatchSize <= 0)
                errors.Add("Batch size must be greater than 0");
            if (LearningRate <= 0 || LearningRate > 1)
                errors.Add("Learning rate must be between 0 and 1");
            if (Dropout < 0 || Dropout > 1)
                errors.Add("Dropout must be between 0 and 1");
            if (HiddenDim <= 0)
                errors.Add("Hidden dimension must be greater than 0");
            if (NumLayers <= 0)
                errors.Add("Number of layers must be greater than 0");

            // TFT-specific validation
            if (ArchitectureType == "tft")
            {
                if (HiddenDim % NumHeads != 0)
                    errors.Add($"Hidden dimension ({HiddenDim}) must be divisible by number of heads ({NumHeads})");
                if (NumHeads <= 0)
                    errors.Add("Number of heads must be greater than 0");
                if (NumAttentionLayers <= 0)
                    errors.Add("Number of attention layers must be greater than 0");
            }

            // Random Forest validation
            if (ModelType == "random_forest")
            {
                if (NumberOfTrees <= 0)
                    errors.Add("Number of trees must be greater than 0");
                if (MaxDepth <= 0)
                    errors.Add("Max depth must be greater than 0");
            }

            return errors;
        }

        /// <summary>
        /// Convert to JSON for Python interop
        /// </summary>
        public string ToJson()
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            return JsonSerializer.Serialize(this, options);
        }

        /// <summary>
        /// Create from JSON
        /// </summary>
        public static TrainingConfiguration FromJson(string json)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            return JsonSerializer.Deserialize<TrainingConfiguration>(json, options);
        }

        /// <summary>
        /// Clone the configuration
        /// </summary>
        public TrainingConfiguration Clone()
        {
            var json = ToJson();
            return FromJson(json);
        }
    }
}
