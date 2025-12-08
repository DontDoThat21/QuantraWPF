using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for logging and retrieving ML model training history
    /// </summary>
    public class ModelTrainingHistoryService
    {
        private readonly QuantraDbContext _context;
        private readonly LoggingService _loggingService;

        public ModelTrainingHistoryService(QuantraDbContext context, LoggingService loggingService)
        {
            _context = context;
            _loggingService = loggingService;
        }

        /// <summary>
        /// Logs a training session with overall metrics
        /// </summary>
        public async Task<int> LogTrainingSessionAsync(ModelTrainingResult trainingResult, string notes = null)
        {
            try
            {
                var history = new ModelTrainingHistory
                {
                    TrainingDate = DateTime.Now,
                    ModelType = trainingResult.ModelType,
                    ArchitectureType = trainingResult.ArchitectureType,
                    SymbolsCount = trainingResult.SymbolsCount,
                    TrainingSamples = trainingResult.TrainingSamples,
                    TestSamples = trainingResult.TestSamples,
                    TrainingTimeSeconds = trainingResult.TrainingTimeSeconds,
                    MAE = trainingResult.Performance?.Mae ?? 0,
                    RMSE = trainingResult.Performance?.Rmse ?? 0,
                    R2Score = trainingResult.Performance?.R2Score ?? 0,
                    Notes = notes,
                    IsActive = true // Mark this model as currently active
                };

                // Deactivate previous models of the same type
                var previousModels = await _context.ModelTrainingHistory
                    .Where(m => m.ModelType == trainingResult.ModelType && 
                               m.ArchitectureType == trainingResult.ArchitectureType &&
                               m.IsActive)
                    .ToListAsync();

                foreach (var prev in previousModels)
                {
                    prev.IsActive = false;
                }

                _context.ModelTrainingHistory.Add(history);
                await _context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Training session logged: {trainingResult.ModelType} ({trainingResult.ArchitectureType}) - R�: {history.R2Score:F4}");

                return history.Id;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error logging training session");
                throw;
            }
        }

        /// <summary>
        /// Logs per-symbol training results
        /// </summary>
        public async Task LogSymbolResultsAsync(int trainingHistoryId, List<(string Symbol, int DataPoints, int TrainingSamples, int TestSamples, bool Included, string ExclusionReason)> symbolResults)
        {
            try
            {
                var results = symbolResults.Select(sr => new SymbolTrainingResult
                {
                    TrainingHistoryId = trainingHistoryId,
                    Symbol = sr.Symbol,
                    DataPointsCount = sr.DataPoints,
                    TrainingSamplesCount = sr.TrainingSamples,
                    TestSamplesCount = sr.TestSamples,
                    IncludedInTraining = sr.Included,
                    ExclusionReason = sr.ExclusionReason,
                    DataStartDate = null,  // Not available in this overload
                    DataEndDate = null     // Not available in this overload
                }).ToList();

                _context.SymbolTrainingResults.AddRange(results);
                await _context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Logged {results.Count} symbol training results");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error logging symbol training results");
                throw;
            }
        }

        /// <summary>
        /// Logs per-symbol training results with full metrics including date ranges
        /// </summary>
        public async Task LogSymbolResultsWithDatesAsync(int trainingHistoryId, List<SymbolTrainingMetric> symbolMetrics)
        {
            try
            {
                var results = symbolMetrics.Select(sm => new SymbolTrainingResult
                {
                    TrainingHistoryId = trainingHistoryId,
                    Symbol = sm.Symbol,
                    DataPointsCount = sm.DataPoints,
                    TrainingSamplesCount = sm.TrainingSamples,
                    TestSamplesCount = sm.TestSamples,
                    IncludedInTraining = sm.Included,
                    ExclusionReason = sm.ExclusionReason,
                    DataStartDate = !string.IsNullOrEmpty(sm.DataStartDate) && DateTime.TryParse(sm.DataStartDate, out var startDate) ? startDate : (DateTime?)null,
                    DataEndDate = !string.IsNullOrEmpty(sm.DataEndDate) && DateTime.TryParse(sm.DataEndDate, out var endDate) ? endDate : (DateTime?)null
                }).ToList();

                _context.SymbolTrainingResults.AddRange(results);
                await _context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Logged {results.Count} symbol training results with date ranges");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error logging symbol training results with dates");
                throw;
            }
        }

        /// <summary>
        /// Gets the most recent training session
        /// </summary>
        public async Task<ModelTrainingHistory> GetLatestTrainingAsync(string modelType = null, string architectureType = null)
        {
            try
            {
                var query = _context.ModelTrainingHistory
                    .OrderByDescending(m => m.TrainingDate)
                    .AsQueryable();

                if (!string.IsNullOrEmpty(modelType))
                {
                    query = query.Where(m => m.ModelType == modelType);
                }

                if (!string.IsNullOrEmpty(architectureType))
                {
                    query = query.Where(m => m.ArchitectureType == architectureType);
                }

                return await query.FirstOrDefaultAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error getting latest training");
                return null;
            }
        }

        /// <summary>
        /// Gets all training history with optional filtering
        /// </summary>
        public async Task<List<ModelTrainingHistory>> GetTrainingHistoryAsync(
            string modelType = null,
            string architectureType = null,
            DateTime? startDate = null,
            DateTime? endDate = null,
            int? limit = null)
        {
            try
            {
                var query = _context.ModelTrainingHistory
                    .OrderByDescending(m => m.TrainingDate)
                    .AsQueryable();

                if (!string.IsNullOrEmpty(modelType))
                {
                    query = query.Where(m => m.ModelType == modelType);
                }

                if (!string.IsNullOrEmpty(architectureType))
                {
                    query = query.Where(m => m.ArchitectureType == architectureType);
                }

                if (startDate.HasValue)
                {
                    query = query.Where(m => m.TrainingDate >= startDate.Value);
                }

                if (endDate.HasValue)
                {
                    query = query.Where(m => m.TrainingDate <= endDate.Value);
                }

                if (limit.HasValue && limit > 0)
                {
                    query = query.Take(limit.Value);
                }

                return await query.ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error getting training history");
                return new List<ModelTrainingHistory>();
            }
        }

        /// <summary>
        /// Gets symbol-level results for a specific training session
        /// </summary>
        public async Task<List<SymbolTrainingResult>> GetSymbolResultsAsync(int trainingHistoryId)
        {
            try
            {
                return await _context.SymbolTrainingResults
                    .Where(sr => sr.TrainingHistoryId == trainingHistoryId)
                    .OrderBy(sr => sr.Symbol)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error getting symbol results for training {trainingHistoryId}");
                return new List<SymbolTrainingResult>();
            }
        }

        /// <summary>
        /// Gets performance comparison across training sessions
        /// </summary>
        public async Task<List<TrainingComparison>> GetPerformanceComparisonAsync(string modelType, string architectureType, int count = 10)
        {
            try
            {
                var sessions = await _context.ModelTrainingHistory
                    .Where(m => m.ModelType == modelType && m.ArchitectureType == architectureType)
                    .OrderByDescending(m => m.TrainingDate)
                    .Take(count)
                    .Select(m => new TrainingComparison
                    {
                        TrainingDate = m.TrainingDate,
                        SymbolsCount = m.SymbolsCount,
                        TrainingSamples = m.TrainingSamples,
                        MAE = m.MAE,
                        RMSE = m.RMSE,
                        R2Score = m.R2Score,
                        TrainingTimeSeconds = m.TrainingTimeSeconds
                    })
                    .ToListAsync();

                return sessions;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error getting performance comparison");
                return new List<TrainingComparison>();
            }
        }

        /// <summary>
        /// Gets the currently active model
        /// </summary>
        public async Task<ModelTrainingHistory> GetActiveModelAsync(string modelType, string architectureType)
        {
            try
            {
                return await _context.ModelTrainingHistory
                    .Where(m => m.ModelType == modelType && 
                               m.ArchitectureType == architectureType &&
                               m.IsActive)
                    .OrderByDescending(m => m.TrainingDate)
                    .FirstOrDefaultAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error getting active model");
                return null;
            }
        }

        /// <summary>
        /// Marks a specific training session as active (and deactivates others)
        /// </summary>
        public async Task<bool> SetActiveModelAsync(int trainingHistoryId)
        {
            try
            {
                var model = await _context.ModelTrainingHistory.FindAsync(trainingHistoryId);
                if (model == null)
                    return false;

                // Deactivate all models of the same type
                var sameTypeModels = await _context.ModelTrainingHistory
                    .Where(m => m.ModelType == model.ModelType && 
                               m.ArchitectureType == model.ArchitectureType &&
                               m.IsActive)
                    .ToListAsync();

                foreach (var m in sameTypeModels)
                {
                    m.IsActive = false;
                }

                model.IsActive = true;
                await _context.SaveChangesAsync();

                _loggingService?.Log("Info", $"Set model {trainingHistoryId} as active");
                return true;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error setting active model {trainingHistoryId}");
                return false;
            }
        }

        /// <summary>
        /// Deletes old training history (keeps last N records per model type)
        /// </summary>
        public async Task<int> CleanupOldHistoryAsync(int keepLast = 20)
        {
            try
            {
                // Group by model type and architecture
                var modelTypes = await _context.ModelTrainingHistory
                    .Select(m => new { m.ModelType, m.ArchitectureType })
                    .Distinct()
                    .ToListAsync();

                int deletedCount = 0;

                foreach (var modelType in modelTypes)
                {
                    var toDelete = await _context.ModelTrainingHistory
                        .Where(m => m.ModelType == modelType.ModelType && 
                                   m.ArchitectureType == modelType.ArchitectureType)
                        .OrderByDescending(m => m.TrainingDate)
                        .Skip(keepLast)
                        .ToListAsync();

                    if (toDelete.Any())
                    {
                        // Also delete associated symbol results
                        var historyIds = toDelete.Select(t => t.Id).ToList();
                        var symbolResults = await _context.SymbolTrainingResults
                            .Where(sr => historyIds.Contains(sr.TrainingHistoryId))
                            .ToListAsync();

                        _context.SymbolTrainingResults.RemoveRange(symbolResults);
                        _context.ModelTrainingHistory.RemoveRange(toDelete);

                        deletedCount += toDelete.Count;
                    }
                }

                if (deletedCount > 0)
                {
                    await _context.SaveChangesAsync();
                    _loggingService?.Log("Info", $"Cleaned up {deletedCount} old training records");
                }

                return deletedCount;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error cleaning up training history");
                return 0;
            }
        }

        /// <summary>
        /// Checks if a trained model is available both in the database and as a local file.
        /// </summary>
        /// <param name="modelType">The model type to check (auto, pytorch, tensorflow, random_forest, tft)</param>
        /// <param name="architectureType">The architecture type to check (lstm, gru, transformer, tft)</param>
        /// <returns>A result indicating whether the model is available and details about its status</returns>
        public async Task<TrainedModelAvailability> CheckTrainedModelAvailabilityAsync(string modelType = "auto", string architectureType = "lstm")
        {
            var result = new TrainedModelAvailability
            {
                RequestedModelType = modelType,
                RequestedArchitectureType = architectureType
            };

            try
            {
                // Resolve 'auto' model type
                string resolvedModelType = modelType?.ToLower() ?? "auto";
                if (resolvedModelType == "auto")
                {
                    // Check for any available trained model, preferring pytorch > tensorflow > random_forest
                    var availableModel = await _context.ModelTrainingHistory
                        .Where(m => m.IsActive)
                        .OrderByDescending(m => m.TrainingDate)
                        .FirstOrDefaultAsync();

                    if (availableModel != null)
                    {
                        resolvedModelType = availableModel.ModelType;
                        result.ResolvedModelType = resolvedModelType;
                        result.ResolvedArchitectureType = availableModel.ArchitectureType;
                    }
                }
                else
                {
                    result.ResolvedModelType = resolvedModelType;
                    result.ResolvedArchitectureType = architectureType;
                }

                // Check database for training history
                var trainingHistory = await GetActiveModelAsync(resolvedModelType, architectureType);
                result.HasDatabaseRecord = trainingHistory != null;
                result.TrainingHistory = trainingHistory;

                if (trainingHistory != null)
                {
                    result.LastTrainingDate = trainingHistory.TrainingDate;
                    result.TrainingHistoryId = trainingHistory.Id;
                }

                // Check for local model file
                string pythonModelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "models");
                result.ModelFilePath = GetModelFilePath(resolvedModelType, pythonModelsDir, result.ResolvedArchitectureType);
                
                // TensorFlow SavedModel format saves as a directory, not a file
                if (resolvedModelType == "tensorflow")
                {
                    result.HasLocalModelFile = !string.IsNullOrEmpty(result.ModelFilePath) && Directory.Exists(result.ModelFilePath);
                }
                else
                {
                    result.HasLocalModelFile = !string.IsNullOrEmpty(result.ModelFilePath) && File.Exists(result.ModelFilePath);
                }

                // Model is available only if both DB record and local file exist
                result.IsModelAvailable = result.HasDatabaseRecord && result.HasLocalModelFile;

                // Generate user-friendly status message
                if (result.IsModelAvailable)
                {
                    result.StatusMessage = $"Trained {resolvedModelType} model available (last trained: {result.LastTrainingDate:g})";
                }
                else if (!result.HasDatabaseRecord && !result.HasLocalModelFile)
                {
                    result.StatusMessage = "No trained model found. Please train a model first using the 'Train Model' button.";
                }
                else if (!result.HasDatabaseRecord)
                {
                    result.StatusMessage = "Model file found but no training record in database. Please retrain the model.";
                }
                else if (!result.HasLocalModelFile)
                {
                    result.StatusMessage = "Training record found but model file is missing. Please retrain the model.";
                }

                _loggingService?.Log("Info", $"Model availability check: {result.StatusMessage}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error checking trained model availability");
                result.StatusMessage = $"Error checking model availability: {ex.Message}";
                result.IsModelAvailable = false;
            }

            return result;
        }

        /// <summary>
        /// Gets the expected model file path based on model type and architecture
        /// </summary>
        private string GetModelFilePath(string modelType, string modelsDir, string architectureType = null)
        {
            if (string.IsNullOrEmpty(modelType) || !Directory.Exists(modelsDir))
                return null;

            // Special case: If model type is "pytorch" and architecture is "tft", use TFT model file
            if (modelType.ToLower() == "pytorch" && architectureType?.ToLower() == "tft")
            {
                return Path.Combine(modelsDir, "tft_model.pt");
            }

            return modelType.ToLower() switch
            {
                "pytorch" => Path.Combine(modelsDir, "stock_pytorch_model.pt"),
                "tensorflow" => Path.Combine(modelsDir, "stock_tensorflow_model"),
                "random_forest" => Path.Combine(modelsDir, "stock_rf_model.pkl"),
                "tft" => Path.Combine(modelsDir, "tft_model.pt"),
                _ => null
            };
        }

        /// <summary>
        /// Checks if any trained model is available (regardless of type)
        /// </summary>
        public async Task<bool> IsAnyTrainedModelAvailableAsync()
        {
            try
            {
                // Check if any active training record exists
                var hasActiveModel = await _context.ModelTrainingHistory
                    .AnyAsync(m => m.IsActive);

                if (!hasActiveModel)
                    return false;

                // Check for any local model files
                string pythonModelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "models");
                
                if (!Directory.Exists(pythonModelsDir))
                    return false;

                // Check for any of the known model files
                // Note: TensorFlow SavedModel format saves as a directory, not a file
                var modelFiles = new[]
                {
                    Path.Combine(pythonModelsDir, "stock_pytorch_model.pt"),
                    Path.Combine(pythonModelsDir, "stock_rf_model.pkl"),
                    Path.Combine(pythonModelsDir, "tft_model.pt")
                };

                // TensorFlow saves as a directory (SavedModel format)
                var tensorflowModelPath = Path.Combine(pythonModelsDir, "stock_tensorflow_model");

                return modelFiles.Any(f => File.Exists(f)) || Directory.Exists(tensorflowModelPath);
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error checking if any trained model is available");
                return false;
            }
        }
    }

    /// <summary>
    /// Result of checking trained model availability
    /// </summary>
    public class TrainedModelAvailability
    {
        /// <summary>
        /// Whether a trained model is available for use (both DB record and local file exist)
        /// </summary>
        public bool IsModelAvailable { get; set; }

        /// <summary>
        /// Whether there's a training history record in the database
        /// </summary>
        public bool HasDatabaseRecord { get; set; }

        /// <summary>
        /// Whether the model file exists locally in python/models/
        /// </summary>
        public bool HasLocalModelFile { get; set; }

        /// <summary>
        /// The requested model type
        /// </summary>
        public string RequestedModelType { get; set; }

        /// <summary>
        /// The requested architecture type
        /// </summary>
        public string RequestedArchitectureType { get; set; }

        /// <summary>
        /// The resolved model type (in case 'auto' was requested)
        /// </summary>
        public string ResolvedModelType { get; set; }

        /// <summary>
        /// The resolved architecture type
        /// </summary>
        public string ResolvedArchitectureType { get; set; }

        /// <summary>
        /// The date when the model was last trained
        /// </summary>
        public DateTime? LastTrainingDate { get; set; }

        /// <summary>
        /// Path to the model file
        /// </summary>
        public string ModelFilePath { get; set; }

        /// <summary>
        /// The training history ID if available
        /// </summary>
        public int? TrainingHistoryId { get; set; }

        /// <summary>
        /// The training history record if available
        /// </summary>
        public ModelTrainingHistory TrainingHistory { get; set; }

        /// <summary>
        /// User-friendly status message
        /// </summary>
        public string StatusMessage { get; set; }
    }

    /// <summary>
    /// Training comparison data for charts/analysis
    /// </summary>
    public class TrainingComparison
    {
        public DateTime TrainingDate { get; set; }
        public int SymbolsCount { get; set; }
        public int TrainingSamples { get; set; }
        public double MAE { get; set; }
        public double RMSE { get; set; }
        public double R2Score { get; set; }
        public double TrainingTimeSeconds { get; set; }
    }
}
