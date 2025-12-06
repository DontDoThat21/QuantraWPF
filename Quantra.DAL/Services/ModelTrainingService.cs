using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Quantra.DAL.Data;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for training ML models using historical data from the database
    /// </summary>
    public class ModelTrainingService
    {
        private readonly LoggingService _loggingService;
        private readonly string _pythonScriptPath;
        private readonly string _connectionString;

        public ModelTrainingService(LoggingService loggingService)
        {
            _loggingService = loggingService;
            _pythonScriptPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "python",
                "train_from_database.py"
            );
            _connectionString = ConvertToOdbcConnectionString(ConnectionHelper.ConnectionString);
            _loggingService.Log("Info", $"ODBC Connection String: {_connectionString}");
        }

        /// <summary>
        /// Converts a .NET SQL Server connection string to ODBC format for Python's pyodbc
        /// </summary>
        private static string ConvertToOdbcConnectionString(string dotNetConnectionString)
        {
            var builder = new Microsoft.Data.SqlClient.SqlConnectionStringBuilder(dotNetConnectionString);
            
            var odbcParts = new List<string>
            {
                "DRIVER={ODBC Driver 17 for SQL Server}",
                $"SERVER={builder.DataSource}",
                $"DATABASE={builder.InitialCatalog}",
                "Trusted_Connection=yes"
            };

            // LocalDB doesn't support encryption in the same way as SQL Server
            // For LocalDB connections, we need to explicitly disable encryption
            bool isLocalDb = builder.DataSource.Contains("localdb", StringComparison.OrdinalIgnoreCase);
            
            if (isLocalDb)
            {
                // LocalDB requires encryption to be disabled or set to optional
                odbcParts.Add("Encrypt=no");
            }
            else if (builder.Encrypt)
            {
                odbcParts.Add("Encrypt=yes");
                
                if (builder.TrustServerCertificate)
                {
                    odbcParts.Add("TrustServerCertificate=yes");
                }
            }

            return string.Join(";", odbcParts);
        }

        /// <summary>
        /// Train a model using all cached historical data from the database
        /// </summary>
        /// <param name="modelType">Model type: 'pytorch', 'tensorflow', 'random_forest', or 'auto'</param>
        /// <param name="architectureType">Architecture: 'lstm', 'gru', or 'transformer'</param>
        /// <param name="maxSymbols">Optional limit on number of symbols to use</param>
        /// <param name="progressCallback">Optional callback for progress updates</param>
        /// <returns>Training results</returns>
        /// <summary>
        /// Train model with full configuration object
        /// </summary>
        public async Task<ModelTrainingResult> TrainModelFromDatabaseAsync(
            Quantra.DAL.Models.TrainingConfiguration config,
            Action<string> progressCallback = null)
        {
            try
            {
                if (!File.Exists(_pythonScriptPath))
                {
                    throw new FileNotFoundException($"Training script not found: {_pythonScriptPath}");
                }

                // Validate configuration
                var errors = config.Validate();
                if (errors.Any())
                {
                    throw new ArgumentException($"Invalid configuration: {string.Join(", ", errors)}");
                }

                progressCallback?.Invoke($"Preparing to train model with configuration: {config.ConfigurationName}...");
                _loggingService.Log("Info", $"Starting model training with config: {config.ConfigurationName}");

                // Create temporary output file
                string tempDir = Path.Combine(Path.GetTempPath(), "Quantra_Training");
                Directory.CreateDirectory(tempDir);
                string outputFile = Path.Combine(tempDir, $"training_results_{Guid.NewGuid()}.json");
                string configFile = Path.Combine(tempDir, $"training_config_{Guid.NewGuid()}.json");

                try
                {
                    // Write configuration to JSON file
                    var configJson = config.ToJson();
                    await File.WriteAllTextAsync(configFile, configJson);

                    // Build arguments - pass config file to Python
                    var arguments = $"\"{_pythonScriptPath}\" " +
                                  $"\"{_connectionString}\" " +
                                  $"\"{outputFile}\" " +
                                  $"--config \"{configFile}\"";

                    // Start Python process
                    var psi = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = arguments,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(_pythonScriptPath)
                    };

                    progressCallback?.Invoke($"Training model: {config.Epochs} epochs, batch size {config.BatchSize}, LR {config.LearningRate}...");

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                            throw new Exception("Failed to start Python process");

                        var errorOutput = new System.Text.StringBuilder();
                        var standardOutput = new System.Text.StringBuilder();

                        // Capture output for progress updates
                        process.OutputDataReceived += (sender, e) =>
                        {
                            if (!string.IsNullOrEmpty(e.Data))
                            {
                                standardOutput.AppendLine(e.Data);
                                progressCallback?.Invoke(e.Data);
                                Debug.WriteLine($"Python: {e.Data}");
                            }
                        };

                        process.ErrorDataReceived += (sender, e) =>
                        {
                            if (!string.IsNullOrEmpty(e.Data))
                            {
                                errorOutput.AppendLine(e.Data);
                                progressCallback?.Invoke($"Warning: {e.Data}");
                                Debug.WriteLine($"Python Error: {e.Data}");
                            }
                        };

                        process.BeginOutputReadLine();
                        process.BeginErrorReadLine();

                        await process.WaitForExitAsync();
                        process.WaitForExit();

                        if (process.ExitCode != 0 || standardOutput.Length == 0)
                        {
                            var errorMessage = $"Python training failed with exit code {process.ExitCode}\n" +
                                             $"Standard output:\n{standardOutput}\n" +
                                             $"Error output:\n{errorOutput}";
                            throw new Exception(errorMessage);
                        }

                        // Read results
                        if (!File.Exists(outputFile))
                        {
                            throw new Exception($"Training script did not create output file\nOutput:\n{standardOutput}\nErrors:\n{errorOutput}");
                        }

                        var jsonResult = await File.ReadAllTextAsync(outputFile);
                        var options = new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true,
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                        };
                        var result = JsonSerializer.Deserialize<ModelTrainingResult>(jsonResult, options);

                        if (result == null || !result.Success)
                        {
                            throw new Exception(result?.Error ?? "Training failed with unknown error");
                        }

                        progressCallback?.Invoke("Model training completed successfully!");
                        _loggingService.Log("Info", $"Model training completed: {result.ModelType} - {result.TrainingSamples} samples");

                        return result;
                    }
                }
                finally
                {
                    // Cleanup temp files
                    try
                    {
                        if (File.Exists(outputFile)) File.Delete(outputFile);
                        if (File.Exists(configFile)) File.Delete(configFile);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Model training failed");
                return new ModelTrainingResult
                {
                    Success = false,
                    Error = ex.Message
                };
            }
        }

        /// <summary>
        /// Train model with legacy parameters (for backward compatibility)
        /// </summary>
        public async Task<ModelTrainingResult> TrainModelFromDatabaseAsync(
            string modelType = "auto",
            string architectureType = "lstm",
            int? maxSymbols = null,
            Action<string> progressCallback = null)
        {
            // Create a configuration from legacy parameters
            var config = Quantra.DAL.Models.TrainingConfiguration.CreateDefault();
            config.ModelType = modelType;
            config.ArchitectureType = architectureType;
            config.MaxSymbols = maxSymbols;

            // Call the new overload
            return await TrainModelFromDatabaseAsync(config, progressCallback);
        }

        /// <summary>
        /// Original implementation (moved below, kept for compatibility)
        /// </summary>
        private async Task<ModelTrainingResult> TrainModelFromDatabaseAsyncLegacy(
            string modelType = "auto",
            string architectureType = "lstm",
            int? maxSymbols = null,
            Action<string> progressCallback = null)
        {
            try
            {
                if (!File.Exists(_pythonScriptPath))
                {
                    throw new FileNotFoundException($"Training script not found: {_pythonScriptPath}");
                }

                progressCallback?.Invoke("Preparing to train model from database...");
                _loggingService.Log("Info", $"Starting model training: {modelType} with {architectureType}");

                // Create temporary output file
                string tempDir = Path.Combine(Path.GetTempPath(), "Quantra_Training");
                Directory.CreateDirectory(tempDir);
                string outputFile = Path.Combine(tempDir, $"training_results_{Guid.NewGuid()}.json");

                try
                {
                    // Build arguments
                    var arguments = $"\"{_pythonScriptPath}\" " +
                                  $"\"{_connectionString}\" " +
                                  $"\"{outputFile}\" " +
                                  $"{modelType} " +
                                  $"{architectureType}";

                    if (maxSymbols.HasValue)
                    {
                        arguments += $" {maxSymbols.Value}";
                    }

                    // Start Python process
                    var psi = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = arguments,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(_pythonScriptPath)
                    };

                    progressCallback?.Invoke("Training model with historical data...");

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                            throw new Exception("Failed to start Python process");

                        var errorOutput = new System.Text.StringBuilder();
                        var standardOutput = new System.Text.StringBuilder();

                        // Capture output for progress updates
                        process.OutputDataReceived += (sender, e) =>
                        {
                            if (!string.IsNullOrEmpty(e.Data))
                            {
                                standardOutput.AppendLine(e.Data);
                                progressCallback?.Invoke(e.Data);
                                Debug.WriteLine($"Python: {e.Data}");
                            }
                        };

                        process.ErrorDataReceived += (sender, e) =>
                        {
                            if (!string.IsNullOrEmpty(e.Data))
                            {
                                errorOutput.AppendLine(e.Data);
                                progressCallback?.Invoke($"Warning: {e.Data}");
                                Debug.WriteLine($"Python Error: {e.Data}");
                            }
                        };

                        process.BeginOutputReadLine();
                        process.BeginErrorReadLine();

                        await process.WaitForExitAsync();

                        // CRITICAL: Wait for async output readers to complete
                        // WaitForExitAsync() can return before async readers finish processing buffered output
                        process.WaitForExit();

                        // Check if we got any output - if not, the script failed silently
                        if (standardOutput.Length == 0 && errorOutput.Length == 0)
                        {
                            var errorMessage = "Python process produced no output. This usually indicates:\n" +
                                             "1. Missing Python dependencies (pyodbc, numpy, tensorflow, pytorch, etc.)\n" +
                                             "2. Python script syntax or import errors\n" +
                                             "3. ODBC Driver 17 for SQL Server not installed\n" +
                                             $"4. Connection string issue: {_connectionString}\n\n" +
                                             $"Try running manually: python \"{_pythonScriptPath}\" \"{_connectionString}\" \"{outputFile}\" {modelType} {architectureType}";
                            throw new Exception(errorMessage);
                        }

                        if (process.ExitCode != 0)
                        {
                            var errorMessage = $"Python training failed with exit code {process.ExitCode}";
                            if (errorOutput.Length > 0)
                            {
                                errorMessage += $"\n\nError output:\n{errorOutput}";
                            }
                            if (standardOutput.Length > 0)
                            {
                                errorMessage += $"\n\nStandard output:\n{standardOutput}";
                            }
                            throw new Exception(errorMessage);
                        }

                        // Read results
                        if (!File.Exists(outputFile))
                        {
                            var errorMessage = "Training script did not create output file";
                            if (errorOutput.Length > 0)
                            {
                                errorMessage += $"\n\nError output:\n{errorOutput}";
                            }
                            if (standardOutput.Length > 0)
                            {
                                errorMessage += $"\n\nStandard output:\n{standardOutput}";
                            }
                            throw new Exception(errorMessage);
                        }

                        var jsonResult = await File.ReadAllTextAsync(outputFile);
                        var options = new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true,
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                        };
                        var result = JsonSerializer.Deserialize<ModelTrainingResult>(jsonResult, options);

                        if (result == null || !result.Success)
                        {
                            throw new Exception(result?.Error ?? "Training failed with unknown error");
                        }

                        progressCallback?.Invoke("Model training completed successfully!");
                        _loggingService.Log("Info", $"Model training completed: {result.ModelType} - {result.TrainingSamples} samples");

                        return result;
                    }
                }
                finally
                {
                    // Cleanup temp file
                    try
                    {
                        if (File.Exists(outputFile))
                            File.Delete(outputFile);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Model training failed");
                progressCallback?.Invoke($"Training failed: {ex.Message}");
                
                return new ModelTrainingResult
                {
                    Success = false,
                    Error = ex.Message
                };
            }
        }
    }

    /// <summary>
    /// Result of model training operation
    /// </summary>
    public class ModelTrainingResult
    {
        [JsonPropertyName("success")]
        public bool Success { get; set; }
        
        [JsonPropertyName("model_type")]
        public string ModelType { get; set; }
        
        [JsonPropertyName("architecture_type")]
        public string ArchitectureType { get; set; }
        
        [JsonPropertyName("symbols_count")]
        public int SymbolsCount { get; set; }
        
        [JsonPropertyName("training_samples")]
        public int TrainingSamples { get; set; }
        
        [JsonPropertyName("test_samples")]
        public int TestSamples { get; set; }
        
        [JsonPropertyName("training_time_seconds")]
        public double TrainingTimeSeconds { get; set; }
        
        [JsonPropertyName("performance")]
        public ModelPerformance Performance { get; set; }
        
        [JsonPropertyName("error")]
        public string Error { get; set; }
        
        [JsonPropertyName("message")]
        public string Message { get; set; }
        
        [JsonPropertyName("symbol_results")]
        public List<SymbolTrainingMetric> SymbolResults { get; set; }
    }

    public class ModelPerformance
    {
        [JsonPropertyName("mse")]
        public double Mse { get; set; }
        
        [JsonPropertyName("mae")]
        public double Mae { get; set; }
        
        [JsonPropertyName("rmse")]
        public double Rmse { get; set; }
        
        [JsonPropertyName("r2_score")]
        public double R2Score { get; set; }
    }

    /// <summary>
    /// Per-symbol training metrics from Python training script
    /// </summary>
    public class SymbolTrainingMetric
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }
        
        [JsonPropertyName("data_points")]
        public int DataPoints { get; set; }
        
        [JsonPropertyName("training_samples")]
        public int TrainingSamples { get; set; }
        
        [JsonPropertyName("test_samples")]
        public int TestSamples { get; set; }
        
        [JsonPropertyName("included")]
        public bool Included { get; set; }
        
        [JsonPropertyName("exclusion_reason")]
        public string ExclusionReason { get; set; }
        
        [JsonPropertyName("data_start_date")]
        public string DataStartDate { get; set; }
        
        [JsonPropertyName("data_end_date")]
        public string DataEndDate { get; set; }
    }
}
