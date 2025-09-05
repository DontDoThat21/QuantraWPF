using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service that enhances stock predictions with OpenAI-generated insights.
    /// </summary>
    public class OpenAIPredictionEnhancementService
    {
        private readonly ILogger _logger;
        private readonly dynamic _apiConfig; // dynamic to avoid hard project refs for DI flexibility
        private readonly dynamic _sentimentConfig; // dynamic to avoid hard project refs for DI flexibility
        private readonly ISocialMediaSentimentService _openAiSentimentService;
        
        /// <summary>
        /// Constructor for OpenAIPredictionEnhancementService
        /// </summary>
        /// <remarks>
        /// Accepts loosely-typed configuration manager to avoid cross-project references.
        /// </remarks>
        public OpenAIPredictionEnhancementService(
            ISocialMediaSentimentService openAiSentimentService,
            object configManager,
            object logger)
        {
            _logger = logger as ILogger ?? Log.ForType<OpenAIPredictionEnhancementService>();
            _openAiSentimentService = openAiSentimentService;

            // Build lightweight config objects from provided configuration manager (if available)
            _apiConfig = BuildApiConfig(configManager);
            _sentimentConfig = BuildSentimentConfig(configManager);
        }
        
        /// <summary>
        /// Enhances a prediction model with OpenAI-generated insights.
        /// </summary>
        public async Task<PredictionModel> EnhancePredictionAsync(PredictionModel prediction, string symbol)
        {
            if (prediction == null || string.IsNullOrEmpty(symbol))
                return prediction;
                
            // Check if OpenAI enabled and API key available
            if (string.IsNullOrEmpty(_apiConfig?.OpenAI?.ApiKey) || !(_sentimentConfig?.OpenAI?.EnableEnhancedPredictionExplanations ?? true))
                return prediction;
                
            try
            {
                _logger.Information("Enhancing prediction for {Symbol} with OpenAI", symbol);
                
                // Fetch recent relevant content for context
                var content = await _openAiSentimentService.FetchRecentContentAsync(symbol, 5);
                if (content == null || content.Count == 0)
                {
                    _logger.Warning("No content available for enhancing prediction for {Symbol}", symbol);
                    return prediction;
                }
                
                // Call Python script to enhance the prediction
                var enhancedPrediction = await CallOpenAIPredictionEnhancementAsync(prediction, content, symbol);
                if (enhancedPrediction != null)
                {
                    // Preserve the original prediction values but add OpenAI insights
                    prediction.OpenAIExplanation = enhancedPrediction.OpenAIExplanation ?? string.Empty;
                    prediction.AnalysisDetails = enhancedPrediction.AnalysisDetails ?? prediction.AnalysisDetails;
                    prediction.UsesOpenAI = true;
                }
                
                return prediction;
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error enhancing prediction for {Symbol}", symbol);
                return prediction; // Return unmodified prediction on error
            }
        }
        
        /// <summary>
        /// Calls the Python helper script to enhance a prediction using OpenAI.
        /// </summary>
        private async Task<PredictionModel> CallOpenAIPredictionEnhancementAsync(PredictionModel prediction, List<string> content, string symbol)
        {
            string pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "openai_sentiment_analysis.py");
            
            try
            {
                if (!File.Exists(pythonScript))
                {
                    _logger.Error("Python script not found at: {Path}", pythonScript);
                    return null;
                }
                
                // Prepare the request data
                var requestData = new
                {
                    type = "enhance_prediction",
                    apiKey = _apiConfig?.OpenAI?.ApiKey,
                    model = _apiConfig?.OpenAI?.Model,
                    prediction = new
                    {
                        symbol,
                        action = prediction.PredictedAction,
                        targetPrice = prediction.TargetPrice,
                        currentPrice = prediction.CurrentPrice,
                        confidence = prediction.Confidence,
                        analysisDetails = prediction.AnalysisDetails
                    },
                    texts = content
                };
                
                var json = JsonSerializer.Serialize(requestData);
                
                string pythonExe = "python";
                
                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{pythonScript}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    RedirectStandardInput = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(pythonScript)
                };
                
                using (var process = Process.Start(psi))
                {
                    if (process == null)
                    {
                        _logger.Error("Failed to start Python process");
                        return null;
                    }
                    
                    // Write JSON request to Python's stdin
                    await process.StandardInput.WriteLineAsync(json);
                    process.StandardInput.Close();
                    
                    var outputTask = process.StandardOutput.ReadToEndAsync();
                    var errorTask = process.StandardError.ReadToEndAsync();
                    
                    await process.WaitForExitAsync();
                    
                    string stdOut = await outputTask;
                    string stdErr = await errorTask;
                    
                    if (process.ExitCode != 0)
                    {
                        _logger.Error("Python script failed with exit code {Code}: {Error}", process.ExitCode, stdErr);
                        return null;
                    }
                    
                    if (string.IsNullOrWhiteSpace(stdOut))
                    {
                        _logger.Error("Python script returned empty output. StdErr: {Error}", stdErr);
                        return null;
                    }
                    
                    try
                    {
                        // Parse the response
                        var response = JsonSerializer.Deserialize<PythonResponse>(stdOut);
                        if (response?.prediction == null)
                        {
                            _logger.Error("Invalid response format from Python script");
                            return null;
                        }
                        
                        return new PredictionModel
                        {
                            Symbol = symbol,
                            AnalysisDetails = response.prediction.analysisDetails,
                            OpenAIExplanation = response.prediction.analysisDetails,
                            UsesOpenAI = true
                        };
                    }
                    catch (JsonException jsonEx)
                    {
                        _logger.Error(jsonEx, "Error parsing Python response: {StdOut}", stdOut);
                        return null;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error in CallOpenAIPredictionEnhancementAsync");
                return null;
            }
        }
        
        /// <summary>
        /// Model for Python response
        /// </summary>
        private class PythonResponse
        {
            public PythonPrediction prediction { get; set; }
            public string error { get; set; }
        }
        
        /// <summary>
        /// Model for Python prediction
        /// </summary>
        private class PythonPrediction
        {
            public string symbol { get; set; }
            public string action { get; set; }
            public double targetPrice { get; set; }
            public double currentPrice { get; set; }
            public double confidence { get; set; }
            public string analysisDetails { get; set; }
            public bool openAiEnhanced { get; set; }
        }

        // --- Configuration helpers (mirror OpenAISentimentService approach) ---
        private dynamic BuildApiConfig(object configManager)
        {
            dynamic root = new System.Dynamic.ExpandoObject();
            dynamic openAi = new System.Dynamic.ExpandoObject();
            openAi.BaseUrl = GetConfigValue(configManager, "ApiConfig:OpenAI:BaseUrl", "https://api.openai.com");
            openAi.ApiKey = GetConfigValue(configManager, "ApiConfig:OpenAI:ApiKey", Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty);
            openAi.Model = GetConfigValue(configManager, "ApiConfig:OpenAI:Model", "gpt-4o-mini");
            openAi.Temperature = GetConfigValue(configManager, "ApiConfig:OpenAI:Temperature", 0.2);
            root.OpenAI = openAi;
            return root;
        }

        private dynamic BuildSentimentConfig(object configManager)
        {
            dynamic root = new System.Dynamic.ExpandoObject();
            dynamic openAi = new System.Dynamic.ExpandoObject();
            openAi.EnableEnhancedPredictionExplanations = GetConfigValue(configManager, "SentimentAnalysisConfig:OpenAI:EnableEnhancedPredictionExplanations", true);
            openAi.MaxTokens = GetConfigValue(configManager, "SentimentAnalysisConfig:OpenAI:MaxTokens", 500);
            root.OpenAI = openAi;
            return root;
        }

        private T GetConfigValue<T>(object configManager, string key, T defaultValue)
        {
            try
            {
                if (configManager == null)
                {
                    return defaultValue;
                }
                var type = configManager.GetType();
                // Try generic GetValue<T>(string key, T defaultValue)
                var method = type.GetMethod("GetValue");
                if (method != null && method.IsGenericMethod)
                {
                    var generic = method.MakeGenericMethod(typeof(T));
                    var result = generic.Invoke(configManager, new object[] { key, defaultValue });
                    if (result is T typed)
                    {
                        return typed;
                    }
                }
                // Try indexer style: configManager[key]
                var indexer = type.GetProperty("Item", new[] { typeof(string) });
                if (indexer != null)
                {
                    var value = indexer.GetValue(configManager, new object[] { key });
                    if (value is T t)
                        return t;
                    if (value != null)
                    {
                        return (T)Convert.ChangeType(value, typeof(T));
                    }
                }
            }
            catch
            {
                // Ignore and fall back
            }
            return defaultValue;
        }
    }
}