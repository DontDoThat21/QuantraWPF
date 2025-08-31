using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.Configuration;
using Quantra.Configuration.Models;
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
        private readonly ILogger<OpenAIPredictionEnhancementService> _logger;
        private readonly ApiConfig _apiConfig;
        private readonly SentimentAnalysisConfig _sentimentConfig;
        private readonly ISocialMediaSentimentService _openAiSentimentService;
        
        /// <summary>
        /// Constructor for OpenAIPredictionEnhancementService
        /// </summary>
        public OpenAIPredictionEnhancementService(
            ILogger<OpenAIPredictionEnhancementService> logger,
            IConfigurationManager configManager,
            ISocialMediaSentimentService openAiSentimentService)
        {
            _logger = logger;
            _apiConfig = configManager.GetSection<ApiConfig>("Api");
            _sentimentConfig = configManager.GetSection<SentimentAnalysisConfig>("SentimentAnalysis");
            _openAiSentimentService = openAiSentimentService;
        }
        
        /// <summary>
        /// Enhances a prediction model with OpenAI-generated insights.
        /// </summary>
        public async Task<PredictionModel> EnhancePredictionAsync(PredictionModel prediction, string symbol)
        {
            if (prediction == null || string.IsNullOrEmpty(symbol))
                return prediction;
                
            if (string.IsNullOrEmpty(_apiConfig?.OpenAI?.ApiKey) || !_sentimentConfig?.OpenAI?.EnableEnhancedPredictionExplanations == true)
                return prediction;
                
            try
            {
                _logger.LogInformation($"Enhancing prediction for {symbol} with OpenAI");
                
                // Fetch recent relevant content for context
                var content = await _openAiSentimentService.FetchRecentContentAsync(symbol, 5);
                if (content.Count == 0)
                {
                    _logger.LogWarning($"No content available for enhancing prediction for {symbol}");
                    return prediction;
                }
                
                // Call Python script to enhance the prediction
                var enhancedPrediction = await CallOpenAIPredictionEnhancementAsync(prediction, content, symbol);
                if (enhancedPrediction != null)
                {
                    // Preserve the original prediction values but add OpenAI insights
                    prediction.OpenAIExplanation = enhancedPrediction.OpenAIExplanation ?? "";
                    prediction.AnalysisDetails = enhancedPrediction.AnalysisDetails ?? prediction.AnalysisDetails;
                    prediction.UsesOpenAI = true;
                }
                
                return prediction;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error enhancing prediction for {symbol}");
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
                    _logger.LogError($"Python script not found at: {pythonScript}");
                    return null;
                }
                
                // Prepare the request data
                var requestData = new
                {
                    type = "enhance_prediction",
                    apiKey = _apiConfig.OpenAI.ApiKey,
                    model = _apiConfig.OpenAI.Model,
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
                        _logger.LogError("Failed to start Python process");
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
                        _logger.LogError($"Python script failed with exit code {process.ExitCode}: {stdErr}");
                        return null;
                    }
                    
                    if (string.IsNullOrWhiteSpace(stdOut))
                    {
                        _logger.LogError($"Python script returned empty output. StdErr: {stdErr}");
                        return null;
                    }
                    
                    try
                    {
                        // Parse the response
                        var response = JsonSerializer.Deserialize<PythonResponse>(stdOut);
                        if (response?.prediction == null)
                        {
                            _logger.LogError("Invalid response format from Python script");
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
                        _logger.LogError(jsonEx, $"Error parsing Python response: {stdOut}");
                        return null;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in CallOpenAIPredictionEnhancementAsync");
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
    }
}