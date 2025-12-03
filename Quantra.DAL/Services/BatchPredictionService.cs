using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Data;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Microsoft.EntityFrameworkCore;
using QuantraLogging = Quantra.CrossCutting.Logging;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for generating predictions for large batches of symbols overnight.
    /// Optimized for processing 12,000+ symbols with progress tracking, error recovery, and resource throttling.
    /// </summary>
    public class BatchPredictionService
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<BatchPredictionService> _logger;
        private readonly RealTimeInferenceService _inferenceService;
        private readonly TechnicalIndicatorService _indicatorService;
        private readonly PredictionAnalysisService _predictionService;
        private readonly PredictionCacheService _cacheService;
        private readonly QuantraLogging.ILogger _loggingService;
        
        // Configuration
        private const int MaxConcurrentPredictions = 10; // Adjust based on system resources
        private const int BatchSize = 100; // Process in batches to avoid memory issues
        private const int RetryAttempts = 3;
        private const int DelayBetweenBatchesMs = 1000; // 1 second delay between batches
        private const string DefaultModelType = "auto";
        
        // Progress tracking
        private int _totalSymbols;
        private int _processedSymbols;
        private int _successfulPredictions;
        private int _failedPredictions;
        private DateTime _batchStartTime;
        private readonly SemaphoreSlim _throttleSemaphore;

        public BatchPredictionService(
            QuantraDbContext context,
            ILogger<BatchPredictionService> logger,
            RealTimeInferenceService inferenceService,
            TechnicalIndicatorService indicatorService,
            PredictionAnalysisService predictionService,
            PredictionCacheService cacheService,
            QuantraLogging.ILogger loggingService)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _inferenceService = inferenceService ?? throw new ArgumentNullException(nameof(inferenceService));
            _indicatorService = indicatorService ?? throw new ArgumentNullException(nameof(indicatorService));
            _predictionService = predictionService ?? throw new ArgumentNullException(nameof(predictionService));
            _cacheService = cacheService ?? throw new ArgumentNullException(nameof(cacheService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
            
            _throttleSemaphore = new SemaphoreSlim(MaxConcurrentPredictions);
        }

        /// <summary>
        /// Generates predictions for all symbols in the database.
        /// Designed to run overnight with progress tracking and error recovery.
        /// </summary>
        public async Task<BatchPredictionResult> GenerateOvernightPredictionsAsync(
            BatchPredictionOptions options = null,
            IProgress<BatchPredictionProgress> progress = null,
            CancellationToken cancellationToken = default)
        {
            options ??= new BatchPredictionOptions();
            _batchStartTime = DateTime.Now;
            _processedSymbols = 0;
            _successfulPredictions = 0;
            _failedPredictions = 0;

            try
            {
                _logger?.LogInformation("Starting overnight batch prediction generation");
                _loggingService?.Information("Starting overnight batch prediction generation");

                // Get all symbols with historical data
                var symbols = await GetSymbolsForProcessingAsync(options, cancellationToken);
                _totalSymbols = symbols.Count;
                
                _logger?.LogInformation("Found {Count} symbols to process", _totalSymbols);
                
                if (_totalSymbols == 0)
                {
                    return new BatchPredictionResult
                    {
                        Success = true,
                        Message = "No symbols found to process",
                        TotalSymbols = 0
                    };
                }

                // Process in batches
                var batches = symbols.Chunk(BatchSize).ToList();
                _logger?.LogInformation("Processing {SymbolCount} symbols in {BatchCount} batches", 
                    _totalSymbols, batches.Count);

                for (int i = 0; i < batches.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    var batch = batches[i];
                    await ProcessBatchAsync(batch, options, progress, cancellationToken);
                    
                    // Delay between batches to avoid overwhelming the system
                    if (i < batches.Count - 1)
                    {
                        await Task.Delay(DelayBetweenBatchesMs, cancellationToken);
                    }
                }

                var duration = DateTime.Now - _batchStartTime;
                var result = new BatchPredictionResult
                {
                    Success = true,
                    TotalSymbols = _totalSymbols,
                    ProcessedSymbols = _processedSymbols,
                    SuccessfulPredictions = _successfulPredictions,
                    FailedPredictions = _failedPredictions,
                    Duration = duration,
                    Message = $"Batch prediction complete: {_successfulPredictions}/{_totalSymbols} successful"
                };

                _logger?.LogInformation("Batch prediction complete: {Success}/{Total} successful in {Duration}",
                    _successfulPredictions, _totalSymbols, duration);
                _loggingService?.Information("Batch prediction complete: {Success}/{Total} successful", _successfulPredictions, _totalSymbols);

                return result;
            }
            catch (OperationCanceledException)
            {
                _logger?.LogWarning("Batch prediction cancelled");
                return new BatchPredictionResult
                {
                    Success = false,
                    Message = "Operation cancelled",
                    TotalSymbols = _totalSymbols,
                    ProcessedSymbols = _processedSymbols,
                    SuccessfulPredictions = _successfulPredictions,
                    FailedPredictions = _failedPredictions
                };
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error during batch prediction generation");
                _loggingService?.Error(ex, "Error during batch prediction generation");
                
                return new BatchPredictionResult
                {
                    Success = false,
                    Message = $"Error: {ex.Message}",
                    TotalSymbols = _totalSymbols,
                    ProcessedSymbols = _processedSymbols,
                    SuccessfulPredictions = _successfulPredictions,
                    FailedPredictions = _failedPredictions
                };
            }
        }

        /// <summary>
        /// Gets symbols that need predictions based on options.
        /// </summary>
        private async Task<List<string>> GetSymbolsForProcessingAsync(
            BatchPredictionOptions options,
            CancellationToken cancellationToken)
        {
            try
            {
                // Get all symbols from StockDataCache table
                var query = _context.StockDataCache
                    .AsNoTracking()
                    .Where(s => !string.IsNullOrEmpty(s.Symbol))
                    .Select(s => s.Symbol)
                    .Distinct();

                // Filter by specific symbols if provided
                if (options.SymbolsToProcess?.Any() == true)
                {
                    query = query.Where(s => options.SymbolsToProcess.Contains(s));
                }

                // Filter by sector if provided
                if (!string.IsNullOrEmpty(options.SectorFilter))
                {
                    // Join with stock symbols to filter by sector
                    var sectorSymbols = await _context.StockSymbols
                        .AsNoTracking()
                        .Where(c => c.Sector == options.SectorFilter)
                        .Select(c => c.Symbol)
                        .ToListAsync(cancellationToken);
                    
                    query = query.Where(s => sectorSymbols.Contains(s));
                }

                // Optionally skip symbols that already have recent predictions
                if (options.SkipRecentPredictions)
                {
                    var cutoffTime = DateTime.Now.AddHours(-options.PredictionAgeThresholdHours);
                    var recentPredictionSymbols = await _context.StockPredictions
                        .AsNoTracking()
                        .Where(p => p.CreatedDate >= cutoffTime)
                        .Select(p => p.Symbol)
                        .Distinct()
                        .ToListAsync(cancellationToken);
                    
                    query = query.Where(s => !recentPredictionSymbols.Contains(s));
                }

                var symbols = await query.ToListAsync(cancellationToken);
                
                // Optionally limit the number of symbols
                if (options.MaxSymbolsToProcess.HasValue && options.MaxSymbolsToProcess.Value > 0)
                {
                    symbols = symbols.Take(options.MaxSymbolsToProcess.Value).ToList();
                }

                return symbols;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error getting symbols for processing");
                throw;
            }
        }

        /// <summary>
        /// Processes a batch of symbols concurrently with throttling.
        /// </summary>
        private async Task ProcessBatchAsync(
            string[] batch,
            BatchPredictionOptions options,
            IProgress<BatchPredictionProgress> progress,
            CancellationToken cancellationToken)
        {
            var tasks = batch.Select(symbol => ProcessSymbolWithThrottlingAsync(
                symbol, options, progress, cancellationToken));
            
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Processes a single symbol with throttling and retry logic.
        /// </summary>
        private async Task ProcessSymbolWithThrottlingAsync(
            string symbol,
            BatchPredictionOptions options,
            IProgress<BatchPredictionProgress> progress,
            CancellationToken cancellationToken)
        {
            await _throttleSemaphore.WaitAsync(cancellationToken);
            
            try
            {
                await ProcessSingleSymbolAsync(symbol, options, progress, cancellationToken);
            }
            finally
            {
                _throttleSemaphore.Release();
            }
        }

        /// <summary>
        /// Processes a single symbol with retry logic.
        /// </summary>
        private async Task ProcessSingleSymbolAsync(
            string symbol,
            BatchPredictionOptions options,
            IProgress<BatchPredictionProgress> progress,
            CancellationToken cancellationToken)
        {
            var sw = Stopwatch.StartNew();
            Exception lastException = null;

            for (int attempt = 1; attempt <= RetryAttempts; attempt++)
            {
                try
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Get technical indicators
                    var indicators = await _indicatorService.GetIndicatorsForPrediction(
                        symbol, options.Timeframe ?? "1day");

                    if (indicators == null || !indicators.Any())
                    {
                        _logger?.LogWarning("No indicators available for {Symbol}, skipping", symbol);
                        Interlocked.Increment(ref _failedPredictions);
                        break;
                    }

                    // Generate prediction
                    var prediction = await GeneratePredictionForSymbolAsync(
                        symbol, indicators, options.ModelType ?? DefaultModelType, cancellationToken);

                    if (prediction != null && !prediction.HasError)
                    {
                        // Save to database
                        await _predictionService.SavePredictionAsync(
                            prediction.ToPredictionModel(), cancellationToken);
                        
                        Interlocked.Increment(ref _successfulPredictions);
                        
                        _logger?.LogDebug("Successfully generated prediction for {Symbol} in {Ms}ms", 
                            symbol, sw.ElapsedMilliseconds);
                    }
                    else
                    {
                        Interlocked.Increment(ref _failedPredictions);
                        _logger?.LogWarning("Prediction failed for {Symbol}: {Error}", 
                            symbol, prediction?.Error ?? "Unknown error");
                    }

                    break; // Success, exit retry loop
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    
                    if (attempt < RetryAttempts)
                    {
                        _logger?.LogWarning(ex, "Error processing {Symbol}, attempt {Attempt}/{Max}", 
                            symbol, attempt, RetryAttempts);
                        await Task.Delay(1000 * attempt, cancellationToken); // Exponential backoff
                    }
                    else
                    {
                        _logger?.LogError(ex, "Failed to process {Symbol} after {Attempts} attempts", 
                            symbol, RetryAttempts);
                        Interlocked.Increment(ref _failedPredictions);
                    }
                }
            }

            // Update progress
            Interlocked.Increment(ref _processedSymbols);
            
            if (progress != null && _processedSymbols % 10 == 0) // Report every 10 symbols
            {
                progress.Report(new BatchPredictionProgress
                {
                    TotalSymbols = _totalSymbols,
                    ProcessedSymbols = _processedSymbols,
                    SuccessfulPredictions = _successfulPredictions,
                    FailedPredictions = _failedPredictions,
                    CurrentSymbol = symbol,
                    ElapsedTime = DateTime.Now - _batchStartTime,
                    EstimatedTimeRemaining = EstimateTimeRemaining()
                });
            }
        }

        /// <summary>
        /// Generates a prediction for a single symbol.
        /// </summary>
        private async Task<PredictionResult> GeneratePredictionForSymbolAsync(
            string symbol,
            Dictionary<string, double> indicators,
            string modelType,
            CancellationToken cancellationToken)
        {
            try
            {
                // Use the real-time inference service to generate prediction
                var result = await _inferenceService.ExecutePythonPredictionAsync(
                    symbol,
                    modelType,
                    cancellationToken: cancellationToken);

                if (result.Success && result.Prediction != null)
                {
                    // Enhance with feature weights (indicators)
                    if (result.Prediction.FeatureWeights == null)
                        result.Prediction.FeatureWeights = new Dictionary<string, double>();
                    
                    foreach (var indicator in indicators)
                    {
                        if (!result.Prediction.FeatureWeights.ContainsKey(indicator.Key))
                            result.Prediction.FeatureWeights[indicator.Key] = indicator.Value;
                    }
                    return result.Prediction;
                }

                return new PredictionResult
                {
                    Symbol = symbol,
                    Error = result.ErrorMessage ?? "Prediction generation failed"
                };
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating prediction for {Symbol}", symbol);
                return new PredictionResult
                {
                    Symbol = symbol,
                    Error = ex.Message
                };
            }
        }

        /// <summary>
        /// Estimates remaining time based on current progress.
        /// </summary>
        private TimeSpan EstimateTimeRemaining()
        {
            if (_processedSymbols == 0) return TimeSpan.Zero;
            
            var elapsed = DateTime.Now - _batchStartTime;
            var avgTimePerSymbol = elapsed.TotalSeconds / _processedSymbols;
            var remaining = _totalSymbols - _processedSymbols;
            
            return TimeSpan.FromSeconds(avgTimePerSymbol * remaining);
        }
    }

    /// <summary>
    /// Options for batch prediction generation.
    /// </summary>
    public class BatchPredictionOptions
    {
        /// <summary>
        /// Specific symbols to process. If null/empty, processes all symbols.
        /// </summary>
        public List<string> SymbolsToProcess { get; set; }

        /// <summary>
        /// Filter by sector (e.g., "Technology", "Healthcare").
        /// </summary>
        public string SectorFilter { get; set; }

        /// <summary>
        /// Maximum number of symbols to process. Null means no limit.
        /// </summary>
        public int? MaxSymbolsToProcess { get; set; }

        /// <summary>
        /// Skip symbols that already have predictions newer than this threshold.
        /// </summary>
        public bool SkipRecentPredictions { get; set; } = true;

        /// <summary>
        /// Threshold in hours for considering a prediction "recent".
        /// Default is 24 hours.
        /// </summary>
        public double PredictionAgeThresholdHours { get; set; } = 24;

        /// <summary>
        /// Timeframe for technical indicators (e.g., "1day", "1week").
        /// </summary>
        public string Timeframe { get; set; } = "1day";

        /// <summary>
        /// Model type to use for predictions (e.g., "auto", "pytorch", "tensorflow").
        /// </summary>
        public string ModelType { get; set; } = "auto";
    }

    /// <summary>
    /// Progress information for batch prediction generation.
    /// </summary>
    public class BatchPredictionProgress
    {
        public int TotalSymbols { get; set; }
        public int ProcessedSymbols { get; set; }
        public int SuccessfulPredictions { get; set; }
        public int FailedPredictions { get; set; }
        public string CurrentSymbol { get; set; }
        public TimeSpan ElapsedTime { get; set; }
        public TimeSpan EstimatedTimeRemaining { get; set; }

        public double ProgressPercentage => 
            TotalSymbols > 0 ? (double)ProcessedSymbols / TotalSymbols * 100 : 0;

        public double SuccessRate =>
            ProcessedSymbols > 0 ? (double)SuccessfulPredictions / ProcessedSymbols * 100 : 0;
    }

    /// <summary>
    /// Result of batch prediction generation.
    /// </summary>
    public class BatchPredictionResult
    {
        public bool Success { get; set; }
        public string Message { get; set; }
        public int TotalSymbols { get; set; }
        public int ProcessedSymbols { get; set; }
        public int SuccessfulPredictions { get; set; }
        public int FailedPredictions { get; set; }
        public TimeSpan Duration { get; set; }
        
        public double SuccessRate =>
            ProcessedSymbols > 0 ? (double)SuccessfulPredictions / ProcessedSymbols * 100 : 0;

        public double PredictionsPerSecond =>
            Duration.TotalSeconds > 0 ? SuccessfulPredictions / Duration.TotalSeconds : 0;
    }

    /// <summary>
    /// Extension methods for converting prediction results.
    /// </summary>
    public static class PredictionResultExtensions
    {
        public static PredictionModel ToPredictionModel(this PredictionResult result)
        {
            return new PredictionModel
            {
                Symbol = result.Symbol,
                PredictedAction = result.PredictedAction ?? result.Action,
                Confidence = result.Confidence,
                CurrentPrice = result.CurrentPrice,
                TargetPrice = result.TargetPrice,
                PredictionDate = result.PredictionDate,
                RiskScore = result.RiskScore,
                ValueAtRisk = result.ValueAtRisk,
                MaxDrawdown = result.MaxDrawdown,
                SharpeRatio = result.SharpeRatio,
                Indicators = result.FeatureWeights ?? new Dictionary<string, double>(),
                ModelType = result.ModelType,
                InferenceTimeMs = result.InferenceTimeMs,
                Error = result.Error
            };
        }
    }
}
