using System;
using System.Collections.Generic;
using System.Diagnostics;
using Serilog;
using Serilog.Context;
using Serilog.Events;

namespace Quantra.CrossCutting.Logging
{
    /// <summary>
    /// Implementation of ILogger using Serilog.
    /// </summary>
    internal class SerilogLogger : ILogger
    {
        private readonly Serilog.ILogger _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="SerilogLogger"/> class.
        /// </summary>
        /// <param name="logger">The underlying Serilog logger.</param>
        public SerilogLogger(Serilog.ILogger logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <inheritdoc />
        public void Verbose(string messageTemplate, params object[] propertyValues)
        {
            _logger.Verbose(messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Debug(string messageTemplate, params object[] propertyValues)
        {
            _logger.Debug(messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Information(string messageTemplate, params object[] propertyValues)
        {
            _logger.Information(messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Warning(string messageTemplate, params object[] propertyValues)
        {
            _logger.Warning(messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Error(string messageTemplate, params object[] propertyValues)
        {
            _logger.Error(messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Error(Exception exception, string messageTemplate, params object[] propertyValues)
        {
            _logger.Error(exception, messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Fatal(string messageTemplate, params object[] propertyValues)
        {
            _logger.Fatal(messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public void Fatal(Exception exception, string messageTemplate, params object[] propertyValues)
        {
            _logger.Fatal(exception, messageTemplate, propertyValues);
        }

        /// <inheritdoc />
        public ILogger ForContext(string propertyName, object value)
        {
            var contextualLogger = _logger.ForContext(propertyName, value);
            return new SerilogLogger(contextualLogger);
        }

        /// <inheritdoc />
        public ILogger ForContext<T>()
        {
            var contextualLogger = _logger.ForContext<T>();
            return new SerilogLogger(contextualLogger);
        }

        /// <inheritdoc />
        public ILogger ForContext(Type source)
        {
            var contextualLogger = _logger.ForContext(source);
            return new SerilogLogger(contextualLogger);
        }

        /// <inheritdoc />
        public IDisposable BeginScope(string operationName, IDictionary<string, object> state = null)
        {
            var disposables = new CompositeDisposable();
            
            // Add operation name to the context
            disposables.Add(LogContext.PushProperty("OperationName", operationName));
            
            // Add any additional state properties
            if (state != null)
            {
                foreach (var item in state)
                {
                    disposables.Add(LogContext.PushProperty(item.Key, item.Value));
                }
            }
            
            // Log beginning of operation at Debug level
            _logger.Debug("Beginning operation {OperationName}", operationName);
            
            return disposables;
        }

        /// <inheritdoc />
        public IDisposable BeginTimedOperation(string operationName, string messageTemplate = "Operation {OperationName} completed in {ElapsedMilliseconds}ms")
        {
            return new TimedOperation(_logger, operationName, messageTemplate);
        }

        /// <summary>
        /// Helper class to manage multiple disposables as one unit.
        /// </summary>
        private class CompositeDisposable : IDisposable
        {
            private readonly List<IDisposable> _disposables = new List<IDisposable>();

            public void Add(IDisposable disposable)
            {
                _disposables.Add(disposable);
            }

            public void Dispose()
            {
                foreach (var disposable in _disposables)
                {
                    disposable.Dispose();
                }
                _disposables.Clear();
            }
        }

        /// <summary>
        /// Helper class to time operations and log the elapsed time on disposal.
        /// </summary>
        private class TimedOperation : IDisposable
        {
            private readonly Serilog.ILogger _logger;
            private readonly string _operationName;
            private readonly string _messageTemplate;
            private readonly Stopwatch _stopwatch;
            private readonly IDisposable _logContext;

            public TimedOperation(Serilog.ILogger logger, string operationName, string messageTemplate)
            {
                _logger = logger;
                _operationName = operationName;
                _messageTemplate = messageTemplate;
                _stopwatch = Stopwatch.StartNew();
                _logContext = LogContext.PushProperty("OperationName", operationName);
                
                // Log start of operation at Debug level
                _logger.Debug("Starting timed operation {OperationName}", operationName);
            }

            public void Dispose()
            {
                _stopwatch.Stop();
                long elapsedMs = _stopwatch.ElapsedMilliseconds;
                
                // Determine appropriate log level based on duration
                LogEventLevel level = elapsedMs switch
                {
                    > 5000 => LogEventLevel.Warning,  // Over 5 seconds is slow
                    > 1000 => LogEventLevel.Information, // Over 1 second is noteworthy
                    _ => LogEventLevel.Debug  // Default to debug level for normal operations
                };
                
                _logger.Write(level, _messageTemplate, _operationName, elapsedMs);
                _logContext.Dispose();
            }
        }
    }
}