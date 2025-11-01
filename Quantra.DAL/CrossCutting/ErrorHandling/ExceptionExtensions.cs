using System;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
//using System.Data.SQLite;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.CrossCutting.ErrorHandling
{
    /// <summary>
    /// Extension methods for exceptions.
    /// </summary>
    public static class ExceptionExtensions
    {
        // Known transient exception types
        private static readonly Type[] _transientExceptionTypes = new[]
        {
            typeof(TimeoutException),
            typeof(SocketException),
            typeof(WebException),
            typeof(HttpRequestException),
            typeof(IOException)//,
            //typeof(SQLiteException)
        };

        // Error codes that typically indicate transient issues for SQLite
        private static readonly int[] _transientSqliteErrorCodes = new[]
        {
            5,   // SQLite_BUSY
            6,   // SQLite_LOCKED
            261, // SQLite_BUSY_RECOVERY
            262, // SQLite_LOCKED_SHAREDCACHE
            517, // SQLite_BUSY_SNAPSHOT
            1555 // SQLite_BUSY_TIMEOUT
        };

        /// <summary>
        /// Determines whether the exception is likely transient (temporary) and can be retried.
        /// </summary>
        public static bool IsTransient(this Exception exception)
        {
            if (exception == null)
                return false;

            // Check if it's a known transient exception type
            var exceptionType = exception.GetType();
            if (_transientExceptionTypes.Contains(exceptionType))
            {
                // For SQLite exceptions, check the error code
                if (exceptionType == typeof(SQLiteException))
                {
                    var sqliteEx = (SQLiteException)exception;
                    return _transientSqliteErrorCodes.Contains(sqliteEx.ErrorCode);
                }
                
                return true;
            }

            // Check for transient HTTP status codes in nested WebException
            if (exception is WebException webEx && webEx.Response is HttpWebResponse response)
            {
                var statusCode = (int)response.StatusCode;
                return statusCode == 429 || statusCode >= 500 && statusCode < 600;
            }
            
            // Check for HttpRequestException that contains a transient status code
            if (exception is HttpRequestException httpEx)
            {
                if (httpEx.Message.Contains("503") || httpEx.Message.Contains("502") || 
                    httpEx.Message.Contains("500") || httpEx.Message.Contains("429"))
                {
                    return true;
                }
            }

            // Check the inner exception as well
            return exception.InnerException != null && IsTransient(exception.InnerException);
        }

        /// <summary>
        /// Gets a list of all nested exceptions including this one.
        /// </summary>
        public static IEnumerable<Exception> GetAllExceptions(this Exception exception)
        {
            if (exception == null)
                yield break;

            yield return exception;
            
            if (exception is AggregateException aggregateEx)
            {
                foreach (var innerEx in aggregateEx.InnerExceptions.SelectMany(e => e.GetAllExceptions()))
                {
                    yield return innerEx;
                }
            }
            else if (exception.InnerException != null)
            {
                foreach (var innerEx in exception.InnerException.GetAllExceptions())
                {
                    yield return innerEx;
                }
            }
        }
        
        /// <summary>
        /// Constructs a descriptive error message from all nested exceptions.
        /// </summary>
        public static string GetFullErrorMessage(this Exception exception)
        {
            if (exception == null)
                return string.Empty;
            
            var messages = new List<string> { exception.Message };
            
            if (exception is AggregateException aggregateEx)
            {
                messages.AddRange(
                    aggregateEx.InnerExceptions
                        .Select(e => e.GetFullErrorMessage())
                        .Where(msg => !string.IsNullOrEmpty(msg)));
            }
            else if (exception.InnerException != null)
            {
                messages.Add(exception.InnerException.GetFullErrorMessage());
            }
            
            return string.Join(" -> ", messages.Where(msg => !string.IsNullOrEmpty(msg)).Distinct());
        }
        
        /// <summary>
        /// Gets the most likely root cause from a chain of exceptions.
        /// </summary>
        public static Exception GetRootCause(this Exception exception)
        {
            if (exception == null)
                return null;
            
            var current = exception;
            
            while (current.InnerException != null)
            {
                current = current.InnerException;
            }
            
            return current;
        }
        
        /// <summary>
        /// Checks if the exception contains any exception of a specific type in its hierarchy.
        /// </summary>
        public static bool ContainsExceptionType<T>(this Exception exception) where T : Exception
        {
            return exception.GetAllExceptions().Any(e => e is T);
        }
        
        /// <summary>
        /// Checks if the exception contains any exception of a specific type in its hierarchy.
        /// </summary>
        public static bool ContainsExceptionType(this Exception exception, Type exceptionType)
        {
            return exception.GetAllExceptions().Any(e => exceptionType.IsInstanceOfType(e));
        }
    }
}