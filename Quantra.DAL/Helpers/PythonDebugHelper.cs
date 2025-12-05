using System;
using System.Diagnostics;

namespace Quantra.DAL.Helpers
{
    /// <summary>
    /// Helper class for managing Python debugpy debugging sessions
    /// </summary>
    public static class PythonDebugHelper
    {
        private const string DebugEnabledEnvVar = "ENABLE_PYTHON_DEBUG";
        private const string DebugPortEnvVar = "DEBUGPY_PORT";
        private const int DefaultDebugPort = 5678;

        /// <summary>
        /// Enables Python debugging for the current process
        /// </summary>
        /// <param name="port">Port number for debugpy to listen on (default: 5678)</param>
        /// <param name="enableBreakpoints">Whether to enable built-in breakpoints in the Python script</param>
        public static void EnablePythonDebugging(int port = DefaultDebugPort, bool enableBreakpoints = true)
        {
            Environment.SetEnvironmentVariable(DebugEnabledEnvVar, "true");
            Environment.SetEnvironmentVariable(DebugPortEnvVar, port.ToString());
            
            if (enableBreakpoints)
            {
                Environment.SetEnvironmentVariable("ENABLE_PYTHON_BREAKPOINTS", "true");
            }

            Debug.WriteLine($"[PythonDebug] Enabled Python debugging on port {port}");
            Debug.WriteLine($"[PythonDebug] Attach debugger to localhost:{port} before the Python script runs");
        }

        /// <summary>
        /// Disables Python debugging for the current process
        /// </summary>
        public static void DisablePythonDebugging()
        {
            Environment.SetEnvironmentVariable(DebugEnabledEnvVar, "false");
            Environment.SetEnvironmentVariable(DebugPortEnvVar, null);
            Environment.SetEnvironmentVariable("ENABLE_PYTHON_BREAKPOINTS", "false");

            Debug.WriteLine("[PythonDebug] Disabled Python debugging");
        }

        /// <summary>
        /// Checks if Python debugging is currently enabled
        /// </summary>
        public static bool IsDebuggingEnabled()
        {
            var enabled = Environment.GetEnvironmentVariable(DebugEnabledEnvVar);
            return enabled?.ToLower() == "true";
        }

        /// <summary>
        /// Gets the current debug port
        /// </summary>
        public static int GetDebugPort()
        {
            var portStr = Environment.GetEnvironmentVariable(DebugPortEnvVar);
            if (int.TryParse(portStr, out int port))
            {
                return port;
            }
            return DefaultDebugPort;
        }

        /// <summary>
        /// Wraps a Python prediction call with automatic debugging setup
        /// </summary>
        /// <remarks>
        /// Only enables debugging in DEBUG builds
        /// </remarks>
        public static IDisposable AutoDebugScope(int port = DefaultDebugPort)
        {
            return new PythonDebugScope(port);
        }

        private class PythonDebugScope : IDisposable
        {
            private readonly bool _wasEnabled;

            public PythonDebugScope(int port)
            {
                _wasEnabled = IsDebuggingEnabled();
                
#if DEBUG
                if (!_wasEnabled)
                {
                    EnablePythonDebugging(port);
                }
#endif
            }

            public void Dispose()
            {
#if DEBUG
                if (!_wasEnabled)
                {
                    DisablePythonDebugging();
                }
#endif
            }
        }
    }
}
