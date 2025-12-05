"""
Utility module for debugpy remote debugging support in Quantra Python scripts.

When the DEBUGPY environment variable is set to "1", this module will:
1. Initialize debugpy on port 5678
2. Wait for a debugger connection (default: 30 seconds)

Usage:
    # At the top of your Python script:
    from debugpy_utils import init_debugpy_if_enabled
    
    # Call early in your main() function:
    def main():
        init_debugpy_if_enabled()
        # ... rest of your code
"""

import os
import sys
import time
import logging

# Configure logging for this module
logger = logging.getLogger('debugpy_utils')

# Default port for Visual Studio Python remote debugging
DEBUGPY_PORT = 5678

# Default timeout for waiting for debugger connection (seconds)
DEBUGPY_WAIT_TIMEOUT = 30


def init_debugpy_if_enabled(port: int = DEBUGPY_PORT, wait_timeout: int = DEBUGPY_WAIT_TIMEOUT) -> bool:
    """
    Initialize debugpy remote debugging if DEBUGPY environment variable is set.
    
    Args:
        port: Port to listen on for debugger connections (default: 5678)
        wait_timeout: Seconds to wait for debugger connection (default: 30)
    
    Returns:
        True if debugpy was initialized and debugger connected, False otherwise
    """
    debugpy_enabled = os.environ.get('DEBUGPY', '').strip()
    
    if debugpy_enabled != '1':
        return False
    
    try:
        import debugpy
    except ImportError:
        # debugpy not installed - print warning but don't fail
        print("DEBUGPY=1 set but debugpy is not installed. Install with: pip install debugpy", 
              file=sys.stderr)
        logger.warning("debugpy is not installed - remote debugging not available")
        return False
    
    try:
        # Start debugpy listener
        debugpy.listen(('0.0.0.0', port))
        print(f"[DEBUGPY] Listening for debugger on port {port}...", file=sys.stderr)
        print(f"[DEBUGPY] Waiting {wait_timeout} seconds for debugger connection...", file=sys.stderr)
        logger.info(f"debugpy listening on port {port}, waiting {wait_timeout}s for connection")
        
        # Wait for debugger connection with timeout
        start_time = time.time()
        while (time.time() - start_time) < wait_timeout:
            if debugpy.is_client_connected():
                print("[DEBUGPY] Debugger connected! Continuing execution...", file=sys.stderr)
                logger.info("Debugger connected successfully")
                return True
            
            # Print countdown every 5 seconds
            elapsed = int(time.time() - start_time)
            remaining = wait_timeout - elapsed
            if elapsed > 0 and elapsed % 5 == 0:
                print(f"[DEBUGPY] Waiting for debugger... {remaining}s remaining", file=sys.stderr)
            
            time.sleep(0.5)
        
        # Timeout reached - continue without debugger
        print(f"[DEBUGPY] Timeout reached. Continuing without debugger.", file=sys.stderr)
        logger.warning("Debugger connection timeout - continuing without debugger")
        return False
        
    except Exception as e:
        # Log error but don't fail - allow script to continue
        print(f"[DEBUGPY] Error initializing debugpy: {e}", file=sys.stderr)
        logger.error(f"Error initializing debugpy: {e}")
        return False


def is_debugpy_enabled() -> bool:
    """
    Check if DEBUGPY environment variable is set to enable remote debugging.
    
    Returns:
        True if DEBUGPY=1, False otherwise
    """
    return os.environ.get('DEBUGPY', '').strip() == '1'
