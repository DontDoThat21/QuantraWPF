#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debugpy Utilities for Remote Python Debugging

This module provides utilities for enabling remote debugging support in Python scripts
using debugpy. It allows attaching a debugger from Visual Studio or VS Code.

Usage:
    1. Set environment variable: DEBUGPY=1
    2. Run the Python script
    3. The script will wait at localhost:5678 for debugger attachment
    4. Attach debugger from Visual Studio (Debug -> Attach to Process -> Python remote)
"""

import os
import sys
import logging

logger = logging.getLogger('debugpy_utils')


def init_debugpy_if_enabled():
    """
    Initialize debugpy remote debugging if the DEBUGPY environment variable is set.
    
    This function:
    - Checks if DEBUGPY environment variable is set to '1' or 'true'
    - Imports debugpy if available
    - Configures debugpy to listen on localhost:5678
    - Waits for a debugger to attach before continuing
    
    Usage in scripts:
        from debugpy_utils import init_debugpy_if_enabled
        init_debugpy_if_enabled()
        # Rest of your code...
    """
    # Check if remote debugging is enabled via environment variable
    debugpy_enabled = os.environ.get('DEBUGPY', '').lower() in ('1', 'true', 'yes')
    
    if not debugpy_enabled:
        return
    
    try:
        import debugpy
        
        # Check if already initialized
        if debugpy.is_client_connected():
            logger.info("Debugpy already connected")
            return
        
        # Configure debugpy
        debugpy_host = os.environ.get('DEBUGPY_HOST', 'localhost')
        debugpy_port = int(os.environ.get('DEBUGPY_PORT', '5678'))
        
        logger.info(f"Enabling remote debugging on {debugpy_host}:{debugpy_port}")
        
        # Listen for debugger connection
        debugpy.listen((debugpy_host, debugpy_port))
        
        logger.info(f"Waiting for debugger to attach on {debugpy_host}:{debugpy_port}...")
        logger.info("Attach debugger from Visual Studio: Debug -> Attach to Process -> Python remote")
        
        # Wait for debugger to attach
        debugpy.wait_for_client()
        
        logger.info("Debugger attached successfully!")
        
        # Optional: Set a breakpoint at the start
        # debugpy.breakpoint()
        
    except ImportError:
        logger.warning("debugpy module not installed. Install with: pip install debugpy")
        logger.warning("Remote debugging will not be available")
    except Exception as e:
        logger.error(f"Error initializing debugpy: {e}")
        logger.warning("Continuing without remote debugging")


def enable_debugpy_breakpoint():
    """
    Set a breakpoint at the current location if debugpy is available.
    
    This is useful for setting conditional breakpoints in code.
    
    Example:
        if some_condition:
            enable_debugpy_breakpoint()
    """
    try:
        import debugpy
        if debugpy.is_client_connected():
            logger.info("Breakpoint hit - debugger should pause here")
            debugpy.breakpoint()
    except ImportError:
        logger.warning("debugpy not available for breakpoint")
    except Exception as e:
        logger.error(f"Error setting debugpy breakpoint: {e}")


if __name__ == "__main__":
    # Test the debugpy utilities
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Testing debugpy utilities")
    logger.info("Set DEBUGPY=1 environment variable to enable remote debugging")
    
    # This will only activate if DEBUGPY env var is set
    init_debugpy_if_enabled()
    
    logger.info("If debugger was attached, you should see messages above")
    logger.info("Script continuing normally...")
    
    # Simulate some work
    import time
    for i in range(5):
        logger.info(f"Working... {i+1}/5")
        time.sleep(1)
    
    logger.info("Test complete!")
