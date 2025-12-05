# Python Debugging with debugpy

This guide explains how to debug the Python stock predictor script from Visual Studio.

## Prerequisites

1. Install debugpy in your Python environment:
   ```bash
   pip install debugpy
   ```

2. Install the Python extension in Visual Studio (if not already installed)

## Enable Debug Mode

### Option 1: Set Environment Variable (Recommended)

Before running your C# application, set the environment variable:

**Windows (PowerShell):**
```powershell
$env:ENABLE_PYTHON_DEBUG = "true"
$env:DEBUGPY_PORT = "5678"
```

**Windows (Command Prompt):**
```cmd
set ENABLE_PYTHON_DEBUG=true
set DEBUGPY_PORT=5678
```

**In C# Code (Alternative):**
Add this before calling the Python script:
```csharp
Environment.SetEnvironmentVariable("ENABLE_PYTHON_DEBUG", "true");
Environment.SetEnvironmentVariable("DEBUGPY_PORT", "5678");
```

### Option 2: Modify Python Script Temporarily

Change this line in `stock_predictor.py`:
```python
ENABLE_DEBUGPY = os.environ.get('ENABLE_PYTHON_DEBUG', 'false').lower() == 'true'
```

To:
```python
ENABLE_DEBUGPY = True  # Force enable for testing
```

## Debugging Steps

### From Visual Studio 2022

1. **Start Your C# Application**
   - Run your Quantra application normally
   - When it calls the Python script, it will pause and wait for debugger attachment

2. **Attach to Python Process**
   - Go to `Debug` ? `Attach to Process...`
   - Change "Connection type" to `Python remote (debugpy)`
   - Set "Connection target" to `localhost:5678` (or your custom port)
   - Click `Attach`

3. **Set Breakpoints**
   - Open `stock_predictor.py` in Visual Studio
   - Click in the left margin to set breakpoints at desired lines
   - The script already has breakpoints at:
     - `create_features()` - Start of feature creation
     - RSI calculation - Where NoneType errors commonly occur
     - `predict_stock()` - Main prediction function
     - Price calculation - Critical price conversion logic
     - `main()` - Script entry point

### From VS Code

1. **Create launch.json Configuration**
   Create `.vscode/launch.json`:
   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: Remote Attach",
               "type": "python",
               "request": "attach",
               "connect": {
                   "host": "localhost",
                   "port": 5678
               },
               "pathMappings": [
                   {
                       "localRoot": "${workspaceFolder}/Quantra/python",
                       "remoteRoot": "."
                   }
               ]
           }
       ]
   }
   ```

2. **Start Debugging**
   - Run your C# application
   - In VS Code, press F5 and select "Python: Remote Attach"
   - Set breakpoints as needed

## Built-in Breakpoint Locations

The script includes conditional breakpoints at strategic locations:

1. **`create_features()`** - Line ~73
   - Examines input data structure
   - Validates DataFrame conversion

2. **RSI Calculation** - Line ~148
   - Most common location for NoneType errors
   - Inspect `gain`, `loss`, and `delta` values

3. **`predict_stock()`** - Line ~1147
   - Main prediction entry point
   - Review incoming features dictionary

4. **Price Calculation** - Line ~1247
   - Verify `predicted_change` is not None
   - Check `current_price` and `target_price`

5. **`main()`** - Line ~1395
   - Script entry point
   - Verify command-line arguments and file paths

## Disabling Breakpoints

To disable the built-in breakpoints while keeping debugpy enabled:

Comment out the breakpoint calls:
```python
# if ENABLE_DEBUGPY:
#     import debugpy
#     debugpy.breakpoint()
```

Or set a different environment variable:
```python
ENABLE_BREAKPOINTS = os.environ.get('ENABLE_PYTHON_BREAKPOINTS', 'false').lower() == 'true'

# Then use:
if ENABLE_DEBUGPY and ENABLE_BREAKPOINTS:
    import debugpy
    debugpy.breakpoint()
```

## Troubleshooting

### "Module not found: debugpy"
```bash
pip install debugpy
```

### "Connection refused"
- Check that the Python script is running and waiting for debugger
- Verify the port number matches (default: 5678)
- Check firewall settings

### "Cannot attach to process"
- Ensure the C# app is actually calling the Python script
- Check that `ENABLE_PYTHON_DEBUG` environment variable is set
- Look for "Waiting for debugger to attach..." message in stderr

### Script Times Out
If the script waits too long for debugger attachment:
- Comment out `debugpy.wait_for_client()` line
- The script will continue without waiting, but you can still attach later

## Performance Impact

**Important:** Disable debug mode in production!

Debug mode adds overhead:
- ~500ms startup delay (waiting for debugger)
- Breakpoints pause execution
- Additional memory usage

Always set `ENABLE_PYTHON_DEBUG=false` or remove the environment variable for production use.

## Example C# Integration

```csharp
// In your PredictionAnalysisService or similar
public async Task<PredictionResult> GetPredictionWithDebugAsync(Dictionary<string, double> features)
{
    // Enable Python debugging (only for development)
    #if DEBUG
    Environment.SetEnvironmentVariable("ENABLE_PYTHON_DEBUG", "true");
    Environment.SetEnvironmentVariable("DEBUGPY_PORT", "5678");
    #endif
    
    try
    {
        var result = await PythonStockPredictor.PredictAsync(features);
        return result;
    }
    finally
    {
        // Clean up
        #if DEBUG
        Environment.SetEnvironmentVariable("ENABLE_PYTHON_DEBUG", "false");
        #endif
    }
}
```

## Tips

1. **Use Conditional Breakpoints**: Set conditions in Visual Studio to break only when specific values occur
2. **Watch Variables**: Add `gain`, `loss`, `delta`, `predicted_change` to Watch window
3. **Log to File**: For post-mortem debugging, log values to a file instead of using breakpoints
4. **Remote Debugging**: Change `0.0.0.0` to specific IP for remote debugging across machines
