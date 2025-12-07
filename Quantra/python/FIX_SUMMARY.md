# Fix Summary: "Using basic feature creation" Issue

## Problem
The Python `stock_predictor.py` was falling back to "Using basic feature creation" instead of using the advanced feature engineering pipeline from `feature_engineering.py`.

## Root Cause
The issue was caused by missing Python dependencies, particularly:
1. **scipy** - Required by scikit-learn for feature selection
2. Potential numpy version incompatibilities
3. Silent import failures that weren't being logged properly

## Changes Made

### 1. Updated `requirements.txt`
- Added `scipy>=1.7.0` as an explicit dependency
- Pinned numpy to `>=1.21.0,<2.0.0` to avoid compatibility issues
- Added `ta-lib` as an optional technical analysis library
- Commented out GPU packages (cudf, cuml, cupy) as optional

**File:** `Quantra\python\requirements.txt`

### 2. Improved Error Handling in `stock_predictor.py`
- Added detailed error logging for import failures
- Shows specific ImportError messages
- Logs full traceback for debugging
- Better differentiation between pipeline loading vs creation

**Changes in:** `Quantra\python\stock_predictor.py`
- Line 50-62: Enhanced import error handling
- Line 135-157: Better pipeline creation/loading logging

### 3. Created Diagnostic Tools

#### `check_dependencies.py`
- Checks all Python dependencies
- Shows which modules are installed and their versions
- Identifies missing required packages
- Tests custom module imports

**Usage:**
```bash
cd Quantra\python
python check_dependencies.py
```

#### `diagnose_environment.ps1` (PowerShell)
- Quick diagnostic for Windows users
- Checks Python, pip, and packages
- Color-coded status indicators
- Provides installation commands for missing packages

**Usage:**
```powershell
cd Quantra\python
.\diagnose_environment.ps1
```

#### `setup_environment.bat` (Windows)
- Automated setup script
- Installs all dependencies
- Runs diagnostic check

**Usage:**
```batch
cd Quantra\python
setup_environment.bat
```

#### `setup_environment.sh` (Linux/Mac)
- Same as Windows script but for Unix systems
- Bash script for automated setup

**Usage:**
```bash
cd Quantra/python
chmod +x setup_environment.sh
./setup_environment.sh
```

### 4. Created Comprehensive README
- Explains all dependencies and their purposes
- Troubleshooting guide
- Usage examples
- Project structure documentation

**File:** `Quantra\python\README.md`

## How to Fix Your Environment

### Option 1: Automated Setup (Recommended)

**Windows:**
```batch
cd Quantra\python
setup_environment.bat
```

**Linux/Mac:**
```bash
cd Quantra/python
chmod +x setup_environment.sh
./setup_environment.sh
```

### Option 2: Manual Fix

1. **Install missing dependencies:**
   ```bash
   pip install scipy>=1.7.0
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python check_dependencies.py
   ```

3. **Test feature engineering:**
   ```bash
   python -c "from feature_engineering import FeatureEngineer; print('OK')"
   ```

### Option 3: Quick Diagnosis

**Windows (PowerShell):**
```powershell
.\diagnose_environment.ps1
```

**Any OS:**
```bash
python check_dependencies.py
```

## What to Look For

### Success Indicators
When feature engineering is working correctly, you should see in the logs:
```
INFO - Feature Engineering module is available
INFO - Using saved feature engineering pipeline with X features
```
or
```
INFO - Created new feature engineering pipeline with X features
```

### Failure Indicators
If feature engineering is NOT working, you'll see:
```
WARNING - Feature Engineering module is not available: <error>
WARNING - Using basic feature creation.
```

## Testing the Fix

### From C# Application
Run a prediction through the Quantra UI and check the logs for:
- "Feature Engineering module is available"
- "Using saved/Created new feature engineering pipeline with X features"

### From Python Directly
```python
from feature_engineering import build_default_pipeline
import pandas as pd

# Test data
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [99, 100, 101, 102, 103],
    'close': [103, 104, 105, 106, 107],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Build pipeline
pipeline = build_default_pipeline(feature_type='balanced')
features = pipeline.fit_transform(data)

print(f"? Generated {features.shape[1]} features successfully")
```

## Benefits of Fix

### 1. Better Diagnostics
- Clear error messages about what's missing
- Easy-to-use diagnostic tools
- Automated setup scripts

### 2. Improved Logging
- Detailed error information
- Traceback for debugging
- Clear status messages

### 3. Documentation
- Comprehensive README
- Troubleshooting guide
- Usage examples

### 4. Easier Maintenance
- Clear dependency list
- Version pinning to avoid conflicts
- Separation of required vs optional packages

## Common Issues After Fix

### Issue 1: Still seeing "Using basic feature creation"
**Solution:**
1. Run `python check_dependencies.py`
2. Check for missing scipy: `pip install scipy`
3. Reinstall all: `pip install --force-reinstall -r requirements.txt`

### Issue 2: Import errors
**Solution:**
1. Ensure you're in the correct directory (`Quantra\python`)
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Verify modules are in the same directory as script

### Issue 3: Version conflicts
**Solution:**
1. Create fresh virtual environment:
   ```bash
   python -m venv quantra_env
   quantra_env\Scripts\activate  # Windows
   # or
   source quantra_env/bin/activate  # Linux/Mac
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Feature Engineering Benefits

When working correctly, the advanced feature engineering provides:

### 1. More Features (~50-100 vs ~15)
- Basic: 15 simple features
- Balanced: 50+ technical indicators
- Full: 100+ comprehensive features

### 2. Better Predictions
- More sophisticated features
- Proper feature scaling
- Feature selection
- Dimensionality reduction

### 3. Consistent Pipeline
- Saved and reusable
- Same features for training and prediction
- Proper data preprocessing

## Next Steps

1. **Run diagnostic tool** to verify your environment
2. **Install missing dependencies** if any
3. **Test feature engineering** with provided examples
4. **Run prediction** through Quantra UI
5. **Check logs** to confirm advanced features are being used

## Support

If you continue to have issues:
1. Review `Quantra\python\README.md` for detailed troubleshooting
2. Check Python logs for error messages
3. Ensure Python 3.8+ is installed
4. Verify all files are in `Quantra\python` directory
5. Try creating a fresh virtual environment

## Files Modified/Created

### Modified:
- `Quantra\python\requirements.txt` - Added scipy, updated numpy pinning
- `Quantra\python\stock_predictor.py` - Improved error handling and logging

### Created:
- `Quantra\python\check_dependencies.py` - Dependency verification tool
- `Quantra\python\diagnose_environment.ps1` - PowerShell diagnostic
- `Quantra\python\setup_environment.bat` - Windows setup script
- `Quantra\python\setup_environment.sh` - Linux/Mac setup script
- `Quantra\python\README.md` - Comprehensive documentation
- `Quantra\python\FIX_SUMMARY.md` - This file

## Conclusion

The "Using basic feature creation" issue was caused by missing dependencies (primarily scipy) and insufficient error logging. The fix adds proper dependency management, diagnostic tools, and improved error reporting to make it easy to identify and resolve Python environment issues.
