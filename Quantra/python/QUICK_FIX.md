# Quick Fix Guide: "Using basic feature creation"

## Problem
Your Python environment is missing dependencies, causing feature engineering to fail.

## Quick Fix (5 minutes)

### Step 1: Open Terminal in Python Directory
```batch
cd Quantra\python
```

### Step 2: Run ONE of these commands

**Option A: Automated Setup (Easiest)**
```batch
setup_environment.bat
```

**Option B: Manual Install**
```batch
pip install scipy>=1.7.0
pip install -r requirements.txt
```

**Option C: Quick Diagnostic First**
```powershell
.\diagnose_environment.ps1
```

### Step 3: Verify
```batch
python check_dependencies.py
```

You should see:
```
? OK                 scipy                         (version: 1.x.x)
? OK                 feature_engineering module
```

## Still Not Working?

### Check 1: Python Version
```batch
python --version
```
Must be 3.8 or later.

### Check 2: Are you in the right directory?
```batch
cd Quantra\python
dir
```
You should see: `requirements.txt`, `feature_engineering.py`, `stock_predictor.py`

### Check 3: Install scipy specifically
```batch
pip install scipy
```

### Check 4: Fresh Install
```batch
pip uninstall numpy scipy scikit-learn pandas -y
pip install -r requirements.txt
```

## How to Know It's Fixed

Run Quantra and check logs for:
```
? "Feature Engineering module is available"
? "Created new feature engineering pipeline with X features"
```

Instead of:
```
? "Using basic feature creation"
```

## Still Having Issues?

Read the full documentation:
- `README.md` - Complete setup guide
- `FIX_SUMMARY.md` - Detailed explanation of changes

Or run diagnostics:
```batch
python check_dependencies.py
.\diagnose_environment.ps1
```

## Need Help?

Common errors and solutions:

### "No module named scipy"
```batch
pip install scipy
```

### "ImportError: DLL load failed"
```batch
pip uninstall numpy -y
pip install numpy==1.23.5
```

### "cannot import name FeatureEngineer"
```batch
# Verify file exists
dir feature_engineering.py

# Test import
python -c "from feature_engineering import FeatureEngineer"
```

## Done!

After fixing, restart Quantra and run a prediction. Check the logs to confirm advanced feature engineering is working.
