#!/bin/bash
#
# Setup script for Quantra GPU Environment
# This script installs and configures all necessary components for GPU acceleration
#

set -e  # Exit on error

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="setup_logs"
mkdir -p $LOG_DIR
MAIN_LOG="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"

# Function for logging
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo -e "$msg" | tee -a "$MAIN_LOG"
}

# Function for section headers
section() {
  echo -e "${BLUE}====================================================${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}====================================================${NC}"
  log "SECTION: $1"
}

# Function to check if command exists
command_exists() {
  command -v "$1" &> /dev/null
}

# Function to check if package is installed (for apt-based systems)
is_package_installed() {
  if command_exists dpkg; then
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "ok installed"
  else
    return 1  # Not apt-based
  fi
}

# Function to get NVIDIA driver version
get_nvidia_driver_version() {
  if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1
  else
    echo "Not installed"
  fi
}

# Function to get CUDA version
get_cuda_version() {
  if [ -x "$(command -v nvcc)" ]; then
    nvcc --version | grep "release" | awk '{print $6}' | cut -c2-
  elif [ -f "/usr/local/cuda/version.txt" ]; then
    cat /usr/local/cuda/version.txt | awk '{print $3}'
  else
    echo "Not installed"
  fi
}

# Check if running with sudo/root
check_sudo() {
  if [ "$EUID" -ne 0 ]; then
    log "${RED}Please run as root or with sudo${NC}"
    exit 1
  fi
}

# Detect OS
detect_os() {
  section "Detecting Operating System"
  
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    log "Detected OS: $OS $VER"
  elif [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
    log "Detected OS: $OS $VER"
  elif [ -f /etc/debian_version ]; then
    OS="Debian"
    VER=$(cat /etc/debian_version)
    log "Detected OS: $OS $VER"
  elif [ -f /etc/redhat-release ]; then
    OS="RedHat"
    VER=$(cat /etc/redhat-release | cut -d ' ' -f 4)
    log "Detected OS: $OS $VER"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="MacOS"
    VER=$(sw_vers -productVersion)
    log "Detected OS: $OS $VER"
  else
    OS="Unknown"
    VER="Unknown"
    log "${YELLOW}Unknown OS, proceeding with caution${NC}"
  fi
  
  # Check if WSL
  if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1
    log "Detected WSL environment"
  else
    IS_WSL=0
  fi
  
  # Export variables
  export OS
  export VER
  export IS_WSL
}

# Check GPU availability
check_gpu() {
  section "Checking GPU Availability"
  
  if command_exists nvidia-smi; then
    log "${GREEN}NVIDIA driver is installed${NC}"
    nvidia-smi | tee -a "$MAIN_LOG"
    DRIVER_VER=$(get_nvidia_driver_version)
    log "Driver version: $DRIVER_VER"
    
    # Check CUDA availability
    CUDA_VER=$(get_cuda_version)
    if [ "$CUDA_VER" != "Not installed" ]; then
      log "${GREEN}CUDA is installed: $CUDA_VER${NC}"
    else
      log "${YELLOW}CUDA is not installed or not in PATH${NC}"
    fi
  else
    log "${RED}NVIDIA driver not found. No GPU detected.${NC}"
    log "${YELLOW}Proceeding with CPU-only installation.${NC}"
    GPU_AVAILABLE=0
  fi
}

# Install system dependencies
install_system_deps() {
  section "Installing System Dependencies"
  
  case $OS in
    "Ubuntu"|"Debian")
      log "Updating package lists..."
      apt-get update -y >> "$MAIN_LOG" 2>&1
      
      log "Installing build tools and dependencies..."
      apt-get install -y build-essential cmake pkg-config \
                         python3-dev python3-pip python3-venv \
                         git wget curl >> "$MAIN_LOG" 2>&1
                         
      # Install additional tools for GPU
      if [ "$GPU_AVAILABLE" -eq 1 ]; then
        log "Installing GPU-related system packages..."
        apt-get install -y nvidia-cuda-toolkit nvidia-cuda-dev \
                           libcudnn8 libcudnn8-dev >> "$MAIN_LOG" 2>&1 || \
        log "${YELLOW}Could not install some CUDA packages. You may need to install them manually.${NC}"
      fi
      ;;
    
    "CentOS"|"RedHat"|"Fedora")
      log "Updating package lists..."
      yum update -y >> "$MAIN_LOG" 2>&1
      
      log "Installing build tools and dependencies..."
      yum groupinstall -y "Development Tools" >> "$MAIN_LOG" 2>&1
      yum install -y cmake python3-devel python3-pip git wget curl >> "$MAIN_LOG" 2>&1
      ;;
    
    "MacOS")
      if command_exists brew; then
        log "Updating Homebrew..."
        brew update >> "$MAIN_LOG" 2>&1
        
        log "Installing dependencies..."
        brew install cmake python git wget >> "$MAIN_LOG" 2>&1
      else
        log "${YELLOW}Homebrew not found. Please install Homebrew first:${NC}"
        log "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
      fi
      ;;
    
    *)
      log "${YELLOW}Unsupported OS for automatic dependency installation.${NC}"
      log "Please install the following packages manually:"
      log "  - Python 3.8 or higher"
      log "  - pip"
      log "  - git"
      log "  - cmake"
      log "  - CUDA Toolkit (if GPU is available)"
      ;;
  esac
}

# Set up Python virtual environment
setup_virtualenv() {
  section "Setting up Python Virtual Environment"
  
  VENV_DIR="venv"
  
  # Create virtual environment if it doesn't exist
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" >> "$MAIN_LOG" 2>&1
    log "${GREEN}Virtual environment created at $VENV_DIR${NC}"
  else
    log "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
  fi
  
  # Source the virtual environment
  log "Activating virtual environment..."
  source "$VENV_DIR/bin/activate" >> "$MAIN_LOG" 2>&1
  
  # Upgrade pip
  log "Upgrading pip..."
  pip install --upgrade pip >> "$MAIN_LOG" 2>&1
}

# Install Python dependencies
install_python_deps() {
  section "Installing Python Dependencies"
  
  # Install basic dependencies
  log "Installing basic Python packages..."
  pip install numpy pandas scikit-learn matplotlib >> "$MAIN_LOG" 2>&1
  
  # Install ML frameworks
  log "Installing machine learning frameworks..."
  
  # TensorFlow (CPU or GPU version based on availability)
  if [ "$GPU_AVAILABLE" -eq 1 ]; then
    log "Installing TensorFlow with GPU support..."
    pip install tensorflow >> "$MAIN_LOG" 2>&1
  else
    log "Installing TensorFlow (CPU)..."
    pip install tensorflow-cpu >> "$MAIN_LOG" 2>&1
  fi
  
  # PyTorch (CPU or GPU version based on availability)
  if [ "$GPU_AVAILABLE" -eq 1 ]; then
    # Install PyTorch with CUDA support
    CUDA_VER=$(get_cuda_version)
    
    case $CUDA_VER in
      11.*)
        log "Installing PyTorch with CUDA 11.x support..."
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 >> "$MAIN_LOG" 2>&1
        ;;
      10.*)
        log "Installing PyTorch with CUDA 10.x support..."
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu102 >> "$MAIN_LOG" 2>&1
        ;;
      *)
        log "${YELLOW}Unknown CUDA version: $CUDA_VER${NC}"
        log "Installing PyTorch with latest CUDA support..."
        pip install torch torchvision torchaudio >> "$MAIN_LOG" 2>&1
        ;;
    esac
  else
    log "Installing PyTorch (CPU)..."
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu >> "$MAIN_LOG" 2>&1
  fi
  
  # Install other ML/DL packages
  log "Installing other ML libraries..."
  pip install optuna plotly hmmlearn >> "$MAIN_LOG" 2>&1
  
  # Install GPU libraries if available
  if [ "$GPU_AVAILABLE" -eq 1 ]; then
    log "Installing GPU acceleration libraries..."
    
    # Try to install RAPIDS libraries
    log "Attempting to install RAPIDS libraries (cuDF, cuML)..."
    pip install cudf-cu11 cuml-cu11 --extra-index-url https://pypi.anaconda.org/rapidsai/label/stable >> "$MAIN_LOG" 2>&1 || \
    log "${YELLOW}Failed to install RAPIDS libraries. This is optional and may need manual installation.${NC}"
    
    # Try to install CuPy
    log "Installing CuPy..."
    pip install cupy-cuda11x >> "$MAIN_LOG" 2>&1 || \
    log "${YELLOW}Failed to install CuPy. This is optional and may need manual installation.${NC}"
    
    # Install pynvml for GPU monitoring
    log "Installing pynvml for GPU monitoring..."
    pip install pynvml >> "$MAIN_LOG" 2>&1
  fi
  
  # Install other Quantra requirements
  log "Installing other Quantra requirements..."
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt >> "$MAIN_LOG" 2>&1
  fi
}

# Verify installation
verify_installation() {
  section "Verifying Installation"
  
  # Create temporary test script
  TEST_SCRIPT="$LOG_DIR/gpu_test.py"
  
  cat > "$TEST_SCRIPT" << 'EOF'
import sys
print(f"Python version: {sys.version}")

# Test NumPy
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not installed")

# Test pandas
try:
    import pandas as pd
    print(f"pandas version: {pd.__version__}")
except ImportError:
    print("pandas not installed")

# Test scikit-learn
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn not installed")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPUs: {[g.name for g in gpus] if gpus else 'None'}")
except ImportError:
    print("TensorFlow not installed")

# Test PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")

# Test CuPy
try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    try:
        a = cp.array([1, 2, 3])
        print("CuPy GPU array creation: Success")
    except Exception as e:
        print(f"CuPy GPU array creation: Failed - {e}")
except ImportError:
    print("CuPy not installed")

# Test RAPIDS (cuDF)
try:
    import cudf
    print(f"cuDF version: {cudf.__version__}")
except ImportError:
    print("cuDF not installed")

# Test pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"pynvml detected {device_count} GPU(s)")
    if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"GPU 0: {name.decode('utf-8')}")
    pynvml.nvmlShutdown()
except ImportError:
    print("pynvml not installed")
except Exception as e:
    print(f"pynvml error: {e}")

print("Installation verification complete")
EOF
  
  log "Running verification script..."
  python "$TEST_SCRIPT" | tee -a "$MAIN_LOG"
}

# Create a configuration
create_config() {
  section "Creating Configuration"
  
  CONFIG_DIR="."
  CONFIG_FILE="$CONFIG_DIR/gpu_config.yaml"
  
  # Only create if it doesn't exist
  if [ -f "$CONFIG_FILE" ]; then
    log "${YELLOW}Configuration file already exists at $CONFIG_FILE${NC}"
    log "Not overwriting existing configuration."
    return
  fi
  
  log "Creating GPU configuration at $CONFIG_FILE"
  
  # Add content to the config file based on detection results
  cat > "$CONFIG_FILE" << EOF
# GPU Configuration for Quantra Trading Platform (Auto-generated)
# Created on $(date)

# General GPU settings
gpu:
  enabled: $([ "$GPU_AVAILABLE" -eq 1 ] && echo "true" || echo "false")
  device_id: 0
  memory_growth: true
  memory_limit: 0.8
  fallback_to_cpu: true
  precision: "float32"

# Framework-specific settings
frameworks:
  pytorch:
    enabled: $([ "$GPU_AVAILABLE" -eq 1 ] && echo "true" || echo "false")
    cuda_visible_devices: "0"
  
  tensorflow:
    enabled: $([ "$GPU_AVAILABLE" -eq 1 ] && echo "true" || echo "false")
    cuda_visible_devices: "0"
    xla_compilation: true
    mixed_precision: true

# Detected environment
environment:
  os: "$OS"
  os_version: "$VER"
  is_wsl: $([ "$IS_WSL" -eq 1 ] && echo "true" || echo "false")
  driver_version: "$(get_nvidia_driver_version)"
  cuda_version: "$(get_cuda_version)"
EOF

  log "${GREEN}Configuration file created successfully${NC}"
}

# Print summary
print_summary() {
  section "Installation Summary"
  
  log "${GREEN}Quantra GPU setup completed!${NC}"
  log "Summary:"
  log "  - OS: $OS $VER"
  
  if [ "$GPU_AVAILABLE" -eq 1 ]; then
    log "  - GPU: Available"
    log "  - NVIDIA Driver: $(get_nvidia_driver_version)"
    log "  - CUDA Version: $(get_cuda_version)"
  else
    log "  - GPU: Not available or not configured"
    log "  - Using CPU-only mode"
  fi
  
  log "  - Configuration: gpu_config.yaml"
  log "  - Log file: $MAIN_LOG"
  
  log ""
  log "${BLUE}Next steps:${NC}"
  log "  1. Review the configuration file and adjust if needed"
  log "  2. Make sure to activate the virtual environment before running: source venv/bin/activate"
  log "  3. Check gpu_utils.py for usage examples and utilities"
  log "  4. Run a test script to verify GPU functionality"
}

# Main function
main() {
  echo -e "${BLUE}=================================================${NC}"
  echo -e "${BLUE}   Quantra Trading Platform GPU Setup Script     ${NC}"
  echo -e "${BLUE}=================================================${NC}"
  
  # Record start time
  START_TIME=$(date +%s)
  
  log "Starting setup at $(date)"
  
  # Detect OS
  detect_os
  
  # Check if sudo is needed (skip on MacOS)
  if [[ "$OS" != "MacOS" ]]; then
    check_sudo
  fi
  
  # Check GPU availability
  GPU_AVAILABLE=0
  if command_exists nvidia-smi; then
    GPU_AVAILABLE=1
  fi
  
  # Check for GPU
  check_gpu
  
  # Install system dependencies
  install_system_deps
  
  # Setup Python environment
  setup_virtualenv
  
  # Install Python packages
  install_python_deps
  
  # Verify installation
  verify_installation
  
  # Create configuration
  create_config
  
  # Calculate execution time
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  log "Setup completed in ${DURATION} seconds"
  
  # Print summary
  print_summary
}

# Run main function
main