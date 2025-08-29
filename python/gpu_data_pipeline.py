"""
GPU-Accelerated Data Pipeline for Quantra Trading Platform

This module provides utilities for GPU-accelerated data preprocessing, feature engineering,
and dataset management for machine learning workflows in the trading platform.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd

# Import local modules
from gpu_utils import GPUManager, get_default_gpu_manager

# Set up logging
logger = logging.getLogger(__name__)


class GPUDataHandler:
    """
    Base class for GPU-accelerated data processing.
    
    Provides utilities for efficiently moving data between CPU and GPU,
    and performing common data processing operations.
    """
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize the GPU data handler.
        
        Args:
            gpu_manager: GPUManager instance for device handling
        """
        self.gpu_manager = gpu_manager or get_default_gpu_manager()
        self._cupy_available = False
        self._cudf_available = False
        self._rapids_available = False
        
        # Try to import GPU data libraries
        try:
            import cupy as cp
            self.cp = cp
            self._cupy_available = True
            logger.info("CuPy successfully imported for GPU array operations")
        except (ImportError, ModuleNotFoundError):
            logger.warning("CuPy not available. Some GPU operations will fall back to CPU.")
            self.cp = None
        
        try:
            import cudf
            self.cudf = cudf
            self._cudf_available = True
            logger.info("cuDF successfully imported for GPU DataFrame operations")
        except (ImportError, ModuleNotFoundError):
            logger.warning("cuDF not available. DataFrame operations will use pandas on CPU.")
            self.cudf = None
        
        try:
            import cuml
            self.cuml = cuml
            self._rapids_available = True
            logger.info("RAPIDS cuML successfully imported for GPU ML operations")
        except (ImportError, ModuleNotFoundError):
            logger.warning("RAPIDS cuML not available. ML operations will use scikit-learn on CPU.")
            self.cuml = None
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if any GPU acceleration library is available."""
        return self._cupy_available or self._cudf_available or self._rapids_available
    
    def to_gpu_array(self, data: np.ndarray) -> Any:
        """
        Convert NumPy array to GPU array (CuPy).
        
        Args:
            data: NumPy array
            
        Returns:
            CuPy array or original NumPy array if CuPy not available
        """
        if self._cupy_available:
            return self.cp.asarray(data)
        return data
    
    def to_cpu_array(self, data: Any) -> np.ndarray:
        """
        Convert GPU array to NumPy array.
        
        Args:
            data: CuPy array or NumPy array
            
        Returns:
            NumPy array
        """
        if self._cupy_available and isinstance(data, self.cp.ndarray):
            return data.get()
        return np.asarray(data)
    
    def to_gpu_dataframe(self, df: pd.DataFrame) -> Any:
        """
        Convert pandas DataFrame to GPU DataFrame (cuDF).
        
        Args:
            df: pandas DataFrame
            
        Returns:
            cuDF DataFrame or original pandas DataFrame if cuDF not available
        """
        if self._cudf_available:
            return self.cudf.DataFrame.from_pandas(df)
        return df
    
    def to_cpu_dataframe(self, df: Any) -> pd.DataFrame:
        """
        Convert GPU DataFrame to pandas DataFrame.
        
        Args:
            df: cuDF DataFrame or pandas DataFrame
            
        Returns:
            pandas DataFrame
        """
        if self._cudf_available and hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df
    
    def is_gpu_array(self, data: Any) -> bool:
        """Check if data is a GPU array."""
        if not self._cupy_available:
            return False
        return isinstance(data, self.cp.ndarray)
    
    def is_gpu_dataframe(self, df: Any) -> bool:
        """Check if df is a GPU DataFrame."""
        if not self._cudf_available:
            return False
        return type(df).__module__.startswith('cudf.')


class GPUDataPipeline:
    """
    Pipeline for GPU-accelerated data processing.
    
    Provides utilities for efficient data loading, preprocessing,
    feature engineering, and batch generation for ML models.
    """
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize the GPU data pipeline.
        
        Args:
            gpu_manager: GPUManager instance for device handling
        """
        self.gpu_manager = gpu_manager or get_default_gpu_manager()
        self.handler = GPUDataHandler(gpu_manager)
        self.processing_times = {}
        self._is_fitted = False
        self._scaler = None
        self._feature_engineering_steps = []
    
    def fit_transform(self, df: pd.DataFrame, 
                     scaler_type: str = 'standard',
                     features: List[str] = None) -> Any:
        """
        Fit the pipeline to the data and transform it.
        
        Args:
            df: Input DataFrame
            scaler_type: Type of scaling to use ('standard', 'minmax', 'robust', None)
            features: List of feature columns to use (None for all)
            
        Returns:
            Transformed data on GPU or CPU
        """
        start_time = time.time()
        
        # Select features
        if features is not None:
            df = df[features].copy()
        
        # Convert to GPU if available
        gpu_df = self.handler.to_gpu_dataframe(df)
        
        # Create scaler based on type
        if scaler_type is not None:
            if self.handler._rapids_available:
                # Use RAPIDS cuML for GPU-accelerated scaling
                if scaler_type == 'standard':
                    self._scaler = self.handler.cuml.preprocessing.StandardScaler()
                elif scaler_type == 'minmax':
                    self._scaler = self.handler.cuml.preprocessing.MinMaxScaler()
                elif scaler_type == 'robust':
                    self._scaler = self.handler.cuml.preprocessing.RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                
                # Fit and transform
                if self.handler.is_gpu_dataframe(gpu_df):
                    transformed = self._scaler.fit_transform(gpu_df)
                else:
                    # Convert to numpy first if not already a GPU DataFrame
                    transformed = self._scaler.fit_transform(gpu_df.values)
            else:
                # Fall back to scikit-learn
                from sklearn import preprocessing
                
                if scaler_type == 'standard':
                    self._scaler = preprocessing.StandardScaler()
                elif scaler_type == 'minmax':
                    self._scaler = preprocessing.MinMaxScaler()
                elif scaler_type == 'robust':
                    self._scaler = preprocessing.RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                
                # Convert back to CPU for scikit-learn
                cpu_df = self.handler.to_cpu_dataframe(gpu_df)
                transformed = self._scaler.fit_transform(cpu_df)
                
                # Convert back to GPU if possible
                transformed = self.handler.to_gpu_array(transformed)
        else:
            # No scaling
            if self.handler.is_gpu_dataframe(gpu_df):
                transformed = gpu_df
            else:
                # Convert to numpy array
                transformed = self.handler.to_gpu_array(gpu_df.values)
        
        self._is_fitted = True
        self.processing_times['fit_transform'] = time.time() - start_time
        
        return transformed
    
    def transform(self, df: pd.DataFrame, features: List[str] = None) -> Any:
        """
        Transform data using the fitted pipeline.
        
        Args:
            df: Input DataFrame
            features: List of feature columns to use (None for all)
            
        Returns:
            Transformed data
        """
        if not self._is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit_transform first.")
        
        start_time = time.time()
        
        # Select features
        if features is not None:
            df = df[features].copy()
        
        # Convert to GPU if available
        gpu_df = self.handler.to_gpu_dataframe(df)
        
        # Apply scaling if scaler exists
        if self._scaler is not None:
            if self.handler._rapids_available and hasattr(self._scaler, '__module__') and self._scaler.__module__.startswith('cuml'):
                # Use RAPIDS cuML scaler directly
                if self.handler.is_gpu_dataframe(gpu_df):
                    transformed = self._scaler.transform(gpu_df)
                else:
                    # Convert to numpy first if not already a GPU DataFrame
                    transformed = self._scaler.transform(gpu_df.values)
            else:
                # Fall back to scikit-learn scaler
                cpu_df = self.handler.to_cpu_dataframe(gpu_df)
                transformed = self._scaler.transform(cpu_df)
                
                # Convert back to GPU if possible
                transformed = self.handler.to_gpu_array(transformed)
        else:
            # No scaling
            if self.handler.is_gpu_dataframe(gpu_df):
                transformed = gpu_df
            else:
                # Convert to numpy array
                transformed = self.handler.to_gpu_array(gpu_df.values)
        
        self.processing_times['transform'] = time.time() - start_time
        return transformed
    
    def batch_generator(self, X: Any, y: Any = None, 
                       batch_size: int = 32,
                       shuffle: bool = True) -> Tuple:
        """
        Generate batches of data for model training/inference.
        
        Args:
            X: Features (array or DataFrame)
            y: Target values (array, DataFrame, or None)
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
            
        Yields:
            Tuple of (X_batch, y_batch) if y is provided, otherwise X_batch
        """
        # Ensure data is on GPU if available
        if not self.handler.is_gpu_array(X):
            X = self.handler.to_gpu_array(X)
        
        if y is not None and not self.handler.is_gpu_array(y):
            y = self.handler.to_gpu_array(y)
        
        # Get length of data
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            
            if y is not None:
                y_batch = y[batch_indices]
                yield X_batch, y_batch
            else:
                yield X_batch
    
    def add_feature_engineering_step(self, name: str, func: Callable) -> None:
        """
        Add a feature engineering step to the pipeline.
        
        Args:
            name: Name of the step
            func: Function to apply to the data
        """
        self._feature_engineering_steps.append((name, func))
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        start_time = time.time()
        
        # Convert to GPU if available
        gpu_df = self.handler.to_gpu_dataframe(df)
        
        # Apply each feature engineering step
        for name, func in self._feature_engineering_steps:
            step_start = time.time()
            gpu_df = func(gpu_df)
            self.processing_times[f'feature_{name}'] = time.time() - step_start
        
        self.processing_times['feature_engineering_total'] = time.time() - start_time
        return gpu_df
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the pipeline.
        
        Returns:
            Dictionary with processing times
        """
        return {
            'is_gpu_available': self.handler.is_gpu_available,
            'cupy_available': self.handler._cupy_available,
            'cudf_available': self.handler._cudf_available,
            'rapids_available': self.handler._rapids_available,
            'processing_times': self.processing_times
        }


# Common GPU-accelerated feature engineering functions

def calculate_moving_averages(df: Any, window_sizes: List[int] = [5, 10, 20], 
                            price_col: str = 'close') -> Any:
    """
    Calculate moving averages for the price column.
    
    Args:
        df: DataFrame (GPU or CPU)
        window_sizes: List of window sizes for moving averages
        price_col: Column name for price data
        
    Returns:
        DataFrame with added moving average columns
    """
    # Handle both cuDF and pandas DataFrames
    result = df.copy()
    
    for window in window_sizes:
        col_name = f'ma_{window}'
        if hasattr(df, 'rolling') and callable(df.rolling):
            # Both pandas and cuDF have rolling functionality
            result[col_name] = df[price_col].rolling(window=window).mean()
        else:
            # Fall back to numpy operations
            values = df[price_col].values
            ma_values = np.convolve(values, np.ones(window)/window, mode='valid')
            padding = np.array([np.nan] * (window - 1))
            ma_values = np.concatenate((padding, ma_values))
            result[col_name] = ma_values
    
    return result


def calculate_rsi(df: Any, window: int = 14, price_col: str = 'close') -> Any:
    """
    Calculate Relative Strength Index (RSI) for the price column.
    
    Args:
        df: DataFrame (GPU or CPU)
        window: Window size for RSI calculation
        price_col: Column name for price data
        
    Returns:
        DataFrame with added RSI column
    """
    result = df.copy()
    
    # Check for cuDF DataFrame
    is_cudf = type(df).__module__.startswith('cudf')
    
    if is_cudf:
        # Use cuDF/CuPy operations
        import cupy as cp
        delta = df[price_col].diff().fillna(0).values
        gain = cp.maximum(delta, 0)
        loss = cp.absolute(cp.minimum(delta, 0))
        
        avg_gain = cp.zeros_like(gain)
        avg_loss = cp.zeros_like(loss)
        
        # First average gain/loss
        avg_gain[window] = cp.mean(gain[:window+1])
        avg_loss[window] = cp.mean(loss[:window+1])
        
        # Calculate subsequent values
        for i in range(window+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Set NaN for the first window values
        rsi[:window] = cp.nan
        
        result['rsi'] = rsi
    else:
        # Fall back to pandas/numpy for CPU implementation
        delta = df[price_col].diff().fillna(0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        result['rsi'] = 100 - (100 / (1 + rs))
    
    return result


def calculate_technical_indicators(df: Any, 
                                  price_col: str = 'close',
                                  volume_col: Optional[str] = 'volume') -> Any:
    """
    Calculate common technical indicators for trading data.
    
    Args:
        df: DataFrame (GPU or CPU)
        price_col: Column name for price data
        volume_col: Column name for volume data (None if not available)
        
    Returns:
        DataFrame with added technical indicator columns
    """
    # Add moving averages (5, 10, 20 days)
    result = calculate_moving_averages(df, [5, 10, 20], price_col)
    
    # Add RSI
    result = calculate_rsi(result, 14, price_col)
    
    # Add Bollinger Bands (20-day, 2 standard deviations)
    ma_20 = result['ma_20']
    
    # Check for cuDF DataFrame
    is_cudf = type(df).__module__.startswith('cudf')
    
    if is_cudf:
        import cupy as cp
        # Calculate standard deviation using CuPy
        rolling_std = cp.zeros_like(ma_20.values)
        valid_indices = ~cp.isnan(ma_20.values)
        
        # For each valid index, calculate standard deviation of previous 20 values
        for i in range(20, len(ma_20)):
            rolling_std[i] = cp.std(df[price_col].values[i-20:i])
    else:
        # Use pandas rolling functionality
        rolling_std = df[price_col].rolling(window=20).std()
    
    result['bollinger_upper'] = ma_20 + (rolling_std * 2)
    result['bollinger_lower'] = ma_20 - (rolling_std * 2)
    
    # Add MACD
    result['macd'] = result['ma_12'] - result['ma_26'] if 'ma_12' in result.columns and 'ma_26' in result.columns else None
    
    # Add on-balance volume if volume data is available
    if volume_col is not None and volume_col in df.columns:
        if is_cudf:
            import cupy as cp
            # Calculate price change direction
            price_change = cp.diff(df[price_col].values)
            price_change[0] = 0  # Set first value to 0
            
            # Create OBV
            obv = cp.zeros_like(price_change)
            
            # OBV increases when price goes up, decreases when price goes down
            obv[1:] = cp.where(price_change[1:] > 0, 
                              df[volume_col].values[1:],
                              cp.where(price_change[1:] < 0,
                                      -df[volume_col].values[1:],
                                      0))
            
            # Cumulative sum
            result['obv'] = cp.cumsum(obv)
        else:
            # Pandas implementation
            price_change = df[price_col].diff()
            obv = df[volume_col] * np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
            result['obv'] = obv.cumsum()
    
    return result


# Example usage functions

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample stock price data for testing."""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    np.random.seed(42)
    
    # Generate random price data
    close = np.random.normal(loc=100, scale=10, size=n_samples).cumsum()
    close = np.abs(close) + 100  # Ensure positive prices
    
    # Add some random noise for other price points
    high = close * (1 + np.random.random(n_samples) * 0.03)
    low = close * (1 - np.random.random(n_samples) * 0.03)
    open_price = low + np.random.random(n_samples) * (high - low)
    
    # Generate trading volume
    volume = np.random.normal(loc=1000000, scale=200000, size=n_samples)
    volume = np.abs(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df


if __name__ == "__main__":
    # Set up logging for script execution
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample data
    logger.info("Creating sample data...")
    sample_data = create_sample_data(1000)
    logger.info(f"Sample data shape: {sample_data.shape}")
    
    # Create GPU data pipeline
    logger.info("Initializing GPU data pipeline...")
    pipeline = GPUDataPipeline()
    
    # Get GPU information
    metrics = pipeline.get_performance_metrics()
    logger.info(f"GPU available: {metrics['is_gpu_available']}")
    logger.info(f"CuPy available: {metrics['cupy_available']}")
    logger.info(f"cuDF available: {metrics['cudf_available']}")
    logger.info(f"RAPIDS available: {metrics['rapids_available']}")
    
    # Apply feature engineering
    logger.info("Adding feature engineering steps...")
    pipeline.add_feature_engineering_step("technical_indicators", 
                                         lambda df: calculate_technical_indicators(df))
    
    logger.info("Applying feature engineering...")
    start_time = time.time()
    enhanced_data = pipeline.apply_feature_engineering(sample_data)
    processing_time = time.time() - start_time
    logger.info(f"Feature engineering completed in {processing_time:.4f} seconds")
    
    # Prepare data for ML
    logger.info("Preparing data for machine learning...")
    
    # Define features to use
    features = ['close', 'ma_5', 'ma_10', 'ma_20', 'rsi']
    features = [f for f in features if f in enhanced_data.columns]
    
    # Create target variable (next day return)
    enhanced_data['next_day_return'] = enhanced_data['close'].pct_change(-1)
    
    # Drop NaN values
    enhanced_data = enhanced_data.dropna()
    
    # Split data into features and target
    X = enhanced_data[features]
    y = enhanced_data['next_day_return']
    
    # Scale data using the pipeline
    logger.info("Scaling data...")
    X_scaled = pipeline.fit_transform(X, scaler_type='standard')
    
    # Generate batches
    logger.info("Generating batches...")
    batch_size = 32
    
    # Convert y to GPU if X is on GPU
    if hasattr(X_scaled, 'device'):
        handler = GPUDataHandler()
        y_gpu = handler.to_gpu_array(y.values)
    else:
        y_gpu = y.values
    
    batch_gen = pipeline.batch_generator(X_scaled, y_gpu, batch_size=batch_size)
    
    # Process a few batches
    for i, (X_batch, y_batch) in enumerate(batch_gen):
        logger.info(f"Batch {i+1}: X shape={X_batch.shape}, y shape={y_batch.shape}")
        if i >= 2:  # Just show a few batches
            break
    
    # Report performance metrics
    logger.info("Performance metrics:")
    metrics = pipeline.get_performance_metrics()
    for key, value in metrics['processing_times'].items():
        logger.info(f"  {key}: {value:.4f} seconds")