"""
Migration Guide for Transitioning CPU Code to GPU-Accelerated Code

This module provides utilities and examples for migrating existing CPU-based
machine learning code in Quantra to GPU-accelerated implementations using
the GPU acceleration utilities.
"""

import logging
import inspect
import textwrap
import time
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np

# Import local modules
from gpu_utils import GPUManager, get_default_gpu_manager, is_gpu_available, get_gpu_info
from gpu_models import GPUModelBase, PyTorchGPUModel, TensorFlowGPUModel, create_gpu_model
from gpu_data_pipeline import GPUDataPipeline

# Set up logging
logger = logging.getLogger(__name__)


class CodeConverter:
    """
    Helper class for converting CPU code snippets to GPU-accelerated code.
    
    Provides utilities for analyzing existing code and suggesting GPU
    alternatives for common patterns.
    """
    
    def __init__(self):
        """Initialize the code converter."""
        # Initialize common patterns and their GPU alternatives
        self.numpy_to_cupy_patterns = {
            'np.array(': 'cp.array(',
            'np.zeros(': 'cp.zeros(',
            'np.ones(': 'cp.ones(',
            'np.random.rand(': 'cp.random.rand(',
            'np.random.randn(': 'cp.random.randn(',
            'np.arange(': 'cp.arange(',
            'np.linspace(': 'cp.linspace(',
            'np.dot(': 'cp.dot(',
            'np.matmul(': 'cp.matmul(',
            'np.mean(': 'cp.mean(',
            'np.std(': 'cp.std(',
            'np.sum(': 'cp.sum(',
            'np.max(': 'cp.max(',
            'np.min(': 'cp.min(',
            'np.argmax(': 'cp.argmax(',
            'np.argmin(': 'cp.argmin(',
            'np.concatenate(': 'cp.concatenate(',
            'np.vstack(': 'cp.vstack(',
            'np.hstack(': 'cp.hstack(',
            'np.split(': 'cp.split(',
            'np.reshape(': 'cp.reshape(',
            'np.transpose(': 'cp.transpose(',
            'np.linalg.': 'cp.linalg.',
            'np.fft.': 'cp.fft.',
            'np.exp(': 'cp.exp(',
            'np.log(': 'cp.log(',
            'np.sin(': 'cp.sin(',
            'np.cos(': 'cp.cos(',
        }
        
        self.pandas_to_cudf_patterns = {
            'pd.DataFrame(': 'cudf.DataFrame(',
            'pd.Series(': 'cudf.Series(',
            'pd.read_csv(': 'cudf.read_csv(',
            'pd.read_parquet(': 'cudf.read_parquet(',
        }
        
        self.sklearn_to_cuml_patterns = {
            'sklearn.preprocessing.StandardScaler': 'cuml.preprocessing.StandardScaler',
            'sklearn.preprocessing.MinMaxScaler': 'cuml.preprocessing.MinMaxScaler',
            'sklearn.preprocessing.RobustScaler': 'cuml.preprocessing.RobustScaler',
            'sklearn.decomposition.PCA': 'cuml.decomposition.PCA', 
            'sklearn.manifold.TSNE': 'cuml.manifold.TSNE',
            'sklearn.cluster.KMeans': 'cuml.cluster.KMeans',
            'sklearn.cluster.DBSCAN': 'cuml.cluster.DBSCAN',
            'sklearn.neighbors.KNeighborsClassifier': 'cuml.neighbors.KNeighborsClassifier',
            'sklearn.neighbors.KNeighborsRegressor': 'cuml.neighbors.KNeighborsRegressor',
            'sklearn.ensemble.RandomForestClassifier': 'cuml.ensemble.RandomForestClassifier',
            'sklearn.ensemble.RandomForestRegressor': 'cuml.ensemble.RandomForestRegressor',
            'sklearn.linear_model.LinearRegression': 'cuml.linear_model.LinearRegression',
            'sklearn.linear_model.LogisticRegression': 'cuml.linear_model.LogisticRegression',
            'sklearn.svm.SVC': 'cuml.svm.SVC',
            'sklearn.svm.SVR': 'cuml.svm.SVR',
        }
        
        self.required_imports = {
            'numpy_to_cupy': 'import cupy as cp',
            'pandas_to_cudf': 'import cudf',
            'sklearn_to_cuml': 'import cuml',
            'torch_cpu_to_gpu': [
                'import torch',
                'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
            ],
            'tensorflow_cpu_to_gpu': [
                'import tensorflow as tf',
                'gpus = tf.config.list_physical_devices("GPU")',
                'if gpus:',
                '    for gpu in gpus:',
                '        tf.config.experimental.set_memory_growth(gpu, True)'
            ]
        }
    
    def suggest_numpy_to_cupy(self, code: str) -> str:
        """
        Convert NumPy code to CuPy equivalent.
        
        Args:
            code: Python code using NumPy
            
        Returns:
            Equivalent code using CuPy
        """
        gpu_code = code
        
        # Add required import
        if 'import numpy as np' in gpu_code and 'import cupy as cp' not in gpu_code:
            gpu_code = self.required_imports['numpy_to_cupy'] + '\n' + gpu_code
        
        # Replace patterns
        for np_pattern, cp_pattern in self.numpy_to_cupy_patterns.items():
            gpu_code = gpu_code.replace(np_pattern, cp_pattern)
        
        # Add code to move data back to CPU when needed
        if '.get()' not in gpu_code and any(p in gpu_code for p in ['cp.array', 'cp.zeros', 'cp.ones']):
            gpu_code += '\n# Move data back to CPU when needed:\n# cpu_array = gpu_array.get()'
        
        return gpu_code
    
    def suggest_pandas_to_cudf(self, code: str) -> str:
        """
        Convert pandas code to cuDF equivalent.
        
        Args:
            code: Python code using pandas
            
        Returns:
            Equivalent code using cuDF
        """
        gpu_code = code
        
        # Add required import
        if 'import pandas as pd' in gpu_code and 'import cudf' not in gpu_code:
            gpu_code = self.required_imports['pandas_to_cudf'] + '\n' + gpu_code
        
        # Replace patterns
        for pd_pattern, cudf_pattern in self.pandas_to_cudf_patterns.items():
            gpu_code = gpu_code.replace(pd_pattern, cudf_pattern)
        
        # Add code to move data back to pandas when needed
        if '.to_pandas()' not in gpu_code and any(p in gpu_code for p in ['cudf.DataFrame', 'cudf.Series']):
            gpu_code += '\n# Move data back to pandas when needed:\n# pandas_df = cudf_df.to_pandas()'
        
        return gpu_code
    
    def suggest_sklearn_to_cuml(self, code: str) -> str:
        """
        Convert scikit-learn code to RAPIDS cuML equivalent.
        
        Args:
            code: Python code using scikit-learn
            
        Returns:
            Equivalent code using cuML
        """
        gpu_code = code
        
        # Add required import
        if 'import sklearn' in gpu_code and 'import cuml' not in gpu_code:
            gpu_code = self.required_imports['sklearn_to_cuml'] + '\n' + gpu_code
        
        # Replace patterns
        for sklearn_pattern, cuml_pattern in self.sklearn_to_cuml_patterns.items():
            gpu_code = gpu_code.replace(sklearn_pattern, cuml_pattern)
        
        # Add user guide reference
        if 'cuml.' in gpu_code:
            gpu_code += '\n# Note: cuML API is similar to scikit-learn but may have slight differences\n'
            gpu_code += '# See RAPIDS documentation: https://docs.rapids.ai/api/cuml/stable/'
        
        return gpu_code
    
    def suggest_torch_cpu_to_gpu(self, code: str) -> str:
        """
        Convert PyTorch CPU code to GPU equivalent.
        
        Args:
            code: Python code using PyTorch on CPU
            
        Returns:
            Equivalent code using PyTorch on GPU
        """
        gpu_code = code
        
        # Add device setup if not present
        if 'import torch' in gpu_code and 'device =' not in gpu_code:
            import_idx = gpu_code.find('import torch')
            end_of_line = gpu_code.find('\n', import_idx)
            
            device_code = '\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            gpu_code = gpu_code[:end_of_line+1] + device_code + gpu_code[end_of_line+1:]
        
        # Replace common patterns
        gpu_code = gpu_code.replace('torch.tensor(', 'torch.tensor(').replace(')', ', device=device)')
        gpu_code = gpu_code.replace('.cuda()', '.to(device)')
        
        # Add .to(device) to model creation
        model_creation_patterns = ['= Model(', '= Net(', '= nn.Sequential(']
        for pattern in model_creation_patterns:
            if pattern in gpu_code:
                idx = gpu_code.find(pattern)
                end_idx = gpu_code.find('\n', idx)
                
                if '.to(device)' not in gpu_code[idx:end_idx]:
                    gpu_code = gpu_code[:end_idx] + '.to(device)' + gpu_code[end_idx:]
        
        # Add note about moving tensors to GPU
        if 'device' in gpu_code and 'to(device)' not in gpu_code:
            gpu_code += '\n# Move tensors to GPU:\n# tensor = tensor.to(device)\n'
        
        return gpu_code
    
    def suggest_tensorflow_cpu_to_gpu(self, code: str) -> str:
        """
        Convert TensorFlow CPU code to GPU equivalent.
        
        Args:
            code: Python code using TensorFlow on CPU
            
        Returns:
            Equivalent code using TensorFlow on GPU
        """
        gpu_code = code
        
        # Add GPU configuration if not present
        if 'import tensorflow as tf' in gpu_code and 'physical_devices' not in gpu_code:
            import_idx = gpu_code.find('import tensorflow as tf')
            end_of_line = gpu_code.find('\n', import_idx)
            
            gpu_config = '\n# Configure GPU memory growth\n'
            gpu_config += 'gpus = tf.config.list_physical_devices("GPU")\n'
            gpu_config += 'if gpus:\n'
            gpu_config += '    for gpu in gpus:\n'
            gpu_config += '        tf.config.experimental.set_memory_growth(gpu, True)\n'
            
            gpu_code = gpu_code[:end_of_line+1] + gpu_config + gpu_code[end_of_line+1:]
        
        # Add mixed precision policy if not present
        if 'mixed_precision' not in gpu_code and 'tensorflow' in gpu_code:
            gpu_code += '\n# Enable mixed precision for faster training\n'
            gpu_code += 'try:\n'
            gpu_code += '    tf.keras.mixed_precision.set_global_policy("mixed_float16")\n'
            gpu_code += 'except:\n'
            gpu_code += '    pass  # Older TensorFlow versions\n'
        
        return gpu_code
    
    def analyze_code(self, code: str) -> Dict[str, bool]:
        """
        Analyze code to determine what GPU conversions are applicable.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary of frameworks detected in the code
        """
        frameworks = {
            'numpy': 'import numpy' in code or 'np.' in code,
            'pandas': 'import pandas' in code or 'pd.' in code,
            'sklearn': 'import sklearn' in code or 'from sklearn' in code,
            'torch': 'import torch' in code,
            'tensorflow': 'import tensorflow' in code or 'import tf' in code,
        }
        
        return frameworks
    
    def convert_code(self, code: str) -> str:
        """
        Convert CPU code to GPU-accelerated equivalent.
        
        Args:
            code: Python code to convert
            
        Returns:
            GPU-accelerated equivalent code
        """
        # Analyze the code
        frameworks = self.analyze_code(code)
        gpu_code = code
        
        # Apply conversions based on detected frameworks
        if frameworks['numpy']:
            gpu_code = self.suggest_numpy_to_cupy(gpu_code)
        
        if frameworks['pandas']:
            gpu_code = self.suggest_pandas_to_cudf(gpu_code)
        
        if frameworks['sklearn']:
            gpu_code = self.suggest_sklearn_to_cuml(gpu_code)
        
        if frameworks['torch']:
            gpu_code = self.suggest_torch_cpu_to_gpu(gpu_code)
        
        if frameworks['tensorflow']:
            gpu_code = self.suggest_tensorflow_cpu_to_gpu(gpu_code)
        
        return gpu_code


class ModelConverter:
    """
    Utility for converting CPU-based ML models to GPU-accelerated versions.
    
    Provides functions to wrap existing models with GPU-accelerated equivalents.
    """
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize the model converter.
        
        Args:
            gpu_manager: GPUManager instance for device handling
        """
        self.gpu_manager = gpu_manager or get_default_gpu_manager()
    
    def convert_sklearn_model(self, model: Any, cuml_equivalent: Optional[str] = None) -> Any:
        """
        Convert scikit-learn model to cuML equivalent.
        
        Args:
            model: scikit-learn model instance
            cuml_equivalent: Name of cuML equivalent class
            
        Returns:
            cuML model instance or original model if conversion failed
        """
        if not is_gpu_available():
            logger.warning("No GPU available. Returning original model.")
            return model
        
        try:
            import cuml
            
            # Get model parameters
            params = model.get_params()
            
            # Determine cuML equivalent if not provided
            if cuml_equivalent is None:
                model_class_name = model.__class__.__name__
                # Try to find matching class in cuML
                if hasattr(cuml, model_class_name):
                    cuml_class = getattr(cuml, model_class_name)
                else:
                    # Try to find in submodules
                    for submodule_name in ['cluster', 'decomposition', 'ensemble',
                                         'linear_model', 'neighbors', 'svm']:
                        submodule = getattr(cuml, submodule_name, None)
                        if submodule and hasattr(submodule, model_class_name):
                            cuml_class = getattr(submodule, model_class_name)
                            break
                    else:
                        logger.warning(f"No cuML equivalent found for {model_class_name}")
                        return model
            else:
                # Use the specified equivalent
                module_path = cuml_equivalent.split('.')
                current_module = cuml
                for submodule in module_path[1:]:  # Skip 'cuml' prefix
                    current_module = getattr(current_module, submodule)
                cuml_class = current_module
            
            # Create cuML model with same parameters
            try:
                gpu_model = cuml_class(**params)
                logger.info(f"Created cuML {cuml_class.__name__} model")
                
                # If the original model is fitted, we'll need to manually fit the GPU model
                if hasattr(model, 'n_features_in_') and hasattr(model, 'feature_names_in_'):
                    logger.warning(f"Original model is fitted. The GPU model needs to be fitted separately.")
                
                return gpu_model
            except Exception as e:
                logger.error(f"Error creating cuML model: {e}")
                return model
                
        except ImportError:
            logger.warning("cuML not installed. Returning original model.")
            return model
    
    def wrap_sklearn_model_for_gpu(self, model: Any) -> Any:
        """
        Create a scikit-learn compatible wrapper for a model with GPU acceleration.
        
        Args:
            model: Original scikit-learn model
            
        Returns:
            Model wrapper with GPU acceleration where possible
        """
        if not is_gpu_available():
            return model
        
        try:
            import cuml
            from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
            
            # Determine if model is a classifier or regressor
            is_classifier = isinstance(model, ClassifierMixin)
            is_regressor = isinstance(model, RegressorMixin)
            
            # Create appropriate wrapper class
            class GPUModelWrapper(BaseEstimator, 
                                ClassifierMixin if is_classifier else RegressorMixin):
                """Wrapper for scikit-learn models that offloads computation to GPU."""
                
                def __init__(self, cpu_model):
                    self.cpu_model = cpu_model
                    self._gpu_data_handler = GPUDataPipeline()
                
                def fit(self, X, y=None, **kwargs):
                    # Start timer
                    start = time.time()
                    
                    try:
                        # Try to move data to GPU
                        X_gpu = self._gpu_data_handler.to_gpu_array(X)
                        y_gpu = self._gpu_data_handler.to_gpu_array(y) if y is not None else None
                        
                        # Try to fit on GPU using cuML equivalent
                        gpu_model = self.convert_sklearn_model(self.cpu_model)
                        if gpu_model is not self.cpu_model:
                            # It's a cuML model - fit it
                            gpu_model.fit(X_gpu, y_gpu, **kwargs)
                            self.gpu_model = gpu_model
                            self._fitted_on_gpu = True
                            logger.info(f"Model fitted on GPU in {time.time() - start:.4f} seconds")
                            return self
                    except Exception as e:
                        logger.warning(f"GPU fit failed: {e}. Falling back to CPU.")
                    
                    # Fallback to CPU
                    self.cpu_model.fit(X, y, **kwargs)
                    self._fitted_on_gpu = False
                    logger.info(f"Model fitted on CPU in {time.time() - start:.4f} seconds")
                    return self
                
                def predict(self, X, **kwargs):
                    # Start timer
                    start = time.time()
                    
                    if hasattr(self, '_fitted_on_gpu') and self._fitted_on_gpu:
                        try:
                            # Try to predict on GPU
                            X_gpu = self._gpu_data_handler.to_gpu_array(X)
                            y_pred = self.gpu_model.predict(X_gpu, **kwargs)
                            y_pred = self._gpu_data_handler.to_cpu_array(y_pred)
                            logger.info(f"Prediction on GPU in {time.time() - start:.4f} seconds")
                            return y_pred
                        except Exception as e:
                            logger.warning(f"GPU prediction failed: {e}. Falling back to CPU.")
                    
                    # Fallback to CPU
                    y_pred = self.cpu_model.predict(X, **kwargs)
                    logger.info(f"Prediction on CPU in {time.time() - start:.4f} seconds")
                    return y_pred
                
                def convert_sklearn_model(self, model):
                    # Use the model converter to get a cuML equivalent
                    converter = ModelConverter()
                    return converter.convert_sklearn_model(model)
                
                # Forward attribute access to underlying model
                def __getattr__(self, name):
                    if name in self.__dict__:
                        return self.__dict__[name]
                    
                    if hasattr(self, '_fitted_on_gpu') and self._fitted_on_gpu and hasattr(self.gpu_model, name):
                        return getattr(self.gpu_model, name)
                    
                    return getattr(self.cpu_model, name)
            
            return GPUModelWrapper(model)
            
        except ImportError:
            logger.warning("cuML not installed. Returning original model.")
            return model
    
    def convert_torch_model(self, model: 'torch.nn.Module') -> 'torch.nn.Module':
        """
        Convert PyTorch model to use GPU.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model moved to GPU if available
        """
        if not is_gpu_available():
            logger.warning("No GPU available. Returning original model.")
            return model
        
        try:
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                model = model.to(device)
                logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
            
            return model
            
        except ImportError:
            logger.warning("PyTorch not installed. Returning original model.")
            return model
    
    def wrap_torch_model_for_gpu(self, model_class, *args, **kwargs):
        """
        Create a GPU-accelerated wrapper for a PyTorch model.
        
        Args:
            model_class: PyTorch model class
            *args: Arguments to pass to model constructor
            **kwargs: Keyword arguments to pass to model constructor
            
        Returns:
            PyTorchGPUModel wrapper
        """
        if not is_gpu_available():
            # Return regular model
            return model_class(*args, **kwargs)
        
        # Define model builder function
        def model_builder(**builder_kwargs):
            # Merge original kwargs with builder_kwargs
            merged_kwargs = {**kwargs, **builder_kwargs}
            return model_class(*args, **merged_kwargs)
        
        # Create wrapped GPU model
        gpu_model = PyTorchGPUModel(model_builder, self.gpu_manager)
        
        return gpu_model
    
    def wrap_tf_model_for_gpu(self, model_builder, *args, **kwargs):
        """
        Create a GPU-accelerated wrapper for a TensorFlow model.
        
        Args:
            model_builder: Function that builds TensorFlow model
            *args: Arguments to pass to model builder
            **kwargs: Keyword arguments to pass to model builder
            
        Returns:
            TensorFlowGPUModel wrapper
        """
        if not is_gpu_available():
            # Return regular model
            return model_builder(*args, **kwargs)
        
        # Define model builder function
        def wrapped_model_builder(**builder_kwargs):
            # Merge original kwargs with builder_kwargs
            merged_kwargs = {**kwargs, **builder_kwargs}
            return model_builder(*args, **merged_kwargs)
        
        # Create wrapped GPU model
        gpu_model = TensorFlowGPUModel(wrapped_model_builder, self.gpu_manager)
        
        return gpu_model


def generate_migration_examples():
    """
    Generate examples of migrating CPU code to GPU-accelerated code.
    
    Returns:
        Dictionary of example names and their code pairs (CPU and GPU)
    """
    examples = {}
    
    # NumPy to CuPy example
    numpy_example = """
    import numpy as np
    
    # Create arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.random.rand(1000, 1000)
    
    # Perform operations
    c = np.dot(b, b.T)
    d = np.mean(c, axis=0)
    e = np.max(d)
    
    result = e * a
    """
    
    # pandas to cuDF example
    pandas_example = """
    import pandas as pd
    
    # Load data
    df = pd.read_csv('data.csv')
    
    # Process data
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Filter and group
    filtered = df[df['volume'] > df['volume'].mean()]
    grouped = filtered.groupby('symbol').agg({
        'close': 'mean',
        'volume': 'sum'
    })
    
    result = grouped.reset_index()
    """
    
    # scikit-learn to cuML example
    sklearn_example = """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Preprocess
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dimensionality reduction
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_pca, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_pca)
    accuracy = (y_pred == y_test).mean()
    """
    
    # PyTorch CPU to GPU example
    pytorch_example = """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Define model
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create model and data
    input_size = 10
    hidden_size = 64
    output_size = 1
    
    model = Net(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X = torch.randn(100, input_size)
    y = torch.randn(100, output_size)
    
    # Train model
    for epoch in range(100):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """
    
    # TensorFlow CPU to GPU example
    tensorflow_example = """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    # Create model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Generate dummy data
    X = tf.random.normal((100, 10))
    y = tf.random.normal((100, 1))
    
    # Train model
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
    
    # Evaluate model
    X_test = tf.random.normal((20, 10))
    y_test = tf.random.normal((20, 1))
    model.evaluate(X_test, y_test)
    
    # Make predictions
    predictions = model.predict(X_test)
    """
    
    # Generate GPU versions
    converter = CodeConverter()
    examples['numpy_to_cupy'] = {
        'cpu_code': textwrap.dedent(numpy_example),
        'gpu_code': textwrap.dedent(converter.suggest_numpy_to_cupy(numpy_example))
    }
    
    examples['pandas_to_cudf'] = {
        'cpu_code': textwrap.dedent(pandas_example),
        'gpu_code': textwrap.dedent(converter.suggest_pandas_to_cudf(pandas_example))
    }
    
    examples['sklearn_to_cuml'] = {
        'cpu_code': textwrap.dedent(sklearn_example),
        'gpu_code': textwrap.dedent(converter.suggest_sklearn_to_cuml(sklearn_example))
    }
    
    examples['pytorch_cpu_to_gpu'] = {
        'cpu_code': textwrap.dedent(pytorch_example),
        'gpu_code': textwrap.dedent(converter.suggest_torch_cpu_to_gpu(pytorch_example))
    }
    
    examples['tensorflow_cpu_to_gpu'] = {
        'cpu_code': textwrap.dedent(tensorflow_example),
        'gpu_code': textwrap.dedent(converter.suggest_tensorflow_cpu_to_gpu(tensorflow_example))
    }
    
    return examples


def print_migration_guide():
    """Print migration guidance for CPU to GPU conversion."""
    
    print("=" * 80)
    print("                   QUANTRA GPU MIGRATION GUIDE")
    print("=" * 80)
    print()
    print("This guide helps you migrate existing CPU-based code to GPU-accelerated code.")
    print("Follow the patterns and examples below to accelerate your machine learning workflows.")
    print()
    
    print("GENERAL MIGRATION STEPS:")
    print("------------------------")
    print("1. Ensure GPU hardware is available and properly configured")
    print("2. Install GPU acceleration libraries (CUDA, cuDNN, etc.)")
    print("3. Replace CPU operations with GPU equivalents")
    print("4. Add code to move data between CPU and GPU as needed")
    print("5. Optimize memory usage and batch processing")
    print("6. Add fallback mechanisms for CPU when GPU is not available")
    print()
    
    print("FRAMEWORK-SPECIFIC MIGRATION:")
    print("----------------------------")
    print("NumPy → CuPy:           Syntax is nearly identical; just replace 'np' with 'cp'")
    print("pandas → cuDF:          Similar API with some restrictions; use 'to_pandas()' to convert back")
    print("scikit-learn → cuML:    Most common algorithms available; API matches scikit-learn")
    print("PyTorch:                Use 'tensor.to(device)' and 'model.to(device)' to move to GPU")
    print("TensorFlow:             Configure memory growth and use mixed precision for performance")
    print()
    
    print("QUANTRA GPU UTILITIES:")
    print("---------------------")
    print("gpu_utils.py:           Base utilities for GPU detection and management")
    print("gpu_models.py:          GPU-accelerated model wrappers for PyTorch/TensorFlow")
    print("gpu_data_pipeline.py:   Optimized data processing pipelines for GPU")
    print("gpu_monitor.py:         Performance monitoring and optimization tools")
    print()
    
    print("CODE MIGRATION EXAMPLES:")
    print("-----------------------")
    examples = generate_migration_examples()
    
    for name, example in examples.items():
        print(f"\n{name.upper()}:\n")
        print("CPU VERSION:")
        print("```python")
        print(example['cpu_code'])
        print("```")
        print("\nGPU VERSION:")
        print("```python")
        print(example['gpu_code'])
        print("```")
        print("\n" + "-" * 40)
    
    print("\nPERFORMANCE TIPS:")
    print("----------------")
    print("1. Minimize data transfers between CPU and GPU")
    print("2. Process data in batches to maximize GPU utilization")
    print("3. Use mixed precision (FP16) where possible for faster computation")
    print("4. Profile your code to identify bottlenecks")
    print("5. Consider multi-GPU strategies for very large workloads")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    # Set up logging for script execution
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Print the migration guide
    print_migration_guide()
    
    # Simple code conversion example
    print("\nCODE CONVERTER EXAMPLE:")
    print("----------------------")
    
    example_code = """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    # Load data
    df = pd.read_csv('stock_data.csv')
    
    # Preprocess
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    
    # Prepare features and target
    X = df[['open', 'high', 'low', 'volume', 'ma_20']].values
    y = df['returns'].values
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    print(f"MSE: {mse:.6f}")
    """
    
    print("Original CPU code:")
    print("```python")
    print(textwrap.dedent(example_code))
    print("```")
    
    # Convert code
    converter = CodeConverter()
    gpu_code = converter.convert_code(example_code)
    
    print("\nConverted GPU code:")
    print("```python")
    print(textwrap.dedent(gpu_code))
    print("```")