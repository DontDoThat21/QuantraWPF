# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Running
```bash
# Build the solution
dotnet build Quantra.sln

# Build specific configurations
dotnet build Quantra.sln --configuration Debug
dotnet build Quantra.sln --configuration Release

# Run the main application
dotnet run --project Quantra/Quantra.csproj

# Run tests
dotnet test Quantra.Tests/Quantra.Tests.csproj
```

### Python ML Components
```bash
# Install Python dependencies
pip install -r python/requirements.txt

# Run Python ML components (from project root)
python python/stock_predictor.py
python python/sentiment_analysis.py
python python/ensemble_learning.py
python python/train_from_database.py
python python/temporal_fusion_transformer.py
```

## Architecture Overview

### Core Structure
- **Quantra** (main WPF application) - Primary trading platform built with .NET 9 and WPF
- **Quantra.DAL** (Data Access Layer) - Separate .NET 9 library handling all data operations
- **Quantra.Helpers** - Shared helper utilities
- **Quantra.Tests** - XUnit test suite for testing core functionality
- **DayTrader** & **DayTrader.Tests** - Day trading module with tests
- **Controls** - Reusable WPF custom controls
- **Database** - Database schema and initialization scripts
- **Documentation** - Project documentation files
- **Migrations** - Database migration scripts
- **python/** - Machine learning components using TensorFlow, PyTorch, and scikit-learn

### Key Architectural Patterns
- **Clean Architecture**: Separation between UI (Quantra) and Data Access (Quantra.DAL) layers
- **MVVM Pattern**: ViewModels in `Quantra/ViewModels/` directory, Views in `Quantra/Views/` directory
- **Dependency Injection**: Service registration in `Quantra/Extensions/ServiceCollectionExtensions.cs`
- **Cross-Cutting Concerns**: Centralized in `Quantra.DAL/CrossCutting/` with logging, monitoring, error handling
- **Repository Pattern**: Data access abstracted through repositories in `Quantra.DAL/Repositories/`
- **Service Layer**: Business logic in `Quantra.DAL/Services/` with interface contracts in `Quantra.DAL/Interfaces/`

### Data Layer (Quantra.DAL)
- **DatabaseMonolith.cs**: Centralized database access layer handling SQLite operations
- **Entity Framework Core**: Primary ORM with SQLite and SQL Server providers
- **Repository Classes**: Abstraction over direct database calls in `Repositories/`
- **Data Context**: Entity Framework DbContext in `Data/` directory
- **Migrations**: EF Core migrations in `Migrations/` directory
- **Models**: Domain entities in `Models/` directory

### UI Framework (Quantra)
- **WPF with Material Design**: Modern UI using MaterialDesignThemes 5.2.1
- **Live Charts**: Real-time charting with LiveCharts.Wpf.Core 0.9.8
- **Dragablz**: Tab control functionality 0.0.3.234
- **CefSharp**: Embedded browser for web content 135.0.220
- **Custom Controls**: Reusable UI components in `Views/` subdirectories
- **Styles**: Centralized styling in `Styles/EnhancedStyles.xaml` and `Styles/Styles.xaml`

### Trading Integration
- **Alpha Vantage API**: Historical data via `Quantra.DAL/Services/AlphaVantageService.cs`
- **Webull API**: Trade execution through `Quantra.DAL/Services/WebullTradingBot.cs`
- **Backtesting Engine**: Comprehensive backtesting in `Quantra.DAL/Services/BacktestingEngine.cs`
- **Trading Engine**: Core trading logic in `Quantra.DAL/TradingEngine/`

### Machine Learning Pipeline
- **C#/Python Interop**: Seamless integration between C# application and Python ML models
- **Core Prediction**: `stock_predictor.py` - Main stock prediction model with LSTM/GRU architectures
- **Sentiment Analysis**: `sentiment_analysis.py` - Multi-source sentiment (YouTube, earnings transcripts, OpenAI)
- **Anomaly Detection**: `anomaly_detection.py` - Market behavior analysis using isolation forests and one-class SVM
- **Ensemble Learning**: `ensemble_learning.py` - Multiple model aggregation for improved predictions
- **Temporal Fusion Transformer**: `temporal_fusion_transformer.py` - Advanced deep learning for time series
- **Market Regime Detection**: `market_regime_detection.py` - Hidden Markov Models for market state detection
- **Reinforcement Learning**: `reinforcement_learning.py` - RL-based trading agents using stable-baselines3
- **Feature Engineering**: `feature_engineering.py` - Advanced feature creation and transformation
- **Hyperparameter Optimization**: `hyperparameter_optimization.py` - Optuna-based model tuning
- **Real-time Inference**: `real_time_inference.py` - Live prediction service integration
- **Model Performance**: `model_performance_tracking.py` - Comprehensive model evaluation and monitoring
- **Database Integration**: `train_from_database.py` - Direct training from Quantra's SQLite database

### Key Services (Quantra.DAL/Services)
- **PredictionService**: Core ML prediction logic and Python interop
- **StockDataCacheService**: Intelligent caching of market data with gzip compression
- **TechnicalIndicatorService**: Technical analysis calculations (RSI, MACD, Bollinger Bands, etc.)
- **NotificationService**: Alert and notification management
- **SettingsService**: Application configuration management
- **AlphaVantageService**: Alpha Vantage API integration
- **WebullTradingBot**: Webull trading platform integration
- **BacktestingEngine**: Strategy backtesting and performance analysis

## Code Organization

### Quantra/ (UI Layer)
- **Adapters/** - Interface adapters between UI and data layer
- **Commands/** - WPF ICommand implementations
- **Configuration/** - Configuration models and settings
- **Controllers/** - UI controllers and coordinators
- **Converters/** - WPF value converters for data binding
- **Examples/** - Example code and usage samples
- **Extensions/** - Extension methods and service registration
- **Managers/** - High-level managers (WindowManager, etc.)
- **Modules/** - Feature modules and plugins
- **Repositories/** - UI-specific repository implementations
- **Styles/** - XAML styles and themes
- **Tests/** - Unit tests
- **Utilities/** - Helper classes and utilities
- **ViewModels/** - MVVM view models with INotifyPropertyChanged
- **Views/** - WPF user controls and windows
- **python/** - Python ML scripts and models

### Quantra.DAL/ (Data Access Layer)
- **CrossCutting/** - Cross-cutting concerns (logging, error handling, monitoring)
- **Data/** - Entity Framework DbContext and data access
- **Enums/** - Enumeration types
- **Interfaces/** - Service and repository interfaces
- **Migrations/** - Entity Framework Core migrations
- **Models/** - Domain entities and data models
- **Notifications/** - Notification models and handlers
- **Repositories/** - Data repository implementations
- **Scripts/** - Database scripts and utilities
- **Services/** - Business logic and external integrations
- **SQL/** - Raw SQL scripts
- **TradingEngine/** - Core trading engine logic
- **Utilities/** - DAL-specific utilities
- **DatabaseMonolith.cs** - Legacy centralized database access (being phased out)

## Key Technologies

### .NET Packages (Quantra)
- **CefSharp.Common.NETCore** 135.0.220 - Embedded browser
- **Dapper** 2.1.66 - Lightweight ORM
- **Dragablz** 0.0.3.234 - Tab control
- **LiveCharts.Core** 0.9.8 & **LiveCharts.Wpf.Core** 0.9.8 - Charting
- **MaterialDesignThemes** 5.2.1 - Material Design UI
- **Microsoft.Extensions.Configuration** 9.0.10 - Configuration management
- **Microsoft.Extensions.DependencyInjection** 9.0.10 - DI container
- **Newtonsoft.Json** 13.0.3 - JSON serialization
- **System.Data.SQLite** 1.0.119 - SQLite database
- **Serilog** 4.1.0 - Structured logging
- **Polly** 8.4.0 - Resilience and retry policies

### .NET Packages (Quantra.DAL)
- **Microsoft.EntityFrameworkCore** 9.0.10 - ORM framework
- **Microsoft.EntityFrameworkCore.Sqlite** 9.0.10 - SQLite provider
- **Microsoft.EntityFrameworkCore.SqlServer** 9.0.10 - SQL Server provider
- **Microsoft.Data.SqlClient** 5.2.2 - SQL client
- **Serilog** 4.0.1 - Structured logging
- **Polly** 8.4.0 - Resilience patterns

### Python Frameworks
- **numpy** >=1.21.0 - Numerical computing
- **pandas** >=1.3.0 - Data manipulation
- **scikit-learn** >=1.0.0 - Classical ML algorithms
- **tensorflow** >=2.9.0 - Deep learning framework
- **torch** >=1.13.0 - PyTorch deep learning
- **optuna** >=3.0.0 - Hyperparameter optimization
- **plotly** >=5.10.0 - Interactive visualizations
- **hmmlearn** >=0.3.0 - Hidden Markov Models
- **gym** >=0.21.0 - Reinforcement learning environments
- **stable-baselines3** >=1.6.0 - RL algorithms
- **yt-dlp** >=2023.1.6 - YouTube data extraction
- **openai-whisper** >=20230314 - Speech recognition
- **openai** >=1.0.0 - OpenAI API client
- **cudf, cuml, cupy** - NVIDIA GPU acceleration (optional)

## Development Guidelines

### Testing Strategy
- **XUnit Framework**: Primary testing framework
- **Test Organization**: Tests mirror main project structure in `Quantra.Tests/`
- **Service Testing**: Focus on testing service layer logic and integrations
- **UI Testing**: Limited UI testing for core user workflows
- **Integration Testing**: Python ML model integration tests

### Configuration Management
- **appsettings.json**: Primary configuration file in `Quantra/`
- **Environment-specific**: appsettings.Development.json, appsettings.Production.json
- **API Keys**: Stored in alphaVantageSettings.json (not in source control)
- **Configuration Models**: Strongly-typed config in `Quantra/Configuration/Models/`

### Python ML Integration
- **Requirements**: Core dependencies in `python/requirements.txt`
- **GPU Support**: Optional CUDA acceleration for TensorFlow/PyTorch
- **Model Persistence**: Trained models stored in `python/models/` or database
- **Feature Engineering**: Advanced feature creation in `python/feature_engineering.py`
- **Database Integration**: Direct SQLite access via `train_from_database.py`

### Performance Considerations
- **Caching**: Extensive use of caching for market data and predictions with gzip compression
- **Async/Await**: Asynchronous operations for UI responsiveness
- **Throttling**: API rate limiting and concurrent task management
- **Memory Management**: Careful handling of large datasets and real-time updates
- **DbContext Threading**: Proper DbContext lifetime management in multi-threaded scenarios

### Error Handling
- **Structured Logging**: Serilog with file and console outputs
- **Resilience**: Polly for retry policies and circuit breakers
- **Error Boundaries**: Centralized error handling in `Quantra.DAL/CrossCutting/ErrorHandling/`
- **User Feedback**: User-friendly error messages and notifications

## Important Notes

### Database Management
- Primary database is SQLite stored in local application data folder
- DatabaseMonolith.cs is legacy code being gradually migrated to Entity Framework Core
- Always use proper DbContext lifetime management (scoped per operation)
- Use DbContextFactory for multi-threaded scenarios
- Database migrations tracked in `Quantra.DAL/Migrations/`

### API Keys and Secrets
- Never commit API keys to source control
- alphaVantageSettings.json is gitignored
- Store sensitive configuration in user secrets or environment variables
- Webull credentials handled securely through authentication service

### Python Integration Setup
- Ensure Python 3.8+ is installed and in PATH
- Install requirements: `pip install -r python/requirements.txt`
- Optional GPU setup for CUDA-enabled TensorFlow/PyTorch
- Python scripts called via Process.Start() from C# services
- JSON used for data exchange between C# and Python components
