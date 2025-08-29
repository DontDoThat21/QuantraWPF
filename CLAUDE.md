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
```

## Architecture Overview

### Core Structure
- **Quantra** (main WPF application) - Primary trading platform built with .NET 8 and WPF
- **Quantra.Tests** - XUnit test suite for testing core functionality
- **python/** - Machine learning components using TensorFlow, PyTorch, and scikit-learn

### Key Architectural Patterns
- **MVVM Pattern**: ViewModels in `ViewModels/` directory, Views in `Views/` directory
- **Dependency Injection**: Service registration in `Extensions/ServiceCollectionExtensions.cs`
- **Cross-Cutting Concerns**: Centralized in `CrossCutting/` with logging, monitoring, error handling
- **Repository Pattern**: Data access abstracted through repositories in `Repositories/`
- **Service Layer**: Business logic in `Services/` with interface contracts in `Services/Interfaces/`

### Data Layer
- **DatabaseMonolith.cs**: Centralized database access layer handling SQLite operations
- **Dapper**: Lightweight ORM for data access
- **Repository Classes**: Abstraction over direct database calls

### UI Framework
- **WPF with Material Design**: Modern UI using MaterialDesignThemes package
- **Live Charts**: Real-time charting with LiveCharts.Wpf.Core
- **Custom Controls**: Reusable UI components in `Views/` subdirectories
- **Styles**: Centralized styling in `Styles/EnhancedStyles.xaml` and `Styles/Styles.xaml`

### Trading Integration
- **Alpha Vantage API**: Historical data via `Services/AlphaVantageService.cs`
- **Webull API**: Trade execution through `Services/WebullTradingBot.cs`
- **Backtesting Engine**: Comprehensive backtesting in `Services/BacktestingEngine.cs`

### Machine Learning Pipeline
- **C#/Python Interop**: Seamless integration between C# application and Python ML models
- **Sentiment Analysis**: Twitter and financial news sentiment using HuggingFace transformers
- **Anomaly Detection**: Market behavior analysis using isolation forests and one-class SVM
- **Ensemble Learning**: Multiple model aggregation for improved predictions
- **Real-time Inference**: Live prediction service integration

### Key Services
- **PredictionService**: Core ML prediction logic
- **StockDataCacheService**: Intelligent caching of market data
- **TechnicalIndicatorService**: Technical analysis calculations
- **NotificationService**: Alert and notification management
- **SettingsService**: Application configuration management

## Development Guidelines

### Testing Strategy
- **XUnit Framework**: Primary testing framework
- **Test Organization**: Tests mirror main project structure in `Quantra.Tests/`
- **Service Testing**: Focus on testing service layer logic and integrations
- **UI Testing**: Limited UI testing for core user workflows

### Configuration Management
- **appsettings.json**: Primary configuration file
- **Environment-specific**: appsettings.Development.json, appsettings.Production.json
- **API Keys**: Stored in alphaVantageSettings.json (not in source control)
- **Configuration Models**: Strongly-typed config in `Configuration/Models/`

### Code Organization
- **Services**: Business logic and external integrations
- **Models**: Domain entities and data transfer objects
- **ViewModels**: MVVM view models with INotifyPropertyChanged
- **Views**: WPF user controls and windows
- **Converters**: Value converters for data binding
- **Utilities**: Helper classes and extension methods

### Python ML Integration
- **Requirements**: Core dependencies in `python/requirements.txt`
- **GPU Support**: Optional CUDA acceleration for TensorFlow/PyTorch
- **Model Persistence**: Trained models stored in `python/models/`
- **Feature Engineering**: Advanced feature creation in `python/feature_engineering.py`

### Performance Considerations
- **Caching**: Extensive use of caching for market data and predictions
- **Async/Await**: Asynchronous operations for UI responsiveness
- **Throttling**: API rate limiting and concurrent task management
- **Memory Management**: Careful handling of large datasets and real-time updates

### Error Handling
- **Structured Logging**: Serilog with file and console outputs
- **Resilience**: Polly for retry policies and circuit breakers
- **Error Boundaries**: Centralized error handling in `CrossCutting/ErrorHandling/`
- **User Feedback**: User-friendly error messages and notifications