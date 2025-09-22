# Quantra Application

## Overview

Quantra is a sophisticated WPF-based algorithmic trading platform providing a comprehensive, highly automated trading dashboard. The application empowers traders with real-time stock analysis, AI-powered predictions, and automated trading capabilities.

**Warning**: This is not financial advice. Trading involves significant risk of loss. This application is for educational and research purposes only.

## Key Features

### Core Trading Strategies
- **Bollinger Bands Strategy**: Mean-reversion trading using standard deviation channels
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Volume-Based Analysis**: Trading strategies based on volume patterns and anomalies
- **Moving Average Strategies**: Simple, Exponential, and Weighted moving averages
- **Fibonacci Retracement**: Key support/resistance level identification
- **Parabolic SAR**: Stop and reverse trend-following indicator
- **Ichimoku Cloud**: Comprehensive trend, momentum, and support/resistance indicator

### Technical Indicators & Visualization
- **Comprehensive Indicator Suite**: 
  - RSI, MACD, Bollinger Bands, ADX, ROC, VWAP, CCI, ATR
  - Williams %R, Stochastic Oscillator, StochRSI, Ultimate Oscillator
  - Bull/Bear Power, On-Balance Volume (OBV), Money Flow Index (MFI)
- **Customizable Indicators**: Create and visualize custom technical indicators
- **Visual Analysis Tools**: Real-time charting and visualization for all indicators

### Machine Learning Integration (Python)
- **Python Integration Framework**: Seamless C# interoperability with Python
- **Sentiment Analysis**: Using HuggingFace transformers for market sentiment evaluation
- **Anomaly Detection**: Identification of unusual market behavior patterns

### Sentiment Analysis
- **Twitter Sentiment Analysis**: Real-time social media sentiment tracking
- **Financial News Analysis**: Integration with major financial news sources

### Backtesting and Performance Analytics
- **Comprehensive Backtesting Engine**: Test strategies against historical data
- **Advanced Performance Metrics**: 
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Information Ratio, Maximum Drawdown
  - CAGR, Profit Factor
- **Benchmark Comparison**: Compare strategy performance against market benchmarks
- **Detailed Trade Analysis**: Analyze individual trades and performance over time

### Alerts and Notifications
- **In-App Alert System**: Configurable alerts with multiple severity levels
- **Alert Management UI**: Comprehensive interface for managing and filtering alerts

### Automated Trading Execution
- **Paper Trading Mode**: Risk-free trading simulation
- **Order Types**: Limit orders, stop-loss, and take-profit functionality
- **Trade Automation**: Configurable for manual approval or fully automated execution

### User Interface & Usability
- **Customizable Dashboard Layout**: Flexible, adaptable trading interface
- **Multi-Tab Organization**: Separate contexts for different trading strategies
- **Persistent Layout Saving**: Automatically saves your preferred configuration
- **Real-Time Data Visualization**: Dynamic updating of charts and indicators

### Application Architecture
- **MVVM Pattern**: Clean separation of UI and business logic
- **Material Design Styling**: Modern, intuitive user interface

### Monitoring and Error Handling
- **Application Logging**: Comprehensive activity and error logging
- **Error Alerts**: Detailed exception information for troubleshooting

## Technical Stack
- **C# WPF Application** using .NET Core 8
- **Python ML Integration** for advanced prediction models
- **Alpha Vantage Premium API** for historical data
- **Webull API** for trade execution
- **Material Design** for modern UI components
- **GitHub Actions** for automated workflows and CI/CD

### GitHub Workflows
- **Auto-Assign Copilot as Reviewer**: Automatically assigns GitHub Copilot as a reviewer on AI-generated or AI-related pull requests
- **Auto-Assign**: Assigns repository owner to new issues and PRs 
- **Proof HTML**: Validates HTML in the repository

## Misc

App uses the Alpha Vantage Premium API for Historicals, Webull API for sending trades currently (likely to be re-written), Benzinga premium. The Benzinga premium features should all be ready to go, but an API key will not be acquired until MVP is ready. 

Application is running on a Ryzen 9 7950X3D, 64GB Ram, RX 9070 XT.

Early PoC:
<img width="2532" height="1371" alt="image" src="https://github.com/user-attachments/assets/550f83ae-5c80-428b-8685-89739cb7fd52" />
<img width="1205" height="807" alt="image" src="https://github.com/user-attachments/assets/63bdc911-8aa1-4de4-8487-0f5e1fb43357" />
<img width="1694" height="945" alt="image" src="https://github.com/user-attachments/assets/85a2ec59-0f15-4a12-926a-8d349729a02b" />
<img width="720" height="405" alt="image" src="https://github.com/user-attachments/assets/5fcd8807-5de7-4358-ad32-5f19795abaa7" />
<img width="502" height="992" alt="img" src="https://github.com/user-attachments/assets/13c7f86d-535a-405a-be0f-1bf6a009ee3d" />
<img width="795" height="1267" alt="image" src="https://github.com/user-attachments/assets/97a35cb4-8500-4ed6-8307-7eb38c14984a" />

