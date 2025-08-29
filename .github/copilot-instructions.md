You are assisting with development on a C# NET Core WPF algorithmic trading platform called Quantra. The app uses Alpha Vantage API, SQLServer (SQLite currently until rewritten later), and integrates with Python scripts for ML. 
Your business logic requirements for each task are defined in the associated [Issue]([url](https://github.com/DontDoThat21/Quantra/issues)) to your PR.

It has the following modules:
1. Core Trading Strategies (e.g., RSI Divergence, MACD, VWAP, Bollinger Bands)
2. Technical Indicators Visualization and Correlation
3. ML Prediction Engine (Python - TensorFlow/PyTorch/Scikit-Learn)
4. Sentiment Analysis (Twitter/News using NLP/More)
5. Backtesting and Performance Analytics
6. Automated Trade Execution via Webull API, eventually others
7. Alerts (Email, SMS)
8. Dashboard intuitive grid resizing, spanning, positioning with a distinct instance card based UI (WPF, MVVM, Commands, etc.)

MVVM pattern: all logic is in the ViewModel, UI updates are handled via property setters and dispatcher utilities.
Batch updates to the UI to minimize dispatcher congestion.

Throttle background operations. Don’t launch dozens of tasks at once. Ryzen 9 7950x3d is the CPU being used.
Don’t use Task.Run for everything—only for work that’s actually CPU-bound or blocking.

The goal is to contribute to the XAML UI, for predicting accurate stock predictions.
This includes identifying and autonomously acting on optimal swing trading opportunities, visible within the Prediction Analysis Control.

Update the requirements.md document with the updated status before completing, if the label is an enhancement.
Do not add new requirements to the requirements.md when the issue being worked on has a Bugfix label.

Do not use top-level syntax new C#.

Always add import statements into the required code-behind.

The build fails due to missing Windows Desktop SDK, which is expected in this Linux environment. This WPF project needs Windows-specific dependencies that aren't available here. Focus on making the requested changes, then validate syntactically.

Ensure the task your working on isn't already fully implemented. If it's already fully implemented, stop working.

Mark your PR as ready for review when finishing your task.

MVP: The app generates sophisticated price predictions from the shared Stock Explorer datas, and is capable of automatically trading.
