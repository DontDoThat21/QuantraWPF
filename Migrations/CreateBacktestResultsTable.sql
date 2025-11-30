-- SQL Script to create BacktestResults table
-- This table stores backtest runs for comparison and analysis

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'BacktestResults')
BEGIN
    CREATE TABLE [dbo].[BacktestResults] (
        [Id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        [Symbol] NVARCHAR(20) NOT NULL,
        [StrategyName] NVARCHAR(200) NOT NULL,
        [TimeFrame] NVARCHAR(50) NULL,
        [StartDate] DATETIME2 NOT NULL,
        [EndDate] DATETIME2 NOT NULL,
        [InitialCapital] REAL NOT NULL,
        [FinalEquity] REAL NOT NULL,
        [TotalReturn] REAL NOT NULL,
        [MaxDrawdown] REAL NOT NULL,
        [WinRate] REAL NULL,
        [TotalTrades] INT NULL,
        [WinningTrades] INT NULL,
        [SharpeRatio] REAL NULL,
        [SortinoRatio] REAL NULL,
        [CAGR] REAL NULL,
        [CalmarRatio] REAL NULL,
        [ProfitFactor] REAL NULL,
        [InformationRatio] REAL NULL,
        [TotalTransactionCosts] REAL NULL,
        [GrossReturn] REAL NULL,
        [EquityCurveJson] NVARCHAR(MAX) NULL,
        [TradesJson] NVARCHAR(MAX) NULL,
        [StrategyParametersJson] NVARCHAR(MAX) NULL,
        [Notes] NVARCHAR(1000) NULL,
        [CreatedAt] DATETIME2 NOT NULL DEFAULT GETDATE(),
        [RunName] NVARCHAR(200) NULL
    );

    PRINT 'BacktestResults table created successfully.';
END
ELSE
BEGIN
    PRINT 'BacktestResults table already exists.';
END
GO

-- Create indexes for efficient querying
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_BacktestResults_Symbol' AND object_id = OBJECT_ID('BacktestResults'))
BEGIN
    CREATE INDEX [IX_BacktestResults_Symbol] ON [dbo].[BacktestResults] ([Symbol]);
    PRINT 'Index IX_BacktestResults_Symbol created.';
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_BacktestResults_StrategyName' AND object_id = OBJECT_ID('BacktestResults'))
BEGIN
    CREATE INDEX [IX_BacktestResults_StrategyName] ON [dbo].[BacktestResults] ([StrategyName]);
    PRINT 'Index IX_BacktestResults_StrategyName created.';
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_BacktestResults_CreatedAt' AND object_id = OBJECT_ID('BacktestResults'))
BEGIN
    CREATE INDEX [IX_BacktestResults_CreatedAt] ON [dbo].[BacktestResults] ([CreatedAt] DESC);
    PRINT 'Index IX_BacktestResults_CreatedAt created.';
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_BacktestResults_Symbol_StrategyName' AND object_id = OBJECT_ID('BacktestResults'))
BEGIN
    CREATE INDEX [IX_BacktestResults_Symbol_StrategyName] ON [dbo].[BacktestResults] ([Symbol], [StrategyName]);
    PRINT 'Index IX_BacktestResults_Symbol_StrategyName created.';
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_BacktestResults_Symbol_CreatedAt' AND object_id = OBJECT_ID('BacktestResults'))
BEGIN
    CREATE INDEX [IX_BacktestResults_Symbol_CreatedAt] ON [dbo].[BacktestResults] ([Symbol], [CreatedAt] DESC);
    PRINT 'Index IX_BacktestResults_Symbol_CreatedAt created.';
END
GO

-- Sample queries for working with backtest results:

-- Get all backtest results for a symbol
-- SELECT * FROM BacktestResults WHERE Symbol = 'AAPL' ORDER BY CreatedAt DESC;

-- Get best performing backtests by total return
-- SELECT TOP 10 Symbol, StrategyName, TotalReturn, SharpeRatio, MaxDrawdown, CreatedAt 
-- FROM BacktestResults ORDER BY TotalReturn DESC;

-- Get backtest comparison for a specific strategy
-- SELECT Symbol, TotalReturn, MaxDrawdown, SharpeRatio, WinRate 
-- FROM BacktestResults WHERE StrategyName = 'SMA Crossover (20/50)' ORDER BY CreatedAt DESC;

-- Get summary statistics by strategy
-- SELECT StrategyName, 
--        COUNT(*) AS BacktestCount,
--        AVG(TotalReturn) AS AvgReturn,
--        AVG(MaxDrawdown) AS AvgDrawdown,
--        AVG(SharpeRatio) AS AvgSharpe
-- FROM BacktestResults 
-- GROUP BY StrategyName
-- ORDER BY AvgReturn DESC;
