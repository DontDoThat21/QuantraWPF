-- Migration: Add Multi-Horizon TFT Support
-- Version: 1.0
-- Date: 2024-12-06
-- Description: Creates tables for storing multi-horizon predictions, feature importance,
--              and temporal attention weights from TFT (Temporal Fusion Transformer) model.
-- Issue: #6 - Update Database Schema for Multi-Horizon Predictions

BEGIN TRANSACTION;

-- Create multi-horizon predictions table
-- Stores predictions for different time horizons (1, 3, 5, 10 days) with confidence intervals
CREATE TABLE StockPredictionHorizons (
    Id INT PRIMARY KEY IDENTITY(1,1),
    PredictionId INT NOT NULL,
    Horizon INT NOT NULL,                       -- Horizon in days (1, 3, 5, 10)
    TargetPrice FLOAT,                          -- Median prediction (50th percentile)
    LowerBound FLOAT,                           -- Lower bound (10th percentile)
    UpperBound FLOAT,                           -- Upper bound (90th percentile)
    Confidence FLOAT,                           -- Horizon-specific confidence (0.0-1.0)
    ExpectedFruitionDate DATETIME2,             -- PredictionDate + Horizon days
    ActualPrice FLOAT NULL,                     -- Filled in after horizon passes
    ActualReturn FLOAT NULL,                    -- (ActualPrice - CurrentPrice) / CurrentPrice
    ErrorPct FLOAT NULL,                        -- (ActualPrice - TargetPrice) / TargetPrice

    CONSTRAINT FK_PredictionHorizons_Prediction FOREIGN KEY (PredictionId)
        REFERENCES StockPredictions(Id) ON DELETE CASCADE
);

CREATE INDEX IX_PredictionHorizons_PredictionId ON StockPredictionHorizons(PredictionId);
CREATE INDEX IX_PredictionHorizons_Horizon ON StockPredictionHorizons(Horizon);
CREATE INDEX IX_PredictionHorizons_FruitionDate ON StockPredictionHorizons(ExpectedFruitionDate);

-- Create feature importance table
-- Stores feature importance weights from TFT variable selection networks
CREATE TABLE PredictionFeatureImportance (
    Id INT PRIMARY KEY IDENTITY(1,1),
    PredictionId INT NOT NULL,
    FeatureName NVARCHAR(100) NOT NULL,         -- Feature name (e.g., RSI, MACD, Volume)
    ImportanceScore FLOAT,                       -- Importance weight from attention mechanism
    FeatureType NVARCHAR(20),                   -- 'static', 'known', or 'observed'

    CONSTRAINT FK_FeatureImportance_Prediction FOREIGN KEY (PredictionId)
        REFERENCES StockPredictions(Id) ON DELETE CASCADE
);

CREATE INDEX IX_FeatureImportance_PredictionId ON PredictionFeatureImportance(PredictionId);
CREATE INDEX IX_FeatureImportance_FeatureName ON PredictionFeatureImportance(FeatureName);

-- Create temporal attention table
-- Stores temporal attention weights showing which past time steps were most influential
CREATE TABLE PredictionTemporalAttention (
    Id INT PRIMARY KEY IDENTITY(1,1),
    PredictionId INT NOT NULL,
    TimeStep INT NOT NULL,                      -- Negative values (e.g., -1 = yesterday)
    AttentionWeight FLOAT,                      -- Attention weight (higher = more influential)

    CONSTRAINT FK_TemporalAttention_Prediction FOREIGN KEY (PredictionId)
        REFERENCES StockPredictions(Id) ON DELETE CASCADE
);

CREATE INDEX IX_TemporalAttention_PredictionId ON PredictionTemporalAttention(PredictionId);

COMMIT TRANSACTION;

-- Add comment describing the schema update
EXEC sp_addextendedproperty
    @name = N'MS_Description',
    @value = N'Multi-horizon predictions from TFT (Temporal Fusion Transformer) model with confidence intervals',
    @level0type = N'SCHEMA', @level0name = N'dbo',
    @level1type = N'TABLE', @level1name = N'StockPredictionHorizons';

EXEC sp_addextendedproperty
    @name = N'MS_Description',
    @value = N'Feature importance weights from TFT variable selection networks',
    @level0type = N'SCHEMA', @level0name = N'dbo',
    @level1type = N'TABLE', @level1name = N'PredictionFeatureImportance';

EXEC sp_addextendedproperty
    @name = N'MS_Description',
    @value = N'Temporal attention weights from TFT interpretable attention mechanism',
    @level0type = N'SCHEMA', @level0name = N'dbo',
    @level1type = N'TABLE', @level1name = N'PredictionTemporalAttention';

GO
