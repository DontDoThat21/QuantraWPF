-- SQL Server Migration Script
-- Add ExpectedFruitionDate, ModelType, ArchitectureType, and TrainingHistoryId columns to StockPredictions table
-- This migration supports the PredictionAnalysis Python Model Analyze button fixes feature

-- Add ExpectedFruitionDate column to track when the prediction is expected to come to fruition
IF NOT EXISTS (SELECT 1 FROM sys.columns 
               WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') 
               AND name = 'ExpectedFruitionDate')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD [ExpectedFruitionDate] DATETIME2(7) NULL;
    
    PRINT 'Added ExpectedFruitionDate column to StockPredictions table';
END
ELSE
BEGIN
    PRINT 'ExpectedFruitionDate column already exists in StockPredictions table';
END
GO

-- Add ModelType column to track which ML model generated the prediction
IF NOT EXISTS (SELECT 1 FROM sys.columns 
               WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') 
               AND name = 'ModelType')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD [ModelType] NVARCHAR(50) NULL;
    
    PRINT 'Added ModelType column to StockPredictions table';
END
ELSE
BEGIN
    PRINT 'ModelType column already exists in StockPredictions table';
END
GO

-- Add ArchitectureType column to track which neural network architecture was used
IF NOT EXISTS (SELECT 1 FROM sys.columns 
               WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') 
               AND name = 'ArchitectureType')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD [ArchitectureType] NVARCHAR(50) NULL;
    
    PRINT 'Added ArchitectureType column to StockPredictions table';
END
ELSE
BEGIN
    PRINT 'ArchitectureType column already exists in StockPredictions table';
END
GO

-- Add TrainingHistoryId column to reference the training session that generated this prediction
IF NOT EXISTS (SELECT 1 FROM sys.columns 
               WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') 
               AND name = 'TrainingHistoryId')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD [TrainingHistoryId] INT NULL;
    
    PRINT 'Added TrainingHistoryId column to StockPredictions table';
END
ELSE
BEGIN
    PRINT 'TrainingHistoryId column already exists in StockPredictions table';
END
GO

-- Create an index on ExpectedFruitionDate for efficient querying of upcoming predictions
IF NOT EXISTS (SELECT 1 FROM sys.indexes 
               WHERE name = 'IX_StockPredictions_ExpectedFruitionDate' 
               AND object_id = OBJECT_ID(N'[dbo].[StockPredictions]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_StockPredictions_ExpectedFruitionDate]
    ON [dbo].[StockPredictions] ([ExpectedFruitionDate] ASC)
    WHERE [ExpectedFruitionDate] IS NOT NULL;
    
    PRINT 'Created index IX_StockPredictions_ExpectedFruitionDate';
END
ELSE
BEGIN
    PRINT 'Index IX_StockPredictions_ExpectedFruitionDate already exists';
END
GO

-- Create an index on ModelType and ArchitectureType for efficient model-based filtering
IF NOT EXISTS (SELECT 1 FROM sys.indexes 
               WHERE name = 'IX_StockPredictions_ModelType_ArchitectureType' 
               AND object_id = OBJECT_ID(N'[dbo].[StockPredictions]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_StockPredictions_ModelType_ArchitectureType]
    ON [dbo].[StockPredictions] ([ModelType] ASC, [ArchitectureType] ASC)
    WHERE [ModelType] IS NOT NULL;
    
    PRINT 'Created index IX_StockPredictions_ModelType_ArchitectureType';
END
ELSE
BEGIN
    PRINT 'Index IX_StockPredictions_ModelType_ArchitectureType already exists';
END
GO

-- Add foreign key constraint to ModelTrainingHistory table (if both tables exist)
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ModelTrainingHistory')
   AND NOT EXISTS (SELECT 1 FROM sys.foreign_keys 
                   WHERE name = 'FK_StockPredictions_TrainingHistoryId')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD CONSTRAINT [FK_StockPredictions_TrainingHistoryId]
    FOREIGN KEY ([TrainingHistoryId]) REFERENCES [dbo].[ModelTrainingHistory] ([Id]);
    
    PRINT 'Added foreign key constraint FK_StockPredictions_TrainingHistoryId';
END
ELSE
BEGIN
    PRINT 'Foreign key constraint FK_StockPredictions_TrainingHistoryId already exists or ModelTrainingHistory table does not exist';
END
GO

PRINT 'Migration completed successfully!';
