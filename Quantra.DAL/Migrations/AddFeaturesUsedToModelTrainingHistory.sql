-- SQL Server Migration Script
-- Add FeaturesUsed column to ModelTrainingHistory table
-- This column stores a JSON list of features used to train the model

-- Add FeaturesUsed column to store JSON list of features
IF NOT EXISTS (SELECT 1 FROM sys.columns 
               WHERE object_id = OBJECT_ID(N'[dbo].[ModelTrainingHistory]') 
               AND name = 'FeaturesUsed')
BEGIN
    ALTER TABLE [dbo].[ModelTrainingHistory]
    ADD [FeaturesUsed] NVARCHAR(MAX) NULL;
    
    PRINT 'Added FeaturesUsed column to ModelTrainingHistory table';
END
ELSE
BEGIN
    PRINT 'FeaturesUsed column already exists in ModelTrainingHistory table';
END
GO

-- Add a regular nullable column for feature count (updated via trigger or application code)
IF NOT EXISTS (SELECT 1 FROM sys.columns 
               WHERE object_id = OBJECT_ID(N'[dbo].[ModelTrainingHistory]') 
               AND name = 'FeatureCount')
BEGIN
    ALTER TABLE [dbo].[ModelTrainingHistory]
    ADD [FeatureCount] INT NULL;
    
    PRINT 'Added FeatureCount column';
    
    -- Update existing rows with feature counts
    UPDATE [dbo].[ModelTrainingHistory]
    SET [FeatureCount] = (
        SELECT COUNT(*) 
        FROM OPENJSON([FeaturesUsed])
    )
    WHERE [FeaturesUsed] IS NOT NULL AND [FeaturesUsed] != '[]';
    
    PRINT 'Updated FeatureCount for existing rows';
END
ELSE
BEGIN
    PRINT 'FeatureCount column already exists';
END
GO

-- Create an index on FeatureCount for efficient filtering
IF NOT EXISTS (SELECT 1 FROM sys.indexes 
               WHERE name = 'IX_ModelTrainingHistory_FeatureCount' 
               AND object_id = OBJECT_ID(N'[dbo].[ModelTrainingHistory]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ModelTrainingHistory_FeatureCount]
    ON [dbo].[ModelTrainingHistory] ([FeatureCount] ASC)
    WHERE [FeatureCount] IS NOT NULL;
    
    PRINT 'Created index IX_ModelTrainingHistory_FeatureCount';
END
ELSE
BEGIN
    PRINT 'Index IX_ModelTrainingHistory_FeatureCount already exists';
END
GO

PRINT 'Migration completed successfully!';
PRINT 'You can now query feature lists like this:';
PRINT '  SELECT Id, ModelType, FeatureCount, FeaturesUsed FROM ModelTrainingHistory';
PRINT '  SELECT feature.value FROM ModelTrainingHistory CROSS APPLY OPENJSON(FeaturesUsed) AS feature WHERE Id = 1';
