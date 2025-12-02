-- Migration script to update BacktestResults table column types from REAL to FLOAT
-- This fixes the issue where ProfitFactor is set to double.MaxValue for backtests with no losses
-- REAL type has max value ~3.4E+38, FLOAT (float(53)) supports full double range ~1.8E+308
-- Run this script against your database to update the column types

USE [master]
GO

PRINT 'Starting BacktestResults column type migration...'
GO

-- Check if BacktestResults table exists
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'BacktestResults')
BEGIN
    PRINT 'BacktestResults table does not exist. Migration not needed.'
    RETURN
END
GO

-- Helper function to check column type
DECLARE @ColumnName NVARCHAR(128)
DECLARE @CurrentType NVARCHAR(128)

-- List of columns to update from REAL to FLOAT
DECLARE @ColumnsToUpdate TABLE (ColumnName NVARCHAR(128))
INSERT INTO @ColumnsToUpdate VALUES 
    ('TotalReturn'),
    ('MaxDrawdown'),
    ('WinRate'),
    ('SharpeRatio'),
    ('SortinoRatio'),
    ('CAGR'),
    ('CalmarRatio'),
    ('ProfitFactor'),
    ('InformationRatio'),
    ('TotalTransactionCosts'),
    ('GrossReturn'),
    ('InitialCapital'),
    ('FinalEquity')

DECLARE column_cursor CURSOR FOR 
    SELECT ColumnName FROM @ColumnsToUpdate

OPEN column_cursor
FETCH NEXT FROM column_cursor INTO @ColumnName

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Check current column type
    SELECT @CurrentType = t.name 
    FROM sys.columns c
    INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
    WHERE c.object_id = OBJECT_ID(N'[dbo].[BacktestResults]') 
    AND c.name = @ColumnName

    IF @CurrentType IS NOT NULL
    BEGIN
        IF @CurrentType = 'real'
        BEGIN
            -- Column exists and is REAL, need to update to FLOAT
            DECLARE @SQL NVARCHAR(MAX)
            SET @SQL = 'ALTER TABLE [dbo].[BacktestResults] ALTER COLUMN [' + @ColumnName + '] FLOAT NULL'
            
            PRINT 'Updating column ' + @ColumnName + ' from REAL to FLOAT...'
            EXEC sp_executesql @SQL
            PRINT 'Successfully updated column ' + @ColumnName
        END
        ELSE IF @CurrentType = 'float'
        BEGIN
            PRINT 'Column ' + @ColumnName + ' is already FLOAT, skipping...'
        END
        ELSE
        BEGIN
            PRINT 'WARNING: Column ' + @ColumnName + ' has unexpected type: ' + @CurrentType
        END
    END
    ELSE
    BEGIN
        PRINT 'Column ' + @ColumnName + ' does not exist in BacktestResults table'
    END

    FETCH NEXT FROM column_cursor INTO @ColumnName
END

CLOSE column_cursor
DEALLOCATE column_cursor

PRINT 'Migration completed successfully!'
PRINT ''
PRINT 'Summary:'
PRINT '- Updated all numeric columns in BacktestResults from REAL to FLOAT'
PRINT '- FLOAT type supports full double range including double.MaxValue'
PRINT '- This fixes DbUpdateException when ProfitFactor = double.MaxValue'
GO
