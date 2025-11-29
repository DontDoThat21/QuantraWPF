# StockPredictions ChatHistoryId and UserQuery Migration

## Overview

This migration adds two new columns to the `StockPredictions` table to support the **MarketChat AI integration** feature. These columns allow predictions to be linked to specific chat conversations and store the original user query that triggered the prediction.

## Problem

When saving predictions through the `PredictionAnalysisService.SavePredictionAsync` method, the following SQL error occurs:

```
SqlException: Invalid column name 'ChatHistoryId'.
Invalid column name 'UserQuery'.
```

This error occurs because:
1. The `StockPredictionEntity` C# class defines these properties
2. The Entity Framework configuration expects these columns
3. **BUT** the actual database table does not have these columns

## Solution

Run the migration script `AddChatHistoryFieldsToStockPredictions.sql` to add the missing columns.

## Columns Added

### 1. ChatHistoryId (INT NULL)
- **Purpose**: Links a prediction to a specific MarketChat conversation
- **Type**: Nullable integer (foreign key to `ChatHistory.Id`)
- **Usage**: Allows tracking which chat session/message triggered a prediction
- **Example**: When a user asks "Run new prediction for TSLA" in MarketChat, the resulting prediction is linked to that chat message

### 2. UserQuery (NVARCHAR(1000) NULL)
- **Purpose**: Stores the original user query that triggered the prediction
- **Type**: Nullable string (max 1000 characters)
- **Usage**: Provides audit trail and context for why a prediction was generated
- **Example**: Stores the exact question like "Run LSTM model for AAPL" or "Generate fresh prediction for MSFT"

## Database Objects Created

1. **Columns**: `ChatHistoryId` and `UserQuery` in `StockPredictions` table
2. **Index**: `IX_StockPredictions_ChatHistoryId` for improved query performance
3. **Foreign Key**: `FK_StockPredictions_ChatHistory` (if ChatHistory table exists)

## How to Run the Migration

### Option 1: SQL Server Management Studio (SSMS)
1. Open SQL Server Management Studio
2. Connect to your database server
3. Open the file: `Quantra.DAL\Migrations\AddChatHistoryFieldsToStockPredictions.sql`
4. Ensure you're connected to the correct database (or modify the `USE [master]` statement)
5. Execute the script (F5)

### Option 2: Command Line (sqlcmd)
```bash
sqlcmd -S (localdb)\MSSQLLocalDB -d QuantraRelational -i "Quantra.DAL\Migrations\AddChatHistoryFieldsToStockPredictions.sql"
```

### Option 3: Azure Data Studio
1. Open Azure Data Studio
2. Connect to your database
3. Open the migration script
4. Click "Run"

## Verification

After running the migration, verify the columns exist:

```sql
-- Check if columns were added
SELECT 
    COLUMN_NAME, 
    DATA_TYPE, 
    CHARACTER_MAXIMUM_LENGTH,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'StockPredictions'
AND COLUMN_NAME IN ('ChatHistoryId', 'UserQuery')
ORDER BY COLUMN_NAME

-- Check if index was created
SELECT name, type_desc 
FROM sys.indexes 
WHERE object_id = OBJECT_ID('StockPredictions') 
AND name = 'IX_StockPredictions_ChatHistoryId'

-- Check if foreign key was created
SELECT name, parent_object_id, referenced_object_id
FROM sys.foreign_keys
WHERE name = 'FK_StockPredictions_ChatHistory'
```

## Related Features

These columns are part of the **MarketChat AI Tool Integration** feature set:

- **MarketChat Story 9**: Python Model Orchestration
  - Allows users to request fresh predictions via chat commands like "Run new prediction for TSLA"
  - Predictions are cached and linked to the chat session
  
- **MarketChat Story 3**: PredictionCache Integration
  - Chat responses indicate whether data came from cache or fresh generation
  - ChatHistoryId helps track which conversations used cached vs. fresh predictions

## Impact on Application

After running this migration:
- ? `PredictionAnalysisService.SavePredictionAsync` will work without SQL errors
- ? MarketChat can store prediction metadata for audit trails
- ? Future features can query predictions by chat session
- ? User queries are preserved for analysis and debugging

## Notes

- Both columns are **nullable** - existing predictions won't be affected
- The foreign key uses `ON DELETE SET NULL` - deleting a chat history won't delete predictions
- The index improves performance when querying predictions by chat history
- If the `ChatHistory` table doesn't exist yet, the foreign key won't be created (script handles this gracefully)

## Rollback (if needed)

To rollback this migration:

```sql
-- Remove foreign key (if exists)
IF EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_StockPredictions_ChatHistory')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    DROP CONSTRAINT [FK_StockPredictions_ChatHistory]
END

-- Remove index
IF EXISTS (SELECT * FROM sys.indexes WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') AND name = N'IX_StockPredictions_ChatHistoryId')
BEGIN
    DROP INDEX [IX_StockPredictions_ChatHistoryId] ON [dbo].[StockPredictions]
END

-- Remove columns
ALTER TABLE [dbo].[StockPredictions]
DROP COLUMN [ChatHistoryId], [UserQuery]
```

## Related Files

- **Entity Model**: `Quantra.DAL\Data\Entities\PredictionEntities.cs` (StockPredictionEntity)
- **EF Configuration**: `Quantra.DAL\Data\Configurations\EntityConfigurations.cs` (StockPredictionConfiguration)
- **Service Using Columns**: `Quantra.DAL\Services\PredictionAnalysisService.cs` (SavePredictionAsync)
- **MarketChat Integration**: `Quantra.DAL\Services\MarketChatService.cs` (ProcessPredictionRequestAsync)
