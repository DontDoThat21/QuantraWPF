-- =====================================================
-- Create All Paper Trading Tables - Master Script
-- Runs all table creation scripts in the correct order
-- =====================================================

USE [QuantraRelational]
GO

PRINT '=========================================================='
PRINT 'Starting Paper Trading Database Schema Creation'
PRINT '=========================================================='
PRINT ''

-- Step 1: Create PaperTradingSessions table (parent table)
PRINT 'STEP 1: Creating PaperTradingSessions table...'
PRINT '----------------------------------------------------------'

-- Drop tables in reverse order of dependencies if they exist
IF OBJECT_ID('dbo.PaperTradingFills', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingFills table...'
    DROP TABLE dbo.PaperTradingFills
END

IF OBJECT_ID('dbo.PaperTradingOrders', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingOrders table...'
    DROP TABLE dbo.PaperTradingOrders
END

IF OBJECT_ID('dbo.PaperTradingPositions', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingPositions table...'
    DROP TABLE dbo.PaperTradingPositions
END

IF OBJECT_ID('dbo.PaperTradingSessions', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingSessions table...'
    DROP TABLE dbo.PaperTradingSessions
END
GO

PRINT ''
PRINT 'Creating PaperTradingSessions table...'

CREATE TABLE dbo.PaperTradingSessions (
    Id INT PRIMARY KEY IDENTITY(1,1),
    SessionId NVARCHAR(36) NOT NULL,
    [Name] NVARCHAR(200) NULL,
    InitialCash DECIMAL(18,4) NOT NULL,
    CashBalance DECIMAL(18,4) NOT NULL,
    RealizedPnL DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    IsActive BIT NOT NULL DEFAULT 1,
    StartedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    EndedAt DATETIME2 NULL,
    LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
)
GO

CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingSessions_SessionId 
ON dbo.PaperTradingSessions(SessionId)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingSessions_IsActive 
ON dbo.PaperTradingSessions(IsActive) 
INCLUDE (SessionId, StartedAt)
GO

PRINT '  ? PaperTradingSessions table created'
PRINT ''

-- Step 2: Create PaperTradingPositions table
PRINT 'STEP 2: Creating PaperTradingPositions table...'
PRINT '----------------------------------------------------------'

CREATE TABLE dbo.PaperTradingPositions (
    Id INT PRIMARY KEY IDENTITY(1,1),
    SessionId INT NOT NULL,
    Symbol NVARCHAR(20) NOT NULL,
    Quantity INT NOT NULL,
    AverageCost DECIMAL(18,4) NOT NULL,
    CurrentPrice DECIMAL(18,4) NOT NULL,
    UnrealizedPnL DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    RealizedPnL DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    AssetType NVARCHAR(20) NOT NULL,
    IsClosed BIT NOT NULL DEFAULT 0,
    OpenedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ClosedAt DATETIME2 NULL,
    LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT FK_PaperTradingPositions_Session 
        FOREIGN KEY (SessionId) 
        REFERENCES dbo.PaperTradingSessions(Id) 
        ON DELETE CASCADE
)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingPositions_SessionId 
ON dbo.PaperTradingPositions(SessionId)
INCLUDE (Symbol, Quantity, IsClosed)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingPositions_SessionId_IsClosed 
ON dbo.PaperTradingPositions(SessionId, IsClosed)
INCLUDE (Symbol, Quantity, AverageCost, CurrentPrice, UnrealizedPnL, RealizedPnL)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingPositions_Symbol 
ON dbo.PaperTradingPositions(Symbol)
INCLUDE (SessionId, Quantity, IsClosed)
GO

CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingPositions_Session_Symbol_Open 
ON dbo.PaperTradingPositions(SessionId, Symbol) 
WHERE IsClosed = 0
GO

PRINT '  ? PaperTradingPositions table created'
PRINT ''

-- Step 3: Create PaperTradingOrders table
PRINT 'STEP 3: Creating PaperTradingOrders table...'
PRINT '----------------------------------------------------------'

CREATE TABLE dbo.PaperTradingOrders (
    Id INT PRIMARY KEY IDENTITY(1,1),
    OrderId NVARCHAR(36) NOT NULL,
    SessionId INT NOT NULL,
    Symbol NVARCHAR(20) NOT NULL,
    OrderType NVARCHAR(20) NOT NULL,
    Side NVARCHAR(10) NOT NULL,
    [State] NVARCHAR(20) NOT NULL,
    Quantity INT NOT NULL,
    FilledQuantity INT NOT NULL DEFAULT 0,
    LimitPrice DECIMAL(18,4) NULL,
    StopPrice DECIMAL(18,4) NULL,
    AvgFillPrice DECIMAL(18,4) NULL,
    TimeInForce NVARCHAR(10) NOT NULL,
    AssetType NVARCHAR(20) NOT NULL,
    RejectReason NVARCHAR(500) NULL,
    Notes NVARCHAR(1000) NULL,
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    SubmittedAt DATETIME2 NULL,
    FilledAt DATETIME2 NULL,
    ExpirationTime DATETIME2 NULL,
    LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT FK_PaperTradingOrders_Session 
        FOREIGN KEY (SessionId) 
        REFERENCES dbo.PaperTradingSessions(Id) 
        ON DELETE CASCADE
)
GO

CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingOrders_OrderId 
ON dbo.PaperTradingOrders(OrderId)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_SessionId 
ON dbo.PaperTradingOrders(SessionId)
INCLUDE (Symbol, OrderType, Side, [State], CreatedAt)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_SessionId_State 
ON dbo.PaperTradingOrders(SessionId, [State])
INCLUDE (OrderId, Symbol, OrderType, Side, Quantity, FilledQuantity)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_Symbol 
ON dbo.PaperTradingOrders(Symbol)
INCLUDE (SessionId, [State], CreatedAt)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_CreatedAt 
ON dbo.PaperTradingOrders(CreatedAt DESC)
INCLUDE (SessionId, OrderId, Symbol, [State])
GO

PRINT '  ? PaperTradingOrders table created'
PRINT ''

-- Step 4: Create PaperTradingFills table
PRINT 'STEP 4: Creating PaperTradingFills table...'
PRINT '----------------------------------------------------------'

CREATE TABLE dbo.PaperTradingFills (
    Id INT PRIMARY KEY IDENTITY(1,1),
    FillId NVARCHAR(36) NOT NULL,
    OrderEntityId INT NOT NULL,
    PositionEntityId INT NULL,
    Symbol NVARCHAR(20) NOT NULL,
    Quantity INT NOT NULL,
    Price DECIMAL(18,4) NOT NULL,
    Side NVARCHAR(10) NOT NULL,
    Commission DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    Slippage DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    Exchange NVARCHAR(50) NULL,
    FillTime DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT FK_PaperTradingFills_Order 
        FOREIGN KEY (OrderEntityId) 
        REFERENCES dbo.PaperTradingOrders(Id) 
        ON DELETE CASCADE,
    
    CONSTRAINT FK_PaperTradingFills_Position 
        FOREIGN KEY (PositionEntityId) 
        REFERENCES dbo.PaperTradingPositions(Id) 
        ON DELETE NO ACTION
)
GO

CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingFills_FillId 
ON dbo.PaperTradingFills(FillId)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingFills_OrderEntityId 
ON dbo.PaperTradingFills(OrderEntityId)
INCLUDE (FillId, Symbol, Quantity, Price, FillTime)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingFills_PositionEntityId 
ON dbo.PaperTradingFills(PositionEntityId)
INCLUDE (Symbol, Quantity, Price, FillTime)
WHERE PositionEntityId IS NOT NULL
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingFills_Symbol 
ON dbo.PaperTradingFills(Symbol)
INCLUDE (OrderEntityId, Quantity, Price, FillTime)
GO

CREATE NONCLUSTERED INDEX IX_PaperTradingFills_FillTime 
ON dbo.PaperTradingFills(FillTime DESC)
INCLUDE (FillId, OrderEntityId, Symbol, Quantity, Price)
GO

PRINT '  ? PaperTradingFills table created'
PRINT ''

-- Verification
PRINT ''
PRINT '=========================================================='
PRINT 'Verifying All Tables...'
PRINT '=========================================================='
PRINT ''

DECLARE @SessionsExists INT = OBJECT_ID('dbo.PaperTradingSessions', 'U')
DECLARE @PositionsExists INT = OBJECT_ID('dbo.PaperTradingPositions', 'U')
DECLARE @OrdersExists INT = OBJECT_ID('dbo.PaperTradingOrders', 'U')
DECLARE @FillsExists INT = OBJECT_ID('dbo.PaperTradingFills', 'U')

IF @SessionsExists IS NOT NULL
    PRINT '? PaperTradingSessions table exists'
ELSE
    PRINT '? PaperTradingSessions table MISSING'

IF @PositionsExists IS NOT NULL
    PRINT '? PaperTradingPositions table exists'
ELSE
    PRINT '? PaperTradingPositions table MISSING'

IF @OrdersExists IS NOT NULL
    PRINT '? PaperTradingOrders table exists'
ELSE
    PRINT '? PaperTradingOrders table MISSING'

IF @FillsExists IS NOT NULL
    PRINT '? PaperTradingFills table exists'
ELSE
    PRINT '? PaperTradingFills table MISSING'

PRINT ''

IF @SessionsExists IS NOT NULL AND @PositionsExists IS NOT NULL 
   AND @OrdersExists IS NOT NULL AND @FillsExists IS NOT NULL
BEGIN
    PRINT '=========================================================='
    PRINT '??? All Paper Trading tables created successfully! ???'
    PRINT '=========================================================='
END
ELSE
BEGIN
    PRINT '=========================================================='
    PRINT '??? Some tables are missing! Please review errors. ???'
    PRINT '=========================================================='
END

PRINT ''
PRINT 'Schema creation complete.'
GO
