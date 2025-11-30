-- SQL Script to create InsiderTransactions cache table
-- Run this in your SQLite database to enable insider transactions caching

CREATE TABLE IF NOT EXISTS InsiderTransactions (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Symbol TEXT NOT NULL,
    FilingDate TEXT NOT NULL,
    TransactionDate TEXT NOT NULL,
    OwnerName TEXT,
    OwnerCik TEXT,
    OwnerTitle TEXT,
    SecurityType TEXT,
    TransactionCode TEXT,
    SharesTraded INTEGER NOT NULL DEFAULT 0,
    PricePerShare REAL NOT NULL DEFAULT 0.0,
    SharesOwnedFollowing INTEGER NOT NULL DEFAULT 0,
    AcquisitionOrDisposal TEXT,
    LastUpdated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(Symbol, FilingDate, TransactionDate, OwnerCik)
);

-- Create index on Symbol for fast lookups
CREATE INDEX IF NOT EXISTS idx_insider_transactions_symbol ON InsiderTransactions(Symbol);

-- Create index on TransactionDate for date-range queries
CREATE INDEX IF NOT EXISTS idx_insider_transactions_date ON InsiderTransactions(TransactionDate);

-- Create index on LastUpdated for cache expiry checks
CREATE INDEX IF NOT EXISTS idx_insider_transactions_lastupdated ON InsiderTransactions(LastUpdated);
