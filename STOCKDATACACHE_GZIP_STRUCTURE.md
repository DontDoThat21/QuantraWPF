# StockDataCache Table - GZip Data Structure Documentation

## Table Schema

```sql
CREATE TABLE [dbo].[StockDataCache] (
    [Symbol]     NVARCHAR(20)  NOT NULL,
    [TimeRange]  NVARCHAR(50)  NOT NULL,
    [Interval]   NVARCHAR(50)  NULL,
    [Data]       NVARCHAR(MAX) NOT NULL,  -- Contains GZip compressed JSON
    [CacheTime]  DATETIME2     NOT NULL
)
```

## Data Column Structure

The `Data` column contains **GZip-compressed JSON** data stored as a Base64-encoded string with a special marker prefix.

### Storage Format

```
GZIP:[Base64-encoded GZip compressed data]
```

**Example:**
```
GZIP:H4sIAAAAAAAACo2Wy27bMBBFf0XQugh...
```

## Compression/Decompression Pipeline

### Compression Process (Writing to Database)

**File:** `StockDataCacheService.cs:189-195`

```csharp
// Step 1: Serialize List<HistoricalPrice> to JSON
var jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(data);

// Step 2: Compress JSON string using GZip
var compressedData = Quantra.Utilities.CompressionHelper.CompressString(jsonData);

// Step 3: Store compressed string in database
```

**Detailed Compression Steps:**

1. **JSON Serialization**: `List<HistoricalPrice>` → JSON string
2. **UTF-8 Encoding**: JSON string → byte array
3. **GZip Compression**: byte array → compressed byte array
4. **Base64 Encoding**: compressed bytes → Base64 string
5. **Marker Addition**: Prepend "GZIP:" prefix
6. **Database Storage**: Store final string in Data column

**Implementation:** `CompressionHelper.cs:21-40`
```csharp
public static string CompressString(string input)
{
    // Convert string to bytes
    byte[] inputBytes = Encoding.UTF8.GetBytes(input);

    using (var memoryStream = new MemoryStream())
    {
        using (var gzipStream = new GZipStream(memoryStream, CompressionMode.Compress))
        {
            gzipStream.Write(inputBytes, 0, inputBytes.Length);
        }

        // Convert compressed data to Base64 string with marker
        byte[] compressedBytes = memoryStream.ToArray();
        return "GZIP:" + Convert.ToBase64String(compressedBytes);
    }
}
```

### Decompression Process (Reading from Database)

**File:** `StockDataCacheService.cs:160-167`

```csharp
var storedData = cacheEntry.Data;
string jsonData;

// Check if data is compressed and decompress if needed
if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
{
    jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
}
else
{
    jsonData = storedData;  // Legacy uncompressed data
}

// Deserialize JSON to List<HistoricalPrice>
var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
```

**Detailed Decompression Steps:**

1. **Check Marker**: Verify string starts with "GZIP:"
2. **Remove Marker**: Strip "GZIP:" prefix
3. **Base64 Decode**: Base64 string → compressed byte array
4. **GZip Decompress**: compressed bytes → original byte array
5. **UTF-8 Decode**: byte array → JSON string
6. **JSON Deserialization**: JSON string → `List<HistoricalPrice>`

**Implementation:** `CompressionHelper.cs:47-73`
```csharp
public static string DecompressString(string compressedInput)
{
    // Check if the data is compressed (has the marker)
    if (!compressedInput.StartsWith("GZIP:"))
        return compressedInput; // Return as-is if not compressed

    // Remove the marker and convert from Base64 to byte array
    string base64 = compressedInput.Substring("GZIP:".Length);
    byte[] compressedBytes = Convert.FromBase64String(base64);

    // Decompress
    using (var memoryStream = new MemoryStream(compressedBytes))
    {
        using (var resultStream = new MemoryStream())
        {
            using (var gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress))
            {
                gzipStream.CopyTo(resultStream);
            }
            return Encoding.UTF8.GetString(resultStream.ToArray());
        }
    }
}
```

## Uncompressed JSON Structure

Before compression, the data is a **JSON array** of `HistoricalPrice` objects.

### HistoricalPrice Model

**File:** `Quantra.DAL\Models\HistoricalPrice.cs`

```csharp
public class HistoricalPrice
{
    public DateTime Date { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public long Volume { get; set; }
    public double AdjClose { get; set; }
}
```

### JSON Example (Before Compression)

```json
[
  {
    "Date": "2024-01-02T00:00:00",
    "Open": 185.64,
    "High": 186.95,
    "Low": 184.30,
    "Close": 185.92,
    "Volume": 82488600,
    "AdjClose": 185.92
  },
  {
    "Date": "2024-01-03T00:00:00",
    "Open": 184.22,
    "High": 185.88,
    "Low": 183.43,
    "Close": 184.25,
    "Volume": 58414480,
    "AdjClose": 184.25
  },
  {
    "Date": "2024-01-04T00:00:00",
    "Open": 182.15,
    "High": 182.76,
    "Low": 180.93,
    "Close": 181.91,
    "Volume": 89621820,
    "AdjClose": 181.91
  }
  // ... more historical price entries
]
```

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE (WRITE) FLOW                         │
└─────────────────────────────────────────────────────────────────────┘

List<HistoricalPrice> (C# Objects)
         │
         │ Newtonsoft.Json.JsonConvert.SerializeObject()
         ▼
JSON String (Uncompressed)
[{"Date":"2024-01-02T00:00:00","Open":185.64,"High":186.95, ...}, ...]
         │
         │ Encoding.UTF8.GetBytes()
         ▼
Byte Array (UTF-8 encoded JSON)
[0x5B, 0x7B, 0x22, 0x44, 0x61, 0x74, 0x65, ...]
         │
         │ GZipStream.Write() (Compression Mode)
         ▼
Compressed Byte Array (GZip)
[0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, ...]
         │
         │ Convert.ToBase64String()
         ▼
Base64 String
"H4sIAAAAAAAACo2Wy27bMBBFf0XQugh..."
         │
         │ Prepend "GZIP:" marker
         ▼
Final Stored String
"GZIP:H4sIAAAAAAAACo2Wy27bMBBFf0XQugh..."
         │
         │ Store in Database
         ▼
[StockDataCache].[Data] NVARCHAR(MAX)


┌─────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL (READ) FLOW                         │
└─────────────────────────────────────────────────────────────────────┘

[StockDataCache].[Data] NVARCHAR(MAX)
         │
         │ Read from Database
         ▼
Stored String
"GZIP:H4sIAAAAAAAACo2Wy27bMBBFf0XQugh..."
         │
         │ Check if StartsWith("GZIP:")
         ▼
[Yes - Compressed Data]
         │
         │ Remove "GZIP:" marker
         ▼
Base64 String
"H4sIAAAAAAAACo2Wy27bMBBFf0XQugh..."
         │
         │ Convert.FromBase64String()
         ▼
Compressed Byte Array (GZip)
[0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, ...]
         │
         │ GZipStream.CopyTo() (Decompress Mode)
         ▼
Decompressed Byte Array (UTF-8)
[0x5B, 0x7B, 0x22, 0x44, 0x61, 0x74, 0x65, ...]
         │
         │ Encoding.UTF8.GetString()
         ▼
JSON String (Uncompressed)
[{"Date":"2024-01-02T00:00:00","Open":185.64,"High":186.95, ...}, ...]
         │
         │ JsonConvert.DeserializeObject<List<HistoricalPrice>>()
         ▼
List<HistoricalPrice> (C# Objects)
```

## Data Size Analysis

### Typical Compression Ratio

For historical stock data (daily OHLCV data):

| Data Type | Uncompressed Size | Compressed Size | Compression Ratio |
|-----------|------------------|-----------------|-------------------|
| 1 year (252 days) | ~15 KB | ~3-4 KB | ~75% reduction |
| 5 years (1260 days) | ~75 KB | ~15-18 KB | ~75% reduction |
| 10 years (2520 days) | ~150 KB | ~30-35 KB | ~75-80% reduction |

**Benefits:**
- Significantly reduced database storage requirements
- Faster database backup/restore operations
- Reduced network transfer time when querying data
- Better SQL Server memory utilization

## Code References

### Entity Definition
- **File:** `Quantra.DAL\Data\Entities\StockEntities.cs:33-52`
- **Class:** `StockDataCache`

### Compression Logic
- **File:** `Quantra\Utilities\CompressionHelper.cs`
- **Methods:**
  - `CompressString()` - lines 21-40
  - `DecompressString()` - lines 47-73
  - `IsCompressed()` - lines 80-83

### Cache Service
- **File:** `Quantra.DAL\Services\StockDataCacheService.cs`
- **Write Method:** `CacheStockData()` - lines 189-220
- **Read Method:** `GetCachedData()` - lines 130-178

### Data Model
- **File:** `Quantra.DAL\Models\HistoricalPrice.cs`
- **Class:** `HistoricalPrice`

## Checking If Data Is Compressed

```csharp
public static bool IsCompressed(string data)
{
    return !string.IsNullOrEmpty(data) && data.StartsWith("GZIP:");
}
```

**Usage in Queries:**

```csharp
var storedData = cacheEntry.Data;

if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
{
    // Data is compressed - decompress before using
    var jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
}
else
{
    // Legacy uncompressed data - use directly
    var jsonData = storedData;
}
```

## Backward Compatibility

The system supports **both compressed and uncompressed data**:

1. **New Data**: Stored with "GZIP:" marker (compressed)
2. **Legacy Data**: Stored without marker (uncompressed JSON)
3. **Detection**: `IsCompressed()` method checks for marker
4. **Handling**: Code automatically decompresses if needed

This ensures smooth migration from old uncompressed data to new compressed data without breaking existing cache entries.

## SQL Query Examples

### View Raw Compressed Data

```sql
SELECT TOP 10
    Symbol,
    TimeRange,
    Interval,
    LEFT(Data, 50) + '...' AS CompressedDataPreview,
    LEN(Data) AS DataLength,
    CacheTime
FROM StockDataCache
ORDER BY CacheTime DESC
```

### Check Compression Status

```sql
SELECT
    Symbol,
    TimeRange,
    CASE
        WHEN Data LIKE 'GZIP:%' THEN 'Compressed'
        ELSE 'Uncompressed'
    END AS CompressionStatus,
    LEN(Data) AS DataLength,
    CacheTime
FROM StockDataCache
ORDER BY CacheTime DESC
```

### Count Compressed vs Uncompressed

```sql
SELECT
    CASE
        WHEN Data LIKE 'GZIP:%' THEN 'Compressed'
        ELSE 'Uncompressed'
    END AS CompressionStatus,
    COUNT(*) AS RecordCount,
    AVG(LEN(Data)) AS AvgDataLength,
    SUM(LEN(Data)) AS TotalDataSize
FROM StockDataCache
GROUP BY CASE
    WHEN Data LIKE 'GZIP:%' THEN 'Compressed'
    ELSE 'Uncompressed'
END
```

## Performance Considerations

### Compression Trade-offs

**Pros:**
- ✅ 75-80% storage reduction
- ✅ Faster database I/O (less data to read/write)
- ✅ Reduced backup/restore time
- ✅ Better memory utilization in SQL Server buffer pool

**Cons:**
- ⚠️ CPU overhead for compression/decompression
- ⚠️ Slightly longer read/write operations
- ⚠️ Cannot query JSON content directly in SQL

**Recommendation:** The storage and I/O benefits far outweigh the CPU cost for historical stock data, especially for large datasets.

### When Compression Is Most Beneficial

1. **Large time ranges** (1+ years of daily data)
2. **Infrequently accessed data** (read once, store long-term)
3. **High volume of symbols** (thousands of cache entries)
4. **Limited database storage** (cost reduction)

## Troubleshooting

### Common Issues

#### Issue: "Invalid Base64 string"
**Cause:** Data is corrupted or marker is missing
**Solution:** Check if data starts with "GZIP:" and is valid Base64

#### Issue: "GZip header not found"
**Cause:** Data is not actually GZip compressed
**Solution:** Verify data was compressed with `CompressString()`

#### Issue: "JSON deserialization failed"
**Cause:** Decompressed string is not valid JSON
**Solution:** Check compression/decompression pipeline integrity

### Validation Script

```csharp
// Test compression round-trip
var testData = new List<HistoricalPrice>
{
    new HistoricalPrice
    {
        Date = DateTime.Now,
        Open = 100.0,
        High = 105.0,
        Low = 99.0,
        Close = 102.0,
        Volume = 1000000,
        AdjClose = 102.0
    }
};

// Compress
var json = JsonConvert.SerializeObject(testData);
var compressed = CompressionHelper.CompressString(json);
Console.WriteLine($"Original: {json.Length} bytes");
Console.WriteLine($"Compressed: {compressed.Length} bytes");

// Decompress
var decompressed = CompressionHelper.DecompressString(compressed);
var restored = JsonConvert.DeserializeObject<List<HistoricalPrice>>(decompressed);

// Verify
Console.WriteLine($"Round-trip success: {json == decompressed}");
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-30
**Author:** System Documentation
**Related Files:**
- `StockEntities.cs`
- `CompressionHelper.cs`
- `StockDataCacheService.cs`
- `HistoricalPrice.cs`
