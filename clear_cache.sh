#!/bin/bash

# Vision Zero Toronto - Cache Clearing Script
# This script clears the processed collision data cache to force fresh data load

echo "=========================================="
echo "Vision Zero Toronto - Clear Cache"
echo "=========================================="
echo ""

# Define cache file paths
CACHE_PARQUET="processed_collisions_cache.parquet"
CACHE_PICKLE="processed_collisions_cache.pkl"

# Check and remove parquet cache
if [ -f "$CACHE_PARQUET" ]; then
    echo "Found: $CACHE_PARQUET"
    FILE_SIZE=$(du -h "$CACHE_PARQUET" | cut -f1)
    echo "  Size: $FILE_SIZE"
    rm -f "$CACHE_PARQUET"
    echo "  ✓ Deleted $CACHE_PARQUET"
else
    echo "Not found: $CACHE_PARQUET (skip)"
fi

echo ""

# Check and remove pickle cache
if [ -f "$CACHE_PICKLE" ]; then
    echo "Found: $CACHE_PICKLE"
    FILE_SIZE=$(du -h "$CACHE_PICKLE" | cut -f1)
    echo "  Size: $FILE_SIZE"
    rm -f "$CACHE_PICKLE"
    echo "  ✓ Deleted $CACHE_PICKLE"
else
    echo "Not found: $CACHE_PICKLE (skip)"
fi

echo ""
echo "=========================================="
echo "Cache cleared successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart your Flask application"
echo "2. The app will regenerate the cache on next startup"
echo "3. Verify the homepage shows 18,351 party records"
echo ""
