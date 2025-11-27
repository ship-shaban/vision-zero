from flask import Flask, render_template, jsonify, send_from_directory, request, Response
from flask_compress import Compress
import pandas as pd
import json
import os
from pathlib import Path
from shapely.geometry import shape, Point
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Union
from werkzeug.datastructures import ImmutableMultiDict

app = Flask(__name__)
Compress(app)  # Enable gzip/brotli compression for all responses

# Add custom Jinja2 filter for number formatting
@app.template_filter('number_format')  # type: ignore
def number_format(value: Union[int, float, str]) -> str:
    """Format number with thousands separator"""
    return "{:,}".format(int(value))

# Configuration
DATA_DIR = Path(__file__).parent  # Use relative path for portability
INPUT_FILE = DATA_DIR / "collision_speed_limit_analysis_UPDATED.csv"  # Fixed filename
WARD_FILE = DATA_DIR / "city-wards" / "City Wards Data - 4326.geojson"
DIVISION_FILE = DATA_DIR / "police-divisions" / "TPS_POLICE_DIVISIONS_-2945003618295158587.geojson"
NEIGHBOURHOOD_FILE = DATA_DIR / "neighbourhoods" / "Neighbourhoods - 4326.geojson"
# PHASE 1 OPT #3: Use Parquet cache (50-70% smaller + faster than JSON)
# Cache version for invalidation (increment when cache format changes)
CACHE_VERSION = 2  # v2: Added duplicate removal in cache generation
# Try Parquet first, fall back to Pickle if parquet libs not available
try:
    import pyarrow  # noqa: F401
    CACHE_FILE = DATA_DIR / f"processed_collisions_cache_v{CACHE_VERSION}.parquet"
    CACHE_FORMAT = 'parquet'
except ImportError:
    try:
        import fastparquet  # noqa: F401
        CACHE_FILE = DATA_DIR / f"processed_collisions_cache_v{CACHE_VERSION}.parquet"
        CACHE_FORMAT = 'parquet'
    except ImportError:
        CACHE_FILE = DATA_DIR / f"processed_collisions_cache_v{CACHE_VERSION}.pkl"
        CACHE_FORMAT = 'pickle'
        print("NOTE: Parquet libraries not available, using pickle cache (install pyarrow or fastparquet for best performance)")

# PHASE 1 OPT #2: Lazy-load GeoJSON data (only load when needed)
# Module-level variables to store loaded GeoJSON (loaded once per request type)
_wards_geojson = None
_divisions_geojson = None
_neighbourhoods_geojson = None

def load_wards_geojson():
    """Lazy-load ward boundaries GeoJSON (only when API endpoint is called)"""
    global _wards_geojson
    if _wards_geojson is None:
        try:
            print("Loading ward boundaries (lazy-load)...")
            with open(WARD_FILE, 'r') as f:
                _wards_geojson = json.load(f)
            print(f"Loaded {len(_wards_geojson['features'])} ward boundaries")
        except FileNotFoundError:
            print(f"ERROR: Ward boundaries file not found at: {WARD_FILE}")
            raise SystemExit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Ward boundaries file is corrupted or invalid JSON: {str(e)}")
            raise SystemExit(1)
    return _wards_geojson

def load_divisions_geojson():
    """Lazy-load police division boundaries GeoJSON (only when API endpoint is called)"""
    global _divisions_geojson
    if _divisions_geojson is None:
        try:
            print("Loading police division boundaries (lazy-load)...")
            with open(DIVISION_FILE, 'r') as f:
                _divisions_geojson = json.load(f)
            print(f"Loaded {len(_divisions_geojson['features'])} police division boundaries")
        except FileNotFoundError:
            print(f"ERROR: Division boundaries file not found at: {DIVISION_FILE}")
            raise SystemExit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Division boundaries file is corrupted or invalid JSON: {str(e)}")
            raise SystemExit(1)
    return _divisions_geojson

def load_neighbourhoods_geojson():
    """Lazy-load neighbourhood boundaries GeoJSON (only when API endpoint is called)"""
    global _neighbourhoods_geojson
    if _neighbourhoods_geojson is None:
        try:
            print("Loading neighbourhood boundaries (lazy-load)...")
            with open(NEIGHBOURHOOD_FILE, 'r') as f:
                _neighbourhoods_geojson = json.load(f)
            print(f"Loaded {len(_neighbourhoods_geojson['features'])} neighbourhood boundaries")
        except FileNotFoundError:
            print(f"ERROR: Neighbourhood boundaries file not found at: {NEIGHBOURHOOD_FILE}")
            raise SystemExit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Neighbourhood boundaries file is corrupted or invalid JSON: {str(e)}")
            raise SystemExit(1)
    return _neighbourhoods_geojson

# Load data once at startup (with caching for faster restarts)
try:
    cache_exists = CACHE_FILE.exists()
    csv_mtime = os.path.getmtime(INPUT_FILE) if INPUT_FILE.exists() else 0
    cache_mtime = os.path.getmtime(CACHE_FILE) if cache_exists else 0
    need_to_save_cache = False

    if cache_exists and cache_mtime > csv_mtime:
        print(f"Loading collision data from {CACHE_FORMAT} cache (fast!)...")
        try:
            # PHASE 1 OPT #3: Load from cache (Parquet or Pickle)
            # Cache contains fully optimized dataframe (pruned columns + categorical dtypes)
            if CACHE_FORMAT == 'parquet':
                df = pd.read_parquet(CACHE_FILE)
            else:  # pickle
                df = pd.read_pickle(CACHE_FILE)
            print(f"Loaded {len(df)} optimized collision records from {CACHE_FORMAT} cache")
        except Exception as e:
            print(f"WARNING: Cache file corrupted, loading from CSV instead...")
            print(f"   Error: {str(e)}")
            need_to_save_cache = True
            df = pd.read_csv(INPUT_FILE, low_memory=False)
            df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()].copy()
            print(f"Loaded {len(df)} collision records from CSV")
    else:
        print("Loading collision data from CSV...")
        need_to_save_cache = True
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()].copy()
        print(f"Loaded {len(df)} collision records")

    # Only do optimization if loading from CSV (cache already has optimized data)
    if need_to_save_cache:
        # PHASE 2 OPT #5: Prune DataFrame columns to only what's needed (reduces memory by ~40%)
        needed_columns = [
            'DATE', 'TIME', 'LATITUDE', 'LONGITUDE', 'STREET1', 'DIVISION',
            'NEIGHBOURHOOD_158', 'ACCLASS', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
            'ROAD_CLASS', 'TRAFFCTL', 'IMPACTYPE', 'PEDESTRIAN', 'CYCLIST',
            'INVTYPE', 'INVAGE', 'INJURY', 'SPEEDING', 'AG_DRIV', 'REDLIGHT',
            'ALCOHOL', 'DRIVCOND', 'DRIVACT', 'VEHTYPE', 'MANOEUVER', 'INITDIR',
            'ACCNUM'  # Required for hybrid collision ID assignment
        ]
        # Only keep columns that exist in the DataFrame
        existing_needed_cols = [col for col in needed_columns if col in df.columns]
        df = df[existing_needed_cols].copy()
        print(f"Pruned DataFrame to {len(existing_needed_cols)} essential columns")

        # PHASE 2 OPT #6: Optimize data types using category dtype (reduces memory by 50-90% for categorical columns)
        categorical_cols = [
            'ACCLASS', 'ROAD_CLASS', 'TRAFFCTL', 'IMPACTYPE', 'LIGHT', 'RDSFCOND',
            'VISIBILITY', 'PEDESTRIAN', 'CYCLIST', 'INVTYPE', 'INJURY', 'SPEEDING',
            'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DRIVCOND', 'DRIVACT', 'VEHTYPE',
            'MANOEUVER', 'INITDIR'
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        print(f"Converted {len([c for c in categorical_cols if c in df.columns])} columns to category dtype")

        # Remove duplicate party records (data quality issue in source)
        records_before = len(df)
        df = df.drop_duplicates().copy()
        duplicates_removed = records_before - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed:,} duplicate party records")
        print(f"Final dataset: {len(df):,} party records")

except FileNotFoundError as e:
    print("\n" + "="*80)
    print("ERROR: Required data file not found!")
    print("="*80)
    print(f"\nMissing file: {INPUT_FILE}")
    print(f"\nPlease ensure the collision data CSV file exists at:")
    print(f"  {INPUT_FILE}")
    print(f"\nData files should be in: {DATA_DIR}")
    print("\nDownload from Toronto Open Data Portal:")
    print("  https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/")
    print("="*80 + "\n")
    raise SystemExit(1)
except Exception as e:
    print("\n" + "="*80)
    print("ERROR: Failed to load collision data!")
    print("="*80)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print(f"\nFile attempted: {INPUT_FILE}")
    print("="*80 + "\n")
    raise SystemExit(1)

# PHASE 1 OPT #1: Fix ward assignment bug - only do expensive processing if cache doesn't exist or is stale
if not (cache_exists and cache_mtime > csv_mtime):
    # Assign wards using proper point-in-polygon
    print("Assigning collisions to wards using point-in-polygon...")

    # Load ward boundaries for assignment (only when needed)
    wards_geojson = load_wards_geojson()

    # Create Shapely geometry objects for each ward
    ward_polygons = {}
    for feature in wards_geojson['features']:
        ward_name = feature['properties']['AREA_DESC']
        ward_code = feature['properties']['AREA_SHORT_CODE']
        polygon = shape(feature['geometry'])
        ward_polygons[ward_name] = {'code': ward_code, 'polygon': polygon}

    print(f"Created {len(ward_polygons)} ward polygon geometries")

    # Assign each collision to a ward using proper point-in-polygon
    ward_assignments = []
    for idx, row in df.iterrows():
        if idx > 0 and idx % 5000 == 0:
            print(f"  Processing collision {idx:,}/{len(df):,}...")

        lat, lon = row['LATITUDE'], row['LONGITUDE']
        point = Point(lon, lat)  # Note: Shapely uses (lon, lat) order
        assigned_ward = None

        # Check each ward polygon
        for ward_name, ward_info in ward_polygons.items():
            if ward_info['polygon'].contains(point):
                assigned_ward = ward_name
                break

        if assigned_ward:
            ward_assignments.append({
                'WARD_DESC': assigned_ward,
                'WARD_NUMBER': ward_polygons[assigned_ward]['code'],
                'WARD_NAME': assigned_ward
            })
        else:
            ward_assignments.append({
                'WARD_DESC': 'Outside Wards',
                'WARD_NUMBER': '00',
                'WARD_NAME': 'Outside Wards'
            })

    df['WARD_DESC'] = [w['WARD_DESC'] for w in ward_assignments]
    df['WARD_NUMBER'] = [w['WARD_NUMBER'] for w in ward_assignments]
    df['WARD_NAME'] = [w['WARD_NAME'] for w in ward_assignments]

    wards_assigned = sum(1 for w in ward_assignments if w['WARD_DESC'] != 'Outside Wards')
    print(f"Assigned {wards_assigned:,} of {len(df):,} collisions to wards (point-in-polygon method)")

# Prepare data
df['COLLISION_DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['YEAR'] = df['COLLISION_DATE'].dt.year

# Note: VULNERABLE_ROAD_USER will be updated after ROAD_USER_CATEGORY is created
# to include Micromobility users
df['VULNERABLE_ROAD_USER_TEMP'] = (
    (df['PEDESTRIAN'] == 'Yes') |
    (df['CYCLIST'] == 'Yes')
).apply(lambda x: 'Yes' if x else 'No')

def get_time_range(time_value: Optional[Union[int, str, float]]) -> str:
    """
    Categorize time into time ranges:
    - Night: 0000 - 0559
    - Morning: 0600 - 1159
    - Afternoon: 1200 - 1759
    - Evening: 1800 - 2359
    """
    if pd.isna(time_value):
        return ''

    try:
        time_int = int(time_value)

        if 0 <= time_int <= 559:
            return 'Night'
        elif 600 <= time_int <= 1159:
            return 'Morning'
        elif 1200 <= time_int <= 1759:
            return 'Afternoon'
        elif 1800 <= time_int <= 2359:
            return 'Evening'
        else:
            return ''  # Invalid time
    except:
        return ''

df['TIME_RANGE'] = df['TIME'].apply(get_time_range)

def categorize_road_user(row: pd.Series) -> str:
    """
    Categorize road users into 6 main categories based on Vision Zero definitions.
    Priority order: Check INVTYPE for explicit types FIRST (handles data quality issues),
    then fall back to PEDESTRIAN/CYCLIST flags if INVTYPE is ambiguous.
    """
    invtype = str(row.get('INVTYPE', '')).lower()

    # Check INVTYPE for specific micromobility devices FIRST (before flags)
    # This includes moped (flagged as CYCLIST), wheelchair (flagged as PEDESTRIAN), skaters (flagged as PEDESTRIAN)
    if 'moped' in invtype or 'wheelchair' in invtype or 'skater' in invtype or 'e-scooter' in invtype or 'skateboard' in invtype:
        return 'Micromobility'

    # Check for motorcycles (motor-powered two-wheeled vehicles with no pedals)
    if 'motorcycle' in invtype:
        return 'Motorcyclist'

    # Check INVTYPE for explicit pedestrian/cyclist designation (handles cases where flags are wrong)
    if 'pedestrian' in invtype:
        return 'Pedestrian'
    if 'cyclist' in invtype:
        return 'Cyclist'

    # Check INVTYPE for driver/passenger BEFORE checking flags (prevents data quality issues)
    # Toronto Open Data sometimes incorrectly sets PEDESTRIAN=Yes for drivers
    if 'driver' in invtype or 'passenger' in invtype:
        return 'Motorist'

    # Fall back to PEDESTRIAN/CYCLIST flags only if INVTYPE didn't match above
    # (for cases where INVTYPE is empty/unknown but flags are set)
    if row.get('PEDESTRIAN') == 'Yes':
        return 'Pedestrian'
    elif row.get('CYCLIST') == 'Yes':
        return 'Cyclist'

    # Check for property owners (vehicles/property involved but person not physically present)
    elif 'owner' in invtype:
        return 'Property Owner'

    else:
        return ''

df['ROAD_USER_CATEGORY'] = df.apply(categorize_road_user, axis=1)

# Update VULNERABLE_ROAD_USER to include Micromobility users
df['VULNERABLE_ROAD_USER'] = (
    (df['PEDESTRIAN'] == 'Yes') |
    (df['CYCLIST'] == 'Yes') |
    (df['ROAD_USER_CATEGORY'] == 'Micromobility')
).apply(lambda x: 'Yes' if x else 'No')

def categorize_age(age_str: Optional[Union[str, float]]) -> str:
    """
    Categorize ages into 4 broad groups using midpoint of age ranges
    """
    if pd.isna(age_str) or age_str == '' or str(age_str).lower() == 'unknown':
        return ''

    age_str = str(age_str).strip()

    # Handle special case "Over 95" - categorize as Seniors
    if age_str.lower().startswith('over'):
        return 'Seniors (65+)'

    # Try to parse age range "X to Y" format and use midpoint
    try:
        parts = age_str.split(' to ')
        if len(parts) == 2:
            # Age range format: calculate midpoint
            age_min = int(parts[0].strip())
            age_max = int(parts[1].strip())
            age = (age_min + age_max) // 2  # Use midpoint
        else:
            # Single number format
            age = int(age_str.split(' ')[0])

        if age <= 14:
            return 'Children (0-14)'
        elif age <= 24:
            return 'Youth (15-24)'
        elif age <= 64:
            return 'Adults (25-64)'
        else:
            return 'Seniors (65+)'
    except:
        return ''

df['AGE_CATEGORY'] = df['INVAGE'].apply(categorize_age)

# Hybrid collision ID assignment (matches analysis report methodology)
# Use ACCNUM where available (74% of data), spatial-temporal clustering for 2015-2019 (26%)
print("Assigning collision IDs using hybrid method (ACCNUM + clustering)...")

df['COLLISION_ID'] = None

# Step 1: Use ACCNUM where populated (2006-2014, 2020-2023)
has_accnum = df['ACCNUM'].notna()
accnum_count = has_accnum.sum()
df.loc[has_accnum, 'COLLISION_ID'] = 'ACCNUM_' + df.loc[has_accnum, 'ACCNUM'].astype(str)
print(f"  ✓ Used ACCNUM for {accnum_count:,} records ({accnum_count/len(df)*100:.1f}%)")

# Step 2: Use spatial-temporal clustering for missing ACCNUM (2015-2019)
missing_accnum = df['ACCNUM'].isna()
missing_count = missing_accnum.sum()

if missing_count > 0:
    # Create cluster keys: DATE + TIME + LAT(4dp) + LON(4dp) for ~11m precision
    def create_cluster_key(row):
        try:
            lat_rounded = round(float(row['LATITUDE']), 4)
            lon_rounded = round(float(row['LONGITUDE']), 4)
            time_int = int(row['TIME']) if pd.notna(row['TIME']) else 0
            time_str = str(time_int).zfill(4)
            time_formatted = f"{time_str[:2]}:{time_str[2:]}"
            date_obj = pd.to_datetime(row['DATE'])
            date_str = date_obj.strftime('%Y-%m-%d')
            return f"{date_str}_{time_formatted}_{lat_rounded}_{lon_rounded}"
        except Exception as e:
            return f"ERROR_{row.name}"

    df.loc[missing_accnum, 'CLUSTER_KEY'] = df.loc[missing_accnum].apply(create_cluster_key, axis=1)
    unique_clusters = df.loc[missing_accnum, 'CLUSTER_KEY'].unique()
    cluster_mapping = {key: f'CLUSTER_{i}' for i, key in enumerate(unique_clusters)}
    df.loc[missing_accnum, 'COLLISION_ID'] = df.loc[missing_accnum, 'CLUSTER_KEY'].map(cluster_mapping)
    df.drop('CLUSTER_KEY', axis=1, inplace=True, errors='ignore')
    print(f"  ✓ Used clustering for {missing_count:,} records ({missing_count/len(df)*100:.1f}%)")

party_counts = df.groupby('COLLISION_ID').size().reset_index(name='NUM_PARTIES_IN_COLLISION')
df = df.merge(party_counts, on='COLLISION_ID', how='left')

unique_collisions = df['COLLISION_ID'].nunique()
print(f"{len(df)} parties representing {unique_collisions} unique collisions")

# Validate collision sizes for data quality
large_collisions = party_counts[party_counts['NUM_PARTIES_IN_COLLISION'] > 10]
if len(large_collisions) > 0:
    print(f"NOTE: {len(large_collisions)} collisions have >10 parties (max: {party_counts['NUM_PARTIES_IN_COLLISION'].max()})")

def clean_value(val: Any) -> str:
    if pd.isna(val):
        return ''
    return str(val)

# Prepare markers data - ONE MARKER PER UNIQUE COLLISION
print("Aggregating data by unique collisions...")
markers_data = []

# Severity order for determining collision-level severity
severity_order = {'Fatal': 3, 'Non-Fatal Injury': 2, 'Property Damage O': 1, '': 0}

for collision_id, collision_group in df.groupby('COLLISION_ID'):
    # Get first row for collision-level data (same for all parties in collision)
    first_row = collision_group.iloc[0]

    # Determine highest severity across all parties
    # Convert categorical back to string for mapping (categorical dtype doesn't support .max() on mapped values)
    max_severity = collision_group['ACCLASS'].astype(str).map(lambda x: severity_order.get(x, 0)).max()
    collision_severity = [k for k, v in severity_order.items() if v == max_severity][0]

    # PHASE 1 OPT #4: Reduce marker data size - aggregate party info but don't include detailed arrays
    # Count party types for filters
    has_pedestrian = any(str(party.get('ROAD_USER_CATEGORY', '')) == 'Pedestrian' for _, party in collision_group.iterrows())
    has_cyclist = any(str(party.get('ROAD_USER_CATEGORY', '')) == 'Cyclist' for _, party in collision_group.iterrows())
    has_vulnerable = any(party.get('VULNERABLE_ROAD_USER') == 'Yes' for _, party in collision_group.iterrows())

    # Get all road user categories involved
    road_user_categories = collision_group['ROAD_USER_CATEGORY'].dropna().unique().tolist()

    # Get all age categories involved
    age_categories = collision_group['AGE_CATEGORY'].dropna().unique().tolist()

    # PHASE 1 OPT #4: Build collision marker with REDUCED fields (remove visibility, light, rdsfcond, speeding, redlight, detailed parties array)
    marker_info = {
        'collision_id': collision_id,
        'lat': round(float(first_row['LATITUDE']), 6),
        'lon': round(float(first_row['LONGITUDE']), 6),
        'year': int(first_row['YEAR']) if pd.notna(first_row['YEAR']) else 2020,
        'date': str(first_row.get('DATE', 'N/A')),
        'time': clean_value(first_row.get('TIME')),
        'time_range': clean_value(first_row.get('TIME_RANGE')),
        'street': str(first_row.get('STREET1', 'N/A'))[:30],
        'ward': clean_value(first_row.get('WARD_DESC')),
        'ward_num': clean_value(first_row.get('WARD_NUMBER')),
        'division': clean_value(first_row.get('DIVISION')),
        'neighbourhood': clean_value(first_row.get('NEIGHBOURHOOD_158')),

        # Collision-level severity (highest among all parties)
        'acclass': collision_severity,

        # Traffic and road info
        'road_class': clean_value(first_row.get('ROAD_CLASS')),
        'traffctl': clean_value(first_row.get('TRAFFCTL')),
        'impactype': clean_value(first_row.get('IMPACTYPE')),

        # Environmental conditions (needed for stats API endpoint compatibility)
        'light': clean_value(first_row.get('LIGHT')),
        'rdsfcond': clean_value(first_row.get('RDSFCOND')),

        # Flags for filtering
        'road_user_category': road_user_categories[0] if road_user_categories else '',  # Primary category
        'road_user_categories': road_user_categories if road_user_categories else [],  # All categories
        'age_category': age_categories[0] if age_categories else '',  # Primary age category
        'age_categories': age_categories if age_categories else [],  # All age categories
        'vulnerable': 'Yes' if has_vulnerable else 'No',

        # Party count (keep count but not detailed party array - saves massive memory)
        'num_parties': len(collision_group)
    }
    markers_data.append(marker_info)

print(f"Prepared {len(markers_data):,} unique collision markers (from {len(df):,} party records)")

# Save processed data to cache for faster future startups
if need_to_save_cache:
    print(f"Saving optimized data to {CACHE_FORMAT} cache...")
    # PHASE 1 OPT #3: Save cache (Parquet preferred, Pickle fallback)
    # Cache stores fully optimized dataframe (pruned + categorical dtypes)
    if CACHE_FORMAT == 'parquet':
        df.to_parquet(CACHE_FILE)
    else:  # pickle
        df.to_pickle(CACHE_FILE)
    print(f"✓ Cache saved ({CACHE_FORMAT} format)")

# Prepare filters metadata
def sort_with_special_values_last(value_list: List[Any]) -> List[str]:
    """Sort values alphabetically but move special values (Unknown, Other, NA, Pending) to the end"""
    def get_sort_key(value_str):
        lower_val = str(value_str).lower().strip()
        # Check for special values that should appear at the end
        special_values = ['unknown', 'other', 'na', 'n/a', 'pending', 'smv other', '']
        if lower_val in special_values or '(unknown)' in lower_val:
            return (1, value_str)  # Sort group 1 (end), then by original value
        return (0, value_str)  # Sort group 0 (normal), then alphabetically

    return sorted([str(v) for v in value_list], key=get_sort_key)

def get_unique_values(column_name: str) -> List[str]:
    values = df[column_name].dropna().unique().tolist()
    # Replace 'Unknown' string with empty string to consolidate unknowns
    values = ['' if str(v).lower().strip() == 'unknown' else v for v in values]
    # Remove duplicates and add empty string if there are any NaN values
    values = list(set(values))
    if df[column_name].isna().sum() > 0 and '' not in values:
        values.append('')
    return sort_with_special_values_last(values)

def sort_age_ranges(age_list: List[str]) -> List[str]:
    """Sort age ranges numerically by their starting age"""
    def get_sort_key(age_str):
        if age_str == 'unknown' or age_str == '':
            return 999  # Put unknown at the end
        if 'Over' in age_str:
            return 96  # "Over 95" goes after "90 to 94"
        # Extract first number from "X to Y" format
        try:
            return int(age_str.split(' ')[0])
        except:
            return 999

    return sorted(age_list, key=get_sort_key)

def sort_wards(ward_list: List[str]) -> List[str]:
    """Sort wards by ward number, with 'Outside' and unknown at the end"""
    def get_sort_key(ward_str):
        ward_str_lower = str(ward_str).lower().strip()

        # Empty or unknown at the end
        if ward_str_lower == '' or ward_str_lower == 'unknown':
            return (999, ward_str)

        # "Outside" wards at the end (but before unknown)
        if 'outside' in ward_str_lower:
            return (998, ward_str)

        # Extract ward number from format "Name (X)"
        import re
        match = re.search(r'\((\d+)\)', ward_str)
        if match:
            ward_num = int(match.group(1))
            return (ward_num, ward_str)

        # If no number found, sort alphabetically at the end
        return (997, ward_str)

    return sorted(ward_list, key=get_sort_key)

filters_metadata = {
    'WARD': sort_wards(get_unique_values('WARD_DESC')),
    'DIVISION': get_unique_values('DIVISION'),
    'NEIGHBOURHOOD': get_unique_values('NEIGHBOURHOOD_158'),
    'TIME_RANGE': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'ROAD_USER_CATEGORY': ['Pedestrian', 'Cyclist', 'Motorcyclist', 'Motorist', 'Micromobility', 'Property Owner', ''],
    'AGE_CATEGORY': ['Children (0-14)', 'Youth (15-24)', 'Adults (25-64)', 'Seniors (65+)', ''],
    'ACCLASS': get_unique_values('ACCLASS'),
    'VISIBILITY': get_unique_values('VISIBILITY'),
    'LIGHT': get_unique_values('LIGHT'),
    'RDSFCOND': get_unique_values('RDSFCOND'),
    'ROAD_CLASS': get_unique_values('ROAD_CLASS'),
    'TRAFFCTL': get_unique_values('TRAFFCTL'),
    'IMPACTYPE': get_unique_values('IMPACTYPE'),
    'INVAGE': sort_age_ranges(get_unique_values('INVAGE')),
    'DRIVCOND': get_unique_values('DRIVCOND'),
    'PEDESTRIAN': ['Yes', 'No', ''],
    'CYCLIST': ['Yes', 'No', ''],
    'VULNERABLE_ROAD_USER': ['Yes', 'No'],
    'SPEEDING': ['Yes', 'No', ''],
    'REDLIGHT': ['Yes', 'No', '']
    # Note: ALCOHOL and AG_DRIV removed - data is redundant/inconsistent with DRIVCOND
}

# ============================================================================
# HELPER FUNCTIONS: Input Validation and Filter Parsing
# ============================================================================

def validate_and_parse_filters(request_args: ImmutableMultiDict) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Validate and parse all filter parameters from request arguments.

    Returns:
        tuple: (parsed_filters dict, error_message str or None)
    """
    parsed = {}

    # Validate years (must be 2000-2030 range, max 100 items)
    years_str = request_args.get('years')
    if years_str:
        try:
            year_list = [int(y.strip()) for y in years_str.split(',')]
            if len(year_list) > 100:
                return None, "Too many years specified (max 100)"
            if not all(2000 <= y <= 2030 for y in year_list):
                return None, "Year values must be between 2000 and 2030"
            parsed['years'] = year_list
        except ValueError:
            return None, "Invalid year format - years must be integers"

    # Validate zoom level
    zoom_str = request_args.get('zoom')
    if zoom_str:
        try:
            zoom = int(zoom_str)
            if not 0 <= zoom <= 20:
                return None, "Zoom level must be between 0 and 20"
            parsed['zoom'] = zoom
        except ValueError:
            return None, "Invalid zoom format - must be an integer"

    # Validate bounds (Toronto area: roughly 43.5-44.0 lat, -79.7 to -79.0 lon)
    bounds_str = request_args.get('bounds')
    if bounds_str:
        try:
            bounds_parts = [float(b.strip()) for b in bounds_str.split(',')]
            if len(bounds_parts) != 4:
                return None, "Bounds must contain exactly 4 values: south,west,north,east"
            south, west, north, east = bounds_parts

            # Validate latitude range (Toronto area)
            if not (42.0 < south < north < 45.0):
                return None, "Invalid latitude bounds (expected Toronto area: 43-44)"

            # Validate longitude range (Toronto area)
            if not (-80.0 < west < east < -78.0):
                return None, "Invalid longitude bounds (expected Toronto area: -79.7 to -79.0)"

            parsed['bounds'] = (south, west, north, east)
        except ValueError:
            return None, "Invalid bounds format - must be numeric values"

    # Validate string list filters (max 100 items each)
    string_filters = {
        'acclass': 'accident class',
        'road_user_category': 'road user category',
        'time_range': 'time range',
        'ward': 'ward',
        'division': 'division',
        'neighbourhood': 'neighbourhood',
        'impactype': 'impact type',
        'road_class': 'road class',
        'traffctl': 'traffic control',
        'light': 'light condition',
        'rdsfcond': 'road surface condition',
        'age_category': 'age category',
        'vulnerable': 'vulnerable road user'
    }

    for param, display_name in string_filters.items():
        value = request_args.get(param)
        if value:
            items = [item.strip() for item in value.split(',')]
            if len(items) > 100:
                return None, f"Too many {display_name} values (max 100)"
            parsed[param] = items

    return parsed, None


def apply_filters(markers: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply validated filters to marker data.

    Args:
        markers: List of marker dictionaries
        filters: Dictionary of parsed and validated filters

    Returns:
        List of filtered markers
    """
    filtered_markers = markers

    # Apply year filter
    if 'years' in filters:
        year_list = filters['years']
        filtered_markers = [m for m in filtered_markers if m['year'] in year_list]

    # Apply accident class filter
    if 'acclass' in filters:
        acclass_list = filters['acclass']
        filtered_markers = [m for m in filtered_markers if m['acclass'] in acclass_list]

    # Apply road user category filter (inclusive - collision matches if ANY category matches)
    if 'road_user_category' in filters:
        road_user_list = filters['road_user_category']
        filtered_markers = [m for m in filtered_markers if any(cat in road_user_list for cat in m.get('road_user_categories', []))]

    # Apply time range filter
    if 'time_range' in filters:
        time_range_list = filters['time_range']
        filtered_markers = [m for m in filtered_markers if m['time_range'] in time_range_list]

    # Apply ward filter
    if 'ward' in filters:
        ward_list = filters['ward']
        filtered_markers = [m for m in filtered_markers if m.get('ward') in ward_list]

    # Apply division filter
    if 'division' in filters:
        division_list = filters['division']
        filtered_markers = [m for m in filtered_markers if m.get('division') in division_list]

    # Apply neighbourhood filter
    if 'neighbourhood' in filters:
        neighbourhood_list = filters['neighbourhood']
        filtered_markers = [m for m in filtered_markers if m.get('neighbourhood') in neighbourhood_list]

    # Apply impact type filter
    if 'impactype' in filters:
        impactype_list = filters['impactype']
        filtered_markers = [m for m in filtered_markers if m.get('impactype') in impactype_list]

    # Apply road class filter
    if 'road_class' in filters:
        road_class_list = filters['road_class']
        filtered_markers = [m for m in filtered_markers if m.get('road_class') in road_class_list]

    # Apply traffic control filter
    if 'traffctl' in filters:
        traffctl_list = filters['traffctl']
        filtered_markers = [m for m in filtered_markers if m.get('traffctl') in traffctl_list]

    # Apply light condition filter
    if 'light' in filters:
        light_list = filters['light']
        filtered_markers = [m for m in filtered_markers if m.get('light') in light_list]

    # Apply road surface condition filter
    if 'rdsfcond' in filters:
        rdsfcond_list = filters['rdsfcond']
        filtered_markers = [m for m in filtered_markers if m.get('rdsfcond') in rdsfcond_list]

    # Apply age category filter (inclusive - collision matches if ANY age category matches)
    if 'age_category' in filters:
        age_category_list = filters['age_category']
        filtered_markers = [m for m in filtered_markers if any(cat in age_category_list for cat in m.get('age_categories', []))]

    # Apply vulnerable road user filter
    if 'vulnerable' in filters:
        vulnerable_list = filters['vulnerable']
        filtered_markers = [m for m in filtered_markers if m.get('vulnerable') in vulnerable_list]

    # Apply bounds filter (for clustering endpoint)
    if 'bounds' in filters:
        south, west, north, east = filters['bounds']
        filtered_markers = [
            m for m in filtered_markers
            if south <= m['lat'] <= north and west <= m['lon'] <= east
        ]

    return filtered_markers

# ============================================================================
# END OF HELPER FUNCTIONS
# ============================================================================

@app.route('/')  # type: ignore
def index() -> str:
    """Default page - clustered map with best performance"""
    return render_template('map_clustered.html',
                         total_collisions=len(markers_data),
                         total_parties=len(df),
                         avg_parties=round(len(df) / len(markers_data), 2) if len(markers_data) > 0 else 0)

@app.route('/simple')  # type: ignore
def simple_map() -> str:
    """Simplified version for testing"""
    return render_template('map_simple.html')

@app.route('/diagnostic')  # type: ignore
def diagnostic_map() -> str:
    """Diagnostic version with detailed console logging"""
    return render_template('map_diagnostic.html')

@app.route('/clustered')  # type: ignore
def clustered_map() -> str:
    """Server-side clustered version - FASTEST performance!"""
    return render_template('map_clustered.html',
                         total_collisions=len(markers_data),
                         total_parties=len(df),
                         avg_parties=round(len(df) / len(markers_data), 2) if len(markers_data) > 0 else 0)

@app.route('/analysis')  # type: ignore
def statistical_analysis() -> Response:
    """Enhanced statistical analysis report with temporal trends & Vision Zero effectiveness"""
    # Dynamically serve the most recent enhanced report
    results_dir = Path('statistical_analysis_results')
    report_files = list(results_dir.glob('enhanced_report_*.html'))
    if report_files:
        # Sort by filename (contains timestamp: YYYYMMDD_HHMMSS) to get the most recent
        # This is more reliable than st_mtime on deployed environments where all files
        # get the same modification time from git checkout
        latest_report = sorted(report_files, reverse=True)[0]
        return send_from_directory('statistical_analysis_results', latest_report.name)
    else:
        # Fallback to latest known report if no reports found
        return send_from_directory('statistical_analysis_results', 'enhanced_report_20251126_155947.html')

@app.route('/advanced')  # type: ignore
def advanced_map() -> str:
    """Advanced map with full filters (may be slow to load)"""
    # Calculate data quality metrics
    outside_wards = len(df[df['WARD_DESC'] == 'Outside Wards'])
    pct_outside_wards = (outside_wards / len(df) * 100) if len(df) > 0 else 0

    # Calculate date range
    date_range = f"{df['YEAR'].min():.0f}-{df['YEAR'].max():.0f}"

    # Calculate missing data percentages
    empty_road_user = len(df[df['ROAD_USER_CATEGORY'] == ''])
    pct_empty_road_user = (empty_road_user / len(df) * 100) if len(df) > 0 else 0

    empty_age = len(df[df['AGE_CATEGORY'] == ''])
    pct_empty_age = (empty_age / len(df) * 100) if len(df) > 0 else 0

    return render_template('map.html',
                         total_parties=len(df),
                         unique_collisions=unique_collisions,
                         outside_wards=outside_wards,
                         pct_outside_wards=round(pct_outside_wards, 1),
                         date_range=date_range,
                         pct_empty_road_user=round(pct_empty_road_user, 1),
                         pct_empty_age=round(pct_empty_age, 1))

@app.route('/api/markers')  # type: ignore
def get_markers() -> Response:
    """Return markers data as JSON with optional filtering"""
    # Support year filtering to reduce initial load
    years = request.args.get('years', None)

    if years:
        # Filter by specific years (e.g., "2020,2021,2022")
        year_list = [int(y) for y in years.split(',')]
        filtered_data = [m for m in markers_data if m['year'] in year_list]
        return jsonify(filtered_data)

    # Support pagination
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', type=int)

    if limit is not None:
        # Return paginated slice
        end = offset + limit if limit else len(markers_data)
        paginated_data = markers_data[offset:end]
        return jsonify({
            'markers': paginated_data,
            'total': len(markers_data),
            'offset': offset,
            'limit': limit,
            'hasMore': end < len(markers_data)
        })

    # Default: return recent 3 years only (much faster!)
    recent_years = [2021, 2022, 2023]
    recent_data = [m for m in markers_data if m['year'] in recent_years]
    print(f"Returning {len(recent_data)} markers from years {recent_years}")
    return jsonify(recent_data)

@app.route('/api/markers/count')  # type: ignore
def get_markers_count() -> Response:
    """Return total marker count"""
    return jsonify({'count': len(markers_data)})

@app.route('/api/stats/summary')  # type: ignore
def get_stats_summary() -> Response:
    """Return pre-aggregated statistics for dashboard (fast!)"""
    # Count by accident class
    acclass_counts = Counter(m['acclass'] for m in markers_data)

    # Count by year
    year_counts = Counter(m['year'] for m in markers_data)

    # Count by road user (inclusive - each collision counted in all applicable categories)
    roaduser_counts = Counter()
    for m in markers_data:
        for category in m.get('road_user_categories', []):
            roaduser_counts[category] += 1

    # Count by time range
    timerange_counts = Counter(m['time_range'] for m in markers_data if m['time_range'])

    # Count by impact type
    impactype_counts = Counter(m['impactype'] for m in markers_data if m['impactype'])

    # Count by road class
    road_class_counts = Counter(m['road_class'] for m in markers_data if m['road_class'])

    # Count by traffic control
    traffctl_counts = Counter(m['traffctl'] for m in markers_data if m['traffctl'])

    # Count by light conditions
    light_counts = Counter(m['light'] for m in markers_data if m['light'])

    # Count by road surface condition
    rdsfcond_counts = Counter(m['rdsfcond'] for m in markers_data if m['rdsfcond'])

    # Count by age category
    age_category_counts = Counter(m.get('age_category', '') for m in markers_data if m.get('age_category'))

    # Count by vulnerable road users
    vulnerable_counts = Counter(m['vulnerable'] for m in markers_data if m['vulnerable'])

    return jsonify({
        'total_records': len(markers_data),
        'unique_collisions': unique_collisions,
        'by_class': dict(acclass_counts),
        'by_year': dict(sorted(year_counts.items())),
        'by_roaduser': dict(roaduser_counts),
        'by_timerange': dict(timerange_counts),
        'by_impactype': dict(impactype_counts),
        'by_road_class': dict(road_class_counts),
        'by_traffctl': dict(traffctl_counts),
        'by_light': dict(light_counts),
        'by_rdsfcond': dict(rdsfcond_counts),
        'by_age_category': dict(age_category_counts),
        'by_vulnerable': dict(vulnerable_counts),
        'date_range': f"{min(year_counts.keys())}-{max(year_counts.keys())}"
    })

@app.route('/api/stats/filtered')  # type: ignore
def get_filtered_stats() -> Union[Response, Tuple[Response, int]]:
    """
    Return filter counts based on current filter selections.
    This makes filter counts dynamic - they update based on active filters.

    Query parameters (all optional):
    - years: Comma-separated list of years
    - acclass: Comma-separated list of accident classes
    - road_user_category: Comma-separated list of road user categories
    - time_range: Comma-separated list of time ranges
    - ward: Comma-separated list of wards
    - division: Comma-separated list of police divisions
    - neighbourhood: Comma-separated list of neighbourhoods
    - impactype: Comma-separated list of impact types
    - road_class: Comma-separated list of road classes
    - traffctl: Comma-separated list of traffic control types
    - light: Comma-separated list of light conditions
    - rdsfcond: Comma-separated list of road surface conditions
    - age_category: Comma-separated list of age categories
    - vulnerable: Comma-separated list (Yes/No for vulnerable road users)
    """
    # Validate and parse filters
    filters, error = validate_and_parse_filters(request.args)
    if error:
        return jsonify({'error': error}), 400

    # Apply filters
    filtered_markers = apply_filters(markers_data, filters)

    # Calculate counts for each filter category based on filtered subset
    acclass_counts = Counter(m['acclass'] for m in filtered_markers)
    year_counts = Counter(m['year'] for m in filtered_markers)

    # Count by road user (inclusive - each collision counted in all applicable categories)
    roaduser_counts = Counter()
    for m in filtered_markers:
        for category in m.get('road_user_categories', []):
            roaduser_counts[category] += 1

    timerange_counts = Counter(m['time_range'] for m in filtered_markers if m['time_range'])
    impactype_counts = Counter(m['impactype'] for m in filtered_markers if m['impactype'])
    road_class_counts = Counter(m['road_class'] for m in filtered_markers if m['road_class'])
    traffctl_counts = Counter(m['traffctl'] for m in filtered_markers if m['traffctl'])
    light_counts = Counter(m['light'] for m in filtered_markers if m['light'])
    rdsfcond_counts = Counter(m['rdsfcond'] for m in filtered_markers if m['rdsfcond'])
    age_category_counts = Counter(m.get('age_category', '') for m in filtered_markers if m.get('age_category'))
    vulnerable_counts = Counter(m['vulnerable'] for m in filtered_markers if m['vulnerable'])

    return jsonify({
        'total_records': len(filtered_markers),
        'unique_collisions': len(filtered_markers),  # Each marker is already a unique collision
        'by_class': dict(acclass_counts),
        'by_year': dict(sorted(year_counts.items())),
        'by_roaduser': dict(roaduser_counts),
        'by_timerange': dict(timerange_counts),
        'by_impactype': dict(impactype_counts),
        'by_road_class': dict(road_class_counts),
        'by_traffctl': dict(traffctl_counts),
        'by_light': dict(light_counts),
        'by_rdsfcond': dict(rdsfcond_counts),
        'by_age_category': dict(age_category_counts),
        'by_vulnerable': dict(vulnerable_counts),
        'date_range': f"{min(year_counts.keys())}-{max(year_counts.keys())}" if year_counts else "N/A"
    })

@app.route('/api/stats/by-ward')  # type: ignore
def get_stats_by_ward() -> Response:
    """Return collision statistics grouped by ward"""
    ward_stats = {}
    for m in markers_data:
        ward = m['ward'] if m['ward'] else 'Unknown'
        if ward not in ward_stats:
            ward_stats[ward] = {'total': 0, 'fatal': 0, 'injury': 0}

        ward_stats[ward]['total'] += 1
        if m['acclass'] == 'Fatal':
            ward_stats[ward]['fatal'] += 1
        elif m['acclass'] == 'Non-Fatal Injury':
            ward_stats[ward]['injury'] += 1

    return jsonify(ward_stats)

@app.route('/api/boundaries/<boundary_type>/<boundary_name>')  # type: ignore
def get_boundary(boundary_type: str, boundary_name: str) -> Union[Response, Tuple[Response, int]]:
    """Return GeoJSON boundary for a specific ward, division, or neighbourhood"""
    try:
        if boundary_type == 'ward':
            geojson_data = load_wards_geojson()  # PHASE 1 OPT #2: Lazy-load
            # Match by AREA_DESC or AREA_NAME for wards
            for feature in geojson_data['features']:
                props = feature['properties']
                area_desc = props.get('AREA_DESC', '')
                area_name = props.get('AREA_NAME', '')
                if area_desc == boundary_name or area_name == boundary_name:
                    return jsonify({
                        'type': 'Feature',
                        'geometry': feature['geometry'],
                        'properties': props
                    })

        elif boundary_type == 'division':
            geojson_data = load_divisions_geojson()  # PHASE 1 OPT #2: Lazy-load
            # Divisions use DIV property (e.g., "D11", "D12")
            for feature in geojson_data['features']:
                props = feature['properties']
                div_code = props.get('DIV', '')
                if div_code == boundary_name:
                    return jsonify({
                        'type': 'Feature',
                        'geometry': feature['geometry'],
                        'properties': props
                    })

        elif boundary_type == 'neighbourhood':
            geojson_data = load_neighbourhoods_geojson()  # PHASE 1 OPT #2: Lazy-load
            # Match by AREA_DESC or AREA_NAME for neighbourhoods
            for feature in geojson_data['features']:
                props = feature['properties']
                area_desc = props.get('AREA_DESC', '')
                area_name = props.get('AREA_NAME', '')
                if area_desc == boundary_name or area_name == boundary_name:
                    return jsonify({
                        'type': 'Feature',
                        'geometry': feature['geometry'],
                        'properties': props
                    })

        else:
            return jsonify({'error': 'Invalid boundary type'}), 400

        return jsonify({'error': 'Boundary not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/trends')  # type: ignore
def get_stats_trends() -> Response:
    """Return year-over-year trends"""
    trends = {}
    for m in markers_data:
        year = m['year']
        if year not in trends:
            trends[year] = {
                'total': 0,
                'fatal': 0,
                'injury': 0,
                'pedestrian': 0,
                'cyclist': 0,
                'speeding': 0
            }

        trends[year]['total'] += 1
        if m['acclass'] == 'Fatal':
            trends[year]['fatal'] += 1
        elif m['acclass'] == 'Non-Fatal Injury':
            trends[year]['injury'] += 1

        # Check if category is in road_user_categories list (inclusive)
        if 'Pedestrian' in m.get('road_user_categories', []):
            trends[year]['pedestrian'] += 1
        if 'Cyclist' in m.get('road_user_categories', []):
            trends[year]['cyclist'] += 1

        # Note: speeding field removed in Phase 1 Opt #4, so we skip this
        # if m.get('speeding') == 'Yes':
        #     trends[year]['speeding'] += 1

    return jsonify(dict(sorted(trends.items())))

@app.route('/api/clusters')  # type: ignore
def get_clusters() -> Union[Response, Tuple[Response, int]]:
    """
    Return clustered collision data based on zoom level and viewport bounds.

    This server-side clustering dramatically improves performance by:
    - Only sending data visible in the current viewport
    - Clustering points at low zoom levels
    - Returning individual markers at high zoom levels

    Query parameters:
    - zoom: Map zoom level (0-18)
    - bounds: Viewport bounds as "south,west,north,east"
    - years: Optional comma-separated list of years (e.g., "2020,2021,2022")
    - acclass: Optional comma-separated list of accident classes
    - road_user_category: Optional comma-separated list of road user categories
    - time_range: Optional comma-separated list of time ranges
    - ward: Optional comma-separated list of wards
    - division: Optional comma-separated list of police divisions
    - neighbourhood: Optional comma-separated list of neighbourhoods
    - impactype: Optional comma-separated list of impact types
    - road_class: Optional comma-separated list of road classes
    - traffctl: Optional comma-separated list of traffic control types
    - light: Optional comma-separated list of light conditions
    - rdsfcond: Optional comma-separated list of road surface conditions
    - age_category: Optional comma-separated list of age categories
    - vulnerable: Optional comma-separated list (Yes/No for vulnerable road users)
    """
    # Validate and parse filters
    filters, error = validate_and_parse_filters(request.args)
    if error:
        return jsonify({'error': error}), 400

    # Get zoom level (validated in helper function if provided)
    zoom = filters.get('zoom', 10)

    # Get bounds (validated in helper function if provided, otherwise use default)
    if 'bounds' not in filters:
        # Default to Toronto area
        filters['bounds'] = (43.58, -79.64, 43.86, -79.12)

    # Unpack bounds for use in response
    south, west, north, east = filters['bounds']

    # Apply filters (including bounds filter)
    markers_in_view = apply_filters(markers_data, filters)

    # Determine cluster grid size based on zoom level
    # Higher zoom = smaller grid = more clusters/individual points
    if zoom >= 16:
        # High zoom: return individual markers (no clustering)
        clusters = [
            {
                'type': 'marker',
                'lat': m['lat'],
                'lon': m['lon'],
                'properties': m
            }
            for m in markers_in_view
        ]
    else:
        # Low/medium zoom: cluster using grid
        # Grid cell size decreases as zoom increases
        grid_size = 0.5 / (2 ** (zoom - 8))  # Adaptive grid size

        # Group markers into grid cells
        grid_clusters = {}
        for m in markers_in_view:
            # Calculate grid cell coordinates
            grid_lat = int(m['lat'] / grid_size)
            grid_lon = int(m['lon'] / grid_size)
            grid_key = f"{grid_lat},{grid_lon}"

            if grid_key not in grid_clusters:
                grid_clusters[grid_key] = {
                    'markers': [],
                    'lat_sum': 0,
                    'lon_sum': 0
                }

            grid_clusters[grid_key]['markers'].append(m)
            grid_clusters[grid_key]['lat_sum'] += m['lat']
            grid_clusters[grid_key]['lon_sum'] += m['lon']

        # Convert grid clusters to response format
        clusters = []
        for grid_key, cluster_data in grid_clusters.items():
            marker_count = len(cluster_data['markers'])

            if marker_count == 1:
                # Single marker - return as individual point
                m = cluster_data['markers'][0]
                clusters.append({
                    'type': 'marker',
                    'lat': m['lat'],
                    'lon': m['lon'],
                    'properties': m
                })
            else:
                # Multiple markers - return as cluster
                # Calculate cluster center (average position)
                center_lat = cluster_data['lat_sum'] / marker_count
                center_lon = cluster_data['lon_sum'] / marker_count

                # Count by accident class for cluster summary
                class_counts = Counter(m['acclass'] for m in cluster_data['markers'])

                clusters.append({
                    'type': 'cluster',
                    'lat': center_lat,
                    'lon': center_lon,
                    'count': marker_count,
                    'properties': {
                        'point_count': marker_count,
                        'fatal': class_counts.get('Fatal', 0),
                        'injury': class_counts.get('Non-Fatal Injury', 0),
                        'property': class_counts.get('Property Damage O', 0)
                    }
                })

    return jsonify({
        'clusters': clusters,
        'total_in_view': len(markers_in_view),
        'total_markers': len(markers_data),
        'zoom': zoom,
        'bounds': {'south': south, 'west': west, 'north': north, 'east': east}
    })

@app.route('/api/wards')  # type: ignore
def get_wards() -> Response:
    """Return ward boundaries as GeoJSON"""
    return jsonify(load_wards_geojson())  # PHASE 1 OPT #2: Lazy-load

@app.route('/api/divisions')  # type: ignore
def get_divisions() -> Response:
    """Return police division boundaries as GeoJSON"""
    return jsonify(load_divisions_geojson())  # PHASE 1 OPT #2: Lazy-load

@app.route('/api/neighbourhoods')  # type: ignore
def get_neighbourhoods() -> Response:
    """Return neighbourhood boundaries as GeoJSON"""
    return jsonify(load_neighbourhoods_geojson())  # PHASE 1 OPT #2: Lazy-load

@app.route('/api/filters')  # type: ignore
def get_filters() -> Response:
    """Return filter metadata"""
    # Debug: Print first 5 wards to verify sorting
    print("DEBUG: First 5 wards:", filters_metadata['WARD'][:5])
    return jsonify(filters_metadata)

if __name__ == '__main__':
    # Configuration from environment variables (safer for production)
    PORT = int(os.getenv('PORT', 5001))
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'

    print("\n" + "="*80)
    print("VISION ZERO MAP SERVER")
    print("="*80)
    print(f"\nStarting server at http://localhost:{PORT}")
    print(f"   Debug mode: {'ON' if DEBUG else 'OFF'}")
    print(f"   Press Ctrl+C to stop the server\n")
    print("="*80 + "\n")
    app.run(debug=DEBUG, port=PORT)
