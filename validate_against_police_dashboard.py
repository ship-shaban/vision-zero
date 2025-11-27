#!/usr/bin/env python3
"""
Validation against Toronto Police Official Dashboard Data
Dashboard shows: 6872 total collisions (2006-2023)
"""

import pandas as pd

print("="*80)
print("VALIDATION AGAINST TORONTO POLICE DASHBOARD")
print("="*80)

# Load data
df = pd.read_csv('collision_speed_limit_analysis_UPDATED.csv')
df['YEAR'] = pd.to_datetime(df['DATE']).dt.year

print(f"\nTotal party records: {len(df):,}")

# Apply hybrid COLLISION_ID
def create_cluster_key(row):
    try:
        lat_rounded = round(float(row['LATITUDE']), 4)
        lon_rounded = round(float(row['LONGITUDE']), 4)
        time_int = int(row['TIME']) if pd.notna(row['TIME']) else 0
        time_str = str(time_int).zfill(4)
        time_formatted = f"{time_str[:2]}:{time_str[2:]}"
        date_str = pd.to_datetime(row['DATE']).strftime('%Y-%m-%d')
        return f"{date_str}_{time_formatted}_{lat_rounded}_{lon_rounded}"
    except:
        return f"ERROR_{row.name}"

df['COLLISION_ID'] = None
has_accnum = df['ACCNUM'].notna()
df.loc[has_accnum, 'COLLISION_ID'] = 'ACCNUM_' + df.loc[has_accnum, 'ACCNUM'].astype(str)

missing_accnum = df['ACCNUM'].isna()
if missing_accnum.sum() > 0:
    df.loc[missing_accnum, 'CLUSTER_KEY'] = df.loc[missing_accnum].apply(create_cluster_key, axis=1)
    unique_clusters = df.loc[missing_accnum, 'CLUSTER_KEY'].unique()
    cluster_mapping = {key: f'CLUSTER_{i}' for i, key in enumerate(unique_clusters)}
    df.loc[missing_accnum, 'COLLISION_ID'] = df.loc[missing_accnum, 'CLUSTER_KEY'].map(cluster_mapping)
    df.drop('CLUSTER_KEY', axis=1, inplace=True)

# Count unique collisions
df_unique = df.drop_duplicates(subset='COLLISION_ID')
total_collisions = len(df_unique)

print(f"\nOur total unique collisions: {total_collisions:,}")
print(f"Police dashboard total: 6,872")
print(f"Difference: {total_collisions - 6872:+,} collision")

# Check 2016 specifically
collisions_2016 = len(df_unique[df_unique['YEAR'] == 2016])
print(f"\n2016 Specific Check:")
print(f"  Our count: {collisions_2016}")
print(f"  Police count: 386")
print(f"  Difference: {collisions_2016 - 386:+,}")

# Year-by-year with party record info
print("\n" + "="*80)
print("DETAILED YEAR-BY-YEAR ANALYSIS")
print("="*80)
print(f"{'Year':<6} {'Collisions':<12} {'Party Records':<15} {'Parties/Collision':<20}")
print("-"*80)

annual_counts = df_unique.groupby('YEAR').size()
party_counts = df.groupby('YEAR').size()

for year in sorted(annual_counts.index):
    collisions = annual_counts[year]
    parties = party_counts[year]
    ratio = parties / collisions
    print(f"{year:<6} {collisions:<12} {parties:<15} {ratio:<20.2f}")

print("-"*80)
print(f"{'TOTAL':<6} {total_collisions:<12} {len(df):<15} {len(df)/total_collisions:<20.2f}")

# Check for potential issues in 2016
print("\n" + "="*80)
print("2016 DETAILED INVESTIGATION")
print("="*80)

df_2016 = df[df['YEAR'] == 2016]
df_2016_unique = df_unique[df_unique['YEAR'] == 2016]

print(f"\n2016 Statistics:")
print(f"  Total party records: {len(df_2016):,}")
print(f"  Unique collisions found: {len(df_2016_unique):,}")
print(f"  Average parties per collision: {len(df_2016)/len(df_2016_unique):.2f}")

# Check ACCNUM vs clustering in 2016
has_accnum_2016 = df_2016['ACCNUM'].notna().sum()
no_accnum_2016 = df_2016['ACCNUM'].isna().sum()

print(f"\n2016 COLLISION_ID Method:")
print(f"  Party records with ACCNUM: {has_accnum_2016:,}")
print(f"  Party records needing clustering: {no_accnum_2016:,}")

# Check for any errors
error_ids = df_2016_unique[df_2016_unique['COLLISION_ID'].str.contains('ERROR', na=False)]
if len(error_ids) > 0:
    print(f"\n⚠️  WARNING: {len(error_ids)} collisions have ERROR in COLLISION_ID")
else:
    print(f"\n✓ No ERROR collision IDs found in 2016")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_diff = total_collisions - 6872
print(f"\nOverall Accuracy:")
print(f"  Total collisions: {total_collisions:,} vs 6,872 (Police)")
print(f"  Difference: {total_diff:+,} collision ({abs(total_diff)/6872*100:.4f}%)")

if abs(total_diff) == 0:
    print(f"  Result: ✅ PERFECT MATCH")
elif abs(total_diff) == 1:
    print(f"  Result: ✅ EXCELLENT (off by only 1 collision)")
elif abs(total_diff) <= 5:
    print(f"  Result: ✅ VERY GOOD (off by {abs(total_diff)} collisions)")
elif abs(total_diff) <= 10:
    print(f"  Result: ✅ GOOD (off by {abs(total_diff)} collisions)")
else:
    print(f"  Result: ⚠️  NEEDS INVESTIGATION (off by {abs(total_diff)} collisions)")

print(f"\n2016 Accuracy:")
print(f"  2016 collisions: {collisions_2016} vs 386 (Police)")
print(f"  Difference: {collisions_2016 - 386:+,} collision")

if collisions_2016 == 386:
    print(f"  Result: ✅ PERFECT MATCH")
elif abs(collisions_2016 - 386) == 1:
    print(f"  Result: ✅ EXCELLENT (off by only 1 collision)")
else:
    print(f"  Result: ⚠️  Difference of {abs(collisions_2016 - 386)} collisions")

print("\n" + "="*80)
