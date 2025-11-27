# Vision Zero Statistical Pattern Analysis - Summary Report
**Generated:** November 3, 2025
**Analysis Period:** Pre-COVID (2006-2019)
**Dataset:** Motor Vehicle Collisions Only (No Speed Limit Data)
**Method:** Pure frequency-based chi-square tests, zero weighting, Bonferroni correction

---

## Executive Summary

Analyzed 15,674 collision records from 2006-2019 (Pre-COVID period) using rigorous statistical methods with complete transparency. Performed 20 independent chi-square tests examining relationships between collision severity and various factors.

### Key Findings

**Strict Criteria (Bonferroni-adjusted p < 0.0005 AND Cramér's V ≥ 0.1):**
- **0 patterns met these extremely stringent criteria**

**Moderate Criteria (p < 0.01 before Bonferroni correction):**
- **10 patterns showed statistically significant associations**

---

## Top Statistical Patterns (Ranked by Effect Size)

### 1. Ward × Collision Severity ⭐⭐⭐
- **Sample:** 15,674 collisions across 25 Toronto wards
- **Statistics:** χ² = 263.06, df = 50, p = 1.06×10⁻³⁰
- **Effect Size:** Cramér's V = 0.092 (Negligible but approaching small)
- **Finding:** Significant geographic disparities in collision severity across wards
- **Ward:**
  - Highest fatal %: Etobicoke North, Scarborough Centre, Scarborough Southwest
  - Includes 126 collisions "Outside Wards" (properly identified via point-in-polygon)

### 2. Accident Location × Collision Severity ⭐⭐
- **Sample:** 10,465 collisions with location data (67% completeness)
- **Statistics:** χ² = 88.97, df = 16, p = 3.87×10⁻¹²
- **Effect Size:** Cramér's V = 0.065 (Negligible)
- **Finding:** Collision severity varies by location type
- **Pattern:** Non-intersection collisions show different severity distribution than at-intersection

### 3. Light Conditions × Collision Severity ⭐⭐
- **Sample:** 15,674 collisions (100% complete data)
- **Statistics:** χ² = 86.61, df = 16, p = 1.05×10⁻¹¹
- **Effect Size:** Cramér's V = 0.053 (Negligible)
- **Finding:** Lighting conditions associated with collision severity
- **Pattern:**
  - Dark conditions: 579 fatal vs 2,792 non-fatal
  - Daylight: 1,093 fatal vs 7,934 non-fatal
  - Different severity distributions by lighting

### 4. Vulnerable Road Users × Light Conditions ⭐⭐
- **Sample:** 15,674 collisions
- **Statistics:** χ² = 87.92, df = 8, p = 1.23×10⁻¹⁵
- **Effect Size:** Cramér's V = 0.075 (Negligible)
- **Finding:** Pedestrian/cyclist involvement varies by lighting
- **Pattern:** Higher vulnerable road user percentage in dawn/dusk periods

### 5. District × Collision Severity ⭐
- **Sample:** 15,656 collisions across 4 districts
- **Statistics:** χ² = 116.74, df = 6, p = 7.89×10⁻²³
- **Effect Size:** Cramér's V = 0.061 (Negligible)
- **Finding:** District-level geographic disparities
- **Districts:**
  - Toronto & East York: 548 fatal, 4,834 non-fatal
  - Scarborough: 624 fatal, 2,905 non-fatal (highest fatal %)
  - Etobicoke York: 487 fatal, 3,129 non-fatal
  - North York: 474 fatal, 2,653 non-fatal

### 6. Vulnerable Road Users × Road Class ⭐
- **Sample:** 15,346 collisions
- **Statistics:** χ² = 73.56, df = 9, p = 3.04×10⁻¹²
- **Effect Size:** Cramér's V = 0.069 (Negligible)
- **Finding:** Pedestrian/cyclist collisions concentrated on certain road types
- **Pattern:** Major arterials have disproportionate share of vulnerable road user collisions

### 7. Road Surface × Collision Severity
- **Sample:** 15,651 collisions
- **Statistics:** χ² = 61.97, df = 16, p = 2.43×10⁻⁷
- **Effect Size:** Cramér's V = 0.044 (Negligible)
- **Finding:** Road surface conditions associated with severity

### 8. Weather Visibility × Collision Severity
- **Sample:** 15,656 collisions
- **Statistics:** χ² = 61.62, df = 14, p = 6.08×10⁻⁸
- **Effect Size:** Cramér's V = 0.044 (Negligible)
- **Finding:** Weather/visibility conditions linked to outcomes

### 9. Traffic Control × Collision Severity
- **Sample:** 15,645 collisions
- **Statistics:** χ² = 60.91, df = 18, p = 1.46×10⁻⁶
- **Effect Size:** Cramér's V = 0.044 (Negligible)
- **Finding:** Severity varies by traffic control type

### 10. Day of Week × Collision Severity
- **Sample:** 15,674 collisions
- **Statistics:** χ² = 40.27, df = 12, p = 6.48×10⁻⁵
- **Effect Size:** Cramér's V = 0.036 (Negligible)
- **Finding:** Weekly patterns in collision severity

---

## Patterns NOT Statistically Significant (p > 0.01)

- **Road Class × Severity:** p = 0.014 (just missed threshold)
- **Season × Severity:** p = 0.068 (no seasonal pattern detected)

---

## Data Quality Notes

### COVID Period Stratification
- **Pre-COVID (2006-2019):** 15,674 records (85.4%) - PRIMARY ANALYSIS
- **COVID (2020-2021):** 1,272 records (6.9%) - Excluded due to anomalous traffic
- **Post-COVID (2022-2023):** 1,405 records (7.7%) - Future validation dataset

### Ward Assignment
- **Method:** Point-in-polygon geometry (Shapely library)
- **Accuracy:** 99.3% (18,225 of 18,351 assigned)
- **Outside Wards:** 126 collisions (0.7%) correctly identified as outside boundaries

### Variable Completeness (Pre-COVID)
- ✓ 100%: ACCLASS, LIGHT, WARD_DESC
- ✓ 99%+: TRAFFCTL, VISIBILITY, RDSFCOND, IMPACTYPE, INVTYPE, DISTRICT
- ✓ 98%: ROAD_CLASS
- ⚠ 67%: ACCLOC (Accident Location)

---

## Statistical Rigor Applied

### Multiple Testing Correction
- **Method:** Bonferroni correction
- **Tests performed:** 20 independent tests
- **Original threshold:** α = 0.01
- **Adjusted threshold:** α = 0.0005
- **Rationale:** Conservative approach to control family-wise error rate

### Effect Size Thresholds
- **Minimum meaningful effect:** Cramér's V ≥ 0.1
- **Interpretation:**
  - <0.1 = Negligible
  - 0.1-0.3 = Small
  - 0.3-0.5 = Medium
  - >0.5 = Large

### Why No Patterns Met Strict Criteria

All detected patterns had **negligible effect sizes** (V < 0.1), meaning:
1. While statistically significant (large sample size detects even tiny effects)
2. The practical/real-world magnitude of associations is very small
3. Most variation in severity NOT explained by these single variables
4. Collision outcomes likely driven by complex multi-factor interactions

---

## Interpretation & Limitations

### Strengths
✓ Large sample: 15,674 Pre-COVID collisions
✓ High data quality: Most variables >99% complete
✓ Rigorous methods: Pure frequency analysis, zero assumptions
✓ Temporal control: COVID period properly stratified
✓ Geographic precision: Point-in-polygon ward assignment

### Limitations
⚠ **Small effect sizes:** All patterns have negligible practical magnitude
⚠ **Single-variable tests:** No multi-variable modeling performed
⚠ **Time categorization issue:** TIME_OF_DAY showed all "Unknown" (needs fixing)
⚠ **Bonferroni conservative:** May miss real patterns due to strict correction
⚠ **Missing speed data:** Analysis excludes speed limit variables as requested

### What This Means

The analysis reveals **statistically detectable but practically small** associations between individual factors and collision severity. Key insights:

1. **Geographic patterns exist** (wards, districts) but explain <1% of severity variance
2. **Environmental conditions matter** (light, weather, road surface) but effects are small
3. **Vulnerable road users** concentrate on major arterials and vary by lighting
4. **No single factor dominates** - collision outcomes are multi-factorial

To find stronger patterns, would need:
- Multi-variable regression models (excluded per "no weighting" constraint)
- Interaction effects (e.g., light × weather × road class combined)
- Temporal trend analysis (not yet performed)
- Risk factor combinations (speeding + alcohol + darkness, etc.)

---

## Next Steps for Deeper Analysis

1. **Fix TIME_OF_DAY categorization** - currently showing all "Unknown"
2. **Temporal trend analysis** - Year-over-year changes (2006-2019)
3. **Interaction effects** - Three-way crosstabulations
4. **Risk factor combinations** - Aggressive driving + alcohol + conditions
5. **Age group analysis** - Pedestrian/cyclist vulnerability by age
6. **Post-COVID validation** - Test if Pre-COVID patterns replicate in 2022-2023

---

## Files Generated

1. **statistical_results_master_20251103_220312.csv** - All 20 test results
2. **significant_patterns_20251103_220312.csv** - Patterns meeting strict criteria (empty)
3. **analysis_summary_20251103_220312.json** - Metadata summary

---

## Conclusion

This analysis provides a **transparent, rigorous, assumption-free** examination of collision patterns using only motor vehicle collision data and ward boundaries. While many statistically significant associations were detected (p < 0.01), **none had practically meaningful effect sizes** (V ≥ 0.1) after conservative multiple testing correction.

The findings suggest collision severity is influenced by many small factors rather than dominated by any single variable. Geographic disparities across wards showed the strongest signal (V=0.092), followed by lighting conditions and vulnerable road user patterns.

**For actionable insights, recommend:**
- Focus on ward-level interventions (highest geographic variance)
- Prioritize vulnerable road user safety on major arterials
- Consider lighting improvements in high-collision areas
- Conduct multi-variable analysis to capture interaction effects
