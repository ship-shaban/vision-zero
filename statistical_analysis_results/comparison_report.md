# Statistical Pattern Comparison Report

## Pre-COVID (2006-2019) vs ALL DATA (2006-2023)

---

## Summary Statistics

| Metric | Pre-COVID (2006-2019) | ALL Data (2006-2023) | Difference |
|--------|----------------------|---------------------|------------|
| Sample Size | 15,346 | 17,895 | +2,549 (+16.6%) |
| Total Tests | 20 | 20 | 0 |
| Significant (p<0.01) | 0 | 10 | 10 |

## Top 10 Patterns by Effect Size (Cramér's V)

| Rank | Pattern | Pre-COVID V | ALL Data V | Change | % Change |
|------|---------|-------------|------------|--------|----------|
| 1 | Vulnerable Road User × Road Class | 0.069 | 0.093 | +0.024 | +34.0% |
| 2 | Ward × Collision Severity | 0.092 | 0.089 | -0.002 | -2.5% |
| 3 | Accident Location × Collision Severity | 0.065 | 0.076 | +0.011 | +16.4% |
| 4 | Vulnerable Road User × Light Conditions | 0.075 | 0.065 | -0.010 | -13.2% |
| 5 | Light Conditions × Collision Severity | 0.053 | 0.064 | +0.011 | +20.8% |
| 6 | District × Collision Severity | 0.061 | 0.062 | +0.001 | +1.5% |
| 7 | Traffic Control × Collision Severity | 0.044 | 0.048 | +0.004 | +9.6% |
| 8 | Weather Visibility × Collision Severity | 0.044 | 0.046 | +0.002 | +4.2% |
| 9 | Road Surface × Collision Severity | 0.044 | 0.042 | -0.002 | -5.6% |
| 10 | Road Class × Collision Severity | 0.033 | 0.035 | +0.002 | +6.5% |

## Ranking Changes (Among Significant Patterns)

Patterns that changed ranking position:

| Pattern | Pre-COVID Rank | ALL Data Rank | Change |
|---------|----------------|---------------|--------|
| Accident Location × Collision Severity | N/A | 3 | → (same) |
| District × Collision Severity | N/A | 6 | → (same) |
| Light Conditions × Collision Severity | N/A | 5 | → (same) |
| Road Surface × Collision Severity | N/A | 9 | → (same) |
| Season × Collision Severity | N/A | 10 | → (same) |
| Traffic Control × Collision Severity | N/A | 7 | → (same) |
| Vulnerable Road User × Light Conditions | N/A | 4 | → (same) |
| Vulnerable Road User × Road Class | N/A | 1 | → (same) |
| Ward × Collision Severity | N/A | 2 | → (same) |
| Weather Visibility × Collision Severity | N/A | 8 | → (same) |

## Key Findings

### Patterns with Largest Effect Size INCREASES:

- **Vulnerable Road User × Road Class**: V increased from 0.069 to 0.093 (+34.0%)
- **Light Conditions × Collision Severity**: V increased from 0.053 to 0.064 (+20.8%)
- **Accident Location × Collision Severity**: V increased from 0.065 to 0.076 (+16.4%)

### Patterns with Largest Effect Size DECREASES:

- **Vulnerable Road User × Light Conditions**: V decreased from 0.075 to 0.065 (-13.2%)
- **Day of Week × Collision Severity**: V decreased from 0.036 to 0.031 (-14.8%)
- **Road Surface × Collision Severity**: V decreased from 0.044 to 0.042 (-5.6%)

## Overall Assessment

### Pattern Stability

- **Average absolute effect size change**: 0.004
- **Average percentage change**: 14.3%

### Top Pattern

- **Pre-COVID #1**: Ward × Collision Severity (V=0.092)
- **ALL Data #1**: Vulnerable Road User × Road Class (V=0.093)
- **Conclusion**: ⚠️ Top pattern changed between analyses

### Robustness Assessment

- **Patterns significant in BOTH analyses**: 0 / 0 (0%)
- **Patterns only significant in Pre-COVID**: 0
- **Patterns only significant in ALL Data**: 10
  - Season × Collision Severity
  - Ward × Collision Severity
  - Vulnerable Road User × Road Class
  - Traffic Control × Collision Severity
  - Weather Visibility × Collision Severity
  - Vulnerable Road User × Light Conditions
  - Accident Location × Collision Severity
  - District × Collision Severity
  - Light Conditions × Collision Severity
  - Road Surface × Collision Severity

## Conclusion

**Pattern Stability**: **MODERATELY STABLE**

Most patterns persist with the inclusion of COVID data, though some modest changes in effect sizes are observed.

No significant patterns found in Pre-COVID analysis for comparison.

### Recommendation

**Consider presenting both analyses:**
- Top patterns differ between datasets
- Effect size changes are substantial
- COVID data appears to meaningfully affect patterns

Clearly document the rationale for any choice made.

---

*Report generated: 2025-11-04 01:22:34*