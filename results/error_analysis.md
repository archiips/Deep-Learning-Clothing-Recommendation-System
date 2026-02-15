# Error Analysis Report

## Executive Summary

This report analyzes the failure modes of the Matrix Factorization recommendation model by examining false positives (items recommended but not relevant) and false negatives (relevant items not recommended).

**Analysis Date:** 2026-02-15
**Model:** Matrix Factorization (Best Performer)
**Sample Size:** 50 false positives, 50 false negatives

---

## 1. False Positive Analysis

**Definition:** Items the model recommended in Top-10 but user did not actually interact with.

### Key Findings:

- **Total False Positives Analyzed:** 50
- **Average Rating:** 4.23 stars
- **Average Popularity:** 561.7 reviews per item
- **Department Distribution:**
  - Tops: 32 (64.0%)
  - Dresses: 18 (36.0%)

### Common Failure Patterns:

❌ Model recommends high-rated items even when not relevant to user

❌ Model heavily biases toward popular items (popularity bias)

❌ Over-representing Tops department


### Interpretation:

The model tends to recommend items that are:
1. **Highly rated** (avg 4.23/5) but not necessarily aligned with user preferences
2. **Popular items** (avg 562 reviews) suggesting popularity bias
3. Items from specific departments may be over-represented

**Root Cause:** The model optimizes for rating prediction accuracy but doesn't capture nuanced user preferences beyond ratings. High-rated popular items are "safe bets" but may not match individual taste.

---

## 2. False Negative Analysis

**Definition:** Items the user actually liked but model failed to recommend in Top-10.

### Key Findings:

- **Total False Negatives Analyzed:** 50
- **Average Rating:** 4.20 stars
- **Average Popularity:** 115.8 reviews per item
- **Average Model Prediction Score:** 3.10
- **Department Distribution:**
  - Tops: 16 (32.0%)
  - Bottoms: 15 (30.0%)
  - Intimate: 9 (18.0%)
  - Dresses: 7 (14.0%)
  - Jackets: 2 (4.0%)
  - Trend: 1 (2.0%)

### Common Failure Patterns:


### Interpretation:

The model struggles with:
1. **Long-tail items** (avg 116 reviews) - insufficient training signal
2. **Niche preferences** that don't align with popularity metrics
3. Items with lower community ratings but high individual fit

**Root Cause:** Sparse data problem - items with fewer reviews have less stable embeddings. Model under-predicts ratings for these items.

---

## 3. User Segment Analysis

### False Positives by User Segment:
- **New Users:** 0 errors
- **Casual Users:** 42 errors
- **Active Users:** 0 errors

### False Negatives by User Segment:
- **New Users:** 1 errors
- **Casual Users:** 47 errors
- **Active Users:** 0 errors


### Interpretation:

- **New users** (0-2 reviews): Suffer from cold-start problem, model lacks sufficient data
- **Casual users** (3-10 reviews): Model has moderate signal but may still miss niche preferences
- **Active users** (11+ reviews): Model performs best but still makes errors on edge cases

---

## 4. Recommendations for Improvement

### Short-term Fixes (Quick Wins):

1. **Diversity Penalty:** Add explicit diversity constraint to reduce over-representation of popular items
   - Limit max 2 items from same department in Top-10
   - Boost long-tail items with recency or novelty factor

2. **Hybrid Approach:** Combine MF with content-based features
   - Add department embeddings to capture category preferences
   - Use review text sentiment for better signal

3. **Re-ranking Layer:** Apply business rules post-prediction
   - Boost items from user's favorite department
   - Filter out departments user has never purchased from

### Medium-term Improvements:

1. **Ensemble Model:** Combine MF + NCF + Content-Based
   - Use MF for general preferences
   - Use content features for cold-start items
   - Use NCF for complex interaction patterns

2. **Negative Sampling Strategy:** Improve training signal
   - Sample hard negatives (high-rated items user didn't interact with)
   - Weight negatives by item popularity to reduce bias

3. **Multi-Objective Optimization:** Balance accuracy with diversity
   - Joint loss: rating prediction + diversity reward + novelty reward

### Long-term Research:

1. **Sequential Models:** Capture temporal patterns (user preferences evolve)
2. **Graph Neural Networks:** Leverage user-item-category graph structure
3. **Contextual Bandits:** Online learning with real-time feedback

---

## 5. Tradeoffs & Business Considerations

### Accuracy vs Diversity:
- Higher precision → more popular items → less diversity → filter bubble
- Higher diversity → more long-tail → lower CTR initially → better exploration

**Recommendation:** Target 60% popular + 40% diverse for balanced exploration-exploitation

### Coverage vs Relevance:
- Recommending more catalog → higher coverage → some irrelevant items
- Recommending only confident predictions → lower coverage → missed opportunities

**Recommendation:** Aim for 40%+ catalog coverage while maintaining >3% Precision@10

### Novelty vs Familiarity:
- Novel items → lower short-term CTR → higher long-term engagement
- Familiar items → higher CTR → risk of boredom

**Recommendation:** A/B test novelty boost (10-20% novel items in recs)

---

## 6. Conclusion

The Matrix Factorization model achieves solid performance (3.79% Precision@10, 37.23% Hit Rate@10) but exhibits clear biases:

✅ **Strengths:**
- Performs well on popular items
- Good at capturing general user preferences
- Fast inference (<0.04ms per user)

❌ **Weaknesses:**
- Popularity bias → over-recommends mainstream items
- Cold-start problem → misses long-tail items
- Limited personalization for niche tastes

**Next Steps:**
1. Implement diversity penalty in re-ranking layer
2. Add content-based features for cold-start items
3. A/B test hybrid model (MF + content features)
4. Monitor coverage and novelty metrics in production

---

**Report Generated:** 2026-02-15
**Model Version:** mf_best.pt
**Evaluation Set:** 857 test users
