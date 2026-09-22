# TikTok Video Engagement Prediction

**Author:** Haya Alharthi
**Competition:** WeCloudData Data Science Bootcamp — In-Class Kaggle Competition
**Task:** Predict the cumulative view count a TikTok video will reach at **Day 30**, using only video metadata, creator statistics, and engagement metrics observed during **Days 0–5**.
**Metric:** RMSE (Root Mean Squared Error) on a hidden test set.

## Final Result

| Model | 5-fold OOF RMSE |
|---|---|
| Constant-mean baseline | 262,365 |
| Linear (latest play count only) | 66,147 |
| Linear + CatBoost residual hybrid | 64,514 |
| Linear + XGBoost residual hybrid | 63,221 |
| Blend of the two hybrids | ~62,000 |
| **Viral Single-Gate (final model)** | **61,410** |

Final submission file: `viral_single_gate_rmse_61410.csv`

## Approach

Forecasting Day-30 views is hard mainly because of a **heavy power-law
distribution**: a small number of videos go viral and reach millions of
views, while most stay in the low hundreds or thousands. A handful of these
outliers dominate RMSE, so the modeling strategy is built around handling
them explicitly rather than relying on a single global model.

### 1. Feature engineering
- **Engagement features**: per-day play/like/comment/share/collect/download
  counts (Days 0–5), first/last observed values, growth rates, and
  engagement ratios (e.g., share-to-view rate).
- **Creator features**: the creator's most recent stats snapshot at or
  before the video's Day-5 cutoff (joined with a leakage-safe `merge_asof`,
  so no future creator data is used).
- **Calendar features**: hour, day of week, day of month, month, and
  weekend flag extracted from the video's creation time.

### 2. Base model — Linear + Tree-Residual Hybrids
`latest_play_count` alone correlates strongly (~0.97) with the Day-30
target, so each base model is built as:
1. A simple **linear regression** on `latest_play_count`.
2. A **tree model (CatBoost or XGBoost)** trained on the linear model's
   *residuals*, using the full feature set.

This keeps the strong linear trend intact and lets the tree model focus on
correcting systematic deviations from it, rather than re-learning the trend
from scratch (which risks overfitting given the skewed target).

Two such hybrids are built (CatBoost-residual, XGBoost-residual) and their
out-of-fold predictions are blended with an RMSE-minimizing weight.

### 3. Viral Two-Stage Correction (final step)
Because a small set of outlier videos drives most of the error, a second
stage explicitly models "will this video go viral":
1. A **classifier** predicts whether a video lands in the top 10% / 5% /
   2.5% of the training target (three candidate thresholds compared).
2. A **specialist regressor**, trained only on videos in that top bucket,
   predicts their (much larger) view counts.
3. The final prediction blends the base model with the specialist's
   output, weighted by the classifier's probability (raised to a tuned
   power) and a tuned gate strength (`gamma`).

All hyperparameters (threshold, power, gamma, blend weights) are selected
using **out-of-fold RMSE only** — the test labels are never used for
tuning.

### What was tried but not used
Several other approaches were explored during development and are not part
of the final pipeline because they did not outperform it on out-of-fold
validation:
- Creator-aware target-encoded residual correction
- Curve-fitting features (exponential/linear/power growth curves) + Ridge
  stacking
- A separate log-growth-multiplier model blended with the viral correction

## Repository Structure

```
.
├── README.md
├── tiktok_engagement_prediction_final.ipynb   # Clean, reproducible final pipeline
└── submissions/
    └── viral_single_gate_rmse_61410.csv       # Final Kaggle submission
```

## Reproducing the Result

1. Open `tiktok_engagement_prediction_final.ipynb` in a **Kaggle Notebook**
   with the competition dataset (`predictive-modelling-ds`) attached.
2. Run all cells top to bottom.
3. The final submission file is written to
   `/kaggle/working/viral_single_gate_rmse_61410.csv`.
4. Upload that file on the competition's **Submit Predictions** page.

## Rules Compliance
- Tabular modeling only (no external scraping, LLMs, or embeddings).
- No data leakage: all features are computed strictly from Days 0–5
  engagement and creator snapshots at or before the Day-5 cutoff.
- Individual effort — no team collaboration on modeling.
