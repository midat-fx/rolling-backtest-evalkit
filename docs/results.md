# Results on real data

Produced by the commands below; the tables are `report.md` from each run, pasted
verbatim. The per-fold metrics are committed as `reports/baseline_<dataset>.csv`
and gated in CI (`make gate-real`, and the pages workflow).

```bash
make fetch
evalkit run --config configs/hospital.yaml --data data/hospital.csv -o out/hospital -j 8
evalkit run --config configs/tourism_monthly.yaml --data data/tourism_monthly.csv -o out/tourism_monthly -j 8
```

Setup: rolling origins every 6 months, 12-month horizon, 5 origins per series; MASE
and RMSSE are scaled by the in-sample seasonal-naive error (lag 12), so
seasonal naive scores ≈ 1 by construction on the training data. Median metrics
are medians over the 5 fold-level means; the significance tables use one MASE
per series (mean over its folds).

## Hospital (767 series, run_id `ea816b669b2a`)

### Median metrics by model

| model | wape | smape | mase | rmsse | bias |
|---|---|---|---|---|---|
| theta | 0.1755 | 0.1751 | 0.7978 | 0.7852 | 0.0153 |
| ses | 0.1824 | 0.1835 | 0.8501 | 0.8305 | 0.0127 |
| holt_winters | 0.1863 | 0.1867 | 0.8241 | 0.8037 | 0.0175 |
| moving_average | 0.1880 | 0.1872 | 0.8580 | 0.8402 | 0.0101 |
| ridge_lags | 0.2080 | 0.2022 | 0.8966 | 0.8721 | 0.0309 |
| seasonal_naive | 0.2134 | 0.2115 | 0.9327 | 0.9210 | 0.0127 |
| holt_damped | 0.2176 | 0.2223 | 1.0120 | 0.9767 | 0.0109 |
| naive | 0.2188 | 0.2191 | 1.0056 | 0.9603 | 0.0099 |
| seasonal_drift | 0.2196 | 0.2173 | 0.9615 | 0.9439 | 0.0295 |
| drift | 0.2373 | 0.2345 | 1.0616 | 0.9994 | 0.0166 |
| holt | 0.2788 | 0.2897 | 1.3151 | 1.2381 | 0.0149 |

**Winner by median WAPE:** theta

### Significance (per-series MASE, 767 series)

| model | mean mase | 95% bootstrap CI | avg rank | DM vs theta (Holm p) |
|---|---|---|---|---|
| theta | 0.7969 | [0.7766, 0.8173] | 3.20 | — |
| holt_winters | 0.8144 | [0.7946, 0.8344] | 4.09 | 0.00013 * |
| ses | 0.8465 | [0.8187, 0.8769] | 3.29 | 3.4e-06 * |
| moving_average | 0.8611 | [0.8367, 0.8873] | 4.42 | < 1e-16 * |
| ridge_lags | 0.9328 | [0.9087, 0.9594] | 6.09 | < 1e-16 * |
| seasonal_naive | 0.9462 | [0.9233, 0.9701] | 6.98 | < 1e-16 * |
| seasonal_drift | 0.9632 | [0.9403, 0.9876] | 7.58 | < 1e-16 * |
| naive | 1.0042 | [0.9726, 1.0371] | 6.78 | < 1e-16 * |
| holt_damped | 1.0229 | [0.9804, 1.0724] | 6.10 | < 1e-16 * |
| drift | 1.0604 | [1.0268, 1.0952] | 8.14 | < 1e-16 * |
| holt | 1.3306 | [1.2676, 1.4045] | 9.33 | < 1e-16 * |

Friedman (Iman-Davenport) p = < 1e-16; Nemenyi critical difference = 0.545 ranks. `*` = DM rejects equal accuracy with theta at 0.05 after Holm adjustment.

## Tourism monthly (366 series, run_id `ccd459f78eeb`)

### Median metrics by model

| model | wape | smape | mase | rmsse | bias |
|---|---|---|---|---|---|
| holt_winters | 0.1662 | 0.1780 | 1.3095 | 1.0884 | -0.0239 |
| ridge_lags | 0.1814 | 0.1925 | 1.4452 | 1.1863 | -0.0115 |
| seasonal_drift | 0.1873 | 0.1967 | 1.4239 | 1.2119 | -0.0078 |
| seasonal_naive | 0.1888 | 0.2034 | 1.4506 | 1.2335 | -0.0413 |
| theta | 0.2038 | 0.2011 | 1.5983 | 1.3693 | -0.0191 |
| ses | 0.3437 | 0.3476 | 3.1511 | 2.7588 | -0.1319 |
| holt_damped | 0.3511 | 0.3579 | 3.2001 | 2.7847 | -0.1276 |
| moving_average | 0.3527 | 0.3622 | 3.2729 | 2.8079 | -0.1409 |
| holt | 0.3693 | 0.3787 | 3.3297 | 2.8809 | -0.1299 |
| drift | 0.3776 | 0.3989 | 3.4003 | 2.9999 | -0.1551 |
| naive | 0.3777 | 0.4014 | 3.4098 | 3.0091 | -0.1714 |

**Winner by median WAPE:** holt_winters

### Significance (per-series MASE, 366 series)

| model | mean mase | 95% bootstrap CI | avg rank | DM vs holt_winters (Holm p) |
|---|---|---|---|---|
| holt_winters | 1.3271 | [1.2679, 1.3869] | 2.41 | — |
| seasonal_drift | 1.4212 | [1.3643, 1.4846] | 3.83 | 2.6e-06 * |
| ridge_lags | 1.4291 | [1.3616, 1.5027] | 3.39 | 5.7e-05 * |
| seasonal_naive | 1.4694 | [1.4084, 1.5373] | 4.25 | 5.3e-13 * |
| theta | 1.5732 | [1.4859, 1.6665] | 3.73 | 3.2e-13 * |
| ses | 3.1672 | [2.9166, 3.4180] | 6.66 | < 1e-16 * |
| holt_damped | 3.2792 | [3.0009, 3.5672] | 7.28 | < 1e-16 * |
| moving_average | 3.3292 | [3.0767, 3.5931] | 8.14 | < 1e-16 * |
| holt | 3.4146 | [3.1466, 3.6823] | 8.41 | < 1e-16 * |
| naive | 3.6261 | [3.3839, 3.8868] | 8.87 | < 1e-16 * |
| drift | 3.6377 | [3.3936, 3.9007] | 9.04 | < 1e-16 * |

Friedman (Iman-Davenport) p = < 1e-16; Nemenyi critical difference = 0.789 ranks. `*` = DM rejects equal accuracy with holt_winters at 0.05 after Holm adjustment.

## Reading the results

- **Different data, different winner.** Theta wins hospital; additive
  Holt-Winters wins tourism, where seasonality is strong and stable. A
  framework that hard-codes a favourite model would get one of them wrong.
- **Significant is not the same as separable.** On hospital, theta beats SES on
  mean per-series MASE (0.7969 vs 0.8465) and the DM test rejects equality
  (Holm p = 3.4e-06), but their average ranks (3.20 vs 3.29) are well inside the
  Nemenyi critical difference (0.545): theta wins by a lot on some series, not on
  most. The report shows both views because they answer different questions.
- **Seasonality dominates tourism.** Every model without a seasonal component
  (naive, drift, SES, Holt, moving average) has mean MASE above 3; every model
  with one is below 1.6.
- **Undamped Holt is the worst model on hospital** (mean MASE 1.33): a 12-month
  linear extrapolation of noisy counts overshoots. Damping (phi = 0.9) brings it
  back to 1.02.
- **Sanity anchors.** The Monash archive reports MASE for these datasets at its
  own horizons and scaling; ours use a 12-month horizon and 5 rolling origins,
  so the numbers are comparable in magnitude only, not equal, and we do not
  quote them as a replication.
