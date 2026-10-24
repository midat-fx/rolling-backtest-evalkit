# Performance

Three changes made the evaluation faster without changing a single output byte:

1. **Vectorised ETS fitting.** Smoothing parameters are chosen by grid search on
   in-sample SSE. Instead of running the recursion once per grid point, the
   recursion runs once over time with the whole grid as a vector
   (`models.ets_fit_forecast`). A test checks it against the scalar textbook
   recursion (`models.ets_sse_reference`) point by point.
2. **Series-major backtest with vectorised metrics.** Each series is forecast
   for all folds and models in one task; metrics are computed on the stacked
   (series, fold, model, horizon) arrays instead of per-cell Python calls.
3. **Parallel series.** `evalkit run -j N` sends series to N processes.
   `pool.map` returns results in series order, so the outputs (metrics, stats,
   report, manifest and its `run_id`) are byte-identical for any N; the
   benchmark below checks every file, and `tests/test_determinism.py` checks it
   in CI.

## Numbers

Measured with `python scripts/benchmark.py` on an Apple-silicon laptop (16
CPUs) that was also running other work (load average ≈ 5), so absolute times
are noisy; ratios are the point. Output pasted verbatim:

### ETS (Holt-Winters) fit, one series of 72 months, 250-point grid

| implementation | time per fit | speed-up |
|---|---|---|
| scalar loop over grid | 11.6 ms | 1.0x |
| vectorised grid | 0.47 ms | 25x |

### Full run (`evalkit run -j N`), 11 models

| dataset | series | workers | wall time | speed-up | outputs identical |
|---|---|---|---|---|---|
| hospital | 767 | 1 | 6.43 s | 1.0x | True |
| hospital | 767 | 2 | 3.53 s | 1.8x | True |
| hospital | 767 | 4 | 2.09 s | 3.1x | True |
| hospital | 767 | 8 | 1.44 s | 4.5x | True |
| tourism_monthly | 366 | 1 | 13.46 s | 1.0x | True |
| tourism_monthly | 366 | 2 | 6.96 s | 1.9x | True |
| tourism_monthly | 366 | 4 | 4.07 s | 3.3x | True |
| tourism_monthly | 366 | 8 | 2.47 s | 5.4x | True |

Backtest only, v1.0.0 vs now, on the bundled synthetic panel (40 series, 13
folds, the original 5 models; best of 5, same machine). v1.0.0 measured from a
`git worktree` of `main`:

| version | `run_backtest` time |
|---|---|
| v1.0.0 (per-cell metric calls) | 55.0 ms |
| this version (stacked arrays) | 17.3 ms |

```bash
git worktree add /tmp/evalkit-v1 main
for src in /tmp/evalkit-v1/src src; do PYTHONPATH=$src python -c "
import time, pandas as pd
from evalkit.backtest import run_backtest
from evalkit.run import load_config
df = pd.read_csv('data/demand.csv'); cfg = load_config('configs/synthetic.yaml')
print(min(__import__('timeit').repeat(lambda: run_backtest(df, cfg), number=1, repeat=5)))"; done
```

## Where the time goes

On hospital with one worker, a profile (`python -m cProfile -s cumtime -m
evalkit.cli run ...`) puts over 80% of the time in the ETS fits (four ETS variants
plus Theta's SES, for 767 series × 5 origins) and almost all the rest in the
other models; statistics, the leakage audit and the HTML report take well
under a second. The next step, if needed, would be fitting all series of a fold
in one vectorised pass (equal-length panels only) rather than more processes.
