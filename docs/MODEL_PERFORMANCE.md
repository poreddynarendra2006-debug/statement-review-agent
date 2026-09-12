# Model performance and evaluation

AuditLens · Team 33 · measured 12 September 2026

Every figure below comes from a run you can repeat: `python -m pytest -q`,
`python -m scripts.check_agents`, and `python -m finsight --benchmark`.

Where a number is weak, it is printed as it came out. A benchmark that only
shows its good results is not a benchmark.

---

## 1. The short version

| Component | What it is | Headline result |
|:--|:--|:--|
| **Validation** | 9 deterministic rules | **Precision 1.00, recall 1.00, F1 1.00** against a planted answer key |
| **Anomaly detection** | Isolation Forest + robust z-scores | **F1 0.42** at its operating point (0.34 under a harder test), against 0.28 for the statistical baseline |
| **Recurring issues** | Cross-year aggregation | **3 of 3** planted repeats found, nothing extra |
| **Peer comparison** | Quartile outlier detection | Flags **7.3%** of company-years; the planted outlier found |
| **Forecasting** | Linear regression, backtested | Median R² **0.19** on synthetic data — honestly weak, see §5 |
| **Review model** | Fine-tuned flan-t5-base | Training loss **1.93 → 0.09**, validation tracking it |
| **Groundedness** | Runtime verification | **12.3%** of the model's numbers unsupported, which is why the check exists |
| **Speed** | End to end | **3.97 s** for 480 company-years |

---

## 2. Validation — the part that must be exact

Measured against `data/dummy_statements_labels.json`, an answer key listing every
defect deliberately planted in the test file.

| Metric | Value |
|:--|:--|
| True positives | 147 |
| False positives | 0 |
| False negatives | 0 |
| **Precision** | **1.0000** |
| **Recall** | **1.0000** |
| **F1** | **1.0000** |
| False alarms on the clean file | **0** |

A perfect score here is not impressive in itself, and we would not present it as
if it were. These are deterministic accounting identities: assets must equal
liabilities plus equity. Anything less than 1.00 would mean the arithmetic was
wrong. It is reported because it is the control that proves the test data and
the answer key agree.

**What it does not measure:** whether the rules are the *right* rules. Nine
checks cannot catch every misstatement, and we do not claim they do.

---

## 3. Anomaly detection — where the real machine learning is

Unsupervised, so there are no labels at training time. To measure it at all, the
benchmark injects 13 known anomalies into 164 real company-years and asks how
many come back.

| Metric | Z-score baseline | **Isolation Forest** |
|:--|:--:|:--:|
| Precision | 0.1646 | **0.3125** |
| Recall | **1.0000** | 0.3846 |
| F1 | 0.2826 | **0.3448** |
| Accuracy | 0.5976 | **0.8841** |
| True positives | 13 | 5 |
| False positives | 66 | 11 |
| False negatives | 0 | 8 |

**Read this honestly.** The baseline catches everything and cries wolf 66 times;
the model catches fewer and is wrong far less often. For a reviewer with limited
hours, 11 false alarms is workable and 66 is not — which is why F1 and accuracy,
not recall alone, are the measures we optimise.

**Why an F1 of 0.34 is a reasonable result here**, and how we would say so:

- Detection is unsupervised. No labels exist in real financial data; the 13
  planted anomalies are a proxy for evaluation only.
- The planted anomalies are deliberately varied in severity. Some are subtle by
  construction, because a benchmark of only obvious cases proves nothing.
- Anomalies are a *prompt to look*, not a verdict. The deterministic rules carry
  the burden of correctness; this layer surfaces what rules cannot express.

**Calibration, which matters more than the F1.** On clean data the model flags
**5.0%** of company-years, and the same on the real Kaggle file. An earlier
setting flagged 37%, which a reviewer would have ignored entirely. 27 engineered
features, all scale-neutral, so a large company and a small one are comparable.

### One artefact worth correcting for

That benchmark plants anomalies in **8%** of rows, while the detector is
calibrated to flag about **5%**. Recall is therefore capped at 5/8 = 62.5%
before the model does anything at all - and 0.625 is exactly what it reaches
when the planted rate matches the rate it is built for:

| Planted rate | Precision | Recall | F1 | Accuracy |
|:--|:--:|:--:|:--:|:--:|
| 8% - more anomalies than the alert budget allows for | 0.3125 | 0.3846 | 0.3448 | 0.8841 |
| **5% - the rate the detector is calibrated for** | **0.3125** | **0.6250** | **0.4167** | **0.9146** |

Precision is identical in both, because the model flags the same rows either
way; only the number of true anomalies changed. So the fair statement is: **F1
0.42 at its operating point, and 0.34 when asked to find more anomalies than its
alert budget permits.** Both are reported here because the second is the more
demanding test and hiding it would be dishonest.

Reproduce with `run_anomaly_benchmark(anomaly_fraction=0.05)`.

The way to raise precision further is better features rather than a different
threshold, and that is roadmap work: the reviewer feedback loop in
`docs/ROADMAP.md` turns every dismissed finding into a label, which is what
would move this from unsupervised to supervised detection.


### The forensic rule layer

Detection runs in two layers. The model finds what is unusual *for a given
file*; five rules find patterns that are suspicious whatever the rest of the
file looks like - revenue that does not become cash, profit with negative
operating cash flow, a collapsed margin, a leverage spike, equity erosion. Each
is a documented indicator: the sales and gross margin indices of the Beneish
M-score, and the accrual divergence behind Dechow's F-score.

| Detector | Flagged | Precision | Recall | F1 |
|:--|:--:|:--:|:--:|:--:|
| Isolation Forest alone | 19 | 0.158 | 0.375 | 0.222 |
| **Forensic rules alone** | 7 | **0.714** | 0.625 | **0.667** |
| Both, as the agent runs | 22 | 0.227 | 0.625 | 0.333 |

**Two things have to be said about that 0.667.**

First, the rules were written while looking at the list of anomaly types this
benchmark plants, and they target those same patterns. The benchmark is
therefore **not an independent test of them**, and the figure is illustrative
rather than a benchmark result.

Second, combining the two detectors does **not** raise the combined F1 - the
union inherits the model's false positives, and 0.333 is the honest number for
how the agent actually runs. Showing only the best findings, certain ones first,
would reach 0.625, at the cost of hiding some model findings from the reviewer.
We chose to show everything.

**What is independent** is the false-positive rate:

| Dataset | Rows | Red flags raised |
|:--|:--:|:--:|
| Clean dummy statements | 480 | **0** |
| Defective dummy statements | 480 | **0** |
| Real Kaggle companies | 161 | 6 - four profit-without-cash, two equity erosion |

Zero across 960 company-years, and on real filings it surfaces genuine accrual
divergences the model missed entirely. A red flag that fires often is not a red
flag.

---

## 4. Recurring issues and peer comparison

Both are mine, and both are measured against planted answers.

| Agent | Test | Result |
|:--|:--|:--|
| Recurring | 3 planted repeats across the file | 3 found, 0 false, precision and recall 1.00 |
| Recurring | A 2-year upload | Correctly skipped — 3 periods are required |
| Peer | Clean dataset | 7.3% of company-years reported, every comparison against 5+ companies |
| Peer | A planted company-year with a margin unlike its industry | Found, ranked top by distance |
| Peer | Fewer than 5 companies | Correctly skipped |

**The calibration decisions behind those numbers**

- Peer comparison at its first setting flagged **37%** of a clean file. Moving
  from standard deviations to interquartile ranges brought it to **7%**, because
  financial ratios are skewed rather than bell-shaped.
- Feeding peer findings into the risk score took clean books from 44 MEDIUM to
  **82 CRITICAL**, so they are deliberately excluded from it.

---

## 5. Forecasting — the weakest number we report

The Trend agent fits a regression per company and metric, then backtests it.

| Metric | Median | Mean |
|:--|:--:|:--:|
| MAE | 2,235 | 3,017 |
| RMSE | 2,455 | 3,484 |
| **R²** | **0.19** | −1.05 |

**We are not going to dress this up.** A median R² of 0.19 means the linear
trend explains a fifth of the variation, and a negative mean means that for some
company-metric pairs the forecast is worse than predicting the average.

Two things are worth saying about it:

1. **The data is synthetic.** Our dummy companies are generated with random
   year-on-year variation, which by construction has no trend to find. Measured
   on the real Kaggle companies the fit is better, because real revenue has
   momentum:

   | Dataset | Median R² | Mean R² | Share above zero |
   |:--|:--:|:--:|:--:|
   | Dummy statements (synthetic) | 0.194 | −1.046 | 52% |
   | **Kaggle, real companies** | **0.363** | **0.048** | **63%** |
2. **The finding does not depend on the forecast being accurate.** What reaches
   the reviewer is a *material deviation* — actual against expected, past a
   materiality threshold. A wide forecast interval makes the agent quieter, not
   wrong.

**What we would do with more time:** hold out the last two years per company and
report R² on unseen data only, and compare against a seasonal-naive baseline. A
forecast that cannot beat "next year looks like this year" is not worth shipping.

---

## 6. The generative model

`google/flan-t5-base`, 250M parameters, fine-tuned on 1,020 examples generated
from our own rule engine — knowledge distillation from a rule-based teacher.

| Measure | Value |
|:--|:--|
| Training loss | 1.93 → **0.09** over 6 epochs |
| Validation loss | 0.17 → **0.09**, tracking training throughout |
| Training time | ~9 minutes on a Colab T4 |
| Overfitting | None visible — the two curves stay together (`training/results/loss_curve.png`) |

**Before and after, on the same validation example**

> **Before:** "Initech Corp. has a recurring issue with rising leverage in the years 2022, 2023, 2024 and 2023."
>
> **After:** "Rising leverage has persisted across 3 consecutive reporting periods (FY2022 to FY2024). A sustained trend of this kind is more significant than any single year's movement and should be addressed in management commentary."

The second is word for word what a reviewer would write.

### Groundedness — the measure that matters most

Does every number in the generated text appear in the input?

| Measure | Value |
|:--|:--|
| Comments quoting only input figures | 48.3% |
| Including figures the input arithmetically supports | 61.7% |
| Numbers generated | 659 |
| **Numbers nothing supports** | **81 (12.3%)** |

**The worst case we found**, and the most useful result of the whole exercise:

> Input: `cost of revenue: 74670` → Written: *"cost of revenue of USD 75,670"*

Right prose, right conclusion, one invented digit. A reviewer would have no way
of knowing.

**So we built the defence rather than hoping.** `agents/groundedness.py` checks
every number in a written review against the computed evidence at runtime, and
any figure that is neither quoted nor derivable from two evidence figures is
reported to the reviewer as unverified. A model that never made this mistake
would have proved nothing; this one proved the check is necessary.

---

## 7. System performance

| Measure | Value |
|:--|:--|
| End to end, 480 company-years | **3.97 s** |
| — Trend | 2.00 s |
| — Review | 0.76 s |
| — Anomaly | 0.65 s |
| — Validation | 0.47 s |
| — Evidence | 0.05 s |
| Real Kaggle file (161 rows) | ~1.4 s |
| Automated tests | **565 passing**, 4 skipped |
| Agents passing their manual check | **10 of 10** |

---

## 8. How to reproduce every number here

```bash
python -m pytest -q                  # the test suite
python -m scripts.check_agents       # all ten agents against planted answers
python -m finsight --benchmark       # the anomaly precision/recall/F1 table
```

The training figures come from `training/finetune_reviewer.ipynb`, and are
recorded in `training/results/metrics.json` alongside the loss curve.
