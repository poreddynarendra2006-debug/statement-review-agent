# Roadmap and development estimate

AuditLens · Team 33 · 12 September 2026

What it took to build, what we deliberately left out, and what we would do with
more time. The estimates are what the work actually cost, not what we hoped it
would.

---

## 1. What the build actually took

Eight people, seven days, 8–14 September 2026.

| Area | Owner | Effort | Outcome |
|:--|:--|:--:|:--|
| Data ingestion | Data Ingestion | ~5 days | CSV and Excel into one record shape, non-financial files rejected |
| Validation rules | Validation | ~5 days | 9 rules: 5 accounting identities, 4 data-quality |
| Trend and forecasting | Trend | ~6 days | YoY, ratios, regression forecast, materiality-graded deviations |
| Anomaly detection | Anomaly | ~6 days | Isolation Forest on 27 engineered features, plain-language explanations |
| Evidence and review | Evidence & Review | ~6 days | Evidence packet, narrative writer, the fine-tuned model |
| Risk and reporting | Risk & Reporting | ~5 days | Explainable 0–100 score, PDF, history, monitoring |
| Frontend | Frontend | ~5 days | 7 screens with sign-in |
| Orchestration, API, deployment | Orchestration & API | ~7 days | Planner, API, auth, 2 agents, groundedness, AWS |
| **Total** | | **~45 person-days** | 565 tests, 10 agents, live on AWS |

**Where the time actually went**, which is the part worth saying out loud:
roughly **a third of the orchestration effort was integration**, not feature
work — folders arriving in the wrong shape, entry points that ran less than
their name promised, field names the frontend guessed at. The lesson we would
carry into a second attempt is to fix the contracts first and write the
integration test before the components exist.

---

## 2. Known limitations, stated plainly

| Limitation | Consequence | Why we accepted it |
|:--|:--|:--|
| **SQLite inside the container** | Reviews and accounts reset on every deployment | A managed database costs money and adds setup; a demo does not need data to survive a restart |
| **One running task** | No horizontal scaling | Two copies would each hold their own database; fixing that needs the database above |
| **The trained model is not served** | The narrative is written by the deterministic fallback | Serving flan-t5 needs PyTorch in the image and more memory than a 2 GB task has |
| **Forecast quality** | Median R² 0.19 on synthetic data | The findings depend on deviation from expectation, not on the forecast being precise |
| **Anomaly F1 0.34** | Some anomalies are missed | Unsupervised, on 13 planted cases; false alarms matter more to a reviewer than perfect recall |
| **English only, one currency per file** | Not usable for a multinational group as-is | Out of scope for the use case |
| **CSV and Excel only** | A PDF statement cannot be uploaded | PDF statement parsing is a project in itself; ruled out of scope on 12 September |

---

## 3. What we would build next

Ordered by value to a reviewer, with honest estimates.

### Near term — 1 to 2 weeks

| # | Item | Estimate | Why it is first |
|:--|:--|:--:|:--|
| 1 | **RDS PostgreSQL** instead of SQLite | 3 days | Removes two limitations at once: data survives deploys, and the service can scale past one task |
| 2 | **Serve the trained model** behind a flag | 3 days | The model exists and is measured; it needs a larger task and a loader, with the deterministic writer as fallback |
| 3 | **Per-company risk score** | 2 days | Today one score covers the whole upload. A reviewer with 60 companies needs to know which to open first |
| 4 | **Uploaded file retention with a 2-day lifecycle** | 2 days | An auditor wants the source document beside the finding; the lifecycle rule keeps it from becoming a data-protection liability |

### Medium term — 1 to 2 months

| # | Item | Estimate | Why |
|:--|:--|:--:|:--|
| 5 | **PDF and scanned statement ingestion** | 2 weeks | The format most statements actually arrive in; needs table extraction and OCR |
| 6 | **Reviewer feedback loop** | 1 week | Every dismissed finding is a label. Enough of them turn anomaly detection from unsupervised into supervised |
| 7 | **Multi-entity consolidation** | 2 weeks | Group accounts with intercompany eliminations, which is where real review effort goes |
| 8 | **Audit trail and e-signature** | 1 week | Who reviewed what, when, and signed off — required before this is usable as working papers |

### Longer term

| # | Item | Why |
|:--|:--|:--|
| 9 | **Industry benchmark data** | Peer comparison is limited to companies inside one upload; external benchmarks make it meaningful for a single company |
| 10 | **Continuous monitoring** | Run the review on each period close rather than on demand, and alert on emerging patterns |
| 11 | **A larger fine-tuned model** | flan-t5-base is 250M parameters. A 1B model would write better prose — and the groundedness check already exists to keep it honest |

---

## 4. What it would cost to run properly

Today's demo costs about **$20 a month**, dominated by the load balancer.

| Scale | Configuration | Estimate |
|:--|:--|:--:|
| Demo, as today | 1 Fargate task, SQLite, ALB | ~$20/month |
| A single audit team | 2 tasks, RDS small, S3 with lifecycle | ~$120/month |
| A firm, with the model served | 4 GPU-less tasks at 4 GB, RDS, S3 | ~$400/month |

The model runs on CPU by design, so serving it costs memory rather than a GPU.
That was a deliberate choice at 250M parameters.

---

## 5. What we would do differently

Four things, and they are all process rather than technology:

1. **Freeze the data contracts on day one.** Most of our integration pain came
   from field names being guessed rather than agreed — the frontend read
   `trend_deviations` while the API sent `material_deviations`.
2. **Write the integration test before the component exists.** Our manual check
   script found broken uploads in seconds, and we built it on day four. On day
   one it would have saved days.
3. **Test the entry point, not just the package.** The Validation role's
   data-quality checks were written, tested and never called, because the
   function the pipeline used ran only half of them.
4. **Measure before believing.** Every calibration decision that turned out to
   matter — the anomaly cap hiding real findings, peer comparison flagging 37%
   of clean data, the model inventing a digit — was invisible until it was
   measured against a known answer.
