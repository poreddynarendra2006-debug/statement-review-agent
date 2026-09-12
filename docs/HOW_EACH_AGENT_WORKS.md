# How each agent works

AuditLens · Team 33 · one diagram per role, drawn from the code as it stands on
12 September 2026.

Read the first diagram for the shape of the whole system, then any single
section for the part you own.

---

## The whole pipeline

```mermaid
flowchart TD
    U["Reviewer uploads a CSV or Excel file"] --> ING["Data Ingestion<br/>maps columns onto one record shape"]
    ING --> PLAN["Planner<br/>decides which agents the data can support"]

    PLAN --> VAL["Validation<br/>5 accounting identities"]
    PLAN --> TRD["Trend<br/>year on year, ratios, forecast"]
    PLAN --> ANO["Anomaly<br/>statistical and model based"]
    PLAN --> REC["Recurring issues<br/>same problem, 3+ years"]
    PLAN --> PEER["Peer comparison<br/>against the same industry"]

    VAL --> EV["Evidence Agent<br/>one packet of verified findings"]
    TRD --> EV
    ANO --> EV
    REC --> EV
    PEER --> EV

    EV --> REV["Review Agent<br/>writes the narrative"]
    REV --> GRD{"Groundedness check<br/>is every number in the evidence?"}
    GRD -->|yes| OUT["AnalysisResult"]
    GRD -->|no| WARN["Warn the reviewer<br/>and name the unsupported figures"] --> OUT

    VAL --> RISK["Risk Scorer<br/>explainable 0-100"]
    TRD --> RISK
    ANO --> RISK
    RISK --> OUT

    OUT --> API["FastAPI<br/>sign-in required"]
    API --> UI["The reviewer's screens"]
    API --> PDF["PDF report"]
    API --> DB[("SQLite<br/>reviews, actions, accounts")]
```

**The rule that shapes everything:** no figure is ever produced by a language
model. Every number is computed in ordinary Python, and the model is given only
figures already established as true. The groundedness check enforces it.

---

## 1. Data Ingestion

**Owner:** Data Ingestion · **Code:** `extraction/`

```mermaid
flowchart TD
    F["Uploaded file"] --> T{"File type?"}
    T -->|".xlsx"| X["Convert the sheet to CSV"]
    T -->|".csv"| P
    T -->|"anything else"| R1["422 - unsupported file type"]
    X --> P["Read the table"]
    P --> M["Map column names<br/>'Share Holder Equity' to shareholder_equity"]
    M --> V{"Does it look like<br/>a financial statement?"}
    V -->|no| R2["422 - not a financial statement"]
    V -->|yes| REC["One FinancialRecord per company-year"]
    REC --> W["Warnings for anything odd<br/>passed on to the reviewer"]
```

**Checked:** reads 480 of 480 rows from the dummy file with values matching the
source, and rejects a non-financial file rather than guessing.

---

## 2. Validation

**Owner:** Validation Agent · **Code:** `validation_agent/`

```mermaid
flowchart TD
    R["Records"] --> C{"Does the row carry<br/>the fields this rule needs?"}
    C -->|no| S["SKIPPED<br/>never a false failure"]
    C -->|yes| E["Compute the expected figure"]
    E --> D{"Difference above<br/>the materiality threshold?"}
    D -->|no| PASS["PASS"]
    D -->|yes| FAIL["FAIL<br/>expected, actual, difference, evidence line"]
```

**The five identities**

| Rule | Identity |
|:--|:--|
| VAL_BS_01 | Total Assets = Total Liabilities + Equity |
| VAL_GP_02 | Gross Profit = Revenue − Cost of Revenue |
| VAL_OP_03 | Operating Income = Gross Profit − Operating Expenses |
| VAL_NI_04 | Net Income = Pre-tax Income − Taxes |
| VAL_CF_05 | Ending Cash = Beginning Cash + Operating + Investing + Financing |

**Checked:** catches 147 of 147 planted errors, with no false alarms on the
clean file.

---

## 3. Trend

**Owner:** Trend Agent · **Code:** `analysis/trend.py` and its neighbours

```mermaid
flowchart TD
    R["Records, 2 or more periods"] --> Y["Year on year movement<br/>per line item"]
    R --> RA["Ratios<br/>liquidity, leverage, profitability, returns"]
    R --> FC["Forecast the next year<br/>linear regression on history"]
    FC --> BT["Compare forecast against actual"]
    BT --> DEV{"Deviation above<br/>the materiality threshold?"}
    DEV -->|yes| MAT["material_deviation = true<br/>this is what risk scoring reads"]
    DEV -->|no| ON["On track"]
    Y --> OUT["yoy, ratios, forecasts,<br/>evaluations, deviations"]
    RA --> OUT
    MAT --> OUT
    ON --> OUT
```

**Skipped when** the file has fewer than 2 periods — a movement needs two points.

**Checked:** revenue growth matches a hand calculation; 57 material deviations
on the defective file.

---

## 4. Anomaly

**Owner:** Anomaly Agent · **Code:** `finsight/`

```mermaid
flowchart TD
    R["Records"] --> N{"24 or more<br/>company-years?"}
    N -->|no| SK["Skipped - with fewer rows<br/>most of them look unusual"]
    N -->|yes| FE["Engineer scale-neutral features<br/>ratios and growth, not raw size"]
    FE --> IF["Isolation Forest<br/>unsupervised model"]
    FE --> RZ["Robust z scores<br/>distance from the median"]
    IF --> CAL["Combine into one score"]
    RZ --> CAL
    CAL --> CAP{"More than 4.8%<br/>of rows flagged?"}
    CAP -->|no| KEEP["Report them"]
    CAP -->|yes| CERT{"Is it beyond<br/>30 deviations?"}
    CERT -->|yes| KEEP2["Always reported<br/>not a judgement call"]
    CERT -->|no| TRIM["Keep only the strongest,<br/>up to the cap"]
    KEEP --> EXP["Plain-language explanation<br/>with the normal range"]
    KEEP2 --> EXP
    TRIM --> EXP
```

**Why the cap exists:** without it an ordinary file fills with borderline
findings. **Why the override exists:** the cap used to hide real ones — a file
with 10 blatant anomalies in 40 rows reported 2.

**Checked:** 5.0% flagged on clean data, the planted anomaly found, small
uploads skipped.

---

## 5. Recurring issues

**Owner:** Orchestration & API · **Code:** `analysis/recurring_issues.py`

```mermaid
flowchart TD
    IN["The other agents' findings"] --> G["Group by company and by<br/>failed rule, anomaly type or metric"]
    G --> C{"Does it appear in<br/>3 or more years?"}
    C -->|no| D["Not recurring"]
    C -->|yes| S{"How many years?"}
    S -->|"3"| M["MEDIUM"]
    S -->|"4 or more"| H["HIGH"]
    M --> O["Recurring issue<br/>with every year listed as evidence"]
    H --> O
```

**Why it matters:** the same check failing for four years running is a
different problem from one bad year, and no single-year agent can see it.

**Checked:** finds 3 of 3 planted repeats, nothing extra, skips a 2-year file.

---

## 6. Peer comparison

**Owner:** Orchestration & API · **Code:** `analysis/peer_comparison.py`

```mermaid
flowchart TD
    R["Records"] --> G["Group by industry and year"]
    G --> N{"5 or more companies<br/>in that group?"}
    N -->|no| W["Widen to the whole year<br/>and say so in the finding"]
    N -->|yes| Q["Quartiles of the group<br/>for 7 ratios"]
    W --> Q
    Q --> D{"Outside the middle half by<br/>3 or more interquartile ranges?"}
    D -->|no| OK["Normal for its peers"]
    D -->|yes| REL{"And a real gap,<br/>not rounding?"}
    REL -->|no| OK
    REL -->|yes| F["Finding: the value,<br/>the peer median and the range"]
```

**Deliberately kept out of the risk score.** A specialist lender genuinely
carries more debt than a software firm; scoring that as a defect took clean
books from 44 MEDIUM to 82 CRITICAL.

**Checked:** 7% of company-years reported on clean data, planted outlier found.

---

## 7. Evidence Agent

**Owner:** Evidence & Review · **Code:** `evidence_agent/`, wired by `agents/evidence_agent.py`

```mermaid
flowchart TD
    V["Validation: only FAIL"] --> P["Evidence packet"]
    T["Trend: only material deviations"] --> P
    A["Anomaly findings"] --> P
    RC["Recurring issues"] --> P
    D["Uploaded document text<br/>screened first"] --> P
    P --> N["Normalise, never recalculate<br/>upstream figures are preserved exactly"]
    N --> FL["Flat list of findings<br/>for the screens and the report"]
    PE["Peer findings"] --> FL
```

**The boundary:** this is the last step before the language model. Only
computed figures pass it, and only the screened version of any uploaded text.

---

## 8. Review Agent and the groundedness check

**Owner:** Evidence & Review, with Orchestration for the check
**Code:** `review_agent/`, `agents/review_agent.py`, `agents/groundedness.py`

```mermaid
flowchart TD
    P["Evidence packet"] --> W["Write the narrative"]
    W --> M{"Which writer?"}
    M -->|"trained model"| MODE1["review_mode = model"]
    M -->|"rule-based fallback"| MODE2["review_mode = heuristic"]
    MODE1 --> G
    MODE2 --> G["Groundedness check"]
    G --> EX["Every number in the text"]
    EX --> Q{"In the computed evidence,<br/>or derived from two figures in it?"}
    Q -->|yes| OK["Supported"]
    Q -->|no| BAD["Unsupported - named in a warning<br/>shown to the reviewer"]
    OK --> R["Narrative plus a groundedness metric"]
    BAD --> R
```

**Why the check exists.** Our own fine-tuned model, on a validation example,
wrote a cost of revenue of **75,670** where the input said **74,670** — right
prose, right conclusion, one invented digit. Measured across 180 examples,
12.3% of the numbers it produced were unsupported. So the narrative is checked
rather than trusted.

---

## 9. Risk scoring

**Owner:** Risk & Reporting · **Code:** `risk_reporting/risk_engine.py`

```mermaid
flowchart TD
    V["Failed checks"] --> S["Points by severity"]
    A["Anomalies"] --> S
    T["Material deviations"] --> S
    S --> DIM["Repeated findings of the same severity<br/>add less each time"]
    DIM --> TOT["Total, capped at 100"]
    TOT --> L{"Level"}
    L -->|"0-24"| LOW["LOW"]
    L -->|"25-49"| MED["MEDIUM"]
    L -->|"50-74"| HIGH["HIGH"]
    L -->|"75-100"| CRIT["CRITICAL"]
    TOT --> EXP["Every point traced to the finding<br/>that earned it"]
```

**Explainable by construction:** the contributors always add up to the score
shown, so a reviewer can ask "why 60?" and get a list.

**Checked:** clean file 44 MEDIUM, defective file 60 HIGH.

---

## 10. Reporting, history and monitoring

**Owner:** Risk & Reporting · **Code:** `database/`, `reports/`

```mermaid
flowchart TD
    R["Finished review"] --> DB[("SQLite<br/>reviews, actions, stage timings")]
    DB --> L["GET /reviews<br/>recent reviews with file name and risk"]
    DB --> O["GET /reviews/id<br/>reopen a past review"]
    DB --> P["GET /reviews/id/report.pdf<br/>generated on request"]
    DB --> M["GET /monitoring/...<br/>stage timings and agent coverage"]
    A["Reviewer accepts, challenges<br/>or annotates a finding"] --> DB
```

**Note for demo day:** the database lives inside the container, so reviews and
accounts reset on every deployment. Demo reviews are seeded automatically;
accounts are not.

---

## 11. Orchestration and API

**Owner:** Orchestration & API · **Code:** `agents/`, `api/`

```mermaid
flowchart TD
    REQ["Request"] --> AUTH{"Signed in?"}
    AUTH -->|no| R401["401 - sign in to continue"]
    AUTH -->|yes| PLAN["Planner examines the data"]
    PLAN --> F["Facts: companies, periods,<br/>rows, statement detail"]
    F --> SEL{"What can this data support?"}
    SEL --> RUN["Run the agents that qualify"]
    SEL --> SKIP["Record why each other agent<br/>did not run"]
    RUN --> GUARD["Screen any uploaded text<br/>for instruction-like content"]
    GUARD --> RES["AnalysisResult"]
    SKIP --> RES
    RES --> TIME["Seconds per stage, recorded"]
```

**Requirements each agent must meet**

| Agent | Needs |
|:--|:--|
| Validation | nothing — it skips checks whose inputs are missing |
| Trend | 2 or more periods |
| Anomaly | 24 or more company-years |
| Recurring | 3 or more periods |
| Peer | 5 or more companies |

**Why it matters:** a reviewer who sees no trend findings must know whether
there were none, or whether the agent never ran. The interface shows both.

---

## 12. The reviewer's screens

**Owner:** Frontend · **Code:** `ui/`

```mermaid
flowchart TD
    SU["Sign up<br/>name, email, role, password"] --> SI["Sign in"]
    SI --> T["Token held in the browser<br/>sent with every request"]
    T --> UP["Upload"]
    UP --> DA["Dashboard<br/>checks, failures, anomalies, risk"]
    DA --> FI["Findings<br/>filter by source, severity, company"]
    FI --> AC["Accept, challenge or annotate<br/>each finding"]
    DA --> RI["Risk matrix<br/>points per contributor"]
    DA --> TR["Ratio trends"]
    DA --> AI["AI summary"]
    DA --> PD["Download the PDF"]
```

Served by the same container as the API, so the screens and the data share one
address and one sign-in.

---

## 13. Deployment

**Owner:** Orchestration & API · **Code:** `Dockerfile`, `docs/DEPLOYMENT.md`

```mermaid
flowchart TD
    G["Code on main"] --> TEST["pytest - 552 tests"]
    TEST --> B["docker build"]
    B --> LOCAL["Run locally, check /health"]
    LOCAL --> ECR["Push to Amazon ECR<br/>tagged with the commit"]
    ECR --> UPD["Update the ECS Express service"]
    UPD --> CAN["Canary: 5% of traffic for 3 minutes"]
    CAN --> HC{"Health checks passing?"}
    HC -->|yes| LIVE["All traffic moves over"]
    HC -->|no| RB["Automatic rollback<br/>the working version stays"]
    LIVE --> MON["CloudWatch alarms<br/>errors, 5xx, downtime, slow responses"]
    MON --> MAIL["Email alert"]
```

**Live:** one Fargate task behind an Application Load Balancer with HTTPS, in
`ap-south-1`. Every release in ECR is a rollback point.

---

## What the numbers were, last time everything ran

| Measure | Value |
|:--|:--|
| Tests | 552 passing |
| Agents passing their manual check | 10 of 10 |
| Findings on the 480-row defective file | 267 |
| — failed accounting checks | 147 |
| — material trend deviations | 57 |
| — anomalies | 24 |
| — recurring issues | 3 |
| — peer comparisons | 36 |
| Risk score | 60 HIGH, against 44 MEDIUM on the clean file |
| Review time | about 3 seconds |
