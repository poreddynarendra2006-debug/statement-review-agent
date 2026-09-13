# High Level Design — AuditLens

**Financial Statement Review Agent**
Cognizant NPN AI & Analytics 2027 · Use case #2 · Team 33

| | |
|:--|:--|
| **Document** | High Level Design (HLD) v1.0 |
| **Date** | 8 September 2026 |
| **Status** | Baseline for build window 8–14 September 2026 |

---

## 1. Purpose and scope

This document describes the architecture of AuditLens: what the system does, how it is decomposed, what crosses each boundary, and which alternatives were considered and rejected.

**In scope:** ingestion of CSV and Excel financial statements, deterministic validation, trend and anomaly analysis, AI-generated review observations, risk scoring, reporting, and deployment.

**Out of scope:** general ledger integration, statutory filing, tax computation, consolidation of multi-entity groups, and any form of investment advice.

---

## 2. Problem statement

Financial statement review is a mandatory control function performed manually by auditors and controllers. It requires checking mathematical accuracy, comparing against prior years, reconciling figures that appear on more than one statement, explaining material variances, and recording review observations.

Performed by hand it is slow, repetitive, and error-prone. A reviewer working through a single company-year takes hours, and consistency degrades across a portfolio.

**Objective:** an AI-powered review agent that validates totals, compares year-on-year performance, highlights inconsistencies, and generates review observations with supporting evidence.

---

## 3. Goals and non-goals

### Goals

| ID | Goal |
|:--|:--|
| G1 | Detect every arithmetic inconsistency in a submitted statement set |
| G2 | Quantify year-on-year movement and flag what exceeds materiality |
| G3 | Identify figures that are anomalous against peers and against the company's own history |
| G4 | Produce narrative review observations, each traceable to a computed figure |
| G5 | Score overall review risk on an explainable 0–100 scale |
| G6 | Keep a human in the loop — every finding can be accepted, challenged or annotated |

### Non-goals

- The system does not allege fraud. It reports inconsistencies and recommends human review.
- The system does not issue investment or audit opinions.
- The system does not replace a qualified reviewer; it prepares their working papers.

---

## 4. Architecture overview

```
                              USER
                                |
                        Upload Statement
                                |
                          ORCHESTRATOR ------> POST /review  (external consumers)
                                |
            +-------------------+-------------------+
            |                   |                   |
       Validation            Trend               Anomaly
         Agent               Agent                Agent
            |                   |                   |
            +-------------------+-------------------+
                                |
====================== LLM BOUNDARY ==============================
   Everything above this line is deterministic Python. No figure
   below it is calculated by a model.
                                |
                          Evidence Agent
                                |
                          Review Agent
                                |
                         Risk Assessment
                                |
                          Final Report
```

The system is a **layered pipeline with a parallel analysis stage**. A single orchestrator sequences the run; the three analysis agents are independent of one another and could execute concurrently.

The defining structural feature is the **LLM boundary**. All computation happens above it. The language model receives a packet of already-verified figures and is responsible only for language: summarising, explaining, and recommending. This is discussed as design decision D1.

---

## 5. Component design

| # | Component | Responsibility | Owner |
|:--|:--|:--|:--|
| C1 | **Ingestion** | Parse CSV and Excel statements; normalise heterogeneous column names into a single record schema | Data Ingestion |
| C2 | **Orchestrator** | Sequence the pipeline, assemble `AnalysisResult`, expose the REST interface, record run duration | Orchestration & API |
| C3 | **Validation Agent** | Five deterministic accounting identities, graded against a configurable materiality threshold | Validation Agent |
| C4 | **Trend Agent** | Year-on-year movement per line item; liquidity, leverage, profitability and return ratios | Trend Agent |
| C5 | **Anomaly Agent** | Cross-figure contradictions and an unsupervised peer-outlier model. Multi-year recurring issues are found separately by `analysis/recurring_issues.py` (Orchestration & API) | Anomaly Agent |
| C5a | **Recurring Issues** | The same failed check, anomaly or deviation for one company in 3 or more years | Orchestration & API |
| C5b | **Peer Comparison** | Company-years sitting far outside their industry and year on margins, returns, liquidity and leverage. Reported to the reviewer, deliberately kept out of the risk score: being unlike one's peers is a reason to look, not a defect | Orchestration & API |
| C6 | **Evidence Agent** | Assemble the verified evidence packet; enforce that nothing unverified reaches the model | Evidence & Review |
| C7 | **Review Agent** | Generate executive summary, per-finding commentary and recommended actions | Evidence & Review |
| C8 | **Risk Scorer** | Explainable 0–100 score with per-finding point attribution | Risk & Reporting |
| C9 | **Reporting** | PDF audit report; SQLite persistence of sessions, findings and reviewer actions | Risk & Reporting |
| C10 | **Presentation** | Seven-screen review workspace with human-in-the-loop controls | Frontend |

### 5.1 Validation rules

| Rule | Identity |
|:--|:--|
| VAL_BS_01 | Total Assets = Total Liabilities + Shareholders' Equity |
| VAL_GP_02 | Gross Profit = Revenue − Cost of Revenue |
| VAL_OP_03 | Operating Income = Gross Profit − Operating Expenses |
| VAL_NI_04 | Net Income = Pre-tax Income − Taxes |
| VAL_CF_05 | Ending Cash = Beginning Cash + Operating + Investing + Financing |

Each rule emits expected value, actual value, difference, severity, the formula applied, and a human-readable evidence string.

---

## 6. Data flow

1. The reviewer uploads a statement set, or selects a demo case.
2. Ingestion parses and normalises it into `FinancialRecord` objects, one per company-year.
3. The orchestrator dispatches those records to the three analysis agents.
4. Each agent returns its own typed findings.
5. The Evidence Agent merges them into a single evidence packet containing only computed figures.
6. The Review Agent turns that packet into narrative.
7. The Risk Scorer attributes points to each finding and produces an overall score.
8. Results are persisted, rendered in the workspace, and exportable as a PDF.
9. The reviewer accepts, challenges or annotates each finding; those actions are persisted.

---

## 7. Data contracts

These six objects are the interfaces between components. They are frozen at the start of the build; new fields must carry defaults so downstream consumers do not break.

| Object | Defined in | Produced by | Consumed by |
|:--|:--|:--|:--|
| `FinancialRecord` | `extraction/pipeline.py` | Ingestion | All three analysis agents |
| `ValidationResult` | `validation_agent/models.py` | Validation Agent | Evidence, Risk |
| `YoYResult` | `analysis/yoy_analysis.py` | Trend Agent | Evidence, Anomaly |
| `AnomalyFinding` | `finsight/core/models.py` | Anomaly Agent | Evidence, Risk |
| Evidence packet | `agents/evidence_agent.py` | Evidence Agent | Review Agent |
| `AnalysisResult` | `agents/orchestrator.py` | Orchestrator | Frontend, Reporting |

---

## 8. Technology stack

| Layer | Technology | Rationale |
|:--|:--|:--|
| Language | Python 3.11+ | Ecosystem fit for data and AI work |
| Data | pandas, numpy | Tabular manipulation |
| Parsing | openpyxl | Excel ingestion (PDF was ruled out of scope on 12 Sep) |
| ML | scikit-learn (IsolationForest) | Unsupervised peer-outlier detection |
| Review model | flan-t5-base fine-tuned by the team (Transformers, PyTorch); deterministic writer in the running service | Our own model; no hosted AI service and no API key |
| API | FastAPI, uvicorn | Typed, documented REST surface |
| UI | HTML, CSS, JavaScript served by FastAPI | The reviewer's screens come from the same service as the API |
| Reporting | ReportLab | Programmatic PDF generation |
| Storage | SQLite | Zero-configuration persistence |
| Packaging | Docker | Reproducible runtime |
| CI/CD | GitHub Actions | Tests and image build on every push; manual deploy to AWS through a short-lived OIDC role |
| Testing | pytest | Unit, integration and benchmark suites |

---

## 9. Key design decisions

### D1 — The language model performs no arithmetic

**Decision:** all figures are computed in deterministic Python. The model receives verified numbers and writes only prose.

**Alternatives considered**

| Option | Why rejected |
|:--|:--|
| Submit the statement to an LLM and ask it to review | Arithmetic is unreliable and failures are silent. In a review context a hallucinated total is worse than no tool at all. |
| LLM with a calculator tool it may invoke | Reintroduces non-determinism and latency. The checks are known in advance, so there is nothing for the model to decide. |

**Consequence:** results are reproducible, auditable, and testable. The model becomes replaceable without affecting correctness.

### D2 — Rules, not a classifier, for validation

**Decision:** validation is implemented as exact accounting identities.

**Alternative:** train a supervised model to predict misstatement. Rejected — no labelled corpus exists, and the relationships are exact identities. A model would be strictly worse than arithmetic.

### D3 — API-first, with the UI as a client

**Decision:** the review engine is exposed as `POST /review`; the workspace consumes that interface.

**Alternative:** a Streamlit monolith. Rejected — it makes the engine unreusable and untestable in isolation, and offers no integration path into an existing finance system.

### D4 — Unsupervised anomaly detection

**Decision:** IsolationForest over ratio vectors, benchmarked against a z-score baseline.

**Alternative:** supervised classification. Rejected — requires labelled anomalies that do not exist for this dataset.

### D5 — Our own model, no hosted AI

**Decision:** no hosted AI service is called. The review narrative is written by a deterministic writer, and a `flan-t5-base` model fine-tuned on our own finding logic is trained and measured (`training/results/`), ready to be served behind a flag.

**Alternative:** a hosted model such as Gemini or OpenAI. Rejected — financial statements are confidential and would leave the system, every request would carry a cost and an API key to protect, and the output would not be reproducible. Measuring our own model found 12.3% of its numbers unsupported, which is why every figure in a written review is checked against the evidence at runtime.

### D6 — Configurable materiality

**Decision:** severity is graded against a threshold expressed as a proportion of a benchmark figure.

**Alternative:** a fixed numeric threshold. Rejected — materiality is scale-dependent; a fixed figure is meaningless across companies of different sizes.

### D7 — Synchronous execution

**Decision:** the pipeline runs synchronously; a full review completes in seconds.

**Alternative:** a queue-backed asynchronous pipeline. Deferred to the roadmap — justified only at portfolio-scale batch volumes.

---

## 10. Non-functional requirements

| Attribute | Target | How it is met |
|:--|:--|:--|
| **Performance** | Deterministic engine under 1s per company-year | Vectorised pandas; no network calls in the engine |
| **Latency** | Full review, including narrative, in seconds | Single model call over a compact evidence packet |
| **Scalability** | Stateless API, horizontally scalable | No server-side session state; storage externalised |
| **Availability** | Functional without internet access | Offline heuristic reviewer |
| **Auditability** | Every finding traceable to its formula and inputs | Evidence and formula fields on every result |
| **Reproducibility** | Identical input yields identical findings | Deterministic engine; fixed seeds in the benchmark |
| **Maintainability** | A component can be replaced without touching others | Frozen data contracts; contract tests in CI |
| **Security** | No credentials in source; no data egress by default | Secrets via environment; offline mode sends nothing |

---

## 11. Deployment architecture

### 11.1 Environments

| Environment | Purpose | Runs on | Data |
|:--|:--|:--|:--|
| **Local** | Development | Developer machine, `uvicorn api.main:app --reload` | Demo CSVs, local SQLite |
| **Container test** | The exact image, before it is released | Docker Desktop, `docker run -p 8000:8000 auditlens` | A fresh SQLite database inside the container |
| **Demo** | The live URL shown to the panel | AWS ECS Express Mode on Fargate, `ap-south-1`, public HTTPS | Seeded demo data; SQLite that resets on every deploy |

There is no production environment. The demo environment is treated *as if* it were production - same image, same configuration mechanism - so the deployment story is honest rather than aspirational.

### 11.2 Runtime topology

```
        Internet  (HTTPS, certificate managed by AWS)
                         |
         Application Load Balancer  --  GET /health every 30 s
                         |
                 HTTP to port 8000
                         |
            +------------+------------+
            |  Fargate task (1 of 1)  |
            |                         |
            |  uvicorn :8000          |
            |       |                 |
            |  FastAPI app -----------+-- /auth, /review, /reviews, /monitoring   (API)
            |       |                 +-- /  static files from ui/                (screens)
            |       |                 |
            |  review engine          |
            |       |                 |
            |  SQLite (task storage)  |
            +-------------------------+
```

**One image, one process, one port.** FastAPI serves both the API and the reviewer's screens, which are static HTML, CSS and JavaScript from `ui/`. The screens call the API on the same origin, so there is no cross-origin configuration to get wrong, and the load balancer routes a single port.

**Rejected alternative:** a separate web server for the screens. It adds a second process and a second port to keep in step, for files that never change at runtime.

### 11.3 Build and release pipeline

```
  code on main
     |
  python -m pytest                  <- every test must pass
     |
  docker build, run locally          <- /health must answer before release
     |
  docker push to ECR                 <- tagged with the commit or a version
     |
  aws ecs update-express-gateway-service  (image only)
     |
  canary deployment  ->  /health passes  ->  live
```

The same steps run by hand (`docs/DEPLOYMENT.md`, section 5) or from GitHub Actions (`.github/workflows/deploy-aws.yml`), which also waits for the service to stabilise and checks the live `/health`. The anomaly benchmark is run on demand with `python -m finsight --benchmark`.

### 11.4 Configuration and secrets

All configuration is by environment variable, never in the image:

| Variable | Purpose | Required |
|:--|:--|:--|
| `AUTH_REQUIRED` | Sign-in for review endpoints; `false` only for local testing | No - defaults to `true` |
| `AUTH_TOKEN_HOURS` | How long a sign-in lasts | No - defaults to 12 |
| `SQLITE_DB_PATH` | Database location | No |
| `MAX_UPLOAD_MB` | Largest upload accepted | No - defaults to 10 |
| `LOG_LEVEL` | Logging verbosity | No |

**No secrets are needed.** The application calls no hosted AI service, and GitHub deploys through a short-lived OIDC role rather than stored AWS keys. `.env.example` documents every variable; `.env` is gitignored.

### 11.5 Hosting

**Chosen: AWS ECS Express Mode.** The image is stored in **Amazon ECR** and run by **ECS Express Mode** on Fargate, which creates the load balancer, HTTPS certificate, security groups, scaling and deployment rollback from three inputs: an image, a task execution role and an infrastructure role.

| AWS service | Role |
|:--|:--|
| **ECR** | Stores the container images; every release is a tagged, reusable rollback point |
| **ECS Express Mode (Fargate)** | Runs the container; canary deployments with automatic rollback |
| **Application Load Balancer + ACM certificate** | Created by Express Mode; terminates HTTPS and health-checks `/health` |
| **IAM** | Task execution role, infrastructure role (with an added read-only inline policy), and an optional GitHub deploy role |
| **CloudWatch Logs** | Receives the application's output |
| **AWS Budgets** | Cost alert at $10 a month |

**Alternatives considered**

| Option | Verdict |
|:--|:--|
| **App Runner** | The original choice. No longer accepts new customers since 30 April 2026; AWS points to ECS Express Mode. |
| **ECS Fargate + ALB configured by hand** | The same runtime as Express Mode, with a day of networking setup. Express Mode builds it with sensible defaults. |
| **EC2 `t3.micro` + Docker** | Cheaper, but TLS, restarts and deploys all become manual. |
| **AWS Lambda** | Rejected - the review service is a long-lived server process, not a request-scoped function. |
| **Amplify** | Rejected - the screens are served by the same container, so a separate frontend host would split one deployment into two. |

**Known constraints, all handled**

- **Cost.** The load balancer costs about $16 a month on its own, plus the running task. A budget alert is set, and the service is deleted once evaluation is over.
- **The task's storage is ephemeral.** SQLite resets on every deploy, so demo reviews are seeded at startup. Maximum tasks is fixed at 1, because each task would otherwise keep its own database. Migrating to RDS PostgreSQL is the roadmap item that removes both constraints.
- **A missing permission in AWS's managed policy.** The infrastructure role could not provision the load balancer without `ec2:DescribeAccountAttributes`; a read-only inline policy adds it (`docs/DEPLOYMENT.md`, section 3.6).

Naming these limits explicitly is better than discovering one on stage.

### 11.6 Health, readiness and rollback

- **Health check:** the load balancer polls `GET /health` every 30 seconds; a task needs 5 passes before it receives traffic. The Dockerfile `HEALTHCHECK` also checks inside the container.
- **Deployments:** canary (5% of traffic for 3 minutes) with a circuit breaker that rolls back, so a failing version does not replace a working one.
- **Manual rollback:** point the service at an earlier image tag with `aws ecs update-express-gateway-service`.
