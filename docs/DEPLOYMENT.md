# Deployment Runbook

FinSight AI · Team 33 · **Owner: Orchestration & API (Member 1)**

Everything needed to take the application from a laptop to a public URL on AWS, and to survive demo day.

---

## 1. What we deploy

One container image running one process on one port:

| Path | Purpose |
|:--|:--|
| `/` | The reviewer's screens — static HTML, CSS and JavaScript from `ui/` |
| `/review`, `/review/upload` | Run a review from JSON records, or from an uploaded CSV / Excel file |
| `/reviews`, `/reviews/{id}/report.pdf`, `/reviews/{id}/actions` | Saved reviews, the PDF audit report, reviewer actions |
| `/monitoring/...` | Run summary, stage timings, agent coverage |
| `/health` | Liveness check for the container host |
| `/docs` | Interactive API documentation |

FastAPI serves the API and the screens together. The screens call the API on the same origin, and every container host we might use routes a single port. The listening port comes from `$PORT`, defaulting to 8000.

On startup the service creates the SQLite tables and seeds demo reviews when the database is empty, so a freshly started container never shows an empty history.

---

## 2. AWS services used

| Service | Why |
|:--|:--|
| **ECR** | Stores the image. CI pushes every build here. |
| **App Runner** | Runs the container. Managed HTTPS, health checks, autoscaling, no cluster to operate. |
| **Secrets Manager** | Holds `GEMINI_API_KEY`. Injected at runtime, never baked into the image. |
| **S3** | Generated PDF reports and uploaded statements. |
| **CloudWatch Logs** | Application logs, and the source for the monitoring view. |
| **IAM + GitHub OIDC** | CI assumes a role to push and deploy — no long-lived AWS keys stored in GitHub. |

> **Cost warning.** App Runner is *not* Free Tier. Budget roughly **$5–25/month** while it runs. If the team has no AWS credits, use the EC2 fallback in section 8. **Stop the service after evaluation** either way.

---

## 3. One-time AWS setup

Do this once, early. It is the part that always takes longer than expected.

### 3.1 Create the image registry

```bash
aws ecr create-repository \
  --repository-name finsight-ai \
  --region ap-south-1
```

Note the returned `repositoryUri` — CI needs it.

### 3.2 Store the API key

```bash
aws secretsmanager create-secret \
  --name finsight/gemini-api-key \
  --secret-string "YOUR_KEY_HERE" \
  --region ap-south-1
```

### 3.3 Create the S3 bucket for reports

```bash
aws s3 mb s3://finsight-reports-team33 --region ap-south-1
```

### 3.4 Let GitHub deploy without stored keys

Create an IAM **OIDC identity provider** for `token.actions.githubusercontent.com`, then a role trusted by it, restricted to this repository. Attach permissions for ECR push and App Runner deploy.

This is worth doing properly rather than pasting an access key into GitHub secrets — an exposed long-lived AWS key is the single most common way student projects get compromised, and the panel may well ask how CI authenticates.

---

## 4. Configuration

Every setting is an environment variable. **Nothing is baked into the image.**

| Variable | Purpose | Required |
|:--|:--|:--|
| `LLM_PROVIDER` | `gemini`, `openai` or `heuristic` | No — defaults to `heuristic` |
| `GEMINI_API_KEY` | Provider credential, from Secrets Manager | No |
| `OPENAI_API_KEY` | Alternative provider credential | No |
| `SQLITE_DB_PATH` | Database file location | No |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | No |
| `AWS_REGION` | Region for S3 and logs | No |
| `AUTH_REQUIRED` | `false` lets review endpoints run without signing in - local testing only | No - defaults to `true` |
| `AUTH_TOKEN_HOURS` | How long a sign-in lasts | No - defaults to `12` |

**The container must start successfully with none of these set.** Without a key it runs the offline heuristic reviewer. This is deliberate: an expired or revoked credential degrades the system instead of breaking it, and the demo cannot fail because of a billing problem.

---

## 5. Deploy

### From CI (normal path)

Push to `main`. The workflow runs tests, builds the image, pushes to ECR, and tells App Runner to deploy.

### By hand (when CI is broken and time is short)

```bash
aws ecr get-login-password --region ap-south-1 \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com

docker build -t finsight-ai .
docker tag finsight-ai:latest <account>.dkr.ecr.ap-south-1.amazonaws.com/finsight-ai:latest
docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/finsight-ai:latest

aws apprunner start-deployment --service-arn <service-arn> --region ap-south-1
```

### Locally, to check the image before pushing

```bash
docker build -t finsight-ai .
docker run -p 8000:8000 --env-file .env finsight-ai
```

Always run the image locally before pushing. A container that works on your machine and fails in App Runner is almost always a missing environment variable, and that is much faster to find locally.

---

## 6. Health and rollback

- **Health check:** the container host polls `GET /health` (set this as the App Runner health check path). Traffic is not routed to an unhealthy container.