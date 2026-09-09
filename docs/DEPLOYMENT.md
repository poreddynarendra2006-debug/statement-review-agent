# Deployment Runbook

FinSight AI · Team 33 · **Owner: Orchestration & API (Member 1)**

Everything needed to take the application from a laptop to a public URL on AWS, and to survive demo day.

---

## 1. What we deploy

One container image running two processes:

| Process | Port | Purpose |
|:--|:--|:--|
| Streamlit | 8501 | The reviewer workspace — the demo surface |
| uvicorn / FastAPI | 8000 | `POST /review`, `GET /health` — the integration surface |

They share the review engine in-process rather than calling each other over the network. `entrypoint.sh` starts uvicorn in the background and Streamlit in the foreground, so the container's life follows the UI process.

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
docker run -p 8501:8501 -p 8000:8000 --env-file .env finsight-ai
```

Always run the image locally before pushing. A container that works on your machine and fails in App Runner is almost always a missing environment variable, and that is much faster to find locally.

---

## 6. Health and rollback

- **Health check:** App Runner polls `/_stcore/health` (Streamlit's own endpoint). Traffic is not routed to an unhealthy container.
- **Readiness:** the review engine has no warm-up, so healthy means ready.
- **Rollback:** every image is tagged with its commit SHA. To roll back, deploy the previous tag. Because images are built in CI and never on a laptop, yesterday's artefact is exactly reproducible.
- **If the container will not start at all:** check CloudWatch Logs first, then run the same image locally with the same environment variables.

---

## 7. Known limits

| Limit | Consequence | What we do |
|:--|:--|:--|
| Ephemeral filesystem | SQLite resets on restart | Seed demo data at startup so a cold container is demonstrable |
| Single-writer SQLite | Concurrent reviewers would contend | Documented; RDS PostgreSQL is the roadmap item |
| Cold start after idle | First request is slow | Warm the URL before the demo |
| LLM provider rate limits | AI narrative may fail under load | Falls back to the offline reviewer rather than erroring |

---

## 8. Fallback: EC2 Free Tier

If there are no AWS credits, `t3.micro` is free for 750 hours a month for twelve months.

1. Launch `t3.micro` with Amazon Linux 2023
2. Install Docker, pull the image from ECR
3. Run the container with `--restart unless-stopped`
4. Put nginx in front for TLS, or accept plain HTTP for an internal demo

Cheaper, but TLS, restarts and deploys all become manual. Decide which route on day one — it changes how the CI workflow ends.

---

## 9. Demo-day runbook

| When | Do |
|:--|:--|
| **T−30 min** | Open the live URL to wake the container from idle |
| **T−25 min** | Run one full review end to end — findings appear, PDF downloads |
| **T−20 min** | Check `GET /health` returns 200 |
| **T−15 min** | Confirm the local container also runs — fallback route two |
| **T−10 min** | Confirm the recorded walkthrough is on the laptop — fallback route three |
| **T−5 min** | Close every other tab. Disable notifications. |
| **On failure** | Switch to local, then to the recording. **Do not debug in front of the panel.** |

Three independent routes to a working demo — hosted, local, recorded. They fail for different reasons, which is the point of having all three.

---

## 10. After evaluation

**Stop the App Runner service.** It bills while it runs, and nobody remembers to check a week later.

```bash
aws apprunner pause-service --service-arn <service-arn> --region ap-south-1
```

Keep the ECR image — it costs almost nothing and means the project can be brought back up for the technical interview if anyone asks to see it.
