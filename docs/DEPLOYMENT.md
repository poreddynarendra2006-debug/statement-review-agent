# Deployment Runbook

AuditLens · Team 33 · **Owner: Orchestration & API**

How AuditLens runs on AWS, how to ship a new version, and how to recover on demo day.

## Current deployment

| | |
|:--|:--|
| **Region** | `ap-south-1`, Asia Pacific (Mumbai) |
| **Service** | ECS Express Mode service `auditlens` in the `default` cluster |
| **Image** | Amazon ECR repository `statement-review`; the first release is tag `v1` |
| **Live URL** | https://au-977afd65e02e450082ab5c24355f6500.ecs.ap-south-1.on.aws (changes only if the service is deleted and created again) |
| **Size** | 1 vCPU, 2 GB memory, exactly 1 running task |
| **Cost alert** | AWS Budget "AuditLens Project", $10 a month, with email alerts |

---

## 1. What we deploy

One container image running one process on one port (8000):

| Path | Purpose |
|:--|:--|
| `/` | The reviewer's screens - static HTML, CSS and JavaScript from `ui/` |
| `/auth/register`, `/auth/login`, `/auth/me`, `/auth/logout` | Reviewer accounts and sign-in |
| `/review`, `/review/upload` | Run a review from JSON records, or from an uploaded CSV / Excel file |
| `/reviews`, `/reviews/{id}/report.pdf`, `/reviews/{id}/actions` | Saved reviews, the PDF report, reviewer actions |
| `/monitoring/...` | Run summary, stage timings, agent coverage |
| `/health` | Liveness check for the load balancer |
| `/docs` | Interactive API documentation |

Review, history and monitoring endpoints need a sign-in token. `/health`, `/docs` and the screens are public.

On startup the app creates its SQLite tables and seeds demo reviews, so a fresh container never shows an empty history.

---

## 2. AWS services used

| Service | What it does for us |
|:--|:--|
| **Amazon ECR** | Stores the container images. Every release is a tagged image. |
| **Amazon ECS Express Mode** (on Fargate) | Runs the container. It also creates the Application Load Balancer with an HTTPS certificate, the security groups, target groups, auto scaling and canary deployments with automatic rollback. |
| **IAM** | `ecsTaskExecutionRole` lets ECS pull the image and write logs. `ecsInfrastructureRoleForExpressServices` lets ECS build the load balancer and networking; it carries one extra inline policy (section 3.6). |
| **CloudWatch Logs** | The app's output. Open it from the service page's **Logs** tab. |
| **AWS Budgets** | Emails an alert before costs pass the limit. |

**Not used, on purpose**

| Service | Why not |
|:--|:--|
| App Runner | No longer accepts new customers since 30 April 2026. AWS recommends ECS Express Mode instead. |
| Secrets Manager | The app uses no API keys. |
| S3 | PDF reports are generated on request from the saved review. |
| Amplify | The screens are served by the same container as the API. |

> **Cost.** Express Mode itself has no charge. You pay for the load balancer (about $16 a month on its own), the running task (1 vCPU, 2 GB) and logs. **Delete everything after evaluation** (section 10).

---

## 3. First-time setup

This is exactly how the current deployment was built. Commands are for PowerShell.

### 3.1 Tools

- **Docker Desktop** - builds and runs the image.
- **AWS CLI v2, version 2.32 or newer** - needed for `aws login`.

Sign the CLI in with your console account. It stores temporary credentials, never access keys:

```powershell
aws login
```
```powershell
aws sts get-caller-identity
```

Turn off the CLI's page-by-page output for the session, or long results stop at `-- More --`:

```powershell
$env:AWS_PAGER = ""
```

### 3.2 Create the image repository

```powershell
aws ecr create-repository --repository-name statement-review --region ap-south-1
```

### 3.3 Build and test the image locally

```powershell
docker build -t auditlens .
```
```powershell
docker run --rm -p 8000:8000 --name auditlens-test auditlens
```

Open http://localhost:8000/health - every installed component should show `true`. Press Ctrl+C to stop.

### 3.4 Push the image

Run these in one PowerShell window; the variable is used by the later commands.

```powershell
$REGISTRY = "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.ap-south-1.amazonaws.com"; $REGISTRY
```
```powershell
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin $REGISTRY
```
```powershell
docker tag auditlens:latest "$REGISTRY/statement-review:v1"; docker push "$REGISTRY/statement-review:v1"
```

ECR may show two untagged entries next to `v1`. They are parts of the same image - do not delete them.

### 3.5 Create the Express Mode service (console)

**ECS console → Express mode**, with the region set to Mumbai:

| Field | Value | Why |
|:--|:--|:--|
| Image URI | **Browse ECR images** → `statement-review` → `v1` | The image to run |
| Task execution role | **Create new role** | Pull the image, write logs |
| Infrastructure role | **Create new role** | Build the load balancer and networking |
| Name | `auditlens` | Part of the resource names; cannot be changed later |
| Container port | `8000` | The port the app listens on. The default 80 fails every health check. |
| Health check path | `/health` | The load balancer calls it every 30 seconds |
| Environment variables | `AUTH_REQUIRED` = `true`, `LOG_LEVEL` = `INFO` | Sign-in stays on in the cloud |
| CPU / Memory | 1 vCPU / 2 GB | A 480-row review takes about 3 seconds |
| Minimum number of tasks | `1` | One copy always running |
| **Maximum number of tasks** | **`1`** | Each copy would have its own SQLite database, so with two or more a user signed in on one copy is asked to sign in again by another |
| Customise networking | unticked | Default VPC with a public HTTPS address |
| Logs | defaults | |

Click **Create**. If it reports *"Unable to assume the service linked role"*, wait a few seconds and click **Create** again.

### 3.6 The extra permission the load balancer needs

On the first deployment every load balancer resource stayed at **Provisioning** with:

```
AccessDenied: ... assumed-role/ecsInfrastructureRoleForExpressServices/ECSGateway
is not authorized to perform: ec2:DescribeAccountAttributes
```

AWS's managed policy `AmazonECSInfrastructureRoleforExpressGatewayServices` (version 6) does not include that action. Add it as an inline policy - every action is a read-only lookup:

**IAM → Roles → `ecsInfrastructureRoleForExpressServices` → Add permissions → Create inline policy → JSON**, name it `AllowLoadBalancerLookups`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowLoadBalancerLookups",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeAccountAttributes",
        "ec2:DescribeInternetGateways",
        "ec2:DescribeAvailabilityZones",
        "ec2:DescribeNetworkInterfaces"
      ],
      "Resource": "*"
    }
  ]
}
```

Express Mode keeps retrying, so nothing needs recreating: within a few minutes every resource turns **Active**, the deployment completes and `/health` answers.

### 3.7 Budget alert

**Billing and Cost Management → Budgets → Create budget**: monthly cost budget, $10, with an email alert (for example at 80% of actual spend). A budget only warns; it never stops the service.

---

## 4. Configuration

Every setting is an environment variable with a safe default. The image itself sets only `PORT=8000` and `SQLITE_DB_PATH=/app/var/finsight_review.db`; nothing secret is baked in.

| Variable | Default | Purpose |
|:--|:--|:--|
| `AUTH_REQUIRED` | `true` | Sign-in for review endpoints. `false` is for local testing only. |
| `AUTH_TOKEN_HOURS` | `12` | How long a sign-in lasts |
| `SQLITE_DB_PATH` | `/app/var/finsight_review.db` in the image | Where reviews, actions and accounts are stored |
| `MAX_UPLOAD_MB` | `10` | Largest upload accepted |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `PORT` | `8000` | Port the server listens on |
| `CORS_ORIGINS` | `*` | Only matters if the screens are served from another address |
| `UI_DIR` | `ui/` | Folder the screens are served from |
| `SEED_ACCOUNT_EMAIL`, `SEED_ACCOUNT_PASSWORD` | unset | An account recreated whenever the app starts, so a deploy doesn't sign the team out. Set both or neither. See 4.1. |
| `SEED_ACCOUNT_NAME`, `SEED_ACCOUNT_ROLE` | `AuditLens Reviewer`, `Senior Financial Auditor` | Name and role for that account |
| `GEMINI_API_KEY`, `OPENAI_API_KEY` | unset | **Leave unset.** By team decision no hosted AI service is used; if one is set, Data Ingestion's column mapper sends column names and sample values to that provider. |

To change a variable on AWS, use **Update service** on the service page in the ECS console.

### 4.1 The account that survives a deploy

Accounts are rows in a SQLite file **inside the task**, so replacing the task throws them away. That is what a deploy does, and on 12 September it signed the team out of the live service mid-afternoon — the account had been created that morning and the deploy was nobody's fault.

Setting `SEED_ACCOUNT_EMAIL` and `SEED_ACCOUNT_PASSWORD` on the service means that account is recreated every time the container starts, so there is always one sign-in that works, however many times we deploy.

What it does **not** do is save anyone else's account, or any saved review — those still go. The password is a service environment variable, readable by anyone with access to the ECS console, so treat it as a shared demo login rather than a personal one. An account somebody registered themselves is never overwritten.

The complete fix is storage that outlives the task — a persistent volume (EFS) mounted at `/app/var`, or a managed database. Both are in `docs/ROADMAP.md`.

---

## 5. Shipping a new version

Rebuild whenever code that runs in the container changes:

| Changed | Redeploy? |
|:--|:--|
| Agent folders, `agents/`, `api/`, `ui/`, `requirements.txt`, `Dockerfile`, `entrypoint.sh` | **Yes** |
| Only `tests/`, `docs/`, `training/` or README files | No - they are not in the image |

Run in one PowerShell window from the project folder:

**1. Get the latest code and make sure nothing is broken**
```powershell
git pull
```
```powershell
python -m pytest -q
```
```powershell
python -m scripts.check_agents
```

**2. Build and check the image**
```powershell
docker build -t auditlens .
```
```powershell
docker run --rm -p 8000:8000 --name auditlens-test auditlens
```

**3. Push it, tagged with the commit**
```powershell
$TAG = git rev-parse --short HEAD; $REGISTRY = "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.ap-south-1.amazonaws.com"
```
```powershell
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin $REGISTRY
```
```powershell
docker tag auditlens:latest "$REGISTRY/statement-review:$TAG"; docker push "$REGISTRY/statement-review:$TAG"
```

**4. Move the service onto it**
```powershell
aws ecs list-services --cluster default --region ap-south-1
```
```powershell
aws ecs update-express-gateway-service --service-arn "PASTE-SERVICE-ARN" --primary-container "image=$REGISTRY/statement-review:$TAG" --region ap-south-1
```
```powershell
aws ecs wait services-stable --cluster default --services auditlens --region ap-south-1
```

Only the image changes; port, health check, variables and scaling stay. The new version takes over without downtime.

**5. Check it** - open `<Live URL>/health`, then register again (see section 8).

---

## 6. Shipping from GitHub Actions (optional)

`.github/workflows/deploy-aws.yml` runs section 5 on GitHub: tests, build, push (tagged with the commit), update the service, wait until stable, check `/health`. Start it from **Actions → Deploy to AWS → Run workflow**. It needs this one-time setup.

### 6.1 Let GitHub sign in to AWS without stored keys

**IAM → Identity providers → Add provider → OpenID Connect**
- Provider URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

### 6.2 Create the deploy role

**IAM → Roles → Create role → Custom trust policy**, named for example `github-deploy-auditlens`. Replace `<account-id>`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Federated": "arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com" },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
        "StringLike": { "token.actions.githubusercontent.com:sub": "repo:poreddynarendra2006-debug/statement-review-agent:*" }
      }
    }
  ]
}
```

Give it this inline permissions policy - exactly the calls the workflow makes:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    {
      "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:InitiateLayerUpload",
                 "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage"],
      "Resource": "arn:aws:ecr:ap-south-1:<account-id>:repository/statement-review"
    },
    {
      "Effect": "Allow",
      "Action": ["ecs:UpdateExpressGatewayService", "ecs:DescribeServices"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": ["arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
                   "arn:aws:iam::<account-id>:role/ecsInfrastructureRoleForExpressServices"],
      "Condition": { "StringLike": { "iam:PassedToService": "ecs*.amazonaws.com" } }
    }
  ]
}
```

### 6.3 Repository variables

**GitHub → Settings → Secrets and variables → Actions → Variables** (none of these are secrets):

| Variable | Value |
|:--|:--|
| `AWS_REGION` | `ap-south-1` |
| `AWS_ROLE_ARN` | the deploy role's ARN |
| `ECR_REPOSITORY` | `statement-review` |
| `ECS_SERVICE_ARN` | `arn:aws:ecs:ap-south-1:<account-id>:service/default/auditlens` |
| `APP_URL` | the Live URL, without a trailing slash |

---

## 7. Health, logs and rollback

- **Health check** - the load balancer calls `GET /health` on port 8000 every 30 seconds. A task needs 5 passes in a row to receive traffic and is marked unhealthy after 2 failures. The Dockerfile's `HEALTHCHECK` also checks inside the container.
- **Deployments** - canary: 5% of traffic goes to the new version for 3 minutes first. A deployment circuit breaker with rollback is on, so a new version that fails its health checks does not replace a working one.
- **Logs** - the service page's **Logs** tab, or CloudWatch log group `/aws/ecs/default/auditlens-…` (a short suffix is added). The first line of a healthy start is `[entrypoint] starting AuditLens on :8000`.
- **Roll back by hand** - point the service at an earlier tag. Every tag in ECR is a rollback point:
  ```powershell
  aws ecs update-express-gateway-service --service-arn "PASTE-SERVICE-ARN" --primary-container "image=$REGISTRY/statement-review:v1" --region ap-south-1
  ```

---

## 8. Limits to know before a demo

- **Data resets on every deploy.** Reviews and accounts live in SQLite inside the task. Deploys are the only thing that has ever replaced it - an account made now still works tomorrow, as long as nobody ships. Set `SEED_ACCOUNT_EMAIL`/`SEED_ACCOUNT_PASSWORD` (section 4.1) so at least one sign-in always survives, and **deploy before the demo, not during it.**
- **Exactly one task.** Keep maximum tasks at 1 while storage is SQLite (section 3.5). Moving to a managed database (RDS PostgreSQL) removes both limits.
- **Large results.** A 480-row review is about 4.8 MB of JSON; responses are compressed to roughly a tenth of that on the wire.
- **A few seconds per review.** About 3 seconds for 480 company-years, most of it in Trend forecasting and Anomaly model training. The first request after a restart is slower while libraries load.

---

## 9. Troubleshooting

| What you see | Cause | Fix |
|:--|:--|:--|
| Load balancer, listener and certificate stuck at **Provisioning** with `AccessDenied ... ec2:DescribeAccountAttributes` | AWS's managed infrastructure policy lacks the action | Add the inline policy in section 3.6 |
| Deployment never finishes; targets unhealthy | Container port left at 80, or wrong health check path | Port `8000`, path `/health` |
| `no basic auth credentials` on `docker push` | Docker is not logged in to ECR in this window, or `$REGISTRY` was empty when logging in | Set `$REGISTRY`, run the `docker login` line again (the login lasts 12 hours) |
| `-- More --` in the terminal | The CLI pages long output | Press `q`; set `$env:AWS_PAGER = ""` |
| `/health` shows an older state after a deploy (for example `"recurring": false`) | The image was built before the latest code | Rebuild, push a new tag, update the service |
| `"Sign in to continue."` | Expected: review endpoints need a token | `/auth/register`, `/auth/login`, then **Authorize** in `/docs` |
| *"Unable to assume the service linked role"* when creating | First-time role setup delay | Wait a few seconds, click **Create** again |
| `/docs` shows *"Could not render responses"* for a large review | Image older than the docs fix | Redeploy the current code |
| Your laptop says the new address does not exist | Local DNS cache | Open it in the browser; it resolves within minutes |

---

## 10. Clean up after evaluation

1. **ECS → Clusters → default → Services → `auditlens` → Delete service.** Express Mode removes the load balancer once no service uses it.
2. **ECR → `statement-review` → Delete** the repository and its images.
3. **CloudWatch → Log groups** → delete `/aws/ecs/default/auditlens-…`.
4. Optional: delete the IAM roles `ecsTaskExecutionRole`, `ecsInfrastructureRoleForExpressServices` and the GitHub deploy role, and the budget.
