"""Guard the container and deployment configuration against drifting back to old designs.

The deployment was first written for a Streamlit app, then for AWS App Runner.
The project has neither now: one FastAPI container on ECS Express Mode. These
checks fail the build if anything reintroduces them, or breaks the single-port
setup the load balancer relies on.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_no_streamlit_left_in_the_container():
    for name in ("Dockerfile", "entrypoint.sh", "requirements.txt"):
        assert "streamlit" not in read(name).lower(), f"{name} still refers to Streamlit"


def test_container_exposes_one_port_with_a_real_health_check():
    dockerfile = read("Dockerfile")
    assert re.search(r"^EXPOSE 8000\s*$", dockerfile, re.M)
    assert "/health" in dockerfile
    assert "_stcore" not in dockerfile


def test_entrypoint_starts_the_api_on_the_hosts_port():
    script = read("entrypoint.sh")
    assert "uvicorn api.main:app" in script
    assert "${PORT:-8000}" in script
    assert "exec uvicorn" in script, "uvicorn must be process 1 to receive stop signals"


def test_entrypoint_keeps_unix_line_endings():
    assert b"\r\n" not in (ROOT / "entrypoint.sh").read_bytes(), \
        "CRLF line endings make the container fail with 'exec format error'"


def test_deploy_workflow_targets_ecs_express_mode():
    workflow = read(".github/workflows/deploy-aws.yml")
    assert "update-express-gateway-service" in workflow
    assert "python -m pytest" in workflow, "never deploy without running the tests"
    assert "/health" in workflow
    assert "apprunner" not in workflow.lower()


def test_no_app_runner_configuration_is_left():
    assert not (ROOT / "apprunner.yaml").exists()
    for name in (".github/workflows/deploy-aws.yml", ".env.example"):
        assert "app runner" not in read(name).lower() and "apprunner" not in read(name).lower()


def test_runbook_documents_the_settings_the_service_depends_on():
    runbook = read("docs/DEPLOYMENT.md")
    for setting in ("8000", "/health", "Maximum number of tasks", "AllowLoadBalancerLookups"):
        assert setting in runbook, f"docs/DEPLOYMENT.md no longer mentions {setting!r}"


def test_upload_and_serving_dependencies_are_listed():
    requirements = read("requirements.txt").lower()
    for package in ("fastapi", "uvicorn", "python-multipart"):
        assert package in requirements


def test_unused_cloud_and_ai_sdks_stay_out_of_the_image():
    requirements = read("requirements.txt").lower()
    for package in ("openai", "google-generativeai", "boto3"):
        assert package not in requirements, f"{package} is not imported anywhere; keep it out of the image"
