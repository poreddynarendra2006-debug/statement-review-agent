"""Guard the container configuration against drifting back to the old design.

The deployment was first written for a Streamlit app that the project no longer
has. These checks fail the build if anything reintroduces it, or breaks the
single-port setup every container host relies on.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_no_streamlit_left_in_the_container():
    for name in ("Dockerfile", "entrypoint.sh", "apprunner.yaml", "requirements.txt"):
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


def test_app_runner_routes_the_api_port():
    assert re.search(r"port:\s*8000", read("apprunner.yaml"))


def test_upload_and_serving_dependencies_are_listed():
    requirements = read("requirements.txt").lower()
    for package in ("fastapi", "uvicorn", "python-multipart"):
        assert package in requirements
