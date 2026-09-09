# FinSight AI - one image, two processes.
#
# Streamlit serves the reviewer workspace on 8501; FastAPI serves the
# integration endpoint on 8000. They share the review engine in-process
# rather than calling each other over the network, so there is no reason
# to split them across two containers at this scale.
#
# Build:  docker build -t finsight-ai .
# Run:    docker run -p 8501:8501 -p 8000:8000 --env-file .env finsight-ai
#
# No API key is required. Without one the application falls back to its
# offline reviewer and stays fully functional.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    API_PORT=8000 \
    UI_PORT=8501

WORKDIR /app

# Dependencies first, so editing source does not invalidate the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

# Run as a non-root user. App Runner does not require it, but a container
# that needs root to start is a finding waiting to happen.
RUN useradd --create-home --uid 10001 finsight && chown -R finsight:finsight /app
USER finsight

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:'+os.environ.get('UI_PORT','8501')+'/_stcore/health', timeout=4).status==200 else sys.exit(1)"

ENTRYPOINT ["./entrypoint.sh"]
