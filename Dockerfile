# FinSight AI - one container, one process, one port.
#
# FastAPI serves the review API and, once the front end is in ui/, the
# reviewer's screens from the same port. Every container host we might deploy
# to routes a single port, and there is no second server to keep in step.
#
# Build:  docker build -t finsight-ai .
# Run:    docker run -p 8000:8000 --env-file .env finsight-ai
#
# No API key is required.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    SQLITE_DB_PATH=/app/var/finsight_review.db

WORKDIR /app

# Dependencies first, so editing source does not invalidate the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user with a writable folder for the SQLite database.
RUN chmod +x entrypoint.sh \
 && useradd --create-home --uid 10001 finsight \
 && mkdir -p /app/var \
 && chown -R finsight:finsight /app
USER finsight

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT','8000'), timeout=4).status == 200 else 1)"

ENTRYPOINT ["./entrypoint.sh"]
