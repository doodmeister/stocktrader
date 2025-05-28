FROM python:3.10-slim

# ——— Environment variables ———
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# ——— Install build dependencies and cleanly install pip packages ———
COPY requirements.txt .  # ✅ Caching benefit here

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libssl-dev \
      libffi-dev \
      libpq-dev \
      libpq5 \
      curl \
 \
 && python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt \
 \
 && apt-get purge -y --auto-remove \
      build-essential \
      libssl-dev \
      libffi-dev \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/* /root/.cache

# ——— Copy project code last to avoid invalidating pip cache ———
COPY . .

# ——— App user and file permissions ———
RUN useradd -m -s /bin/bash appuser \
 && touch /app/trading.log \
 && chown -R appuser:appuser /app

USER appuser

# ——— Healthcheck & port exposure ———
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

EXPOSE ${STREAMLIT_SERVER_PORT}

# ——— Entrypoint ———
CMD ["streamlit", "run", "streamlit_dashboard.py", "--logger.level=info"]
