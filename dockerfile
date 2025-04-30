# Dockerfile

FROM python:3.10-slim

# ——— Environment variables ———
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ——— Install OS dependencies & build tools ———
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libssl-dev \
      libffi-dev \
      libpq-dev \
      libpq5 \
      curl \
 && rm -rf /var/lib/apt/lists/*

# ——— Create non-root user ———
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# ——— Copy & install Python dependencies ———
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ——— Copy app code & switch to appuser ———
COPY --chown=appuser:appuser . .

USER appuser

# ——— Healthcheck & ports ———
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

EXPOSE ${STREAMLIT_SERVER_PORT}

# ——— Entrypoint ———
CMD ["streamlit", "run", "streamlit_dashboard.py", "--logger.level=info"]
