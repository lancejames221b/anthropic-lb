FROM python:3.12-slim AS base

LABEL maintainer="Lance James, Unit 221B, Inc."
LABEL description="anthropic-lb — Usage-aware load balancer for the Anthropic API with PII redaction"

WORKDIR /app

# System deps for potential presidio/spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Presidio + spaCy for Phase 2 NER
ARG INSTALL_PRESIDIO=false
COPY requirements-presidio.txt .
RUN if [ "$INSTALL_PRESIDIO" = "true" ]; then \
    pip install --no-cache-dir -r requirements-presidio.txt && \
    python -m spacy download en_core_web_lg; \
    fi

COPY proxy.py .
COPY patterns.json .
COPY synthetic_pools.json .
COPY dashboard.html .

# Default config — override with env vars or docker-compose
ENV ANTHROPIC_LB_PORT=8891
ENV ANTHROPIC_LB_PII=regex
ENV ANTHROPIC_LB_PII_RESPONSE=detokenize
ENV ANTHROPIC_LB_PII_PATTERNS=/app/patterns.json
ENV ANTHROPIC_LB_PII_AUDIT_LOG=/app/logs/pii-audit.log
ENV ANTHROPIC_LB_KEYS=/app/keys.json

EXPOSE 8891

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8891/health || exit 1

VOLUME ["/app/logs"]

CMD ["python3", "proxy.py"]
