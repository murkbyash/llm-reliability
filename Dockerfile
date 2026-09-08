# Multi-stage lightweight Dockerfile for LLM Reliability Analyzer

# Stage 1: Build wheel
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip build

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN python -m build --wheel --outdir /build/dist

# Stage 2: Minimal runtime image
FROM python:3.11-slim AS runner

LABEL org.opencontainers.image.title="LLM Reliability Analyzer" \
      org.opencontainers.image.description="Local-first root-cause diagnosis engine for LLM and RAG systems" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/murkbyash/llm-reliability"

# Create non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

WORKDIR /data

# Copy built wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Switch to non-root user
USER appuser:appgroup

ENTRYPOINT ["llm-reliability"]
CMD ["--help"]

