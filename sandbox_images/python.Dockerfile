# Base image for executing Python repository test suites.
# Kept deliberately thin: no compilers beyond what wheels commonly need, no
# curl/wget, no shell utilities that make exfiltration convenient.
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --create-home --shell /bin/sh sandbox \
 && mkdir -p /workspace \
 && chown -R 1000:1000 /workspace

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 1000:1000
WORKDIR /workspace
CMD ["python", "-m", "pytest", "-q"]
