FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir setuptools wheel uv

# Copy metadata first
COPY pyproject.toml uv.lock README.md ./

# Copy the new organized folders
COPY clinical_triage/ ./clinical_triage/
COPY server/ ./server/

# Install the project as a package
RUN pip install --no-cache-dir .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# Run using the new entry point script
CMD ["clinical-triage-server"]